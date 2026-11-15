using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// Per-scene NeuS reconstruction (Wang et al. 2021): optimizes a <see cref="NeuSField{T}"/> against posed views with
/// the reference's loss, sampling, optimizer and learning-rate schedule (<c>exp_runner.py</c> <c>Runner.train</c>).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Each iteration takes the next view of a per-epoch random permutation, draws <c>BatchSize</c> pixels uniformly at
/// random, renders their rays and steps Adam (β = (0.9, 0.999), ε = 10⁻⁸, no clipping) on
/// <c>L = L1_color / Σmask + 0.1·Eikonal + 0.1·BCE(Σw, mask)</c>, with a linear warm-up over 5000 iterations and a cosine
/// decay to 5% at 300000. With <c>MaskWeight = 0</c> every pixel counts as foreground, as in the reference. An optional
/// normal term (<c>NormalWeight</c>) supervises the rendered normal with the views' normal maps, which is how
/// Wonder3D fuses its generated normals; it is off by default because NeuS itself has no normal supervision.
/// </para>
/// <para>
/// The rest of Wonder3D's geometry fusion (instant-nsr-pl <c>systems/neus_ortho.py</c>) is available through
/// <see cref="NeuSOptions"/> and off by default: a squared color term, per-view weights on the mask and normal terms,
/// its outlier-dropping ranking (each term keeps only the given fraction of its rays, smallest error first), the
/// geometry-aware normal weighting, the visibility cut on the target normals, and the sparsity and 3D
/// normal-smoothness regularizers on random points of the unit cube.
/// </para>
/// <para>
/// The views must already be in NeuS's normalized frame: the object inside the unit sphere around the origin
/// (the reference's <c>scale_mat</c> normalization).
/// </para>
/// <para><b>For Beginners:</b> Give it photos of one object from known camera positions and call <see cref="Fit"/>;
/// the result is a field you can sample on a grid and turn into a mesh with marching cubes.</para>
/// </remarks>
internal sealed class NeuSReconstructor<T>
{
    private readonly IEngine _engine;
    private readonly Random _random;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly Dictionary<NeuSView<T>, HostView> _hostViews = new();
    private int[] _permutation = Array.Empty<int>();
    private IReadOnlyList<NeuSView<T>>? _permutedViews;

    /// <summary>Creates a reconstructor with a freshly initialized field.</summary>
    /// <param name="options">The configuration, or null for the paper's.</param>
    /// <param name="engine">The engine, or null for the current one.</param>
    /// <param name="optimizerFactory">Builds the optimizer from the field's options, or null for the reference's Adam
    /// with its warm-up and cosine schedule.</param>
    public NeuSReconstructor(NeuSOptions? options = null, IEngine? engine = null,
        Func<NeuSOptions, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? optimizerFactory = null)
    {
        Options = options ?? new NeuSOptions();
        Options.Validate();
        _engine = engine ?? AiDotNetEngine.Current;
        _random = Options.Seed.HasValue ? RandomHelper.CreateSeededRandom(Options.Seed.Value) : RandomHelper.CreateSecureRandom();
        Field = new NeuSField<T>(Options, _random, _engine);
        _optimizer = optimizerFactory?.Invoke(Options) ?? CreateReferenceOptimizer(Options);
    }

    /// <summary>The configuration.</summary>
    public NeuSOptions Options { get; }

    /// <summary>The field being optimized.</summary>
    public NeuSField<T> Field { get; }

    /// <summary>The number of completed iterations.</summary>
    public int Iteration { get; private set; }

    /// <summary>Runs <paramref name="iterations"/> iterations (default <c>Options.Iterations</c>) and returns the field.</summary>
    /// <param name="views">The posed views.</param>
    /// <param name="iterations">The iteration count, or null for the configured total.</param>
    /// <param name="onStep">Receives each iteration's index and loss.</param>
    public NeuSField<T> Fit(IReadOnlyList<NeuSView<T>> views, int? iterations = null, Action<int, T>? onStep = null)
    {
        int count = iterations ?? Options.Iterations;
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(iterations));
        for (int i = 0; i < count; i++)
        {
            int iteration = Iteration;
            var loss = TrainStep(views);
            onStep?.Invoke(iteration, loss);
        }

        return Field;
    }

    /// <summary>One NeuS iteration; returns the loss before the update.</summary>
    /// <param name="views">The posed views.</param>
    public T TrainStep(IReadOnlyList<NeuSView<T>> views)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        if (views.Count == 0) throw new ArgumentException("At least one view is needed.", nameof(views));
        var view = NextView(views);
        var host = Host(view);
        var batch = DrawBatch(host, Options.BatchSize);
        int perturbSeed = _random.Next();
        // Drawn after the perturbation seed, and only when a term reads them, so a plain NeuS run draws exactly the
        // reference's sequence.
        if (Options.SparsityWeight > 0 || Options.NormalSmoothWeight > 0) DrawRegularizationPoints(batch);
        double annealRatio = Options.CosineAnnealRatio(Iteration);

        Tensor<T> Objective()
        {
            var perturb = Options.Perturb ? RandomHelper.CreateSeededRandom(perturbSeed) : null;
            return Loss(batch, Field.Render(batch.Origins, batch.Directions, annealRatio, perturb));
        }

        var loss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => Field.Parameters, Objective, _optimizer, applyOptimizerRegularization: false);
        Iteration++;
        if (_optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled) scheduled.StepScheduler();
        return loss;
    }

    /// <summary>
    /// The training loss for one rendered batch: NeuS's color, Eikonal and mask terms, plus the normal, squared-color,
    /// sparsity and smoothness terms of Wonder3D's geometry fusion where their weights are positive.
    /// </summary>
    internal Tensor<T> Loss(NeuSRayBatch batch, NeuSRenderResult<T> render)
    {
        var e = _engine;
        var o = Options;
        int rays = batch.Count;
        var allRays = new double[rays];
        for (int r = 0; r < rays; r++) allRays[r] = 1.0;
        var colorRays = o.ColorOnForegroundOnly ? batch.Mask : allRays;
        var target = NeuSHost<T>.FromDoubles(batch.Colors, rays, 3);
        var colorError = e.TensorSubtract(render.Color, target);

        // NeuS: Σ|ĉ − c| over the foreground rays / (Σ mask + 1e-5).
        var loss = e.TensorMultiplyScalar(render.GradientError, NeuSHost<T>.Scalar(o.EikonalWeight));
        if (o.ColorL1Weight > 0)
        {
            var perRay = e.ReduceSum(e.TensorAbs(colorError), new[] { 1 }, true);
            loss = e.TensorAdd(loss, Weighted(RankedMean(perRay, colorRays, o.ColorKeepRatio, 1.0, 1e-5), o.ColorL1Weight));
        }

        // Wonder3D: F.mse_loss(..., reduction='none').sum(dim=1), ranked with rgb_p_ratio, mean.
        if (o.ColorMseWeight > 0)
        {
            var perRay = e.ReduceSum(e.TensorMultiply(colorError, colorError), new[] { 1 }, true);
            loss = e.TensorAdd(loss, Weighted(RankedMean(perRay, colorRays, o.ColorKeepRatio, 1.0, 1e-5), o.ColorMseWeight));
        }

        if (o.MaskWeight > 0)
        {
            var mask = NeuSHost<T>.FromDoubles(batch.Mask, rays, 1);
            var w = e.TensorClamp(render.WeightSum, NeuSHost<T>.Scalar(1e-3), NeuSHost<T>.Scalar(1.0 - 1e-3));
            var oneMinusMask = e.TensorAddScalar(e.TensorNegate(mask), NeuSHost<T>.Scalar(1.0));
            var oneMinusW = e.TensorAddScalar(e.TensorNegate(w), NeuSHost<T>.Scalar(1.0));
            var logLikelihood = e.TensorAdd(e.TensorMultiply(mask, e.TensorLog(w)), e.TensorMultiply(oneMinusMask, e.TensorLog(oneMinusW)));
            // Per-ray binary cross-entropy, averaged over the kept rays (all of them in NeuS) and scaled by the view's
            // weight (Wonder3D's ranking_loss(..., extra_weights=view_weights)).
            var bce = RankedMean(e.TensorNegate(logLikelihood), allRays, o.MaskKeepRatio, batch.ViewWeight, 0.0);
            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(bce, NeuSHost<T>.Scalar(o.MaskWeight)));
        }

        if (o.NormalWeight > 0 && batch.Normals is not null)
        {
            var targetNormals = NeuSHost<T>.FromDoubles(batch.Normals, rays, 3);
            var dot = e.ReduceSum(e.TensorMultiply(render.Normal, targetNormals), new[] { 1 }, true);
            // F.cosine_similarity: x·y / max(‖x‖‖y‖, 1e-8).
            var denominator = e.TensorMax(e.TensorMultiply(NeuSRenderer<T>.Norm(e, render.Normal), NeuSRenderer<T>.Norm(e, targetNormals)),
                NeuSHost<T>.Epsilon(1e-8));
            var cosine = e.TensorDivide(dot, denominator);
            var misalignment = e.TensorAddScalar(e.TensorNegate(cosine), NeuSHost<T>.Scalar(1.0));
            var (normalRays, geometryWeights) = NormalSupervision(batch.Directions, batch.Normals, batch.Mask,
                o.NormalVisibleCosine, o.GeometryAwareNormals);
            Tensor<T> normalLoss;
            if (geometryWeights is not null)
            {
                // Wonder3D geo_aware: errors · exp(|cos|) / Σ exp(|cos|), then ranking_loss(..., type='sum').
                var weighted = e.TensorMultiply(misalignment, NeuSHost<T>.FromDoubles(geometryWeights, rays, 1));
                normalLoss = RankedSum(weighted, normalRays, o.NormalKeepRatio, batch.ViewWeight);
            }
            else
            {
                normalLoss = RankedMean(misalignment, normalRays, o.NormalKeepRatio, batch.ViewWeight, 1e-5);
            }

            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(normalLoss, NeuSHost<T>.Scalar(o.NormalWeight)));
        }

        if (batch.RandomPoints is not null && (o.SparsityWeight > 0 || o.NormalSmoothWeight > 0))
        {
            int n = batch.RandomPoints.Length / 3;
            bool smooth = o.NormalSmoothWeight > 0 && batch.PerturbedPoints is not null;
            var output = Field.SdfNetwork.Forward(e, NeuSHost<T>.FromDoubles(batch.RandomPoints, n, 3), null, null,
                withGradient: smooth);
            if (o.SparsityWeight > 0)
            {
                // exp(−s·|f(p)|).mean()
                var sdf = e.Reshape(output.Sdf, new[] { n, 1 });
                var decay = e.TensorExp(e.TensorMultiplyScalar(e.TensorAbs(sdf), NeuSHost<T>.Scalar(-o.SparsityScale)));
                var sparsity = e.TensorMultiplyScalar(e.ReduceSum(decay, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / n));
                loss = e.TensorAdd(loss, e.TensorMultiplyScalar(sparsity, NeuSHost<T>.Scalar(o.SparsityWeight)));
            }

            if (smooth && batch.PerturbedPoints is not null)
            {
                var gradient = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
                var perturbed = Field.SdfNetwork.Forward(e, NeuSHost<T>.FromDoubles(batch.PerturbedPoints, n, 3), null, null,
                    withGradient: true).Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
                // (∇f(p) − ∇f(p + ε)).abs().mean() over every component.
                var difference = e.TensorAbs(e.TensorSubtract(gradient, perturbed));
                var smoothness = e.TensorMultiplyScalar(e.ReduceSum(difference, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / (3.0 * n)));
                loss = e.TensorAdd(loss, e.TensorMultiplyScalar(smoothness, NeuSHost<T>.Scalar(o.NormalSmoothWeight)));
            }
        }

        return loss;
    }

    /// <summary>
    /// The indicator of the <paramref name="keepRatio"/> fraction of the candidates with the smallest error, the
    /// selection of Wonder3D's <c>ranking_loss</c>: <c>⌊keepRatio · n⌋</c> of the <c>n</c> candidates (Python's
    /// <c>int()</c>), ties broken by index.
    /// </summary>
    /// <param name="errors">One error per element.</param>
    /// <param name="candidates">1 where an element may be kept, 0 where it is excluded beforehand.</param>
    /// <param name="keepRatio">The fraction to keep, in (0, 1].</param>
    /// <param name="kept">Receives the number kept.</param>
    /// <returns>1 for every kept element, 0 elsewhere.</returns>
    /// <remarks>
    /// The reference sorts the errors and then gathers the sorted list at the first <c>k</c> sort indices
    /// (<c>torch.index_select(error, 0, indices[:k])</c> on the already sorted <c>error</c>), which picks values by
    /// their original positions rather than the k smallest. Its comment ("only sum relatively small errors") and its
    /// weight gather (the weights of <c>indices[:k]</c>, which ARE the k smallest) both state the intent this follows.
    /// </remarks>
    internal static double[] KeepSmallest(double[] errors, double[] candidates, double keepRatio, out int kept)
    {
        if (errors is null) throw new ArgumentNullException(nameof(errors));
        if (candidates is null) throw new ArgumentNullException(nameof(candidates));
        if (errors.Length != candidates.Length) throw new ArgumentException("One candidate flag per error is needed.", nameof(candidates));
        if (!(keepRatio > 0 && keepRatio <= 1)) throw new ArgumentOutOfRangeException(nameof(keepRatio));
        var indices = new List<int>(errors.Length);
        for (int i = 0; i < errors.Length; i++)
            if (candidates[i] > 0) indices.Add(i);
        var keep = new double[errors.Length];
        kept = (int)(keepRatio * indices.Count);
        if (kept == indices.Count)
        {
            foreach (int i in indices) keep[i] = 1.0;
            return keep;
        }

        // A total order (error, then index) so the selection is deterministic whatever the sort algorithm.
        var order = indices.ToArray();
        Array.Sort(order, (a, b) =>
        {
            int byError = errors[a].CompareTo(errors[b]);
            return byError != 0 ? byError : a.CompareTo(b);
        });
        for (int j = 0; j < kept; j++) keep[order[j]] = 1.0;
        return keep;
    }

    /// <summary>
    /// The rays that supervise normals and, when <paramref name="geometryAware"/>, Wonder3D's per-ray weights
    /// <c>exp(|cos θ|) / Σ_rays exp(|cos θ|)</c>, θ between the ray and its target normal, with <c>cos θ</c> zeroed
    /// above <paramref name="visibleCosine"/> (reference: <c>cosines[cosines &gt; -0.1] = 0</c>, mask
    /// <c>fg_mask &amp; cosines &lt; -0.1</c>).
    /// </summary>
    /// <param name="directions">The ray directions, <c>[R, 3]</c>.</param>
    /// <param name="normals">The target normals, <c>[R, 3]</c>.</param>
    /// <param name="mask">The 0/1 foreground mask, <c>[R]</c>.</param>
    /// <param name="visibleCosine">The visibility limit, or null for none.</param>
    /// <param name="geometryAware">Whether to compute the weights.</param>
    internal static (double[] Rays, double[]? Weights) NormalSupervision(double[] directions, double[] normals,
        double[] mask, double? visibleCosine, bool geometryAware)
    {
        int rays = mask.Length;
        var supervised = new double[rays];
        var cosines = new double[rays];
        for (int r = 0; r < rays; r++)
        {
            double dot = 0, dd = 0, nn = 0;
            for (int c = 0; c < 3; c++)
            {
                double d = directions[r * 3 + c], n = normals[r * 3 + c];
                dot += d * n;
                dd += d * d;
                nn += n * n;
            }

            double cosine = dot / Math.Max(Math.Sqrt(dd) * Math.Sqrt(nn), 1e-8);
            bool visible = visibleCosine is not double limit || cosine < limit;
            supervised[r] = mask[r] > 0 && visible ? 1.0 : 0.0;
            cosines[r] = visibleCosine is double cut && cosine > cut ? 0.0 : cosine;
        }

        if (!geometryAware) return (supervised, null);
        var weights = new double[rays];
        double total = 0;
        for (int r = 0; r < rays; r++)
        {
            weights[r] = Math.Exp(Math.Abs(cosines[r]));
            total += weights[r];
        }

        for (int r = 0; r < rays; r++) weights[r] /= total;
        return (supervised, weights);
    }

    // Σ over the kept rays of error · scale / (kept + guard); the kept set is every candidate when keepRatio is 1.
    private Tensor<T> RankedMean(Tensor<T> errors, double[] candidates, double keepRatio, double scale, double guard)
    {
        var keep = Selection(errors, candidates, keepRatio, out int kept);
        var sum = _engine.ReduceSum(_engine.TensorMultiply(errors, NeuSHost<T>.FromDoubles(keep, keep.Length, 1)), new[] { 0, 1 }, false);
        return _engine.TensorMultiplyScalar(sum, NeuSHost<T>.Scalar(scale / (kept + guard)));
    }

    // Σ over the kept rays of error · scale.
    private Tensor<T> RankedSum(Tensor<T> errors, double[] candidates, double keepRatio, double scale)
    {
        var keep = Selection(errors, candidates, keepRatio, out _);
        var sum = _engine.ReduceSum(_engine.TensorMultiply(errors, NeuSHost<T>.FromDoubles(keep, keep.Length, 1)), new[] { 0, 1 }, false);
        return _engine.TensorMultiplyScalar(sum, NeuSHost<T>.Scalar(scale));
    }

    private static double[] Selection(Tensor<T> errors, double[] candidates, double keepRatio, out int kept)
    {
        if (keepRatio >= 1.0)
        {
            // Every candidate: no need to read the errors back.
            kept = 0;
            var all = new double[candidates.Length];
            for (int i = 0; i < all.Length; i++)
                if (candidates[i] > 0) { all[i] = 1.0; kept++; }
            return all;
        }

        return KeepSmallest(NeuSHost<T>.ToDoubles(errors), candidates, keepRatio, out kept);
    }

    private Tensor<T> Weighted(Tensor<T> term, double weight)
        => weight == 1.0 ? term : _engine.TensorMultiplyScalar(term, NeuSHost<T>.Scalar(weight));

    /// <summary>
    /// Draws the sparsity and smoothness terms' points uniformly in the unit cube (reference
    /// <c>torch.rand([2048, 3]) * 2 - 1</c>) and their N(0, σ²) perturbations.
    /// </summary>
    private void DrawRegularizationPoints(NeuSRayBatch batch)
    {
        int n = Options.RegularizationPoints;
        var points = new double[n * 3];
        for (int i = 0; i < points.Length; i++) points[i] = _random.NextDouble() * 2.0 - 1.0;
        double[]? perturbed = null;
        if (Options.NormalSmoothWeight > 0)
        {
            perturbed = new double[n * 3];
            for (int i = 0; i < perturbed.Length; i++)
                perturbed[i] = points[i] + NeuSHost<T>.NextGaussian(_random) * Options.NormalSmoothNoise;
        }

        batch.RandomPoints = points;
        batch.PerturbedPoints = perturbed;
    }

    /// <summary>Draws <paramref name="count"/> pixels of one view uniformly at random (reference <c>gen_random_rays_at</c>).</summary>
    internal NeuSRayBatch DrawBatch(HostView view, int count)
    {
        var origins = new double[count * 3];
        var directions = new double[count * 3];
        var colors = new double[count * 3];
        var mask = new double[count];
        var normals = view.Normals is null ? null : new double[count * 3];
        var origin = new double[3];
        var direction = new double[3];
        bool useMask = Options.MaskWeight > 0;
        for (int k = 0; k < count; k++)
        {
            int x = _random.Next(view.Width);
            int y = _random.Next(view.Height);
            int pixel = y * view.Width + x;
            view.Camera.Ray(x, y, origin, direction);
            for (int c = 0; c < 3; c++)
            {
                origins[k * 3 + c] = origin[c];
                directions[k * 3 + c] = direction[c];
                colors[k * 3 + c] = view.Colors[pixel * 3 + c];
                if (normals is not null && view.Normals is not null) normals[k * 3 + c] = view.Normals[pixel * 3 + c];
            }

            // NeuS binarizes the mask at 0.5 when the mask loss is on, and treats every pixel as foreground otherwise.
            mask[k] = !useMask ? 1.0 : view.Mask is not null && view.Mask[pixel] > 0.5 ? 1.0 : 0.0;
        }

        return new NeuSRayBatch(origins, directions, colors, mask, normals) { ViewWeight = view.Weight };
    }

    internal HostView Host(NeuSView<T> view)
    {
        if (_hostViews.TryGetValue(view, out var host)) return host;
        if (Options.MaskWeight > 0 && view.Mask is null)
            throw new ArgumentException("MaskWeight > 0 needs a mask on every view; set MaskWeight to 0 to train without masks.");
        host = new HostView(view.Camera, view.Width, view.Height, NeuSHost<T>.ToDoubles(view.Image),
            view.Mask is null ? null : NeuSHost<T>.ToDoubles(view.Mask),
            view.Normals is null ? null : NeuSHost<T>.ToDoubles(view.Normals), view.Weight);
        _hostViews[view] = host;
        return host;
    }

    private NeuSView<T> NextView(IReadOnlyList<NeuSView<T>> views)
    {
        // A fresh permutation per pass over the views (reference get_image_perm, renewed every len(images) steps).
        if (!ReferenceEquals(views, _permutedViews) || _permutation.Length != views.Count)
        {
            _permutedViews = views;
            _permutation = Array.Empty<int>();
        }

        int position = Iteration % views.Count;
        if (position == 0 || _permutation.Length == 0)
        {
            _permutation = new int[views.Count];
            for (int i = 0; i < _permutation.Length; i++) _permutation[i] = i;
            for (int i = _permutation.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (_permutation[i], _permutation[j]) = (_permutation[j], _permutation[i]);
            }
        }

        return views[_permutation[position]];
    }

    private static IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateReferenceOptimizer(NeuSOptions options)
    {
        // The schedule's step k gives the factor of iteration k − 1; stepping once here makes iteration 0 use
        // factor(0), as the reference's update_learning_rate() before the first iteration does.
        var schedule = new LambdaLRScheduler(options.LearningRate, step => options.LearningRateFactor(Math.Max(0, step - 1)));
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            model: null,
            options: new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                UseAdaptiveBetas = false,
                UseAdaptiveLearningRate = false,
                UseAMSGrad = false,
                AnomalyGuardMode = AdamAnomalyGuardMode.Never,
                EnableGradientClipping = false,
                LearningRateScheduler = schedule,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        optimizer.StepScheduler();
        return optimizer;
    }

    /// <summary>A view's pixels on the host.</summary>
    internal sealed class HostView
    {
        public HostView(NeuSCamera camera, int width, int height, double[] colors, double[]? mask, double[]? normals,
            double weight = 1.0)
        {
            Weight = weight;
            Camera = camera;
            Width = width;
            Height = height;
            Colors = colors;
            Mask = mask;
            Normals = normals;
        }

        public NeuSCamera Camera { get; }

        public int Width { get; }

        public int Height { get; }

        public double[] Colors { get; }

        public double[]? Mask { get; }

        public double[]? Normals { get; }

        public double Weight { get; }
    }
}

/// <summary>A batch of training rays with their targets, on the host.</summary>
internal sealed class NeuSRayBatch
{
    public NeuSRayBatch(double[] origins, double[] directions, double[] colors, double[] mask, double[]? normals)
    {
        Origins = origins;
        Directions = directions;
        Colors = colors;
        Mask = mask;
        Normals = normals;
    }

    /// <summary>Ray origins, <c>[R, 3]</c>.</summary>
    public double[] Origins { get; }

    /// <summary>Unit ray directions, <c>[R, 3]</c>.</summary>
    public double[] Directions { get; }

    /// <summary>Target colors, <c>[R, 3]</c>.</summary>
    public double[] Colors { get; }

    /// <summary>The 0/1 foreground mask, <c>[R]</c>.</summary>
    public double[] Mask { get; }

    /// <summary>Target world normals, <c>[R, 3]</c>, or null.</summary>
    public double[]? Normals { get; }

    /// <summary>The ray count.</summary>
    public int Count => Mask.Length;

    /// <summary>The weight of the view the rays come from (1 unless the view sets one).</summary>
    public double ViewWeight { get; init; } = 1.0;

    /// <summary>The sparsity and smoothness terms' random points, <c>[P, 3]</c>, or null when neither term is on.</summary>
    public double[]? RandomPoints { get; set; }

    /// <summary>The perturbed random points of the smoothness term, <c>[P, 3]</c>, or null.</summary>
    public double[]? PerturbedPoints { get; set; }
}
