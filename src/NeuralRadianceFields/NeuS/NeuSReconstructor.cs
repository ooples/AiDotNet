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

    /// <summary>The training loss for one rendered batch.</summary>
    internal Tensor<T> Loss(NeuSRayBatch batch, NeuSRenderResult<T> render)
    {
        var e = _engine;
        int rays = batch.Count;
        double maskSum = 0;
        foreach (double m in batch.Mask) maskSum += m;
        var mask = NeuSHost<T>.FromDoubles(batch.Mask, rays, 1);
        var target = NeuSHost<T>.FromDoubles(batch.Colors, rays, 3);
        var colorError = e.TensorMultiply(e.TensorSubtract(render.Color, target), mask);
        var colorLoss = e.TensorMultiplyScalar(e.ReduceSum(e.TensorAbs(colorError), new[] { 0, 1 }, false),
            NeuSHost<T>.Scalar(1.0 / (maskSum + 1e-5)));
        var loss = e.TensorAdd(colorLoss, e.TensorMultiplyScalar(render.GradientError, NeuSHost<T>.Scalar(Options.EikonalWeight)));

        if (Options.MaskWeight > 0)
        {
            var w = e.TensorClamp(render.WeightSum, NeuSHost<T>.Scalar(1e-3), NeuSHost<T>.Scalar(1.0 - 1e-3));
            var oneMinusMask = e.TensorAddScalar(e.TensorNegate(mask), NeuSHost<T>.Scalar(1.0));
            var oneMinusW = e.TensorAddScalar(e.TensorNegate(w), NeuSHost<T>.Scalar(1.0));
            var logLikelihood = e.TensorAdd(e.TensorMultiply(mask, e.TensorLog(w)), e.TensorMultiply(oneMinusMask, e.TensorLog(oneMinusW)));
            var bce = e.TensorMultiplyScalar(e.ReduceSum(logLikelihood, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(-1.0 / rays));
            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(bce, NeuSHost<T>.Scalar(Options.MaskWeight)));
        }

        if (Options.NormalWeight > 0 && batch.Normals is not null)
        {
            var targetNormals = NeuSHost<T>.FromDoubles(batch.Normals, rays, 3);
            var dot = e.ReduceSum(e.TensorMultiply(render.Normal, targetNormals), new[] { 1 }, true);
            // F.cosine_similarity: x·y / max(‖x‖‖y‖, 1e-8).
            var denominator = e.TensorMax(e.TensorMultiply(NeuSRenderer<T>.Norm(e, render.Normal), NeuSRenderer<T>.Norm(e, targetNormals)),
                NeuSHost<T>.Epsilon(1e-8));
            var cosine = e.TensorDivide(dot, denominator);
            var misalignment = e.TensorMultiply(e.TensorAddScalar(e.TensorNegate(cosine), NeuSHost<T>.Scalar(1.0)), mask);
            var normalLoss = e.TensorMultiplyScalar(e.ReduceSum(misalignment, new[] { 0, 1 }, false),
                NeuSHost<T>.Scalar(1.0 / (maskSum + 1e-5)));
            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(normalLoss, NeuSHost<T>.Scalar(Options.NormalWeight)));
        }

        return loss;
    }

    /// <summary>Draws <paramref name="count"/> pixels of one view uniformly at random (reference <c>gen_random_rays_at</c>).</summary>
    internal NeuSRayBatch DrawBatch(HostView view, int count)
    {
        var origins = new double[count * 3];
        var directions = new double[count * 3];
        var colors = new double[count * 3];
        var mask = new double[count];
        var normals = view.Normals is null ? null : new double[count * 3];
        var direction = new double[3];
        bool useMask = Options.MaskWeight > 0;
        for (int k = 0; k < count; k++)
        {
            int x = _random.Next(view.Width);
            int y = _random.Next(view.Height);
            int pixel = y * view.Width + x;
            view.Camera.RayDirection(x, y, direction);
            for (int c = 0; c < 3; c++)
            {
                origins[k * 3 + c] = view.Camera.Center[c];
                directions[k * 3 + c] = direction[c];
                colors[k * 3 + c] = view.Colors[pixel * 3 + c];
                if (normals is not null && view.Normals is not null) normals[k * 3 + c] = view.Normals[pixel * 3 + c];
            }

            // NeuS binarizes the mask at 0.5 when the mask loss is on, and treats every pixel as foreground otherwise.
            mask[k] = !useMask ? 1.0 : view.Mask is not null && view.Mask[pixel] > 0.5 ? 1.0 : 0.0;
        }

        return new NeuSRayBatch(origins, directions, colors, mask, normals);
    }

    internal HostView Host(NeuSView<T> view)
    {
        if (_hostViews.TryGetValue(view, out var host)) return host;
        if (Options.MaskWeight > 0 && view.Mask is null)
            throw new ArgumentException("MaskWeight > 0 needs a mask on every view; set MaskWeight to 0 to train without masks.");
        host = new HostView(view.Camera, view.Width, view.Height, NeuSHost<T>.ToDoubles(view.Image),
            view.Mask is null ? null : NeuSHost<T>.ToDoubles(view.Mask),
            view.Normals is null ? null : NeuSHost<T>.ToDoubles(view.Normals));
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
        public HostView(NeuSCamera camera, int width, int height, double[] colors, double[]? mask, double[]? normals)
        {
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
}
