using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Optimizers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// SyncDreamer's NeuS reconstruction from its generated views (<c>renderer/renderer.py RendererTrainer</c> with
/// <c>configs/neus.yaml</c>), training a shared <see cref="NeuSField{T}"/> with SyncDreamer's renderer variant.
/// </summary>
/// <remarks>
/// <para>
/// Every pixel of every view is one ray (<c>_construct_ray_batch</c>). Each step takes the next
/// <see cref="SyncDreamerOptions.ReconstructionRayBatchSize"/> rays of a random permutation of all rays and the next
/// <see cref="SyncDreamerOptions.ReconstructionForegroundRayBatchSize"/> of a permutation of the pseudo-foreground rays
/// (pixels whose color is not above 0.99 in all three channels); a permutation is renewed when the next batch would
/// reach its end. The loss is
/// <c>0.5 * mean(sqrt(|c - c_gt|^2 + 0.001)) + 0.1 * Eikonal + 1.0 * mean((sum w - mask)^2)</c>, the last term only when
/// the views carry masks. AdamW (torch defaults) runs at 5e-4 with <c>schedule_fn</c>: a linear warm-up from 0.02 over
/// the warm-up steps, then a factor <c>0.02^(1 / (total // 1000))</c> per 1000 steps.
/// </para>
/// <para>
/// The official trainer predicts masks with CarveKit (<c>BackgroundRemoval</c>); there is no CarveKit here, so masks
/// come from the caller (or the views' alpha channels).
/// </para>
/// </remarks>
internal sealed class SyncDreamerNeuSTrainer<T>
{
    private const double SoftL1Epsilon = 0.001;            // render_with_loss: epsilon = 0.001
    private const double ForegroundThreshold = 0.99;       // pseudo_fg_mask: rgb > 0.99
    private const double WarmUpInitialFactor = 0.02;       // schedule_fn: warm_up_init
    private const double FinalFactor = 0.02;               // schedule_fn: final_lr
    private const int DecayInterval = 1000;                // schedule_fn: interval
    private const double AdamWeightDecay = 0.01;           // torch.optim.AdamW default

    private readonly SyncDreamerOptions _options;
    private readonly IEngine _engine;
    private readonly Random _random;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    public SyncDreamerNeuSTrainer(SyncDreamerOptions options, NeuSOptions fieldOptions, int seed, IEngine engine)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (fieldOptions is null) throw new ArgumentNullException(nameof(fieldOptions));
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _random = RandomHelper.CreateSeededRandom(seed);
        Field = new NeuSField<T>(fieldOptions, _random, engine);
        _optimizer = CreateOptimizer();
    }

    /// <summary>The field being optimized.</summary>
    public NeuSField<T> Field { get; }

    /// <summary>The completed steps.</summary>
    public int Step { get; private set; }

    /// <summary>The <c>schedule_fn</c> learning-rate factor at <paramref name="step"/>.</summary>
    public static double LearningRateFactor(int step, int totalSteps, int warmUpSteps)
    {
        if (step < warmUpSteps)
            return (double)step / warmUpSteps * (1.0 - WarmUpInitialFactor) + WarmUpInitialFactor;
        int times = totalSteps / DecayInterval;
        double ratio = times == 0 ? 1.0 : Math.Pow(FinalFactor, 1.0 / times);
        return Math.Pow(ratio, step / DecayInterval);
    }

    /// <summary>The ray set of the views: every pixel of every view, view-major then row-major.</summary>
    /// <param name="cameras">The view cameras.</param>
    /// <param name="colors">Each view's colors, [H * W * 3] row-major in [0, 1].</param>
    /// <param name="masks">Each view's foreground mask, [H * W] in [0, 1], or null for none.</param>
    /// <param name="size">The image side.</param>
    public static SyncDreamerRaySet BuildRays(IReadOnlyList<NeuSCamera> cameras, IReadOnlyList<double[]> colors,
        IReadOnlyList<double[]>? masks, int size)
    {
        int views = cameras.Count, pixels = size * size, total = views * pixels;
        var origins = new double[total * 3];
        var directions = new double[total * 3];
        var rgb = new double[total * 3];
        var mask = new double[total];
        var direction = new double[3];
        var foreground = new List<int>();
        for (int v = 0; v < views; v++)
        {
            var camera = cameras[v];
            for (int row = 0; row < size; row++)
            {
                for (int col = 0; col < size; col++)
                {
                    int pixel = row * size + col, ray = v * pixels + pixel;
                    camera.RayDirection(col, row, direction);
                    int bright = 0;
                    for (int c = 0; c < 3; c++)
                    {
                        origins[ray * 3 + c] = camera.Center[c];
                        directions[ray * 3 + c] = direction[c];
                        double value = colors[v][pixel * 3 + c];
                        rgb[ray * 3 + c] = value;
                        if (value > ForegroundThreshold) bright++;
                    }

                    mask[ray] = masks is null ? 0.0 : masks[v][pixel];
                    if (bright != 3) foreground.Add(ray);
                }
            }
        }

        return new SyncDreamerRaySet(origins, directions, rgb, mask, masks is not null, foreground.ToArray());
    }

    /// <summary>Runs <paramref name="steps"/> steps on <paramref name="rays"/> and returns the field.</summary>
    public NeuSField<T> Fit(SyncDreamerRaySet rays, int steps)
    {
        if (rays is null) throw new ArgumentNullException(nameof(rays));
        int batch = _options.ReconstructionRayBatchSize, foregroundBatch = _options.ReconstructionForegroundRayBatchSize;
        var all = new RayCursor(rays.Count, batch, _random);
        var foreground = new RayCursor(rays.Foreground.Length, foregroundBatch, _random);
        for (int i = 0; i < steps; i++)
        {
            var indices = new List<int>(batch + foregroundBatch);
            indices.AddRange(all.Next());
            if (foregroundBatch > 0)
                foreach (int k in foreground.Next()) indices.Add(rays.Foreground[k]);
            TrainStep(rays.Select(indices));
        }

        return Field;
    }

    private void TrainStep(SyncDreamerRayBatch batch)
    {
        int anneal = _options.ReconstructionAnnealIterations;
        double annealRatio = anneal <= 0 ? 1.0 : Math.Min(1.0, (double)Step / anneal);
        int perturbSeed = _random.Next();

        Tensor<T> Objective()
        {
            var render = Field.Render(batch.Origins, batch.Directions, annealRatio, RandomHelper.CreateSeededRandom(perturbSeed));
            return Loss(batch, render);
        }

        AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => Field.Parameters, Objective, _optimizer, applyOptimizerRegularization: false);
        Step++;
        if (_optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled) scheduled.StepScheduler();
    }

    /// <summary>The training loss of one rendered batch (<c>NeuSRenderer.render_with_loss</c>).</summary>
    internal Tensor<T> Loss(SyncDreamerRayBatch batch, NeuSRenderResult<T> render)
    {
        var e = _engine;
        int rays = batch.Count;
        var target = NeuSHost<T>.FromDoubles(batch.Colors, rays, 3);
        var difference = e.TensorSubtract(target, render.Color);
        var squared = e.ReduceSum(e.TensorMultiply(difference, difference), new[] { 1 }, true);
        var softL1 = e.TensorSqrt(e.TensorAddScalar(squared, NeuSHost<T>.Scalar(SoftL1Epsilon)));
        var rgbLoss = e.TensorMultiplyScalar(e.ReduceSum(softL1, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / rays));
        var loss = e.TensorAdd(
            e.TensorMultiplyScalar(rgbLoss, NeuSHost<T>.Scalar(_options.ReconstructionRgbWeight)),
            e.TensorMultiplyScalar(render.GradientError, NeuSHost<T>.Scalar(_options.ReconstructionEikonalWeight)));
        if (batch.HasMask && _options.ReconstructionMaskWeight > 0)
        {
            var maskError = e.TensorSubtract(render.WeightSum, NeuSHost<T>.FromDoubles(batch.Mask, rays, 1));
            var maskLoss = e.TensorMultiplyScalar(e.ReduceSum(e.TensorMultiply(maskError, maskError), new[] { 0, 1 }, false),
                NeuSHost<T>.Scalar(1.0 / rays));
            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(maskLoss, NeuSHost<T>.Scalar(_options.ReconstructionMaskWeight)));
        }

        return loss;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateOptimizer()
    {
        int total = _options.ReconstructionIterations, warmUp = _options.ReconstructionWarmUpIterations;
        // LambdaLR: step k of the schedule gives the factor of step k - 1; stepping once before training makes step 0
        // use schedule_fn(0), as LambdaLR's construction does.
        var schedule = new LambdaLRScheduler(_options.ReconstructionLearningRate,
            step => LearningRateFactor(Math.Max(0, step - 1), total, warmUp));
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            model: null,
            options: new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.ReconstructionLearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                WeightDecay = AdamWeightDecay,
                UseAMSGrad = false,
                UseAdaptiveBetas = false,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
                LearningRateScheduler = schedule,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        optimizer.StepScheduler();
        return optimizer;
    }

    // A random permutation of [0, count) read batch by batch (_shuffle_train_batch / training_step).
    private sealed class RayCursor
    {
        private readonly int _count;
        private readonly int _batch;
        private readonly Random _random;
        private readonly int[] _order;
        private int _position;

        public RayCursor(int count, int batch, Random random)
        {
            _count = count;
            _batch = batch;
            _random = random;
            _order = new int[count];
            Shuffle();
        }

        public int[] Next()
        {
            int take = Math.Max(0, Math.Min(_batch, _count - _position));
            var indices = new int[take];
            Array.Copy(_order, _position, indices, 0, take);
            _position += _batch;
            if (_position + _batch >= _count) Shuffle();
            return indices;
        }

        private void Shuffle()
        {
            _position = 0;
            for (int i = 0; i < _count; i++) _order[i] = i;
            for (int i = _count - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (_order[i], _order[j]) = (_order[j], _order[i]);
            }
        }
    }
}

/// <summary>Every ray of SyncDreamer's views on the host, with the pseudo-foreground subset.</summary>
internal sealed class SyncDreamerRaySet
{
    public SyncDreamerRaySet(double[] origins, double[] directions, double[] colors, double[] mask, bool hasMask, int[] foreground)
    {
        Origins = origins;
        Directions = directions;
        Colors = colors;
        Mask = mask;
        HasMask = hasMask;
        Foreground = foreground;
    }

    public double[] Origins { get; }

    public double[] Directions { get; }

    public double[] Colors { get; }

    public double[] Mask { get; }

    public bool HasMask { get; }

    /// <summary>The rays whose color is not above 0.99 in every channel.</summary>
    public int[] Foreground { get; }

    public int Count => Mask.Length;

    public SyncDreamerRayBatch Select(IReadOnlyList<int> indices)
    {
        int n = indices.Count;
        var origins = new double[n * 3];
        var directions = new double[n * 3];
        var colors = new double[n * 3];
        var mask = new double[n];
        for (int k = 0; k < n; k++)
        {
            int ray = indices[k];
            Array.Copy(Origins, ray * 3, origins, k * 3, 3);
            Array.Copy(Directions, ray * 3, directions, k * 3, 3);
            Array.Copy(Colors, ray * 3, colors, k * 3, 3);
            mask[k] = Mask[ray];
        }

        return new SyncDreamerRayBatch(origins, directions, colors, mask, HasMask);
    }
}

/// <summary>One training batch of rays.</summary>
internal sealed class SyncDreamerRayBatch
{
    public SyncDreamerRayBatch(double[] origins, double[] directions, double[] colors, double[] mask, bool hasMask)
    {
        Origins = origins;
        Directions = directions;
        Colors = colors;
        Mask = mask;
        HasMask = hasMask;
    }

    public double[] Origins { get; }

    public double[] Directions { get; }

    public double[] Colors { get; }

    public double[] Mask { get; }

    public bool HasMask { get; }

    public int Count => Mask.Length;
}
