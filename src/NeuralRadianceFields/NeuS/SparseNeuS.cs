using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// SparseNeuS (Long et al. 2022, arXiv 2206.05737): generalizable neural surface reconstruction from a few posed
/// views in one forward pass, at the reference's coarse level (<c>general_lod0.conf</c>).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Geometry.</b> A shared FPN (<see cref="SparseNeuSFeatureNet{T}"/>) turns each view into a 56-channel feature map;
/// a 3×3 convolution with InPlaceABN compresses it to 16 channels. Every voxel of the 96³ volume over [−1, 1]³ seen by
/// at least two views gathers its projected features from every view and keeps their variance and mean (32 channels,
/// reference <c>aggregate_multiview_features</c>); the 3-D U-Net <see cref="SparseNeuSCostRegNet{T}"/> regularizes this
/// into a 16-channel geometry encoding. The SDF at a point is the latent SDF MLP of the point and the encoding's
/// trilinear sample there.
/// </para>
/// <para>
/// <b>Appearance.</b> <see cref="SparseNeuSRenderingNetwork{T}"/> blends the source views' colors (IBRNet-style), and
/// the NeuS renderer integrates them with the unbiased SDF weights, a learned variance (init 0.2), stratified samples,
/// four rounds of up-sampling, and the slope clipped to ±10.
/// </para>
/// <para>
/// <b>Weights.</b> The architecture is complete and trainable (<see cref="TrainStep"/>: color L1 over the rays more
/// than 8 samples of which two views see, 0.02·sparseness ramped over 50000 iterations, 0.1·Eikonal, and after 50000
/// iterations 0.01·background opacity; Adam at 2e-4 halved at 100k, 150k, 200k). Meaningful reconstructions need a
/// trained checkpoint, loaded through <see cref="LoadParameters"/> under the reference's module names
/// (<c>pyramid_feature_network_geometry_lod0.*</c>, <c>sdf_network_lod0.*</c>, <c>variance_network_lod0.variance</c>,
/// <c>rendering_network_lod0.*</c>); torchsparse's <c>[27, in, out]</c> kernels must be converted to the dense
/// <c>[out, in, 3, 3, 3]</c> layout (transposed blocks: <c>[in, out, 3, 3, 3]</c>) first.
/// </para>
/// </remarks>
internal sealed class SparseNeuS<T>
{
    private const string FeaturePrefix = "pyramid_feature_network_geometry_lod0.";
    private const string SdfPrefix = "sdf_network_lod0.";
    private const string CompressPrefix = SdfPrefix + "compress_layer";
    private const string RegNetPrefix = SdfPrefix + "sparse_costreg_net.";
    private const string VarianceName = "variance_network_lod0.variance";
    private const string RenderingPrefix = "rendering_network_lod0.";

    private readonly IEngine _engine;
    private readonly Random _random;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>Creates the network with freshly initialized weights.</summary>
    /// <param name="options">The configuration, or null for the reference's.</param>
    /// <param name="engine">The engine, or null for the current one.</param>
    /// <param name="optimizerFactory">Builds the training optimizer, or null for the reference's Adam schedule.</param>
    public SparseNeuS(SparseNeuSOptions? options = null, IEngine? engine = null,
        Func<SparseNeuSOptions, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? optimizerFactory = null)
    {
        Options = options ?? new SparseNeuSOptions();
        Options.Validate();
        _engine = engine ?? AiDotNetEngine.Current;
        _random = Options.Seed.HasValue ? RandomHelper.CreateSeededRandom(Options.Seed.Value) : RandomHelper.CreateSecureRandom();
        var o = Options;
        FeatureNet = new SparseNeuSFeatureNet<T>(_random);
        Compress = new SparseNeuSConvBlock<T>(2, SparseNeuSFeatureNet<T>.FusedChannels, o.CompressedFeatureDim, 3, 1, 1,
            false, true, SparseNeuSActivation.LeakyReLU, _random);
        CostRegNet = new SparseNeuSCostRegNet<T>(2 * o.CompressedFeatureDim, o.GeometryFeatureDim, _random);
        SdfNetwork = NeuSSdfMlp<T>.CreateLatent(o.SdfHiddenDim, o.SdfLayers, o.GeometryFeatureDim, new[] { 4 }, o.SdfFrequencies,
            o.SdfInitialSphereRadius, geometricInit: true, weightNorm: true, o.SoftplusBeta, _random, "sdf_layer.");
        Variance = NeuSHost<T>.FromDoubles(new[] { o.InitialVariance }, 1);
        RenderingNetwork = new SparseNeuSRenderingNetwork<T>(o.GeometryFeatureDim, SparseNeuSFeatureNet<T>.FusedChannels,
            o.AntiAliasPooling, _random);
        _optimizer = optimizerFactory?.Invoke(o) ?? CreateReferenceOptimizer(o);
    }

    /// <summary>The configuration.</summary>
    public SparseNeuSOptions Options { get; }

    /// <summary>The image feature pyramid.</summary>
    public SparseNeuSFeatureNet<T> FeatureNet { get; }

    /// <summary>The 56 → 16 feature compression (<c>compress_layer</c>).</summary>
    public SparseNeuSConvBlock<T> Compress { get; }

    /// <summary>The cost-volume regularization network.</summary>
    public SparseNeuSCostRegNet<T> CostRegNet { get; }

    /// <summary>The latent SDF MLP.</summary>
    public NeuSSdfMlp<T> SdfNetwork { get; }

    /// <summary>The variance parameter, one element.</summary>
    public Tensor<T> Variance { get; private set; }

    /// <summary>The color blending network.</summary>
    public SparseNeuSRenderingNetwork<T> RenderingNetwork { get; }

    /// <summary>The completed training iterations.</summary>
    public int Iteration { get; private set; }

    /// <summary>The trainable tensors.</summary>
    public IReadOnlyList<Tensor<T>> Parameters
        => FeatureNet.Parameters.Concat(Compress.Parameters).Concat(CostRegNet.Parameters).Concat(SdfNetwork.Parameters)
            .Concat(new[] { Variance }).Concat(RenderingNetwork.Parameters).ToList();

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => FeatureNet.ParameterCount + Compress.ParameterCount + CostRegNet.ParameterCount
        + SdfNetwork.ParameterCount + 1 + RenderingNetwork.ParameterCount;

    /// <summary>The parameters (and batch-norm statistics) under the reference's module names.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters()
    {
        foreach (var p in FeatureNet.NamedParameters(FeaturePrefix)) yield return p;
        foreach (var p in Compress.NamedParameters(CompressPrefix)) yield return p;
        foreach (var p in CostRegNet.NamedParameters(RegNetPrefix)) yield return p;
        foreach (var p in SdfNetwork.NamedParameters()) yield return new KeyValuePair<string, Tensor<T>>(SdfPrefix + p.Key, p.Value);
        yield return new KeyValuePair<string, Tensor<T>>(VarianceName, Variance);
        foreach (var p in RenderingNetwork.NamedParameters(RenderingPrefix)) yield return p;
    }

    /// <summary>Loads parameters by the names <see cref="NamedParameters"/> lists; unknown names throw.</summary>
    /// <param name="parameters">Name to tensor.</param>
    public void LoadParameters(IReadOnlyDictionary<string, Tensor<T>> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        foreach (var entry in parameters)
        {
            string name = entry.Key;
            bool loaded = FeatureNet.TryLoad(FeaturePrefix, name, entry.Value)
                || Compress.TryLoad(CompressPrefix, name, entry.Value)
                || CostRegNet.TryLoad(RegNetPrefix, name, entry.Value)
                || (name.StartsWith(SdfPrefix, StringComparison.Ordinal) && SdfNetwork.TryLoad(name.Substring(SdfPrefix.Length), entry.Value))
                || RenderingNetwork.TryLoad(RenderingPrefix, name, entry.Value)
                || (name == VarianceName && LoadVariance(entry.Value));
            if (!loaded) throw new ArgumentException($"Unknown SparseNeuS parameter '{name}'.", nameof(parameters));
        }
    }

    /// <summary>Encodes a scene from posed views without recording a tape (inference).</summary>
    /// <param name="views">Two or more views of equal size, H and W divisible by 4, in the normalized frame.</param>
    public SparseNeuSScene<T> Reconstruct(IReadOnlyList<NeuSView<T>> views)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Encode(views, training: false);
    }

    /// <summary>
    /// Builds the geometry volume and the color-blend inputs (reference <c>get_conditional_volume</c> at lod 0).
    /// </summary>
    /// <param name="views">The views.</param>
    /// <param name="training">Whether batch normalization uses batch statistics.</param>
    internal SparseNeuSScene<T> Encode(IReadOnlyList<NeuSView<T>> views, bool training)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        if (views.Count == 0) throw new ArgumentException("At least one view is needed.", nameof(views));
        var e = _engine;
        var o = Options;
        int height = views[0].Height, width = views[0].Width;
        foreach (var v in views)
            if (v.Height != height || v.Width != width) throw new ArgumentException("Every view must have the same size.", nameof(views));
        int viewCount = views.Count;
        int pixels = height * width;

        var images = e.TensorStack(views.Select(v => e.TensorPermute(v.Image, new[] { 2, 0, 1 })).ToArray(), 0); // [V, 3, H, W]
        var fused = FeatureNet.Forward(e, images, training);                                                      // [V, 56, H, W]
        var compressed = Compress.Forward(e, fused, training);                                                    // [V, 16, H, W]

        var features = new Tensor<T>[viewCount];
        var compressedFlat = new Tensor<T>[viewCount];
        var colors = new Tensor<T>[viewCount];
        for (int v = 0; v < viewCount; v++)
        {
            features[v] = PixelRows(e, fused, v, SparseNeuSFeatureNet<T>.FusedChannels, height, width);
            compressedFlat[v] = PixelRows(e, compressed, v, o.CompressedFeatureDim, height, width);
            colors[v] = e.Reshape(views[v].Image, new[] { pixels, 3 });
        }

        // Voxel visibility: the reference keeps voxels seen by more than min(1, V − 1) views.
        var dims = o.VolumeDims;
        int voxels = dims[0] * dims[1] * dims[2];
        var centers = new double[voxels * 3];
        int q = 0;
        for (int x = 0; x < dims[0]; x++)
            for (int y = 0; y < dims[1]; y++)
                for (int z = 0; z < dims[2]; z++)
                {
                    centers[q++] = o.VolumeOrigin[0] + x * o.VoxelSize;
                    centers[q++] = o.VolumeOrigin[1] + y * o.VoxelSize;
                    centers[q++] = o.VolumeOrigin[2] + z * o.VoxelSize;
                }

        var projectedU = new double[viewCount][];
        var projectedV = new double[viewCount][];
        var visibleCount = new int[voxels];
        for (int v = 0; v < viewCount; v++)
        {
            projectedU[v] = new double[voxels];
            projectedV[v] = new double[voxels];
            var camera = views[v].Camera;
            for (int i = 0; i < voxels; i++)
            {
                var (pu, pv, depth) = camera.Project(centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]);
                projectedU[v][i] = pu;
                projectedV[v][i] = pv;
                if (depth > 0 && pu >= 0 && pu <= width - 1 && pv >= 0 && pv <= height - 1) visibleCount[i]++;
            }
        }

        int minimumViews = Math.Min(1, viewCount - 1);
        var activeList = new List<int>();
        var active = new double[voxels];
        for (int i = 0; i < voxels; i++)
            if (visibleCount[i] > minimumViews)
            {
                active[i] = 1.0;
                activeList.Add(i);
            }

        int activeCount = activeList.Count;
        if (activeCount == 0) throw new InvalidOperationException("No voxel is visible in enough views; check the cameras' normalization.");

        // Variance and mean of the compressed features over the views (aggregate_multiview_features): the sums run
        // over every view's sample, the count over the views that see the voxel.
        Tensor<T>? sum = null, squares = null;
        for (int v = 0; v < viewCount; v++)
        {
            var u = new double[activeCount];
            var w = new double[activeCount];
            for (int k = 0; k < activeCount; k++)
            {
                u[k] = projectedU[v][activeList[k]];
                w[k] = projectedV[v][activeList[k]];
            }

            var sampled = NeuSGridInterpolation<T>.Bilinear(u, w, height, width).Sample(e, compressedFlat[v]);
            sum = sum is null ? sampled : e.TensorAdd(sum, sampled);
            var sq = e.TensorMultiply(sampled, sampled);
            squares = squares is null ? sq : e.TensorAdd(squares, sq);
        }

        var inverseCount = new double[activeCount];
        for (int k = 0; k < activeCount; k++) inverseCount[k] = 1.0 / (visibleCount[activeList[k]] + 1e-5);
        var inv = NeuSHost<T>.FromDoubles(inverseCount, activeCount, 1);
        var mean = e.TensorMultiply(sum ?? throw new InvalidOperationException("No views."), inv);
        var variance = e.TensorSubtract(e.TensorMultiply(squares ?? throw new InvalidOperationException("No views."), inv), e.TensorMultiply(mean, mean));
        var cost = e.TensorConcatenate(new[] { variance, mean }, 1); // [Na, 32]

        // Scatter to the dense grid: inactive voxels read an appended zero row.
        int channels = 2 * o.CompressedFeatureDim;
        var withZero = e.TensorConcatenate(new[] { cost, NeuSHost<T>.Filled(0.0, 1, channels) }, 0);
        var map = new int[voxels];
        for (int i = 0; i < voxels; i++) map[i] = activeCount;
        for (int k = 0; k < activeCount; k++) map[activeList[k]] = k;
        var dense = e.TensorIndexSelect(withZero, NeuSHost<T>.Indices(map), 0); // [XYZ, 32]
        var denseVolume = e.Reshape(e.TensorTranspose(dense), new[] { 1, channels, dims[0], dims[1], dims[2] });

        var encoded = CostRegNet.Forward(e, denseVolume, active, dims, training); // [1, 16, X, Y, Z]
        var volume = e.TensorTranspose(e.Reshape(encoded, new[] { o.GeometryFeatureDim, voxels })); // [XYZ, 16]
        var cameras = views.Select(v => v.Camera).ToList();
        return new SparseNeuSScene<T>(this, e, volume, active, features, colors, cameras, width, height);
    }

    /// <summary>
    /// One generalizable training iteration: encode the scene from every view, render <c>BatchSize</c> random rays of
    /// view <paramref name="targetView"/> blending the other views, and step the optimizer. Returns the loss.
    /// </summary>
    /// <param name="views">The scene's views (the reference trains with 5, the target first).</param>
    /// <param name="targetView">The view whose pixels supervise the step.</param>
    public T TrainStep(IReadOnlyList<NeuSView<T>> views, int targetView = 0)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        if (views.Count < 2) throw new ArgumentException("Training needs the target view and at least one other.", nameof(views));
        if (targetView < 0 || targetView >= views.Count) throw new ArgumentOutOfRangeException(nameof(targetView));
        var o = Options;
        var target = views[targetView];
        var colors = NeuSHost<T>.ToDoubles(target.Image);
        var mask = target.Mask is null ? null : NeuSHost<T>.ToDoubles(target.Mask);
        int rays = o.BatchSize;
        var origins = new double[rays * 3];
        var directions = new double[rays * 3];
        var targetColors = new double[rays * 3];
        var targetMask = new double[rays];
        var direction = new double[3];
        for (int k = 0; k < rays; k++)
        {
            int x = _random.Next(target.Width), y = _random.Next(target.Height);
            int pixel = y * target.Width + x;
            target.Camera.RayDirection(x, y, direction);
            for (int c = 0; c < 3; c++)
            {
                origins[k * 3 + c] = target.Camera.Center[c];
                directions[k * 3 + c] = direction[c];
                targetColors[k * 3 + c] = colors[pixel * 3 + c];
            }

            targetMask[k] = mask is null ? 1.0 : mask[pixel];
        }

        var randomPoints = new double[o.SparseLossPoints * 3];
        for (int i = 0; i < randomPoints.Length; i++) randomPoints[i] = _random.NextDouble() * 2.0 - 1.0;
        int perturbSeed = _random.Next();
        int iteration = Iteration;

        Tensor<T> Objective()
        {
            var scene = Encode(views, training: true);
            var perturb = o.Perturb ? RandomHelper.CreateSeededRandom(perturbSeed) : null;
            var render = scene.Render(origins, directions, target.Camera.Center, targetView, o.CosineAnnealRatio(iteration), perturb);
            return Loss(scene, render, targetColors, targetMask, randomPoints, iteration);
        }

        var loss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(() => Parameters, Objective, _optimizer,
            applyOptimizerRegularization: false);
        Iteration++;
        if (_optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled) scheduled.StepScheduler();
        return loss;
    }

    /// <summary>The reference's <c>cal_losses_sdf</c> at lod 0.</summary>
    internal Tensor<T> Loss(SparseNeuSScene<T> scene, SparseNeuSRenderResult<T> render, double[] targetColors,
        double[] targetMask, double[] randomPoints, int iteration)
    {
        var e = _engine;
        var o = Options;
        int rays = render.ColorValid.Length;
        var valid = new double[rays];
        double validCount = 0;
        for (int r = 0; r < rays; r++)
        {
            valid[r] = render.ColorValid[r] ? 1.0 : 0.0;
            validCount += valid[r];
        }

        // F.l1_loss(..., reduction='mean') over the valid rays' three channels.
        var validT = NeuSHost<T>.FromDoubles(valid, rays, 1);
        var error = e.TensorMultiply(e.TensorAbs(e.TensorSubtract(render.Color, NeuSHost<T>.FromDoubles(targetColors, rays, 3))), validT);
        var loss = e.TensorMultiplyScalar(e.ReduceSum(error, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / Math.Max(1.0, 3.0 * validCount)));

        // Sparseness: exp(−decay |f|) at random points and at the samples.
        var randomSdf = scene.Query(randomPoints, withGradient: false).Output.Sdf;
        var decay = NeuSHost<T>.Scalar(-o.SparseDecay);
        var sparseRandom = MeanAll(e, e.TensorExp(e.TensorMultiplyScalar(e.TensorAbs(randomSdf), decay)));
        var sparseSamples = MeanAll(e, e.TensorExp(e.TensorMultiplyScalar(e.TensorAbs(render.Sdf), decay)));
        var sparse = e.TensorMultiplyScalar(e.TensorAdd(sparseRandom, sparseSamples), NeuSHost<T>.Scalar(0.5));
        loss = e.TensorAdd(loss, e.TensorMultiplyScalar(sparse, NeuSHost<T>.Scalar(o.Ramp(iteration, o.SparseWeight))));
        loss = e.TensorAdd(loss, e.TensorMultiplyScalar(render.GradientError, NeuSHost<T>.Scalar(o.EikonalWeight)));

        // Background opacity, only after ForegroundLossStart and when over 2% of the rays are background.
        double background = 0;
        foreach (double m in targetMask) background += m < 0.5 ? 1.0 : 0.0;
        if (o.ForegroundWeight > 0 && iteration >= o.ForegroundLossStart && background / rays > 0.02)
        {
            var isBackground = new double[rays];
            for (int r = 0; r < rays; r++) isBackground[r] = targetMask[r] < 0.5 ? 1.0 : 0.0;
            var residual = e.TensorMultiply(e.TensorAbs(e.TensorSubtract(render.WeightSum, NeuSHost<T>.FromDoubles(targetMask, rays, 1))),
                NeuSHost<T>.FromDoubles(isBackground, rays, 1));
            var fgBg = e.TensorMultiplyScalar(e.ReduceSum(residual, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / background));
            loss = e.TensorAdd(loss, e.TensorMultiplyScalar(fgBg, NeuSHost<T>.Scalar(o.Ramp(iteration, o.ForegroundWeight))));
        }

        return loss;
    }

    private static Tensor<T> MeanAll(IEngine e, Tensor<T> x)
    {
        var axes = new int[x.Shape.Length];
        for (int i = 0; i < axes.Length; i++) axes[i] = i;
        return e.ReduceMean(x, axes, false);
    }

    private static Tensor<T> PixelRows(IEngine e, Tensor<T> maps, int view, int channels, int height, int width)
    {
        var slice = e.TensorSlice(maps, new[] { view, 0, 0, 0 }, new[] { 1, channels, height, width });
        return e.TensorTranspose(e.Reshape(slice, new[] { channels, height * width })); // [H·W, C]
    }

    private bool LoadVariance(Tensor<T> value)
    {
        if (value is null || value.Length != 1) throw new ArgumentException("The variance is a single value.", nameof(value));
        Variance = value.Reshape(1);
        return true;
    }

    private static IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateReferenceOptimizer(SparseNeuSOptions options)
    {
        var schedule = new LambdaLRScheduler(options.LearningRate, step => options.LearningRateMultiplier(Math.Max(0, step - 1)));
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
}
