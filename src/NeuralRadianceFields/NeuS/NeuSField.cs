using AiDotNet.NeuralRadianceFields.Encodings;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// A NeuS scene: the SDF network, the color network and the S-density's variance, with NeuS's renderer and a dense
/// SDF grid sampler for mesh extraction (Wang et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <see cref="Render"/> is the reference's <c>NeuSRenderer.render</c> without the NeRF++ background model (the
/// <c>wmask</c> configuration): uniform samples between the unit-sphere chord's ends, <c>UpSampleSteps</c> rounds of
/// SDF-guided up-sampling with sharpness 64·2ⁱ, then the unbiased weights of <see cref="NeuSRenderer{T}"/> over every
/// sample. The SDF is queried with its spatial gradient so the normal-dependent color and the Eikonal residual both
/// differentiate into the weights.
/// </para>
/// <para>
/// The field is trained by <see cref="NeuSReconstructor{T}"/>; untrained, it is the geometric initialization's sphere
/// of radius <c>SdfInitialSphereRadius</c>. Reference checkpoints load through <see cref="LoadParameters"/> under the
/// reference's names (<c>sdf_network_fine.lin0.weight_v</c>, <c>variance_network_fine.variance</c>, …).
/// </para>
/// <para>
/// instant-nsr-pl's variant (Wonder3D's reconstruction) is selected through the options: the hash-grid backbone,
/// finite-difference gradients (<see cref="NeuSGradientMode.FiniteDifference"/>), the <c>VolumeRadiance</c> color
/// network and nerfacc's occupancy-grid ray marching (<see cref="NeuSRaySampler.OccupancyGrid"/>).
/// </para>
/// </remarks>
internal sealed class NeuSField<T> : ISignedDistanceField<T>
{
    private const string SdfPrefix = "sdf_network_fine.";
    private const string ColorPrefix = "color_network_fine.";
    private const string VarianceName = "variance_network_fine.variance";
    private const string HashGridPrefix = "sdf_network_fine.hash_grid.";

    private readonly IEngine _engine;

    /// <summary>Creates a field with freshly initialized networks.</summary>
    /// <param name="options">The configuration (validated here).</param>
    /// <param name="random">The initialization generator.</param>
    /// <param name="engine">The engine, or null for the current one.</param>
    public NeuSField(NeuSOptions options, Random random, IEngine? engine = null)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        if (random is null) throw new ArgumentNullException(nameof(random));
        options.Validate();
        _engine = engine ?? AiDotNetEngine.Current;
        if (options.SdfBackbone == NeuSSdfBackbone.HashGrid)
        {
            var grid = new MultiresolutionHashGrid<T>(options.HashGridLevels, options.HashGridFeaturesPerLevel,
                options.HashGridLog2TableSize, options.HashGridBaseResolution, options.HashGridPerLevelScale, random);
            HashGrid = grid;
            SdfNetwork = NeuSSdfMlp<T>.CreateHashGrid(options.SdfHiddenDim, options.SdfHiddenLayers, options.SdfFeatureDim + 1,
                grid.OutputDims, options.SdfInitialSphereRadius, options.GeometricInit, options.WeightNorm,
                options.InsideOutside, options.SoftplusBeta, random);
        }
        else
        {
            SdfNetwork = NeuSSdfMlp<T>.CreateNeuS(options.SdfHiddenDim, options.SdfHiddenLayers, options.SdfFeatureDim + 1,
                options.SdfSkipLayers, options.SdfFrequencies, options.SdfInitialSphereRadius, options.SdfScale,
                options.GeometricInit, options.WeightNorm, options.InsideOutside, options.SoftplusBeta, random);
        }

        bool radiance = options.RenderingMode == NeuSRenderingMode.VolumeRadiance;
        // VolumeRadiance reads the SDF network's whole output (input_feature_dim = feature_dim + 3 with the normal).
        ColorNetwork = new NeuSRenderingNetwork<T>(radiance ? options.SdfFeatureDim + 1 : options.SdfFeatureDim,
            options.RenderingMode, options.ColorHiddenDim, options.ColorHiddenLayers, options.ViewFrequencies,
            options.WeightNorm, options.SqueezeColor, random,
            radiance && options.RadianceUsesViewDirection ? options.RadianceShDegree : 0);
        Variance = NeuSHost<T>.FromDoubles(new[] { options.InitialVariance }, 1);
        if (options.Sampler == NeuSRaySampler.OccupancyGrid && options.UseOccupancyGrid)
            OccupancyGrid = new NeuSOccupancyGrid(options.OccupancyGridResolution, options.SceneRadius);
        UpdateStep(0);
    }

    /// <summary>The configuration.</summary>
    public NeuSOptions Options { get; }

    /// <summary>The SDF network.</summary>
    public NeuSSdfMlp<T> SdfNetwork { get; }

    /// <summary>The color network.</summary>
    public NeuSRenderingNetwork<T> ColorNetwork { get; }

    /// <summary>The hash grid encoding the SDF network's input, or null for NeuS's frequency embedding.</summary>
    public MultiresolutionHashGrid<T>? HashGrid { get; }

    /// <summary>The occupancy grid the marcher prunes with, or null (hierarchical sampling, or pruning off).</summary>
    public NeuSOccupancyGrid? OccupancyGrid { get; }

    /// <summary>The current finite-difference step (see <see cref="NeuSOptions.FiniteDifferenceEpsilonAt"/>).</summary>
    public double FiniteDifferenceEpsilon { get; private set; }

    /// <summary>
    /// The per-iteration schedules of instant-nsr-pl's <c>update_step</c> that live in the field: the hash grid's
    /// progressive band and the finite-difference step.
    /// </summary>
    /// <param name="iteration">The training iteration.</param>
    public void UpdateStep(int iteration)
    {
        UpdateHashGridLevels(iteration);
        FiniteDifferenceEpsilon = Options.FiniteDifferenceEpsilonAt(iteration);
    }

    /// <summary>
    /// nerfacc's <c>occupancy_grid.every_n_step</c> with instant-nsr-pl's <c>occ_eval_fn</c>: the S-density opacity of a
    /// marching step at each point, <c>clip((Φ(f + δ/2) − Φ(f − δ/2) + 1e-5) / (Φ(f + δ/2) + 1e-5), 0, 1)</c> with
    /// <c>Φ(u) = sigmoid(s·u)</c> and δ the marching step. Returns whether the grid was refreshed.
    /// </summary>
    /// <param name="iteration">The training iteration.</param>
    /// <param name="random">The cell and jitter generator.</param>
    public bool UpdateOccupancyGrid(int iteration, Random random)
    {
        if (OccupancyGrid is not { } grid) return false;
        var o = Options;
        return grid.EveryNStep(iteration, o.OccupancyGridUpdateInterval, OccupancyAlpha, o.OccupancyGridThreshold,
            o.OccupancyGridEmaDecay, o.OccupancyGridWarmUpIterations, random);
    }

    /// <summary>The occupancy evaluation of <see cref="UpdateOccupancyGrid"/> at <c>N</c> points (<c>N · 3</c> values).</summary>
    internal double[] OccupancyAlpha(double[] points)
    {
        if (points is null || points.Length % 3 != 0) throw new ArgumentException("The points must be N · 3 values.", nameof(points));
        int n = points.Length / 3;
        var alpha = new double[n];
        double s = Math.Min(1e6, Math.Max(1e-6, Math.Exp(10.0 * NeuSHost<T>.ToDoubles(Variance)[0])));
        double half = Options.MarchingStepSize * 0.5;
        int chunk = Math.Max(1, Options.GridChunkSize);
        for (int start = 0; start < n; start += chunk)
        {
            int count = Math.Min(chunk, n - start);
            var block = new double[count * 3];
            Array.Copy(points, start * 3, block, 0, count * 3);
            var sdf = HostSdf(block, count);
            for (int i = 0; i < count; i++)
            {
                double prev = Sigmoid((sdf[i] + half) * s), next = Sigmoid((sdf[i] - half) * s);
                alpha[start + i] = Math.Min(1.0, Math.Max(0.0, (prev - next + 1e-5) / (prev + 1e-5)));
            }
        }

        return alpha;
    }

    private static double Sigmoid(double u) => u >= 0 ? 1.0 / (1.0 + Math.Exp(-u)) : Math.Exp(u) / (1.0 + Math.Exp(u));

    /// <summary>The geometry's trainable tensors: the hash grid's tables (if any) and the SDF network.</summary>
    public IReadOnlyList<Tensor<T>> GeometryParameters
        => (HashGrid?.Tables ?? Array.Empty<Tensor<T>>()).Concat(SdfNetwork.Parameters).ToList();

    /// <summary>
    /// Sets the hash grid's active levels for <paramref name="iteration"/> (instant-nsr-pl's progressive band):
    /// <c>min(start + max(iteration − startIteration, 0) / updateIterations, levels)</c>. No effect without a grid.
    /// </summary>
    /// <param name="iteration">The training iteration.</param>
    public void UpdateHashGridLevels(int iteration)
    {
        if (HashGrid is not { } grid) return;
        grid.ActiveLevels = Options.ActiveHashGridLevels(iteration);
    }

    /// <summary>
    /// The SDF network at <c>[N, 3]</c> points with whatever encodes them: NeuS's embedding, or the hash grid over the
    /// scene box <c>[−r, r]³</c> (instant-nsr-pl's <c>contract_to_unisphere</c> to the unit cube, then
    /// <c>include_xyz</c> with <c>x / r</c>). The gradient is the analytic one (the embedding's or the grid's spatial
    /// Jacobian through the network) or, with <see cref="NeuSGradientMode.FiniteDifference"/>, instant-nsr-pl's
    /// central differences <c>(f(x + εeᵢ) − f(x − εeᵢ)) / 2ε</c> with the offset points clamped to the box.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="points">The points.</param>
    /// <param name="withGradient">Whether to compute ∇ₓ SDF.</param>
    public NeuSSdfOutput<T> QueryNetwork(IEngine engine, Tensor<T> points, bool withGradient)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (!withGradient || Options.GradientMode == NeuSGradientMode.Analytic) return QueryCore(engine, points, withGradient);

        var output = QueryCore(engine, points, withGradient: false);
        int n = points.Shape[0];
        var host = NeuSHost<T>.ToDoubles(points);
        double eps = FiniteDifferenceEpsilon, r = Options.SceneRadius;
        // offsets [+x, −x, +y, −y, +z, −z] per point, clamped to [−r, r].
        var offsets = new double[n * 18];
        for (int p = 0; p < n; p++)
            for (int k = 0; k < 6; k++)
                for (int c = 0; c < 3; c++)
                {
                    double v = host[p * 3 + c] + (c == k / 2 ? (k % 2 == 0 ? eps : -eps) : 0.0);
                    offsets[(p * 6 + k) * 3 + c] = Math.Max(-r, Math.Min(r, v));
                }

        var shifted = QueryCore(engine, NeuSHost<T>.FromDoubles(offsets, n * 6, 3), withGradient: false).Sdf;
        // Rows (point, axis), columns (+, −): grad = 0.5 · (sdf[..., 0::2] − sdf[..., 1::2]) / ε.
        var pairs = engine.Reshape(shifted, new[] { n * 3, 2 });
        var plus = engine.Reshape(engine.TensorSlice(pairs, new[] { 0, 0 }, new[] { n * 3, 1 }), new[] { n, 3 });
        var minus = engine.Reshape(engine.TensorSlice(pairs, new[] { 0, 1 }, new[] { n * 3, 1 }), new[] { n, 3 });
        var gradient = engine.TensorMultiplyScalar(engine.TensorSubtract(plus, minus), NeuSHost<T>.Scalar(0.5 / eps));
        return new NeuSSdfOutput<T>(output.Sdf, output.Features, gradient);
    }

    // The network with its encoding and, when asked, the analytic gradient.
    private NeuSSdfOutput<T> QueryCore(IEngine engine, Tensor<T> points, bool withGradient)
    {
        if (HashGrid is not { } grid) return SdfNetwork.Forward(engine, points, null, null, withGradient);
        double r = Options.SceneRadius;
        var host = NeuSHost<T>.ToDoubles(points);
        var unit = new double[host.Length];
        for (int i = 0; i < host.Length; i++) unit[i] = (host[i] / r + 1.0) * 0.5;
        var (features, jacobian) = grid.Encode(engine, unit, withGradient);
        Tensor<T>[]? spatial = null;
        if (jacobian is not null)
        {
            // The network sees x̂ = x / r; d(unit)/dx̂ = 1/2.
            spatial = new Tensor<T>[3];
            for (int d = 0; d < 3; d++) spatial[d] = engine.TensorMultiplyScalar(jacobian[d], NeuSHost<T>.Scalar(0.5));
        }

        if (r == 1.0) return SdfNetwork.Forward(engine, points, features, spatial, withGradient);
        var scaled = SdfNetwork.Forward(engine, engine.TensorMultiplyScalar(points, NeuSHost<T>.Scalar(1.0 / r)), features,
            spatial, withGradient);
        // ∇ₓf = ∇ₓ̂f / r.
        var gradient = scaled.Gradient is null ? null : engine.TensorMultiplyScalar(scaled.Gradient, NeuSHost<T>.Scalar(1.0 / r));
        return new NeuSSdfOutput<T>(scaled.Sdf, scaled.Features, gradient);
    }

    /// <summary>
    /// The color network's normal and feature inputs: the gradient and the features, or for
    /// <see cref="NeuSRenderingMode.VolumeRadiance"/> the unit normal and the SDF network's whole output.
    /// </summary>
    private (Tensor<T> Normals, Tensor<T> Features) ColorInputs(IEngine engine, NeuSSdfOutput<T> output, Tensor<T> gradient)
    {
        if (Options.RenderingMode != NeuSRenderingMode.VolumeRadiance) return (gradient, output.Features);
        int n = gradient.Shape[0];
        var unitNormals = engine.TensorDivide(gradient, NeuSRenderer<T>.Norm(engine, gradient));
        var sdf = engine.Reshape(output.Sdf, new[] { n, 1 });
        var all = output.Features.Shape[output.Features.Shape.Length - 1] == 0
            ? sdf
            : engine.TensorConcatenate(new[] { sdf, output.Features }, 1);
        return (unitNormals, all);
    }

    /// <summary>The variance parameter v, one element; the sharpness is <c>exp(10·v)</c>.</summary>
    public Tensor<T> Variance { get; private set; }

    /// <summary>The trainable tensors: SDF network, color network, variance.</summary>
    public IReadOnlyList<Tensor<T>> Parameters
        => GeometryParameters.Concat(ColorNetwork.Parameters).Concat(new[] { Variance }).ToList();

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => (HashGrid?.ParameterCount ?? 0) + SdfNetwork.ParameterCount + ColorNetwork.ParameterCount + 1;

    /// <summary>The parameters under the reference checkpoint's names.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters()
    {
        if (HashGrid is not null)
            foreach (var p in HashGrid.NamedParameters(HashGridPrefix)) yield return p;
        foreach (var p in SdfNetwork.NamedParameters()) yield return new KeyValuePair<string, Tensor<T>>(SdfPrefix + p.Key, p.Value);
        foreach (var p in ColorNetwork.NamedParameters()) yield return new KeyValuePair<string, Tensor<T>>(ColorPrefix + p.Key, p.Value);
        yield return new KeyValuePair<string, Tensor<T>>(VarianceName, Variance);
    }

    /// <summary>
    /// Loads parameters by the names <see cref="NamedParameters"/> lists (a converted reference checkpoint). Every
    /// name must be known; missing names keep their current values.
    /// </summary>
    /// <param name="parameters">Name to tensor.</param>
    public void LoadParameters(IReadOnlyDictionary<string, Tensor<T>> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        foreach (var entry in parameters)
        {
            bool loaded = HashGrid is not null && entry.Key.StartsWith(HashGridPrefix, StringComparison.Ordinal)
                ? HashGrid.TryLoad(HashGridPrefix, entry.Key, entry.Value)
                : entry.Key.StartsWith(SdfPrefix, StringComparison.Ordinal)
                ? SdfNetwork.TryLoad(entry.Key.Substring(SdfPrefix.Length), entry.Value)
                : entry.Key.StartsWith(ColorPrefix, StringComparison.Ordinal)
                    ? ColorNetwork.TryLoad(entry.Key.Substring(ColorPrefix.Length), entry.Value)
                    : entry.Key == VarianceName && LoadVariance(entry.Value);
            if (!loaded) throw new ArgumentException($"Unknown NeuS parameter '{entry.Key}'.", nameof(parameters));
        }
    }

    /// <inheritdoc/>
    public Tensor<T> QuerySdf(Tensor<T> points)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        var sdf = QueryNetwork(_engine, points, withGradient: false).Sdf;
        return _engine.Reshape(sdf, new[] { points.Shape[0] });
    }

    /// <inheritdoc/>
    public Tensor<T> SampleGrid(int resolution, SdfGridBounds bounds)
        => NeuSHost<T>.SampleGrid(_engine, QuerySdf, resolution, bounds, Options.GridChunkSize);

    /// <summary>The unit outward normals ∇f/‖∇f‖ at <c>[N, 3]</c> points, without recording a tape.</summary>
    /// <param name="points">The points.</param>
    public Tensor<T> QueryNormals(Tensor<T> points)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var gradient = QueryNetwork(_engine, points, withGradient: true).Gradient
            ?? throw new InvalidOperationException("The SDF network returned no gradient.");
        return _engine.TensorDivide(gradient, NeuSRenderer<T>.Norm(_engine, gradient));
    }

    /// <summary>
    /// The color network's output at <c>[N, 3]</c> points seen along <paramref name="viewDirections"/>, without
    /// recording a tape: what a surface point looks like from that direction (for texturing an extracted mesh, pass
    /// the negated normal as the view direction).
    /// </summary>
    /// <param name="points">The points, <c>[N, 3]</c>.</param>
    /// <param name="viewDirections">Unit view directions, <c>[N, 3]</c>.</param>
    public Tensor<T> QueryColor(Tensor<T> points, Tensor<T> viewDirections)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var output = QueryNetwork(_engine, points, withGradient: true);
        var gradient = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
        var (normals, features) = ColorInputs(_engine, output, gradient);
        return ColorNetwork.Forward(_engine, points, normals, viewDirections, features);
    }

    /// <summary>
    /// Renders a batch of rays (reference <c>NeuSRenderer.render</c> with <c>n_outside = 0</c>).
    /// </summary>
    /// <param name="origins">Ray origins, <c>[R, 3]</c> on the host.</param>
    /// <param name="directions">Unit ray directions, <c>[R, 3]</c> on the host.</param>
    /// <param name="cosAnnealRatio">The slope annealing ratio (1 after annealing).</param>
    /// <param name="perturb">The generator of the per-ray sample shift, or null for none (evaluation).</param>
    public NeuSRenderResult<T> Render(double[] origins, double[] directions, double cosAnnealRatio, Random? perturb)
    {
        if (origins is null) throw new ArgumentNullException(nameof(origins));
        if (directions is null) throw new ArgumentNullException(nameof(directions));
        if (origins.Length != directions.Length || origins.Length % 3 != 0)
            throw new ArgumentException("Origins and directions must both be [R, 3].", nameof(directions));
        int rays = origins.Length / 3;
        var o = Options;
        if (o.Sampler == NeuSRaySampler.OccupancyGrid) return RenderMarching(origins, directions, cosAnnealRatio, perturb);
        bool syncDreamer = o.RendererVariant == NeuSRendererVariant.SyncDreamer;
        var (near, far) = NeuSSampling.NearFarFromSphere(origins, directions, o.NearFarRadius);
        var z = NeuSSampling.UniformDepths(near, far, o.Samples, perturb);
        int samples = o.Samples;
        if (o.ImportanceSamples > 0)
        {
            var sdf = HostSdf(NeuSSampling.Points(origins, directions, z, samples), rays * samples);
            int perStep = o.ImportanceSamples / o.UpSampleSteps;
            for (int i = 0; i < o.UpSampleSteps; i++)
            {
                var sectionValid = syncDreamer
                    ? NeuSSampling.SectionsTouchingCube(origins, directions, z, samples, o.CubeHalfSide)
                    : null;
                var newZ = NeuSSampling.UpSample(origins, directions, z, sdf, samples, perStep, 64.0 * Math.Pow(2.0, i),
                    -1e3, sectionValid, zeroInvalidAlpha: false);
                bool last = i + 1 == o.UpSampleSteps;
                double[]? newSdf = last ? null : HostSdf(NeuSSampling.Points(origins, directions, newZ, perStep), rays * perStep);
                var merged = NeuSSampling.Merge(z, last ? null : sdf, samples, newZ, newSdf, perStep);
                z = merged.Z;
                sdf = merged.Values ?? sdf;
                samples += perStep;
            }
        }

        // NeuS assumes the region of interest is the unit sphere: the last section is 2 / n_samples long. SyncDreamer
        // repeats the previous section instead.
        var (dists, mid) = NeuSRenderer<T>.Sections(z, samples, syncDreamer ? (double?)null : 2.0 / o.Samples);
        int n = rays * samples;
        var pointsHost = NeuSSampling.Points(origins, directions, mid, samples);
        var dirsHost = new double[n * 3];
        var relaxInside = new double[n];
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < samples; s++)
            {
                int i = r * samples + s;
                double radius2 = 0;
                for (int c = 0; c < 3; c++)
                {
                    dirsHost[i * 3 + c] = directions[r * 3 + c];
                    radius2 += pointsHost[i * 3 + c] * pointsHost[i * 3 + c];
                }

                relaxInside[i] = syncDreamer
                    ? (Math.Abs(pointsHost[i * 3]) <= o.CubeHalfSide && Math.Abs(pointsHost[i * 3 + 1]) <= o.CubeHalfSide
                        && Math.Abs(pointsHost[i * 3 + 2]) <= o.CubeHalfSide ? 1.0 : 0.0)
                    : Math.Sqrt(radius2) < 1.2 ? 1.0 : 0.0;
            }

        var e = _engine;
        var points = NeuSHost<T>.FromDoubles(pointsHost, n, 3);
        var dirs = NeuSHost<T>.FromDoubles(dirsHost, n, 3);
        var output = QueryNetwork(e, points, withGradient: true);
        var gradients = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
        // SyncDreamer's color network reads -dirs (color_network(points, gradients, -dirs, feature_vector)).
        var (colorNormals, colorFeatures) = ColorInputs(e, output, gradients);
        var sampledColor = ColorNetwork.Forward(e, points, colorNormals, syncDreamer ? e.TensorNegate(dirs) : dirs, colorFeatures);
        var trueCos = e.ReduceSum(e.TensorMultiply(dirs, gradients), new[] { 1 }, true);
        // SyncDreamer evaluates only the samples inside its cube; the others keep zero opacity.
        var alpha = NeuSRenderer<T>.Alpha(e, output.Sdf, trueCos, NeuSHost<T>.FromDoubles(dists, n, 1),
            NeuSRenderer<T>.InverseS(e, Variance), cosAnnealRatio, null,
            syncDreamer ? NeuSHost<T>.FromDoubles(relaxInside, n, 1) : null);
        var weights = NeuSRenderer<T>.Weights(e, e.Reshape(alpha, new[] { rays, samples }));
        var weightSum = e.ReduceSum(weights, new[] { 1 }, true);
        var color = NeuSRenderer<T>.Composite(e, weights, sampledColor);
        if (o.BackgroundColor is not null)
        {
            var background = NeuSHost<T>.FromDoubles(o.BackgroundColor, 1, 3);
            var uncovered = e.TensorAddScalar(e.TensorNegate(weightSum), NeuSHost<T>.Scalar(1.0));
            color = e.TensorAdd(color, e.TensorMultiply(uncovered, background));
        }

        var unitNormals = e.TensorDivide(gradients, NeuSRenderer<T>.Norm(e, gradients));
        return new NeuSRenderResult<T>
        {
            Color = color,
            Weights = weights,
            WeightSum = weightSum,
            Depth = NeuSRenderer<T>.Composite(e, weights, NeuSHost<T>.FromDoubles(mid, n, 1)),
            Normal = NeuSRenderer<T>.Composite(e, weights, unitNormals),
            GradientError = NeuSRenderer<T>.EikonalError(e, gradients, relaxInside, perSampleEpsilon: syncDreamer),
            Sdf = e.Reshape(output.Sdf, new[] { rays, samples }),
            MidDepths = mid,
            SamplesPerRay = samples,
            SampleCount = n,
        };
    }

    /// <summary>
    /// The marched samples of <paramref name="rays"/> rays (nerfacc v0.3 <c>ray_marching</c> with <c>cone_angle = 0</c>
    /// and <c>alpha_thre = 0</c>): each ray's slab interval with the box <c>[−r, r]³</c> (near clamped at 0, shifted by
    /// U[0, δ) when stratified), cut into steps of δ = <see cref="NeuSOptions.MarchingStepSize"/> whose midpoints lie
    /// before the far end, keeping the steps whose midpoint is in an occupied cell.
    /// </summary>
    /// <returns>Each sample's ray, its interval start and the per-ray sample counts.</returns>
    internal (int[] RayIndices, double[] Starts, int[] Counts) MarchSamples(double[] origins, double[] directions, Random? perturb)
    {
        int rays = origins.Length / 3;
        double r = Options.SceneRadius, step = Options.MarchingStepSize;
        var rayIndices = new List<int>();
        var starts = new List<double>();
        var counts = new int[rays];
        var grid = OccupancyGrid;
        for (int ray = 0; ray < rays; ray++)
        {
            double near = 0.0, far = double.PositiveInfinity;
            bool hit = true;
            for (int c = 0; c < 3 && hit; c++)
            {
                double o = origins[ray * 3 + c], d = directions[ray * 3 + c];
                if (Math.Abs(d) < 1e-12)
                {
                    if (o < -r || o > r) hit = false;
                    continue;
                }

                double t1 = (-r - o) / d, t2 = (r - o) / d;
                near = Math.Max(near, Math.Min(t1, t2));
                far = Math.Min(far, Math.Max(t1, t2));
            }

            // The stratified shift is drawn for every ray, as torch.rand_like(t_min) does.
            double shift = perturb is null ? 0.0 : perturb.NextDouble() * step;
            if (!hit || far <= near) continue;
            double t0 = near + shift;
            for (long k = 0; ; k++)
            {
                double start = t0 + k * step, mid = start + 0.5 * step;
                if (!(mid < far)) break;
                double x = origins[ray * 3] + mid * directions[ray * 3];
                double y = origins[ray * 3 + 1] + mid * directions[ray * 3 + 1];
                double z = origins[ray * 3 + 2] + mid * directions[ray * 3 + 2];
                if (grid is not null && !grid.IsOccupied(x, y, z)) continue;
                rayIndices.Add(ray);
                starts.Add(start);
                counts[ray]++;
            }
        }

        return (rayIndices.ToArray(), starts.ToArray(), counts);
    }

    /// <summary>
    /// instant-nsr-pl's <c>NeuSModel.forward_</c> on the marched samples: the SDF and its gradient at each step's
    /// midpoint, NeuS's alpha with the step δ, nerfacc's <c>render_weight_from_alpha</c> (weights get zero once the
    /// transmittance drops below <see cref="NeuSOptions.TransmittanceEarlyStop"/>) and the per-ray sums. The packed
    /// samples are laid out per ray, padded to the longest ray with zero-opacity entries.
    /// </summary>
    private NeuSRenderResult<T> RenderMarching(double[] origins, double[] directions, double cosAnnealRatio, Random? perturb)
    {
        var o = Options;
        var e = _engine;
        int rays = origins.Length / 3;
        double step = o.MarchingStepSize;
        var (rayIndices, starts, counts) = MarchSamples(origins, directions, perturb);
        int n = rayIndices.Length;
        int width = Math.Max(1, counts.Length == 0 ? 1 : counts.Max());
        // Padded layout: slot (ray, j) holds the ray's j-th sample, or index n (a zero row) past its end.
        var slots = new int[rays * width];
        for (int i = 0; i < slots.Length; i++) slots[i] = n;
        var cursor = new int[rays];
        var midHost = new double[n];
        for (int i = 0; i < n; i++)
        {
            int ray = rayIndices[i];
            slots[ray * width + cursor[ray]++] = i;
            midHost[i] = starts[i] + 0.5 * step;
        }

        var paddedMid = new double[rays * width];
        for (int i = 0; i < slots.Length; i++) paddedMid[i] = slots[i] < n ? midHost[slots[i]] : 0.0;
        var slotIndex = NeuSHost<T>.Indices(slots);

        Tensor<T> Pad(Tensor<T> packed, int channels)
        {
            var extended = n == 0
                ? new Tensor<T>(new[] { 1, channels })
                : e.TensorConcatenate(new[] { packed, new Tensor<T>(new[] { 1, channels }) }, 0);
            return e.TensorIndexSelect(extended, slotIndex, 0);
        }

        Tensor<T> color;
        Tensor<T> weights, weightSum, normal, gradientError, sdfPadded;
        if (n == 0)
        {
            weights = new Tensor<T>(new[] { rays, width });
            weightSum = new Tensor<T>(new[] { rays, 1 });
            color = new Tensor<T>(new[] { rays, 3 });
            normal = new Tensor<T>(new[] { rays, 3 });
            gradientError = new Tensor<T>(new[] { 1 });
            sdfPadded = new Tensor<T>(new[] { rays, width });
        }
        else
        {
            var pointsHost = new double[n * 3];
            var dirsHost = new double[n * 3];
            for (int i = 0; i < n; i++)
                for (int c = 0; c < 3; c++)
                {
                    int ray = rayIndices[i];
                    dirsHost[i * 3 + c] = directions[ray * 3 + c];
                    pointsHost[i * 3 + c] = origins[ray * 3 + c] + midHost[i] * directions[ray * 3 + c];
                }

            var points = NeuSHost<T>.FromDoubles(pointsHost, n, 3);
            var dirs = NeuSHost<T>.FromDoubles(dirsHost, n, 3);
            var output = QueryNetwork(e, points, withGradient: true);
            var gradients = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
            var (colorNormals, colorFeatures) = ColorInputs(e, output, gradients);
            var sampledColor = ColorNetwork.Forward(e, points, colorNormals, dirs, colorFeatures);
            var unitNormals = e.TensorDivide(gradients, NeuSRenderer<T>.Norm(e, gradients));
            // get_alpha's true_cos uses the normalized gradient.
            var trueCos = e.ReduceSum(e.TensorMultiply(dirs, unitNormals), new[] { 1 }, true);
            var dists = new double[n];
            for (int i = 0; i < n; i++) dists[i] = step;
            var sdfColumn = e.Reshape(output.Sdf, new[] { n, 1 });
            var alpha = NeuSRenderer<T>.Alpha(e, sdfColumn, trueCos, NeuSHost<T>.FromDoubles(dists, n, 1),
                NeuSRenderer<T>.InverseS(e, Variance), cosAnnealRatio, null, null);
            var paddedAlpha = e.Reshape(Pad(alpha, 1), new[] { rays, width });
            weights = NeuSRenderer<T>.Weights(e, paddedAlpha);
            if (o.TransmittanceEarlyStop > 0)
                weights = e.TensorMultiply(weights, NeuSHost<T>.FromDoubles(
                    EarlyStopMask(NeuSHost<T>.ToDoubles(paddedAlpha), rays, width, o.TransmittanceEarlyStop), rays, width));
            weightSum = e.ReduceSum(weights, new[] { 1 }, true);
            color = NeuSRenderer<T>.Composite(e, weights, Pad(sampledColor, 3));
            normal = NeuSRenderer<T>.Composite(e, weights, Pad(unitNormals, 3));
            // ((‖∇f‖ − 1)²).mean() over the marched samples.
            var everySample = new double[n];
            for (int i = 0; i < n; i++) everySample[i] = 1.0;
            gradientError = NeuSRenderer<T>.EikonalError(e, gradients, everySample);
            sdfPadded = e.Reshape(Pad(sdfColumn, 1), new[] { rays, width });
        }

        if (o.BackgroundColor is not null)
        {
            var background = NeuSHost<T>.FromDoubles(o.BackgroundColor, 1, 3);
            var uncovered = e.TensorAddScalar(e.TensorNegate(weightSum), NeuSHost<T>.Scalar(1.0));
            color = e.TensorAdd(color, e.TensorMultiply(uncovered, background));
        }

        return new NeuSRenderResult<T>
        {
            Color = color,
            Weights = weights,
            WeightSum = weightSum,
            Depth = NeuSRenderer<T>.Composite(e, weights, NeuSHost<T>.FromDoubles(paddedMid, rays * width, 1)),
            Normal = normal,
            GradientError = gradientError,
            Sdf = sdfPadded,
            MidDepths = paddedMid,
            SamplesPerRay = width,
            SampleCount = n,
        };
    }

    /// <summary>
    /// nerfacc's early stop: 1 where the transmittance before a sample, <c>Π_{j&lt;i}(1 − αⱼ)</c>, is at least
    /// <paramref name="epsilon"/>, 0 from the first sample where it is below.
    /// </summary>
    internal static double[] EarlyStopMask(double[] alpha, int rays, int width, double epsilon)
    {
        var mask = new double[rays * width];
        for (int ray = 0; ray < rays; ray++)
        {
            double transmittance = 1.0;
            for (int j = 0; j < width; j++)
            {
                int i = ray * width + j;
                if (transmittance < epsilon) break;
                mask[i] = 1.0;
                transmittance *= 1.0 - alpha[i];
            }
        }

        return mask;
    }

    /// <summary>Renders a full image from <paramref name="camera"/>, <c>[H, W, 3]</c>, without recording a tape.</summary>
    /// <param name="camera">The camera.</param>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    /// <param name="raysPerBatch">Rays rendered together.</param>
    public Tensor<T> RenderImage(NeuSCamera camera, int width, int height, int raysPerBatch = 512)
    {
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The image must be non-empty.");
        if (raysPerBatch <= 0) throw new ArgumentOutOfRangeException(nameof(raysPerBatch));
        int total = width * height;
        var pixels = new double[total * 3];
        var origin = new double[3];
        var direction = new double[3];
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            for (int start = 0; start < total; start += raysPerBatch)
            {
                int count = Math.Min(raysPerBatch, total - start);
                var origins = new double[count * 3];
                var directions = new double[count * 3];
                for (int k = 0; k < count; k++)
                {
                    int p = start + k;
                    camera.Ray(p % width, p / width, origin, direction);
                    for (int c = 0; c < 3; c++)
                    {
                        origins[k * 3 + c] = origin[c];
                        directions[k * 3 + c] = direction[c];
                    }
                }

                var rendered = NeuSHost<T>.ToDoubles(Render(origins, directions, 1.0, null).Color);
                Array.Copy(rendered, 0, pixels, start * 3, count * 3);
            }
        }

        return NeuSHost<T>.FromDoubles(pixels, height, width, 3);
    }

    private double[] HostSdf(double[] points, int count)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return NeuSHost<T>.ToDoubles(QuerySdf(NeuSHost<T>.FromDoubles(points, count, 3)));
    }

    private bool LoadVariance(Tensor<T> value)
    {
        if (value is null || value.Length != 1) throw new ArgumentException("The variance is a single value.", nameof(value));
        Variance = value.Reshape(1);
        return true;
    }
}
