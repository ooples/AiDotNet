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
/// </remarks>
internal sealed class NeuSField<T> : ISignedDistanceField<T>
{
    private const string SdfPrefix = "sdf_network_fine.";
    private const string ColorPrefix = "color_network_fine.";
    private const string VarianceName = "variance_network_fine.variance";

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
        SdfNetwork = NeuSSdfMlp<T>.CreateNeuS(options.SdfHiddenDim, options.SdfHiddenLayers, options.SdfFeatureDim + 1,
            options.SdfSkipLayers, options.SdfFrequencies, options.SdfInitialSphereRadius, options.SdfScale,
            options.GeometricInit, options.WeightNorm, options.InsideOutside, options.SoftplusBeta, random);
        ColorNetwork = new NeuSRenderingNetwork<T>(options.SdfFeatureDim, options.RenderingMode, options.ColorHiddenDim,
            options.ColorHiddenLayers, options.ViewFrequencies, options.WeightNorm, options.SqueezeColor, random);
        Variance = NeuSHost<T>.FromDoubles(new[] { options.InitialVariance }, 1);
    }

    /// <summary>The configuration.</summary>
    public NeuSOptions Options { get; }

    /// <summary>The SDF network.</summary>
    public NeuSSdfMlp<T> SdfNetwork { get; }

    /// <summary>The color network.</summary>
    public NeuSRenderingNetwork<T> ColorNetwork { get; }

    /// <summary>The variance parameter v, one element; the sharpness is <c>exp(10·v)</c>.</summary>
    public Tensor<T> Variance { get; private set; }

    /// <summary>The trainable tensors: SDF network, color network, variance.</summary>
    public IReadOnlyList<Tensor<T>> Parameters
        => SdfNetwork.Parameters.Concat(ColorNetwork.Parameters).Concat(new[] { Variance }).ToList();

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => SdfNetwork.ParameterCount + ColorNetwork.ParameterCount + 1;

    /// <summary>The parameters under the reference checkpoint's names.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters()
    {
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
            bool loaded = entry.Key.StartsWith(SdfPrefix, StringComparison.Ordinal)
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
        var sdf = SdfNetwork.Forward(_engine, points, null, null, withGradient: false).Sdf;
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
        var gradient = SdfNetwork.Forward(_engine, points, null, null, withGradient: true).Gradient
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
        var output = SdfNetwork.Forward(_engine, points, null, null, withGradient: true);
        var gradient = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
        return ColorNetwork.Forward(_engine, points, gradient, viewDirections, output.Features);
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
        var (near, far) = NeuSSampling.NearFarFromSphere(origins, directions);
        var z = NeuSSampling.UniformDepths(near, far, o.Samples, perturb);
        int samples = o.Samples;
        if (o.ImportanceSamples > 0)
        {
            var sdf = HostSdf(NeuSSampling.Points(origins, directions, z, samples), rays * samples);
            int perStep = o.ImportanceSamples / o.UpSampleSteps;
            for (int i = 0; i < o.UpSampleSteps; i++)
            {
                var newZ = NeuSSampling.UpSample(origins, directions, z, sdf, samples, perStep, 64.0 * Math.Pow(2.0, i),
                    -1e3, null);
                bool last = i + 1 == o.UpSampleSteps;
                double[]? newSdf = last ? null : HostSdf(NeuSSampling.Points(origins, directions, newZ, perStep), rays * perStep);
                var merged = NeuSSampling.Merge(z, last ? null : sdf, samples, newZ, newSdf, perStep);
                z = merged.Z;
                sdf = merged.Values ?? sdf;
                samples += perStep;
            }
        }

        // NeuS assumes the region of interest is the unit sphere: the last section is 2 / n_samples long.
        var (dists, mid) = NeuSRenderer<T>.Sections(z, samples, 2.0 / o.Samples);
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

                relaxInside[i] = Math.Sqrt(radius2) < 1.2 ? 1.0 : 0.0;
            }

        var e = _engine;
        var points = NeuSHost<T>.FromDoubles(pointsHost, n, 3);
        var dirs = NeuSHost<T>.FromDoubles(dirsHost, n, 3);
        var output = SdfNetwork.Forward(e, points, null, null, withGradient: true);
        var gradients = output.Gradient ?? throw new InvalidOperationException("The SDF network returned no gradient.");
        var sampledColor = ColorNetwork.Forward(e, points, gradients, dirs, output.Features);
        var trueCos = e.ReduceSum(e.TensorMultiply(dirs, gradients), new[] { 1 }, true);
        var alpha = NeuSRenderer<T>.Alpha(e, output.Sdf, trueCos, NeuSHost<T>.FromDoubles(dists, n, 1),
            NeuSRenderer<T>.InverseS(e, Variance), cosAnnealRatio, null, null);
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
            GradientError = NeuSRenderer<T>.EikonalError(e, gradients, relaxInside),
            Sdf = e.Reshape(output.Sdf, new[] { rays, samples }),
            MidDepths = mid,
            SamplesPerRay = samples,
        };
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
