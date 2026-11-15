namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// One scene encoded by <see cref="SparseNeuS{T}"/>: the regularized geometry volume built from the input views,
/// with the SparseNeuS renderer and a dense SDF grid sampler for mesh extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SDF at a point is the latent SDF MLP applied to the point and the geometry volume's trilinear sample there
/// (reference <c>SparseSdfNetwork.sdf</c>). Points whose nearest voxel is not active (seen by fewer than two views)
/// are empty space: the renderer gives them SDF 100 and zero opacity, as <c>render_core</c> does, and
/// <see cref="SampleGrid"/> reports 100 there so marching cubes leaves them outside.
/// </para>
/// </remarks>
internal sealed class SparseNeuSScene<T> : ISignedDistanceField<T>
{
    private const double EmptySdf = 100.0;

    private readonly SparseNeuS<T> _network;
    private readonly IEngine _engine;
    private readonly IReadOnlyList<Tensor<T>> _features;
    private readonly IReadOnlyList<Tensor<T>> _colors;
    private readonly IReadOnlyList<NeuSCamera> _cameras;

    internal SparseNeuSScene(SparseNeuS<T> network, IEngine engine, Tensor<T> volume, double[] active,
        IReadOnlyList<Tensor<T>> features, IReadOnlyList<Tensor<T>> colors, IReadOnlyList<NeuSCamera> cameras, int width, int height)
    {
        _network = network;
        _engine = engine;
        Volume = volume;
        Active = active;
        _features = features;
        _colors = colors;
        _cameras = cameras;
        Width = width;
        Height = height;
    }

    /// <summary>The geometry encoding, <c>[X·Y·Z, C]</c>, x slowest.</summary>
    public Tensor<T> Volume { get; }

    /// <summary>The 0/1 active-voxel mask (seen by at least two views), <c>X·Y·Z</c>.</summary>
    public double[] Active { get; }

    /// <summary>The source image width.</summary>
    public int Width { get; }

    /// <summary>The source image height.</summary>
    public int Height { get; }

    /// <summary>The number of source views.</summary>
    public int ViewCount => _cameras.Count;

    private SparseNeuSOptions Options => _network.Options;

    /// <inheritdoc/>
    public Tensor<T> QuerySdf(Tensor<T> points)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        int n = points.Shape[0];
        var host = NeuSHost<T>.ToDoubles(points);
        var sdf = Query(host, withGradient: false).Output.Sdf;
        var keep = NeuSHost<T>.FromDoubles(NearestActive(host), n, 1);
        return _engine.Reshape(Masked(sdf, keep, EmptySdf), new[] { n });
    }

    /// <inheritdoc/>
    public Tensor<T> SampleGrid(int resolution, SdfGridBounds bounds)
        => NeuSHost<T>.SampleGrid(_engine, QuerySdf, resolution, bounds, Options.GridChunkSize);

    /// <summary>The SDF network's output and latent at host points.</summary>
    internal (NeuSSdfOutput<T> Output, Tensor<T> Latent, NeuSGridInterpolation<T> Interpolation) Query(double[] points, bool withGradient)
    {
        var o = Options;
        var coordinates = new double[points.Length];
        for (int i = 0; i < points.Length; i++) coordinates[i] = (points[i] - o.VolumeOrigin[i % 3]) / o.VoxelSize;
        var interpolation = NeuSGridInterpolation<T>.Trilinear(coordinates, o.VolumeDims, o.VoxelSize, withGradient);
        var latent = interpolation.Sample(_engine, Volume);
        var jacobian = withGradient ? interpolation.SpatialDerivatives(_engine, Volume) : null;
        var output = _network.SdfNetwork.Forward(_engine, NeuSHost<T>.FromDoubles(points, points.Length / 3, 3), latent, jacobian, withGradient);
        return (output, latent, interpolation);
    }

    /// <summary>
    /// Renders a batch of rays (reference <c>SparseNeuSRenderer.render</c> at lod 0).
    /// </summary>
    /// <param name="origins">Ray origins, <c>[R, 3]</c>.</param>
    /// <param name="directions">Unit ray directions, <c>[R, 3]</c>.</param>
    /// <param name="queryCenter">The rendering camera's center (for the ray-direction differences).</param>
    /// <param name="excludedView">A source view left out of the color blend (the reference view during training), or −1.</param>
    /// <param name="cosAnnealRatio">The slope annealing ratio.</param>
    /// <param name="perturb">The stratified-jitter generator, or null.</param>
    public SparseNeuSRenderResult<T> Render(double[] origins, double[] directions, double[] queryCenter, int excludedView,
        double cosAnnealRatio, Random? perturb)
    {
        if (origins is null || directions is null) throw new ArgumentNullException(nameof(origins));
        if (queryCenter is null || queryCenter.Length != 3) throw new ArgumentException("The query center needs 3 values.", nameof(queryCenter));
        var o = Options;
        var e = _engine;
        int rays = origins.Length / 3;
        var (near, far) = NeuSSampling.NearFarFromSphere(origins, directions);
        double sampleDist = 0;
        for (int r = 0; r < rays; r++) sampleDist += (far[r] - near[r]) / o.Samples;
        sampleDist /= Math.Max(1, rays);
        var z = NeuSSampling.StratifiedDepths(near, far, o.Samples, perturb);
        int samples = o.Samples;
        if (o.ImportanceSamples > 0)
        {
            double[] sdf;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                sdf = NeuSHost<T>.ToDoubles(Query(NeuSSampling.Points(origins, directions, z, samples), false).Output.Sdf);
            int perStep = o.ImportanceSamples / o.UpSampleSteps;
            for (int i = 0; i < o.UpSampleSteps; i++)
            {
                var pointMask = NearestActive(NeuSSampling.Points(origins, directions, z, samples));
                var sectionValid = new bool[rays * (samples - 1)];
                for (int r = 0; r < rays; r++)
                    for (int s = 0; s < samples - 1; s++)
                        sectionValid[r * (samples - 1) + s] = pointMask[r * samples + s] > 0 && pointMask[r * samples + s + 1] > 0;
                var newZ = NeuSSampling.UpSample(origins, directions, z, sdf, samples, perStep, 64.0 * Math.Pow(2.0, i), -10.0, sectionValid);
                double[] newSdf;
                using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                    newSdf = NeuSHost<T>.ToDoubles(Query(NeuSSampling.Points(origins, directions, newZ, perStep), false).Output.Sdf);
                var merged = NeuSSampling.Merge(z, sdf, samples, newZ, newSdf, perStep);
                z = merged.Z;
                sdf = merged.Values ?? sdf;
                samples += perStep;
            }
        }

        var (dists, mid) = NeuSRenderer<T>.Sections(z, samples, sampleDist);
        int n = rays * samples;
        var points = NeuSSampling.Points(origins, directions, mid, samples);
        var dirsHost = new double[n * 3];
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < samples; s++)
                for (int c = 0; c < 3; c++)
                    dirsHost[(r * samples + s) * 3 + c] = directions[r * 3 + c];

        var keepHost = NearestActive(points);
        double kept = 0;
        foreach (double k in keepHost) kept += k;
        if (kept < 1)
            for (int i = 0; i < Math.Min(100, n); i++) keepHost[i] = 1.0; // the reference's guard for empty batches

        var keep = NeuSHost<T>.FromDoubles(keepHost, n, 1);
        var (output, latent, interpolation) = Query(points, withGradient: true);
        var sdfAll = Masked(output.Sdf, keep, EmptySdf);
        var gradients = e.TensorMultiply(output.Gradient ?? throw new InvalidOperationException("No SDF gradient."), keep);
        var dirs = NeuSHost<T>.FromDoubles(dirsHost, n, 3);

        // Color blending over the source views.
        var (rgbFeatures, rayDiff, viewMask, views) = Project(points, queryCenter, excludedView, interpolation);
        var sampledColor = _network.RenderingNetwork.Forward(e, latent, rgbFeatures, rayDiff, viewMask, views);

        var trueCos = e.ReduceSum(e.TensorMultiply(dirs, gradients), new[] { 1 }, true);
        var alpha = NeuSRenderer<T>.Alpha(e, sdfAll, trueCos, NeuSHost<T>.FromDoubles(dists, n, 1),
            NeuSRenderer<T>.InverseS(e, _network.Variance), cosAnnealRatio, 10.0, keep);
        var weights = NeuSRenderer<T>.Weights(e, e.Reshape(alpha, new[] { rays, samples }));
        var colorValid = new bool[rays];
        for (int r = 0; r < rays; r++)
        {
            int seen = 0;
            for (int s = 0; s < samples; s++)
            {
                int count = 0;
                for (int v = 0; v < views; v++) count += viewMask[(r * samples + s) * views + v] > 0 ? 1 : 0;
                if (count >= 2) seen++;
            }

            colorValid[r] = seen > 8;
        }

        var unitNormals = e.TensorDivide(gradients, NeuSRenderer<T>.Norm(e, gradients));
        return new SparseNeuSRenderResult<T>
        {
            Color = NeuSRenderer<T>.Composite(e, weights, sampledColor),
            Weights = weights,
            WeightSum = e.ReduceSum(weights, new[] { 1 }, true),
            Depth = NeuSRenderer<T>.Composite(e, weights, NeuSHost<T>.FromDoubles(mid, n, 1)),
            Normal = NeuSRenderer<T>.Composite(e, weights, unitNormals),
            GradientError = NeuSRenderer<T>.EikonalError(e, gradients, keepHost),
            Sdf = e.Reshape(sdfAll, new[] { rays, samples }),
            MidDepths = mid,
            SamplesPerRay = samples,
            ColorValid = colorValid,
        };
    }

    /// <summary>Renders an image from <paramref name="camera"/>, <c>[H, W, 3]</c>, blending every source view, without a tape.</summary>
    /// <param name="camera">The camera.</param>
    /// <param name="width">The width.</param>
    /// <param name="height">The height.</param>
    /// <param name="raysPerBatch">Rays rendered together.</param>
    public Tensor<T> RenderImage(NeuSCamera camera, int width, int height, int raysPerBatch = 512)
    {
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        if (width <= 0 || height <= 0 || raysPerBatch <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        int total = width * height;
        var pixels = new double[total * 3];
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
                    camera.RayDirection((start + k) % width, (start + k) / width, direction);
                    for (int c = 0; c < 3; c++)
                    {
                        origins[k * 3 + c] = camera.Center[c];
                        directions[k * 3 + c] = direction[c];
                    }
                }

                var rendered = NeuSHost<T>.ToDoubles(Render(origins, directions, camera.Center, -1, 1.0, null).Color);
                Array.Copy(rendered, 0, pixels, start * 3, count * 3);
            }
        }

        return NeuSHost<T>.FromDoubles(pixels, height, width, 3);
    }

    /// <summary>
    /// The rendering network's colour at surface points (the reference's <c>validate_colored_mesh</c>:
    /// <c>Projector.compute_view_independent</c> then <c>rendering_network</c>), blending every source view as seen
    /// from <paramref name="queryCenter"/>, without a tape.
    /// </summary>
    /// <param name="points">The points, <c>[P, 3]</c> row-major (host).</param>
    /// <param name="queryCenter">The query camera's center (the reference's <c>query_c2w</c>).</param>
    /// <returns>The colours, <c>[P, 3]</c>.</returns>
    public Tensor<T> PointColors(double[] points, double[] queryCenter)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (points.Length % 3 != 0) throw new ArgumentException("Points need three coordinates each.", nameof(points));
        if (queryCenter is null || queryCenter.Length != 3) throw new ArgumentException("The query center needs 3 values.", nameof(queryCenter));
        int total = points.Length / 3;
        var colors = new double[total * 3];
        int chunk = Options.GridChunkSize;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            for (int start = 0; start < total; start += chunk)
            {
                int count = Math.Min(chunk, total - start);
                var part = new double[count * 3];
                Array.Copy(points, start * 3, part, 0, count * 3);
                var (_, latent, interpolation) = Query(part, withGradient: false);
                var (rgbFeatures, rayDiff, viewMask, views) = Project(part, queryCenter, -1, interpolation);
                var rgb = _network.RenderingNetwork.Forward(_engine, latent, rgbFeatures, rayDiff, viewMask, views);
                // Copied out of the query's arena: the next chunk may reuse its buffers.
                Array.Copy(NeuSHost<T>.ToDoubles(rgb), 0, colors, start * 3, count * 3);
            }
        }

        return NeuSHost<T>.FromDoubles(colors, total, 3);
    }

    /// <summary>Whether each point's nearest voxel is active (nearest-neighbor <c>grid_sample</c> of the mask volume).</summary>
    internal double[] NearestActive(double[] points)
    {
        var o = Options;
        var dims = o.VolumeDims;
        int n = points.Length / 3;
        var result = new double[n];
        for (int p = 0; p < n; p++)
        {
            int ix = (int)Math.Round((points[p * 3] - o.VolumeOrigin[0]) / o.VoxelSize, MidpointRounding.ToEven);
            int iy = (int)Math.Round((points[p * 3 + 1] - o.VolumeOrigin[1]) / o.VoxelSize, MidpointRounding.ToEven);
            int iz = (int)Math.Round((points[p * 3 + 2] - o.VolumeOrigin[2]) / o.VoxelSize, MidpointRounding.ToEven);
            bool inside = ix >= 0 && ix < dims[0] && iy >= 0 && iy < dims[1] && iz >= 0 && iz < dims[2];
            result[p] = inside ? Active[(ix * dims[1] + iy) * dims[2] + iz] : 0.0;
        }

        return result;
    }

    /// <summary>
    /// The per-view inputs of the color network (reference <c>Projector.compute</c>): sampled colors and features,
    /// direction differences and visibility, point-major over the supporting views.
    /// </summary>
    private (Tensor<T> RgbFeatures, double[] RayDiff, double[] Mask, int Views) Project(double[] points, double[] queryCenter,
        int excludedView, NeuSGridInterpolation<T> volumeInterpolation)
    {
        var e = _engine;
        int n = points.Length / 3;
        var supporting = new List<int>();
        for (int v = 0; v < _cameras.Count; v++)
            if (v != excludedView) supporting.Add(v);
        if (supporting.Count == 0) throw new InvalidOperationException("Color blending needs at least one supporting view.");
        int views = supporting.Count;

        // pts_geometry_masks: inside the volume and the trilinearly sampled validity is positive.
        var geometryValid = volumeInterpolation.SampleHost(Active);
        var o = Options;
        var rayDiff = new double[n * views * 4];
        var mask = new double[n * views];
        var perView = new Tensor<T>[views];
        var u = new double[n];
        var vv = new double[n];
        for (int k = 0; k < views; k++)
        {
            var camera = _cameras[supporting[k]];
            var inImage = new bool[n];
            for (int p = 0; p < n; p++)
            {
                double x = points[p * 3], y = points[p * 3 + 1], zc = points[p * 3 + 2];
                var (pu, pv, depth) = camera.Project(x, y, zc);
                u[p] = pu;
                vv[p] = pv;
                inImage[p] = depth > 0 && pu >= 0 && pu <= Width - 1 && pv >= 0 && pv <= Height - 1;
                bool insideVolume = true;
                for (int c = 0; c < 3; c++)
                {
                    double g = (points[p * 3 + c] - o.VolumeOrigin[c]) / o.VoxelSize;
                    insideVolume &= g >= 0 && g <= o.VolumeDims[c] - 1;
                }

                mask[p * views + k] = inImage[p] && insideVolume && geometryValid[p] > 0 ? 1.0 : 0.0;

                // IBRNet's ray difference between the query ray and the supporting camera's ray to the point.
                var toQuery = Unit(queryCenter[0] - x, queryCenter[1] - y, queryCenter[2] - zc);
                var toSupport = Unit(camera.Center[0] - x, camera.Center[1] - y, camera.Center[2] - zc);
                double dx = toQuery[0] - toSupport[0], dy = toQuery[1] - toSupport[1], dz = toQuery[2] - toSupport[2];
                double norm = Math.Max(Math.Sqrt(dx * dx + dy * dy + dz * dz), 1e-6);
                int b = (p * views + k) * 4;
                rayDiff[b] = dx / norm;
                rayDiff[b + 1] = dy / norm;
                rayDiff[b + 2] = dz / norm;
                rayDiff[b + 3] = toQuery[0] * toSupport[0] + toQuery[1] * toSupport[1] + toQuery[2] * toSupport[2];
            }

            var bilinear = NeuSGridInterpolation<T>.Bilinear(u, vv, Height, Width);
            int view = supporting[k];
            perView[k] = e.TensorConcatenate(new[] { bilinear.Sample(e, _colors[view]), bilinear.Sample(e, _features[view]) }, 1);
        }

        var stacked = e.TensorStack(perView, 1); // [P, V, 3 + C]
        return (e.Reshape(stacked, new[] { n * views, stacked.Shape[2] }), rayDiff, mask, views);
    }

    private Tensor<T> Masked(Tensor<T> values, Tensor<T> keep, double fill)
    {
        var e = _engine;
        var dropped = e.TensorAddScalar(e.TensorNegate(keep), NeuSHost<T>.Scalar(1.0));
        return e.TensorAdd(e.TensorMultiply(values, keep), e.TensorMultiplyScalar(dropped, NeuSHost<T>.Scalar(fill)));
    }

    private static double[] Unit(double x, double y, double z)
    {
        double norm = Math.Sqrt(x * x + y * y + z * z) + 1e-6;
        return new[] { x / norm, y / norm, z / norm };
    }
}

/// <summary>A SparseNeuS render: the NeuS outputs plus which rays the color blend could see.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class SparseNeuSRenderResult<T>
{
    /// <summary>The rendered colors, <c>[R, 3]</c>.</summary>
    public Tensor<T> Color { get; init; } = new Tensor<T>(new[] { 0, 3 });

    /// <summary>The weights, <c>[R, S]</c>.</summary>
    public Tensor<T> Weights { get; init; } = new Tensor<T>(new[] { 0, 0 });

    /// <summary>The accumulated opacity, <c>[R, 1]</c>.</summary>
    public Tensor<T> WeightSum { get; init; } = new Tensor<T>(new[] { 0, 1 });

    /// <summary>The expected depth, <c>[R, 1]</c>.</summary>
    public Tensor<T> Depth { get; init; } = new Tensor<T>(new[] { 0, 1 });

    /// <summary>The rendered normal, <c>[R, 3]</c>.</summary>
    public Tensor<T> Normal { get; init; } = new Tensor<T>(new[] { 0, 3 });

    /// <summary>The masked Eikonal residual, one element.</summary>
    public Tensor<T> GradientError { get; init; } = new Tensor<T>(new[] { 1 });

    /// <summary>The SDF at the samples (100 off the active voxels), <c>[R, S]</c>.</summary>
    public Tensor<T> Sdf { get; init; } = new Tensor<T>(new[] { 0, 0 });

    /// <summary>The section midpoint depths, <c>[R·S]</c>.</summary>
    public double[] MidDepths { get; init; } = Array.Empty<double>();

    /// <summary>The samples per ray.</summary>
    public int SamplesPerRay { get; init; }

    /// <summary>Per ray, whether more than 8 samples are seen by at least two views (the reference's color mask).</summary>
    public bool[] ColorValid { get; init; } = Array.Empty<bool>();
}
