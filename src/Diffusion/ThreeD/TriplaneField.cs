using AiDotNet.Enums;
using AiDotNet.NeuralRadianceFields.Helpers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The 3D field one image decodes to in <see cref="TripoSRModel{T}"/>: three axis-aligned feature planes and the NeRF
/// MLP that turns a sampled point into a density and a colour, with dense-grid sampling, point extraction and volume
/// rendering on top.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// A query point p in [-r, r]^3 is scaled to [-1, 1]^3 and projected onto the XY, XZ and YZ planes; each plane is
/// sampled bilinearly (<c>grid_sample</c>, <c>align_corners=False</c>, zero padding), the three features are
/// concatenated (or averaged) and decoded by the MLP into a raw density and three colour features. The density is
/// <c>activation(raw + bias)</c> (TripoSR: <c>exp(raw - 1)</c>) and the colour is <c>sigmoid(features)</c>
/// (Tochilkin et al. 2024, <c>TriplaneNeRFRenderer.query_triplane</c>).
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "3D object" TripoSR predicted. Ask it about any 3D point and it tells you how
/// solid that point is and what colour it has. Sampling it on a regular grid gives a voxel picture of the object.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var field = model.Reconstruct(image);
/// var grid = field.SampleDensityGrid(64, field.Radius);   // [64, 64, 64] densities
/// var cloud = field.SamplePointCloud(64, field.Radius, 25.0);   // [1, P, 6] xyz + rgb
/// </code>
/// </example>
public sealed class TriplaneField<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly TsrNeRFDecoder<T> _decoder;
    private readonly TriplaneFeatureReduction _reduction;
    private readonly TriplaneDensityActivation _densityActivation;
    private readonly T _densityBias;
    private readonly T _inverseRadius;
    private readonly int _chunkSize;
    private readonly int _samplesPerRay;

    internal TriplaneField(Tensor<T> planes, TsrNeRFDecoder<T> decoder, TripoSROptions options)
    {
        if (planes is null) throw new ArgumentNullException(nameof(planes));
        if (decoder is null) throw new ArgumentNullException(nameof(decoder));
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (planes.Rank != 4 || planes.Shape[0] != 3)
            throw new ArgumentException("Triplanes must be [3, channels, height, width].", nameof(planes));
        Planes = planes;
        _decoder = decoder;
        _reduction = options.FeatureReduction;
        _densityActivation = options.DensityActivation;
        _densityBias = NumOps.FromDouble(options.DensityBias);
        Radius = options.Radius;
        _inverseRadius = NumOps.FromDouble(1.0 / options.Radius);
        _chunkSize = options.QueryChunkSize;
        _samplesPerRay = options.SamplesPerRay;
    }

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The feature planes [3, C, H, W]: XY, XZ and YZ, in that order.</summary>
    public Tensor<T> Planes { get; }

    /// <summary>The half-extent r of the cube [-r, r]^3 the planes span.</summary>
    public double Radius { get; }

    /// <summary>The channels of each plane.</summary>
    public int Channels => Planes.Shape[1];

    /// <summary>
    /// The decoder input of each point [N, 3C] (concatenated XY, XZ, YZ features) or [N, C] (their mean).
    /// </summary>
    /// <param name="points">Points [N, 3] in world units; outside [-r, r]^3 the planes read as zero.</param>
    public Tensor<T> SampleFeatures(Tensor<T> points)
    {
        ValidatePoints(points);
        int count = points.Shape[0];
        int channels = Channels;
        var scaled = Engine.TensorMultiplyScalar(points, _inverseRadius);
        var x = Engine.TensorNarrow(scaled, 1, 0, 1).Contiguous();
        var y = Engine.TensorNarrow(scaled, 1, 1, 1).Contiguous();
        var z = Engine.TensorNarrow(scaled, 1, 2, 1).Contiguous();
        // grid_sample reads (u, v) as (width, height): plane 0 is indexed by (x, y), plane 1 by (x, z), plane 2 by (y, z).
        var grid = Engine.TensorConcatenate(new[]
        {
            Engine.Reshape(Engine.TensorConcatenate(new[] { x, y }, 1), new[] { 1, 1, count, 2 }),
            Engine.Reshape(Engine.TensorConcatenate(new[] { x, z }, 1), new[] { 1, 1, count, 2 }),
            Engine.Reshape(Engine.TensorConcatenate(new[] { y, z }, 1), new[] { 1, 1, count, 2 })
        }, 0);
        var sampled = Engine.GridSample(Planes, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
        var perPlane = Engine.Reshape(sampled, new[] { 3, channels, count });
        if (_reduction == TriplaneFeatureReduction.Mean)
        {
            var mean = Engine.TensorMultiplyScalar(Engine.ReduceSum(perPlane, new[] { 0 }, false), NumOps.FromDouble(1.0 / 3.0));
            return Engine.TensorPermute(mean, new[] { 1, 0 }).Contiguous();
        }

        // "Np Cp () N -> N (Np Cp)": plane-major, then channel.
        return Engine.Reshape(Engine.TensorPermute(perPlane, new[] { 2, 0, 1 }).Contiguous(), new[] { count, 3 * channels });
    }

    /// <summary>
    /// The field at <paramref name="points"/> [N, 3]: a tensor [N, 4] holding the activated density in column 0 and
    /// the RGB colour in [0, 1] in columns 1 to 3. Points are decoded in chunks of the configured query size.
    /// </summary>
    public Tensor<T> Query(Tensor<T> points)
    {
        ValidatePoints(points);
        int count = points.Shape[0];
        if (count == 0) return new Tensor<T>(new[] { 0, 4 });
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < count; start += _chunkSize)
        {
            int length = Math.Min(_chunkSize, count - start);
            chunks.Add(QueryChunk(Engine.TensorNarrow(points, 0, start, length).Contiguous()));
        }

        return chunks.Count == 1 ? chunks[0] : Engine.TensorConcatenate(chunks.ToArray(), 0);
    }

    private Tensor<T> QueryChunk(Tensor<T> points)
    {
        var raw = _decoder.Forward(SampleFeatures(points));
        var density = ActivateDensity(Engine.TensorAddScalar(Engine.TensorNarrow(raw, 1, 0, 1).Contiguous(), _densityBias));
        var colour = Engine.TensorSigmoid(Engine.TensorNarrow(raw, 1, 1, 3).Contiguous());
        return Engine.TensorConcatenate(new[] { density, colour }, 1);
    }

    private Tensor<T> ActivateDensity(Tensor<T> x) => _densityActivation switch
    {
        TriplaneDensityActivation.Exp => Engine.TensorExp(x),
        TriplaneDensityActivation.Softplus => Engine.Softplus(x),
        TriplaneDensityActivation.ReLU => Engine.TensorReLU(x),
        _ => throw new NotSupportedException($"Unknown density activation {_densityActivation}.")
    };

    /// <summary>
    /// The activated density on the regular grid <c>linspace(-bounds, bounds, resolution)</c> on each axis, as
    /// [resolution, resolution, resolution] indexed [x, y, z] (TripoSR's <c>MarchingCubeHelper</c> grid,
    /// <c>meshgrid(indexing="ij")</c>).
    /// </summary>
    /// <param name="resolution">Samples per axis, at least 2.</param>
    /// <param name="bounds">The half-extent of the sampled cube; TripoSR samples its own radius.</param>
    public Tensor<T> SampleDensityGrid(int resolution, double bounds)
    {
        var axis = GridAxis(resolution, bounds);
        int slab = resolution * resolution;
        var slabs = new Tensor<T>[resolution];
        var coordinates = new double[slab * 3];
        for (int i = 0; i < resolution; i++)
        {
            // One x-slab of the grid at a time, so the points never exceed resolution^2 rows.
            for (int j = 0; j < resolution; j++)
            {
                for (int k = 0; k < resolution; k++)
                {
                    int row = 3 * ((j * resolution) + k);
                    coordinates[row] = axis[i];
                    coordinates[row + 1] = axis[j];
                    coordinates[row + 2] = axis[k];
                }
            }

            var points = new Tensor<double>(new[] { slab, 3 }, new Vector<double>(coordinates)).Cast<T>();
            slabs[i] = Engine.TensorNarrow(Query(points), 1, 0, 1).Contiguous();
        }

        return Engine.Reshape(Engine.TensorConcatenate(slabs, 0), new[] { resolution, resolution, resolution });
    }

    /// <summary>
    /// The grid samples whose density exceeds <paramref name="threshold"/>, as a point cloud [1, P, 6] of XYZ and the
    /// decoded RGB at each point. With <paramref name="maxPoints"/>, the densest points are kept (ties by grid order).
    /// An empty field gives [1, 0, 6].
    /// </summary>
    /// <param name="resolution">Grid samples per axis, at least 2.</param>
    /// <param name="bounds">The half-extent of the sampled cube.</param>
    /// <param name="threshold">The occupancy density threshold; TripoSR extracts its surface at 25.</param>
    /// <param name="maxPoints">The most points to return, or null for every occupied sample.</param>
    public Tensor<T> SamplePointCloud(int resolution, double bounds, double threshold, int? maxPoints = null)
    {
        if (maxPoints is int cap && cap < 0) throw new ArgumentOutOfRangeException(nameof(maxPoints), "The point budget cannot be negative.");
        var axis = GridAxis(resolution, bounds);
        var density = SampleDensityGrid(resolution, bounds).Cast<double>().ToArray();
        var occupied = new List<int>();
        for (int index = 0; index < density.Length; index++)
            if (density[index] > threshold) occupied.Add(index);

        if (maxPoints is int budget && occupied.Count > budget)
        {
            occupied.Sort((a, b) =>
            {
                int byDensity = density[b].CompareTo(density[a]);
                return byDensity != 0 ? byDensity : a.CompareTo(b);
            });
            occupied.RemoveRange(budget, occupied.Count - budget);
            occupied.Sort();
        }

        int count = occupied.Count;
        if (count == 0) return new Tensor<T>(new[] { 1, 0, 6 });
        var coordinates = new double[count * 3];
        int slab = resolution * resolution;
        for (int p = 0; p < count; p++)
        {
            int index = occupied[p];
            coordinates[3 * p] = axis[index / slab];
            coordinates[(3 * p) + 1] = axis[(index / resolution) % resolution];
            coordinates[(3 * p) + 2] = axis[index % resolution];
        }

        var points = new Tensor<double>(new[] { count, 3 }, new Vector<double>(coordinates)).Cast<T>();
        var colours = Engine.TensorNarrow(Query(points), 1, 1, 3).Contiguous();
        return Engine.Reshape(Engine.TensorConcatenate(new[] { points, colours }, 1), new[] { 1, count, 6 });
    }

    /// <summary>
    /// Volume-renders rays through the field as TripoSR's <c>TriplaneNeRFRenderer</c> does: each ray is clipped to
    /// the cube (shrunk by 0.1%), sampled at the midpoints of <c>SamplesPerRay</c> equal intervals, and composited
    /// with <see cref="VolumeRenderer{T}"/> using the renderer's interval width 1 / SamplesPerRay. Rays that miss the
    /// cube render black.
    /// </summary>
    /// <param name="origins">Ray origins, three values per ray.</param>
    /// <param name="directions">Unit ray directions, three values per ray.</param>
    /// <returns>The colours [N, 3].</returns>
    public Tensor<T> RenderRays(double[] origins, double[] directions)
    {
        if (origins is null) throw new ArgumentNullException(nameof(origins));
        if (directions is null) throw new ArgumentNullException(nameof(directions));
        if (origins.Length != directions.Length || origins.Length % 3 != 0)
            throw new ArgumentException("Origins and directions must both hold three values per ray.", nameof(directions));
        int rays = origins.Length / 3;
        var near = new double[rays];
        var far = new double[rays];
        var gather = new int[rays];
        var valid = new List<int>();
        for (int r = 0; r < rays; r++)
        {
            if (IntersectCube(origins, directions, r, Radius, out near[r], out far[r]))
            {
                valid.Add(r);
                gather[r] = valid.Count;
            }
        }

        if (valid.Count == 0) return new Tensor<T>(new[] { rays, 3 });

        int samples = _samplesPerRay;
        var positions = new double[valid.Count * samples * 3];
        for (int v = 0; v < valid.Count; v++)
        {
            int r = valid[v];
            for (int s = 0; s < samples; s++)
            {
                double mid = (s + 0.5) / samples;
                double depth = (near[r] * (1.0 - mid)) + (far[r] * mid);
                int row = 3 * ((v * samples) + s);
                for (int a = 0; a < 3; a++)
                    positions[row + a] = origins[(3 * r) + a] + (depth * directions[(3 * r) + a]);
            }
        }

        var field = Query(new Tensor<double>(new[] { valid.Count * samples, 3 }, new Vector<double>(positions)).Cast<T>());
        var density = Engine.TensorNarrow(field, 1, 0, 1).Contiguous();
        var rgb = Engine.TensorNarrow(field, 1, 1, 3).Contiguous();
        // The official renderer's deltas are t_vals[1:] - t_vals[:-1] over the normalized ray parameter: 1 / S.
        var widths = new double[valid.Count * samples];
        for (int i = 0; i < widths.Length; i++) widths[i] = 1.0 / samples;
        var deltas = new Tensor<double>(new[] { valid.Count, samples }, new Vector<double>(widths)).Cast<T>();
        var rendered = VolumeRenderer<T>.Render(Engine, rgb, density, deltas);

        // Row 0 is black for rays that miss the cube; ray r reads row gather[r].
        var padded = Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, 3 }), rendered }, 0);
        return Engine.TensorGather(padded, new Tensor<int>(new[] { rays }, new Vector<int>(gather)), 0);
    }

    /// <summary>
    /// Renders the view of a camera orbiting the origin and looking at it, as TripoSR's
    /// <c>get_spherical_cameras</c> places it (z up, OpenGL camera axes, pixel-centre rays).
    /// </summary>
    /// <param name="azimuthDegrees">The camera azimuth about the z axis, in degrees.</param>
    /// <param name="elevationDegrees">The camera elevation above the XY plane, in degrees.</param>
    /// <param name="cameraDistance">The camera's distance from the origin.</param>
    /// <param name="fieldOfViewDegrees">The vertical field of view, in degrees.</param>
    /// <param name="height">The image height in pixels.</param>
    /// <param name="width">The image width in pixels.</param>
    /// <returns>The image [1, 3, height, width].</returns>
    public Tensor<T> RenderView(double azimuthDegrees, double elevationDegrees, double cameraDistance,
        double fieldOfViewDegrees, int height, int width)
    {
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "The height must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The width must be positive.");
        if (!(fieldOfViewDegrees > 0 && fieldOfViewDegrees < 180))
            throw new ArgumentOutOfRangeException(nameof(fieldOfViewDegrees), "The field of view must be in (0, 180) degrees.");
        double azimuth = azimuthDegrees * Math.PI / 180.0;
        double elevation = elevationDegrees * Math.PI / 180.0;
        var position = new[]
        {
            cameraDistance * Math.Cos(elevation) * Math.Cos(azimuth),
            cameraDistance * Math.Cos(elevation) * Math.Sin(azimuth),
            cameraDistance * Math.Sin(elevation)
        };
        var lookAt = Normalize(-position[0], -position[1], -position[2]);
        var right = Normalize(Cross(lookAt, new[] { 0.0, 0.0, 1.0 }));
        var up = Normalize(Cross(right, lookAt));
        double focal = 0.5 * height / Math.Tan(0.5 * fieldOfViewDegrees * Math.PI / 180.0);

        int pixels = height * width;
        var origins = new double[pixels * 3];
        var directions = new double[pixels * 3];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                // get_ray_directions: ((i - cx) / f, -(j - cy) / f, -1) with pixel centres; world = R d, R = [right, up, -lookAt].
                double dx = (column + 0.5 - (0.5 * width)) / focal;
                double dy = -(row + 0.5 - (0.5 * height)) / focal;
                var d = Normalize(
                    (right[0] * dx) + (up[0] * dy) + lookAt[0],
                    (right[1] * dx) + (up[1] * dy) + lookAt[1],
                    (right[2] * dx) + (up[2] * dy) + lookAt[2]);
                int offset = 3 * ((row * width) + column);
                for (int a = 0; a < 3; a++)
                {
                    origins[offset + a] = position[a];
                    directions[offset + a] = d[a];
                }
            }
        }

        var colours = RenderRays(origins, directions);
        var image = Engine.TensorPermute(Engine.Reshape(colours, new[] { height, width, 3 }), new[] { 2, 0, 1 }).Contiguous();
        return Engine.Reshape(image, new[] { 1, 3, height, width });
    }

    // rays_intersect_bbox: slab test against the cube shrunk by 0.1%, near clamped at 0, valid when far - near > 0.01.
    // A direction component below 1e-6 in magnitude is replaced by +1e-6, as the official code does.
    internal static bool IntersectCube(double[] origins, double[] directions, int ray, double radius, out double near, out double far)
    {
        double bound = (1.0 - 1.0e-3) * radius;
        near = double.NegativeInfinity;
        far = double.PositiveInfinity;
        for (int a = 0; a < 3; a++)
        {
            double o = origins[(3 * ray) + a];
            double d = directions[(3 * ray) + a];
            if (Math.Abs(d) < 1e-6) d = 1e-6;
            double t0 = (bound - o) / d;
            double t1 = (-bound - o) / d;
            near = Math.Max(near, Math.Min(t0, t1));
            far = Math.Min(far, Math.Max(t0, t1));
        }

        near = Math.Max(near, 0.0);
        bool valid = far - near > 0.01;
        if (!valid)
        {
            near = 0.0;
            far = 0.0;
        }

        return valid;
    }

    // torch.linspace(-bounds, bounds, resolution).
    private static double[] GridAxis(int resolution, double bounds)
    {
        if (resolution < 2) throw new ArgumentOutOfRangeException(nameof(resolution), "A grid needs at least two samples per axis.");
        if (!(bounds > 0) || double.IsInfinity(bounds)) throw new ArgumentOutOfRangeException(nameof(bounds), "The bounds must be positive and finite.");
        var axis = new double[resolution];
        double step = 2.0 * bounds / (resolution - 1);
        for (int i = 0; i < resolution; i++) axis[i] = -bounds + (i * step);
        return axis;
    }

    private static void ValidatePoints(Tensor<T> points)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (points.Rank != 2 || points.Shape[1] != 3)
            throw new ArgumentException("Points must be [N, 3].", nameof(points));
    }

    private static double[] Cross(double[] a, double[] b) => new[]
    {
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0])
    };

    private static double[] Normalize(double[] v) => Normalize(v[0], v[1], v[2]);

    private static double[] Normalize(double x, double y, double z)
    {
        double norm = Math.Sqrt((x * x) + (y * y) + (z * z));
        return norm > 0 ? new[] { x / norm, y / norm, z / norm } : new[] { x, y, z };
    }
}
