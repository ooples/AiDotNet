using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// SyncDreamer's fixed camera layout and the projective geometry of its 3D-aware feature attention: the target views,
/// the viewpoint embedding, the warp of the spatial volume into each view, and each view's frustum volume.
/// </summary>
/// <remarks>
/// <para>
/// View i sits at azimuth <c>i * 360 / N</c> degrees (index 0 at azimuth 0, the input view's azimuth) and the fixed
/// target elevation, <see cref="SyncDreamerOptions.CameraDistance"/> from the origin, looking at the origin with world
/// up +z, in the OpenCV camera convention (x right, y down, z forward) of <see cref="NeuSCamera"/>. Its intrinsics are
/// <c>K = [[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]]</c> at image side S (<c>meta_info/camera-16.pkl</c>: f = 280 and
/// principal point 128 at 256).
/// </para>
/// <para>
/// The spatial volume holds V^3 voxels over [-l, l]^3; voxel [i, j, k] (depth, height, width) is the point
/// (x, y, z) = (s_k, s_j, s_i) with <c>s = linspace(-l, l, V)</c>, as <c>construct_spatial_volume</c> lays it out. The
/// warp of a voxel into a view is <c>get_warp_coordinates</c>: the projection with K scaled to the latent side, the
/// depth clamped below at 1e-4, and pixel coordinates normalized by <c>(side - 1) / 2</c> (align-corners sampling).
/// A frustum volume is <c>create_target_volume</c>: the latent pixel grid at D depths evenly spaced in camera z over
/// [distance - h, distance + h], unprojected into the world and divided by l to sample the spatial volume.
/// </para>
/// <para>
/// All of this is camera geometry: it depends on the configuration only, so it is computed once in double precision.
/// </para>
/// </remarks>
internal sealed class SyncDreamerCameraLayout
{
    private readonly double[][] _rotationWorldToCamera;
    private readonly double[][] _translation;
    private readonly double[][] _centers;

    public SyncDreamerCameraLayout(SyncDreamerOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        Options = options;
        int n = options.ViewCount;
        ImageSize = options.ImageSize;
        LatentSize = options.ImageSize / options.VaeDownsampleFactor;
        Focal = options.FocalLength;
        Principal = options.ImageSize / 2.0;
        AzimuthDegrees = new double[n];
        Azimuths = new double[n];
        _rotationWorldToCamera = new double[n][];
        _translation = new double[n][];
        _centers = new double[n][];
        double elevation = options.TargetElevationDegrees * Math.PI / 180.0;
        for (int i = 0; i < n; i++)
        {
            double degrees = i * 360.0 / n;
            AzimuthDegrees[i] = degrees;
            double azimuth = degrees * Math.PI / 180.0;
            // (azs + pi) % (2 pi) - pi: the reference wraps every azimuth into [-pi, pi).
            Azimuths[i] = PositiveModulo(azimuth + Math.PI, 2.0 * Math.PI) - Math.PI;
            double r = options.CameraDistance;
            var center = new[]
            {
                r * Math.Cos(elevation) * Math.Cos(azimuth),
                r * Math.Cos(elevation) * Math.Sin(azimuth),
                r * Math.Sin(elevation),
            };
            var camera = NeuSCamera.LookAt(center, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }, Focal, ImageSize, ImageSize);
            // The camera-to-world rotation's columns are the camera axes; its transpose maps world to camera.
            var c2w = camera.Rotation;
            var w2c = new[]
            {
                c2w[0], c2w[3], c2w[6],
                c2w[1], c2w[4], c2w[7],
                c2w[2], c2w[5], c2w[8],
            };
            _rotationWorldToCamera[i] = w2c;
            _translation[i] = new[]
            {
                -(w2c[0] * center[0] + w2c[1] * center[1] + w2c[2] * center[2]),
                -(w2c[3] * center[0] + w2c[4] * center[1] + w2c[5] * center[2]),
                -(w2c[6] * center[0] + w2c[7] * center[1] + w2c[8] * center[2]),
            };
            _centers[i] = center;
        }
    }

    /// <summary>The configuration the layout was built from.</summary>
    public SyncDreamerOptions Options { get; }

    /// <summary>The number of views.</summary>
    public int ViewCount => Azimuths.Length;

    /// <summary>The image side.</summary>
    public int ImageSize { get; }

    /// <summary>The latent side (and the frustum volume's side).</summary>
    public int LatentSize { get; }

    /// <summary>The focal length in pixels at <see cref="ImageSize"/>.</summary>
    public double Focal { get; }

    /// <summary>The principal point (both axes) in pixels at <see cref="ImageSize"/>.</summary>
    public double Principal { get; }

    /// <summary>Each view's azimuth in degrees, in [0, 360).</summary>
    public double[] AzimuthDegrees { get; }

    /// <summary>Each view's azimuth in radians wrapped into [-pi, pi), as the reference stores it.</summary>
    public double[] Azimuths { get; }

    /// <summary>View <paramref name="view"/>'s world-to-camera rotation, 3x3 row-major.</summary>
    public double[] RotationWorldToCamera(int view) => (double[])_rotationWorldToCamera[view].Clone();

    /// <summary>View <paramref name="view"/>'s world-to-camera translation.</summary>
    public double[] Translation(int view) => (double[])_translation[view].Clone();

    /// <summary>View <paramref name="view"/>'s camera center in world space.</summary>
    public double[] Center(int view) => (double[])_centers[view].Clone();

    /// <summary>View <paramref name="view"/>'s camera for NeuS, with the layout's intrinsics.</summary>
    public NeuSCamera CameraForNeuS(int view)
    {
        var w2c = _rotationWorldToCamera[view];
        var c2w = new[]
        {
            w2c[0], w2c[3], w2c[6],
            w2c[1], w2c[4], w2c[7],
            w2c[2], w2c[5], w2c[8],
        };
        return new NeuSCamera(Focal, Focal, Principal, Principal, c2w, _centers[view]);
    }

    /// <summary>
    /// The viewpoint embedding of every target view relative to the input view, [N, 4] row-major:
    /// <c>(d_elevation, sin d_azimuth, cos d_azimuth, 0)</c> with Zero123's negated elevations
    /// (<c>get_viewpoint_embedding</c>).
    /// </summary>
    public double[] ViewpointEmbedding()
    {
        int n = ViewCount;
        double inputElevation = -Options.InputElevationDegrees * Math.PI / 180.0;
        double targetElevation = -Options.TargetElevationDegrees * Math.PI / 180.0;
        double inputAzimuth = Azimuths[0];
        var embedding = new double[n * 4];
        for (int i = 0; i < n; i++)
        {
            double da = Azimuths[i] - inputAzimuth;
            embedding[i * 4] = targetElevation - inputElevation;
            embedding[i * 4 + 1] = Math.Sin(da);
            embedding[i * 4 + 2] = Math.Cos(da);
            embedding[i * 4 + 3] = 0.0;
        }

        return embedding;
    }

    /// <summary>
    /// The normalized image coordinates of every spatial-volume voxel in view <paramref name="view"/>'s
    /// <paramref name="featureSide"/>-pixel feature map, [V, V * V, 2] row-major (x then y), the grid
    /// <c>F.grid_sample(..., align_corners=True)</c> reads.
    /// </summary>
    public double[] SpatialVolumeWarp(int view, int featureSide)
    {
        int v = Options.SpatialVolumeSize;
        double length = Options.SpatialVolumeLength;
        double ratio = featureSide / (double)ImageSize;
        double fx = Focal * ratio, cx = Principal * ratio;
        double half = (featureSide - 1) / 2.0;
        var r = _rotationWorldToCamera[view];
        var t = _translation[view];
        var grid = new double[v * v * v * 2];
        int p = 0;
        for (int i = 0; i < v; i++)
        {
            double z = Linspace(-length, length, v, i);
            for (int j = 0; j < v; j++)
            {
                double y = Linspace(-length, length, v, j);
                for (int k = 0; k < v; k++)
                {
                    double x = Linspace(-length, length, v, k);
                    double camX = r[0] * x + r[1] * y + r[2] * z + t[0];
                    double camY = r[3] * x + r[4] * y + r[5] * z + t[1];
                    double camZ = r[6] * x + r[7] * y + r[8] * z + t[2];
                    double depth = camZ < 1e-4 ? 1e-4 : camZ;
                    double u = (fx * camX + cx * camZ) / depth;
                    double w = (fx * camY + cx * camZ) / depth;
                    grid[p++] = u / half - 1.0;
                    grid[p++] = w / half - 1.0;
                }
            }
        }

        return grid;
    }

    /// <summary>
    /// View <paramref name="view"/>'s frustum volume as normalized spatial-volume coordinates, [D, S, S, 3] row-major
    /// (x, y, z divided by the spatial volume's half extent), with S the latent side; the grid
    /// <c>F.grid_sample(..., align_corners=True)</c> reads from the spatial volume.
    /// </summary>
    public double[] FrustumGrid(int view)
    {
        int d = Options.FrustumVolumeDepth, s = LatentSize;
        double near = Options.CameraDistance - Options.FrustumVolumeLength;
        double far = Options.CameraDistance + Options.FrustumVolumeLength;
        double ratio = s / (double)ImageSize;
        double f = Focal * ratio, c = Principal * ratio;
        double length = Options.SpatialVolumeLength;
        var r = _rotationWorldToCamera[view];
        var center = _centers[view];
        var grid = new double[d * s * s * 3];
        int p = 0;
        for (int depthIndex = 0; depthIndex < d; depthIndex++)
        {
            double depth = Linspace(near, far, d, depthIndex);
            for (int row = 0; row < s; row++)
            {
                for (int col = 0; col < s; col++)
                {
                    // K^-1 (col * depth, row * depth, depth), then camera to world: R^T cam + C.
                    double camX = (col - c) * depth / f;
                    double camY = (row - c) * depth / f;
                    double camZ = depth;
                    double x = r[0] * camX + r[3] * camY + r[6] * camZ + center[0];
                    double y = r[1] * camX + r[4] * camY + r[7] * camZ + center[1];
                    double z = r[2] * camX + r[5] * camY + r[8] * camZ + center[2];
                    grid[p++] = x / length;
                    grid[p++] = y / length;
                    grid[p++] = z / length;
                }
            }
        }

        return grid;
    }

    /// <summary>The depth of frustum slice <paramref name="index"/> (<c>linspace(near, far, D)</c>).</summary>
    public double FrustumDepth(int index)
        => Linspace(Options.CameraDistance - Options.FrustumVolumeLength, Options.CameraDistance + Options.FrustumVolumeLength,
            Options.FrustumVolumeDepth, index);

    private static double Linspace(double start, double end, int count, int index)
        => count == 1 ? start : start + (end - start) * index / (count - 1);

    private static double PositiveModulo(double value, double modulus)
    {
        double result = value % modulus;
        return result < 0 ? result + modulus : result;
    }
}
