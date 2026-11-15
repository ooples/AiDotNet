using AiDotNet.NeuralRadianceFields.Data;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// A pinhole camera in the OpenCV convention NeuS and SparseNeuS use: x right, y down, z forward, a camera-to-world
/// rotation and a camera center in world space.
/// </summary>
/// <remarks>
/// <para>
/// Pixel (x, y) at integer coordinates looks along <c>R · normalize(K⁻¹ [x, y, 1])</c> from the camera center, as
/// NeuS's <c>gen_random_rays_at</c> builds rays, and a world point projects to <c>K · Rᵀ (p − C)</c>, as SparseNeuS's
/// <c>back_project_sparse_type</c> projects voxels. This is the same convention as <see cref="ImageView{T}"/> with the
/// principal point given explicitly.
/// </para>
/// <para><b>For Beginners:</b> The camera says where the photo was taken from and which way it was pointing, plus
/// the lens (focal length and image center). Both NeuS and SparseNeuS expect the scene to have been scaled so the
/// object sits inside the unit sphere around the origin.</para>
/// </remarks>
internal sealed class NeuSCamera
{
    /// <summary>Creates a camera.</summary>
    /// <param name="focalX">The horizontal focal length in pixels.</param>
    /// <param name="focalY">The vertical focal length in pixels.</param>
    /// <param name="principalX">The principal point's x in pixels.</param>
    /// <param name="principalY">The principal point's y in pixels.</param>
    /// <param name="cameraToWorld">The camera-to-world rotation, 3×3 row-major (9 values).</param>
    /// <param name="center">The camera center in world space (3 values).</param>
    public NeuSCamera(double focalX, double focalY, double principalX, double principalY, double[] cameraToWorld,
        double[] center)
    {
        if (!(focalX > 0) || !(focalY > 0)) throw new ArgumentOutOfRangeException(nameof(focalX), "Focal lengths must be positive.");
        if (cameraToWorld is null || cameraToWorld.Length != 9)
            throw new ArgumentException("The rotation must have 9 values (3×3 row-major).", nameof(cameraToWorld));
        if (center is null || center.Length != 3) throw new ArgumentException("The center must have 3 values.", nameof(center));
        FocalX = focalX;
        FocalY = focalY;
        PrincipalX = principalX;
        PrincipalY = principalY;
        Rotation = (double[])cameraToWorld.Clone();
        Center = (double[])center.Clone();
    }

    /// <summary>The horizontal focal length in pixels.</summary>
    public double FocalX { get; }

    /// <summary>The vertical focal length in pixels.</summary>
    public double FocalY { get; }

    /// <summary>The principal point's x in pixels.</summary>
    public double PrincipalX { get; }

    /// <summary>The principal point's y in pixels.</summary>
    public double PrincipalY { get; }

    /// <summary>The camera-to-world rotation, 3×3 row-major.</summary>
    public double[] Rotation { get; }

    /// <summary>The camera center in world space.</summary>
    public double[] Center { get; }

    /// <summary>
    /// The camera of an <see cref="ImageView{T}"/>: its rotation and position, its focal length for both axes, and the
    /// principal point at the image center <c>((W − 1) / 2, (H − 1) / 2)</c> that <see cref="ImageView{T}"/> assumes.
    /// </summary>
    /// <param name="view">The view.</param>
    [AiDotNet.Attributes.ScalarKernel("Reads the nine rotation and three position values of one camera.")]
    public static NeuSCamera FromImageView<T>(ImageView<T> view)
    {
        if (view is null) throw new ArgumentNullException(nameof(view));
        var ops = MathHelper.GetNumericOperations<T>();
        var rotation = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                rotation[r * 3 + c] = ops.ToDouble(view.CameraRotation[r, c]);
        var center = new[]
        {
            ops.ToDouble(view.CameraPosition[0]), ops.ToDouble(view.CameraPosition[1]), ops.ToDouble(view.CameraPosition[2]),
        };
        double focal = view.ResolveFocalLengthInPixels();
        return new NeuSCamera(focal, focal, (view.Width - 1) * 0.5, (view.Height - 1) * 0.5, rotation, center);
    }

    /// <summary>The unit world direction of the ray through pixel (x, y).</summary>
    /// <param name="x">The pixel column.</param>
    /// <param name="y">The pixel row.</param>
    /// <param name="direction">Receives the direction (3 values).</param>
    public void RayDirection(double x, double y, double[] direction)
    {
        double px = (x - PrincipalX) / FocalX;
        double py = (y - PrincipalY) / FocalY;
        double norm = Math.Sqrt(px * px + py * py + 1.0);
        px /= norm;
        py /= norm;
        double pz = 1.0 / norm;
        for (int r = 0; r < 3; r++)
            direction[r] = Rotation[r * 3] * px + Rotation[r * 3 + 1] * py + Rotation[r * 3 + 2] * pz;
    }

    /// <summary>Projects a world point into the image.</summary>
    /// <param name="x">World x.</param>
    /// <param name="y">World y.</param>
    /// <param name="z">World z.</param>
    /// <returns>The pixel coordinates and the camera-space depth.</returns>
    public (double U, double V, double Depth) Project(double x, double y, double z)
    {
        double dx = x - Center[0], dy = y - Center[1], dz = z - Center[2];
        // Rᵀ (p − C): the rotation's columns are the camera axes in world space.
        double cx = Rotation[0] * dx + Rotation[3] * dy + Rotation[6] * dz;
        double cy = Rotation[1] * dx + Rotation[4] * dy + Rotation[7] * dz;
        double cz = Rotation[2] * dx + Rotation[5] * dy + Rotation[8] * dz;
        if (cz == 0.0) return (double.NaN, double.NaN, 0.0);
        return (FocalX * cx / cz + PrincipalX, FocalY * cy / cz + PrincipalY, cz);
    }

    /// <summary>
    /// A camera at <paramref name="center"/> looking at <paramref name="target"/> with world <paramref name="up"/>,
    /// in the OpenCV convention (the image's y axis points along −up).
    /// </summary>
    /// <param name="center">The camera center.</param>
    /// <param name="target">The point the optical axis passes through.</param>
    /// <param name="up">The world up direction.</param>
    /// <param name="focal">The focal length in pixels.</param>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    public static NeuSCamera LookAt(double[] center, double[] target, double[] up, double focal, int width, int height)
    {
        if (center is null || target is null || up is null) throw new ArgumentNullException(nameof(center));
        var forward = Normalize(new[] { target[0] - center[0], target[1] - center[1], target[2] - center[2] });
        // OpenCV axes: right = normalize(forward × up), down = forward × right.
        var right = Normalize(Cross(forward, up));
        var down = Cross(forward, right);
        var rotation = new[]
        {
            right[0], down[0], forward[0],
            right[1], down[1], forward[1],
            right[2], down[2], forward[2],
        };
        return new NeuSCamera(focal, focal, (width - 1) * 0.5, (height - 1) * 0.5, rotation, center);
    }

    private static double[] Cross(double[] a, double[] b)
        => new[] { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };

    private static double[] Normalize(double[] v)
    {
        double n = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (n == 0.0) throw new ArgumentException("A camera axis is degenerate (zero length).");
        return new[] { v[0] / n, v[1] / n, v[2] / n };
    }
}
