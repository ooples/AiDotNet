using AiDotNet.NeuralRadianceFields.Data;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>How a <see cref="NeuSCamera"/> maps pixels to rays.</summary>
internal enum NeuSProjection
{
    /// <summary>A pinhole camera: every ray starts at the camera center (NeuS, SparseNeuS).</summary>
    Perspective,

    /// <summary>
    /// An orthographic camera: every ray runs along the optical axis from its own point on the image plane through
    /// the camera center (Wonder3D's <c>get_ortho_ray_directions_origins</c>).
    /// </summary>
    Orthographic,
}

/// <summary>
/// A pinhole or orthographic camera in the OpenCV convention NeuS and SparseNeuS use: x right, y down, z forward, a camera-to-world
/// rotation and a camera center in world space.
/// </summary>
/// <remarks>
/// <para>
/// Pixel (x, y) at integer coordinates looks along <c>R · normalize(K⁻¹ [x, y, 1])</c> from the camera center, as
/// NeuS's <c>gen_random_rays_at</c> builds rays, and a world point projects to <c>K · Rᵀ (p − C)</c>, as SparseNeuS's
/// <c>back_project_sparse_type</c> projects voxels. This is the same convention as <see cref="ImageView{T}"/> with the
/// principal point given explicitly.
/// </para>
/// <para>
/// An orthographic camera (<see cref="Orthographic"/>) keeps the same axes, but <see cref="FocalX"/> and
/// <see cref="FocalY"/> are pixels per world unit: pixel (x, y) is the ray from
/// <c>C + R · [(x − cx) / fx, (y − cy) / fy, 0]</c> along <c>R · [0, 0, 1]</c>, and a world point projects to
/// <c>(fx · x_c + cx, fy · y_c + cy)</c> with <c>(x_c, y_c, z_c) = Rᵀ (p − C)</c>. Wonder3D's views are of this kind:
/// a <c>W</c>-pixel image spanning <c>[−1, 1]</c> on the image plane is <c>fx = W / 2</c> with the principal point at
/// <c>(W − 1) / 2</c>, which puts pixel i's ray at <c>((i + 0.5) / W − 0.5) · 2</c> as its reference does.
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
        : this(focalX, focalY, principalX, principalY, cameraToWorld, center, NeuSProjection.Perspective)
    {
    }

    /// <summary>Creates a camera of either projection.</summary>
    /// <param name="focalX">The horizontal focal length in pixels, or pixels per world unit when orthographic.</param>
    /// <param name="focalY">The vertical focal length in pixels, or pixels per world unit when orthographic.</param>
    /// <param name="principalX">The principal point's x in pixels.</param>
    /// <param name="principalY">The principal point's y in pixels.</param>
    /// <param name="cameraToWorld">The camera-to-world rotation, 3×3 row-major (9 values).</param>
    /// <param name="center">The camera center in world space (3 values).</param>
    /// <param name="projection">The projection.</param>
    public NeuSCamera(double focalX, double focalY, double principalX, double principalY, double[] cameraToWorld,
        double[] center, NeuSProjection projection)
    {
        if (projection != NeuSProjection.Perspective && projection != NeuSProjection.Orthographic)
            throw new ArgumentOutOfRangeException(nameof(projection), "Unknown projection.");
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
        Projection = projection;
    }

    /// <summary>The projection.</summary>
    public NeuSProjection Projection { get; }

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

    /// <summary>
    /// The ray through pixel (x, y): its world origin and unit world direction. A perspective ray starts at the
    /// camera center; an orthographic ray starts at the pixel's point on the image plane through the center.
    /// </summary>
    /// <param name="x">The pixel column.</param>
    /// <param name="y">The pixel row.</param>
    /// <param name="origin">Receives the origin (3 values).</param>
    /// <param name="direction">Receives the direction (3 values).</param>
    public void Ray(double x, double y, double[] origin, double[] direction)
    {
        if (origin is null) throw new ArgumentNullException(nameof(origin));
        RayDirection(x, y, direction);
        if (Projection == NeuSProjection.Perspective)
        {
            for (int r = 0; r < 3; r++) origin[r] = Center[r];
            return;
        }

        double px = (x - PrincipalX) / FocalX;
        double py = (y - PrincipalY) / FocalY;
        for (int r = 0; r < 3; r++)
            origin[r] = Center[r] + Rotation[r * 3] * px + Rotation[r * 3 + 1] * py;
    }

    /// <summary>
    /// The unit world direction of the ray through pixel (x, y); the optical axis for every pixel of an orthographic
    /// camera.
    /// </summary>
    /// <param name="x">The pixel column.</param>
    /// <param name="y">The pixel row.</param>
    /// <param name="direction">Receives the direction (3 values).</param>
    public void RayDirection(double x, double y, double[] direction)
    {
        if (direction is null) throw new ArgumentNullException(nameof(direction));
        if (Projection == NeuSProjection.Orthographic)
        {
            for (int r = 0; r < 3; r++) direction[r] = Rotation[r * 3 + 2];
            return;
        }

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
        if (Projection == NeuSProjection.Orthographic)
            return (FocalX * cx + PrincipalX, FocalY * cy + PrincipalY, cz);
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

    /// <summary>
    /// An orthographic camera at <paramref name="center"/> looking at <paramref name="target"/> with world
    /// <paramref name="up"/> (OpenCV axes, as <see cref="LookAt"/>), whose <paramref name="width"/>-pixel-wide image
    /// spans <c>[−halfExtent, halfExtent]</c> horizontally on the image plane, with square pixels.
    /// </summary>
    /// <param name="center">The camera center; the image plane passes through it.</param>
    /// <param name="target">The point the optical axis passes through.</param>
    /// <param name="up">The world up direction.</param>
    /// <param name="halfExtent">Half the image plane's width in world units (Wonder3D: 1).</param>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    public static NeuSCamera Orthographic(double[] center, double[] target, double[] up, double halfExtent, int width,
        int height)
    {
        if (!(halfExtent > 0) || double.IsInfinity(halfExtent))
            throw new ArgumentOutOfRangeException(nameof(halfExtent), "The half extent must be positive and finite.");
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The image must be non-empty.");
        var axes = LookAt(center, target, up, 1.0, width, height);
        double pixelsPerUnit = width / (2.0 * halfExtent);
        return new NeuSCamera(pixelsPerUnit, pixelsPerUnit, (width - 1) * 0.5, (height - 1) * 0.5, axes.Rotation,
            axes.Center, NeuSProjection.Orthographic);
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
