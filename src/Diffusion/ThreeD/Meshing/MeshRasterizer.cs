using AiDotNet.NeuralRadianceFields.Models;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Per-pixel result of <see cref="MeshRasterizer.Rasterize"/>: the visible triangle, its perspective-correct
/// barycentric coordinates and the camera-space depth.
/// </summary>
internal sealed class MeshRasterization
{
    internal MeshRasterization(int width, int height)
    {
        Width = width;
        Height = height;
        Face = new int[width * height];
        Bary1 = new double[width * height];
        Bary2 = new double[width * height];
        Depth = new double[width * height];
        BaryDerivatives = new double[4 * width * height];
        for (int i = 0; i < Face.Length; i++)
            Face[i] = -1;
        ScreenX = Array.Empty<double>();
        ScreenY = Array.Empty<double>();
        VertexDepth = Array.Empty<double>();
    }

    /// <summary>Image width.</summary>
    public int Width { get; }

    /// <summary>Image height.</summary>
    public int Height { get; }

    /// <summary>The visible triangle per pixel, or -1 (nvdiffrast's <c>rast[..., 3] - 1</c>).</summary>
    public int[] Face { get; }

    /// <summary>Barycentric weight of the triangle's second vertex (nvdiffrast's <c>rast[..., 0]</c>).</summary>
    public double[] Bary1 { get; }

    /// <summary>Barycentric weight of the triangle's third vertex (nvdiffrast's <c>rast[..., 1]</c>).</summary>
    public double[] Bary2 { get; }

    /// <summary>Camera-space depth of the visible surface.</summary>
    public double[] Depth { get; }

    /// <summary>
    /// The pixel-space derivatives of the two barycentrics [H, W, 4] (d b1/dx, d b1/dy, d b2/dx, d b2/dy), nvdiffrast's
    /// <c>rast_db</c>.
    /// </summary>
    public double[] BaryDerivatives { get; }

    /// <summary>Projected x of every vertex, in pixels (pixel centres at integers).</summary>
    public double[] ScreenX { get; internal set; }

    /// <summary>Projected y of every vertex, in pixels.</summary>
    public double[] ScreenY { get; internal set; }

    /// <summary>Camera-space depth of every vertex.</summary>
    public double[] VertexDepth { get; internal set; }

    /// <summary>Whether a triangle lies in front of the near plane, so it is drawn.</summary>
    public bool IsDrawable(int i0, int i1, int i2)
        => VertexDepth[i0] > MeshRasterizer.NearPlane && VertexDepth[i1] > MeshRasterizer.NearPlane
           && VertexDepth[i2] > MeshRasterizer.NearPlane;

    /// <summary>Twice the signed screen-space area of a triangle (its winding as seen by the camera).</summary>
    public double SignedArea(int i0, int i1, int i2)
        => ((ScreenX[i1] - ScreenX[i0]) * (ScreenY[i2] - ScreenY[i0])) - ((ScreenX[i2] - ScreenX[i0]) * (ScreenY[i1] - ScreenY[i0]));

    /// <summary>
    /// The pixel-space derivatives of an interpolated attribute ([H, W, width, 2]: d/dx then d/dy per component),
    /// <c>dr.interpolate(..., rast_db, diff_attrs='all')</c>'s second output.
    /// </summary>
    public double[] InterpolateDerivatives(double[] attributes, int width, int[] attributeFaces)
    {
        if (attributes is null) throw new ArgumentNullException(nameof(attributes));
        if (attributeFaces is null) throw new ArgumentNullException(nameof(attributeFaces));
        var result = new double[Face.Length * width * 2];
        for (int p = 0; p < Face.Length; p++)
        {
            int f = Face[p];
            if (f < 0) continue;
            int i0 = attributeFaces[3 * f], i1 = attributeFaces[(3 * f) + 1], i2 = attributeFaces[(3 * f) + 2];
            double b1x = BaryDerivatives[4 * p], b1y = BaryDerivatives[(4 * p) + 1];
            double b2x = BaryDerivatives[(4 * p) + 2], b2y = BaryDerivatives[(4 * p) + 3];
            for (int k = 0; k < width; k++)
            {
                double a0 = attributes[(i0 * width) + k];
                double d1 = attributes[(i1 * width) + k] - a0, d2 = attributes[(i2 * width) + k] - a0;
                result[(((p * width) + k) * 2)] = (b1x * d1) + (b2x * d2);
                result[(((p * width) + k) * 2) + 1] = (b1y * d1) + (b2y * d2);
            }
        }

        return result;
    }

    /// <summary>
    /// Interpolates a per-vertex attribute of <paramref name="width"/> values (<c>dr.interpolate</c>), with
    /// <paramref name="attributeFaces"/> giving each triangle's attribute indices.
    /// </summary>
    /// <returns>[H, W, width]; zero where no triangle is visible.</returns>
    public double[] Interpolate(double[] attributes, int width, int[] attributeFaces)
    {
        if (attributes is null) throw new ArgumentNullException(nameof(attributes));
        if (attributeFaces is null) throw new ArgumentNullException(nameof(attributeFaces));
        var result = new double[Face.Length * width];
        for (int p = 0; p < Face.Length; p++)
        {
            int f = Face[p];
            if (f < 0) continue;
            double b1 = Bary1[p], b2 = Bary2[p], b0 = 1.0 - b1 - b2;
            int i0 = attributeFaces[3 * f], i1 = attributeFaces[(3 * f) + 1], i2 = attributeFaces[(3 * f) + 2];
            for (int k = 0; k < width; k++)
            {
                result[(p * width) + k] = (b0 * attributes[(i0 * width) + k]) + (b1 * attributes[(i1 * width) + k])
                    + (b2 * attributes[(i2 * width) + k]);
            }
        }

        return result;
    }
}

/// <summary>
/// A z-buffered triangle rasterizer with the projection of the Gaussian rasterizer (<see cref="GaussianSplatCamera"/>),
/// standing in for nvdiffrast's <c>dr.rasterize</c> in DreamGaussian's mesh stage.
/// </summary>
/// <remarks>
/// <para>
/// Pixel centres are sampled at integer coordinates with the principal point at the image centre, the convention of
/// the Gaussian renderer, so a mesh and the Gaussians it came from line up pixel for pixel. Both windings are drawn
/// (nvdiffrast does not cull), barycentrics are perspective-correct and the nearest surface wins. Triangles with a
/// vertex at or behind the near plane are skipped rather than clipped; DreamGaussian's cameras orbit at a radius of 2
/// or more around a mesh inside the unit cube, so none of its triangles reaches the plane. The barycentrics' pixel-space
/// derivatives (nvdiffrast's <c>rast_db</c>) are the analytic derivatives of the perspective-correct barycentrics of the
/// visible triangle.
/// </para>
/// </remarks>
internal static class MeshRasterizer
{
    /// <summary>The near plane, DreamGaussian's <c>OrbitCamera.near</c>.</summary>
    internal const double NearPlane = 0.01;

    /// <summary>
    /// Rasterizes <paramref name="faces"/> over <paramref name="vertices"/> ([V, 3] and [F, 3], flattened).
    /// </summary>
    public static MeshRasterization Rasterize(double[] vertices, int[] faces, GaussianSplatCamera camera)
    {
        if (vertices is null) throw new ArgumentNullException(nameof(vertices));
        if (faces is null) throw new ArgumentNullException(nameof(faces));
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        int width = camera.Width, height = camera.Height;
        var result = new MeshRasterization(width, height);
        int vertexCount = vertices.Length / 3;
        var screenX = new double[vertexCount];
        var screenY = new double[vertexCount];
        var depth = new double[vertexCount];
        var r = camera.CameraToWorld;
        var c = camera.Position;
        double fx = camera.FocalX, fy = camera.FocalY, cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
        for (int v = 0; v < vertexCount; v++)
        {
            double px = vertices[3 * v] - c[0], py = vertices[(3 * v) + 1] - c[1], pz = vertices[(3 * v) + 2] - c[2];
            double tx = (r[0] * px) + (r[3] * py) + (r[6] * pz);
            double ty = (r[1] * px) + (r[4] * py) + (r[7] * pz);
            double tz = (r[2] * px) + (r[5] * py) + (r[8] * pz);
            depth[v] = tz;
            screenX[v] = tz > NearPlane ? (fx * tx / tz) + cx : 0.0;
            screenY[v] = tz > NearPlane ? (fy * ty / tz) + cy : 0.0;
        }

        result.ScreenX = screenX;
        result.ScreenY = screenY;
        result.VertexDepth = depth;
        var zBuffer = new double[width * height];
        for (int i = 0; i < zBuffer.Length; i++)
            zBuffer[i] = double.MaxValue;

        int faceCount = faces.Length / 3;
        for (int f = 0; f < faceCount; f++)
        {
            int i0 = faces[3 * f], i1 = faces[(3 * f) + 1], i2 = faces[(3 * f) + 2];
            double z0 = depth[i0], z1 = depth[i1], z2 = depth[i2];
            if (z0 <= NearPlane || z1 <= NearPlane || z2 <= NearPlane)
                continue;
            double x0 = screenX[i0], y0 = screenY[i0], x1 = screenX[i1], y1 = screenY[i1], x2 = screenX[i2], y2 = screenY[i2];
            double area = ((x1 - x0) * (y2 - y0)) - ((x2 - x0) * (y1 - y0));
            if (area == 0.0)
                continue;
            int minX = Math.Max(0, (int)Math.Ceiling(Math.Min(x0, Math.Min(x1, x2))));
            int maxX = Math.Min(width - 1, (int)Math.Floor(Math.Max(x0, Math.Max(x1, x2))));
            int minY = Math.Max(0, (int)Math.Ceiling(Math.Min(y0, Math.Min(y1, y2))));
            int maxY = Math.Min(height - 1, (int)Math.Floor(Math.Max(y0, Math.Max(y1, y2))));
            double inverseArea = 1.0 / area;
            // Screen-linear barycentrics are affine in (x, y): constant derivatives per triangle.
            double l1dx = (y2 - y0) * inverseArea, l1dy = -(x2 - x0) * inverseArea;
            double l2dx = -(y1 - y0) * inverseArea, l2dy = (x1 - x0) * inverseArea;
            double l0dx = -l1dx - l2dx, l0dy = -l1dy - l2dy;
            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    // Screen-space barycentrics from the edge functions; both windings are accepted.
                    double l1 = (((x - x0) * (y2 - y0)) - ((x2 - x0) * (y - y0))) * inverseArea;
                    double l2 = (((x1 - x0) * (y - y0)) - ((x - x0) * (y1 - y0))) * inverseArea;
                    double l0 = 1.0 - l1 - l2;
                    if (l0 < 0.0 || l1 < 0.0 || l2 < 0.0)
                        continue;
                    double w0 = l0 / z0, w1 = l1 / z1, w2 = l2 / z2;
                    double sum = w0 + w1 + w2;
                    double z = 1.0 / sum;
                    int p = (y * width) + x;
                    if (z >= zBuffer[p])
                        continue;
                    zBuffer[p] = z;
                    result.Face[p] = f;
                    result.Bary1[p] = w1 * z;
                    result.Bary2[p] = w2 * z;
                    result.Depth[p] = z;
                    // b_i = (l_i / z_i) / S with S = sum_j l_j / z_j: d b_i = (d l_i / z_i - b_i dS) / S.
                    double sdx = (l0dx / z0) + (l1dx / z1) + (l2dx / z2);
                    double sdy = (l0dy / z0) + (l1dy / z1) + (l2dy / z2);
                    double b1 = w1 * z, b2 = w2 * z;
                    result.BaryDerivatives[4 * p] = ((l1dx / z1) - (b1 * sdx)) * z;
                    result.BaryDerivatives[(4 * p) + 1] = ((l1dy / z1) - (b1 * sdy)) * z;
                    result.BaryDerivatives[(4 * p) + 2] = ((l2dx / z2) - (b2 * sdx)) * z;
                    result.BaryDerivatives[(4 * p) + 3] = ((l2dy / z2) - (b2 * sdy)) * z;
                }
            }
        }

        return result;
    }
}
