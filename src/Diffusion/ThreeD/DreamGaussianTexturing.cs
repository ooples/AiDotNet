using AiDotNet.Attributes;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.NeuralRadianceFields.Models;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// A mesh unwrapped into a UV atlas, with the per-vertex normals of the reference's <c>Mesh.auto_normal</c>: the
/// fixed geometry of DreamGaussian's texture stage.
/// </summary>
internal sealed class UvTexturedMesh
{
    internal UvTexturedMesh(UvAtlas atlas, double[] normals)
    {
        Atlas = atlas;
        Normals = normals;
    }

    /// <summary>The atlas (vertices split at seams, faces, texture coordinates).</summary>
    public UvAtlas Atlas { get; }

    /// <summary>Unit normals [V', 3] of the atlas vertices.</summary>
    public double[] Normals { get; }
}

/// <summary>
/// The rasterized mesh seen from one camera: which pixels it covers, the texture coordinate there and the cosine
/// between the surface normal and the direction to the camera (the reference's <c>alpha</c>, <c>texc</c> and
/// <c>viewcos</c>).
/// </summary>
internal sealed class MeshTextureView
{
    internal MeshTextureView(int width, int height, bool[] covered, double[] uv, double[] viewCosine)
    {
        Width = width;
        Height = height;
        Covered = covered;
        Uv = uv;
        ViewCosine = viewCosine;
    }

    /// <summary>Image width.</summary>
    public int Width { get; }

    /// <summary>Image height.</summary>
    public int Height { get; }

    /// <summary>Whether a triangle covers each pixel.</summary>
    public bool[] Covered { get; }

    /// <summary>Texture coordinates [H, W, 2] (u, v).</summary>
    public double[] Uv { get; }

    /// <summary>Normal-to-camera cosine [H, W].</summary>
    public double[] ViewCosine { get; }
}

/// <summary>
/// DreamGaussian's texture stage on fixed geometry: per-vertex normals, the multi-view back-projection into the UV
/// texture (<c>main.py save_model('geo+tex')</c>), the nearest-neighbour inpainting of its border, and the textured
/// renderer and its gradient with respect to the raw (pre-sigmoid) texture that <c>main2.py</c> optimises.
/// </summary>
/// <remarks>
/// Textures are [H, W, 3] row-major; texture coordinate v runs along the rows, as nvdiffrast's <c>dr.texture</c> and
/// the reference's <c>grid_put</c> index them. Sampling is bilinear with wrap-around addressing (nvdiffrast's default
/// boundary mode); nvdiffrast's mipmapped minification (<c>linear-mipmap-linear</c>) and analytic antialiasing are not
/// reproduced.
/// </remarks>
internal static class DreamGaussianTexturing
{
    /// <summary>The reference's <c>trunc_rev_sigmoid</c> clamp.</summary>
    internal const double InverseSigmoidEpsilon = 1e-6;

    /// <summary>
    /// The reference's <c>Mesh.auto_normal</c>: each vertex sums the (area-weighted) normals of its faces; a vertex whose
    /// sum is degenerate gets (0, 0, 1).
    /// </summary>
    [ScalarKernel("Scatters each face normal to its three vertices, as the reference's scatter_add_ does.")]
    internal static double[] AutoNormals(double[] vertices, int[] faces)
    {
        if (vertices is null) throw new ArgumentNullException(nameof(vertices));
        if (faces is null) throw new ArgumentNullException(nameof(faces));
        var normals = new double[vertices.Length];
        for (int f = 0; f < faces.Length / 3; f++)
        {
            int i0 = faces[3 * f], i1 = faces[(3 * f) + 1], i2 = faces[(3 * f) + 2];
            double ax = vertices[3 * i1] - vertices[3 * i0], ay = vertices[(3 * i1) + 1] - vertices[(3 * i0) + 1];
            double az = vertices[(3 * i1) + 2] - vertices[(3 * i0) + 2];
            double bx = vertices[3 * i2] - vertices[3 * i0], by = vertices[(3 * i2) + 1] - vertices[(3 * i0) + 1];
            double bz = vertices[(3 * i2) + 2] - vertices[(3 * i0) + 2];
            double nx = (ay * bz) - (az * by), ny = (az * bx) - (ax * bz), nz = (ax * by) - (ay * bx);
            foreach (int v in new[] { i0, i1, i2 })
            {
                normals[3 * v] += nx;
                normals[(3 * v) + 1] += ny;
                normals[(3 * v) + 2] += nz;
            }
        }

        for (int v = 0; v < normals.Length / 3; v++)
        {
            double x = normals[3 * v], y = normals[(3 * v) + 1], z = normals[(3 * v) + 2];
            double squared = (x * x) + (y * y) + (z * z);
            if (!(squared > 1e-20))
            {
                x = 0.0;
                y = 0.0;
                z = 1.0;
                squared = 1.0;
            }

            double length = Math.Sqrt(Math.Max(squared, 1e-20));
            normals[3 * v] = x / length;
            normals[(3 * v) + 1] = y / length;
            normals[(3 * v) + 2] = z / length;
        }

        return normals;
    }

    /// <summary>Unwraps the mesh and gives every atlas vertex the normal of the vertex it copies.</summary>
    internal static UvTexturedMesh Prepare(double[] vertices, int[] faces, UvAtlasOptions atlasOptions)
    {
        var sourceNormals = AutoNormals(vertices, faces);
        var atlas = UvAtlasUnwrapper.Unwrap(vertices, faces, atlasOptions);
        var normals = new double[atlas.Vertices.Length];
        for (int v = 0; v < atlas.SourceVertex.Length; v++)
        {
            int s = atlas.SourceVertex[v];
            normals[3 * v] = sourceNormals[3 * s];
            normals[(3 * v) + 1] = sourceNormals[(3 * s) + 1];
            normals[(3 * v) + 2] = sourceNormals[(3 * s) + 2];
        }

        return new UvTexturedMesh(atlas, normals);
    }

    /// <summary>Rasterizes the mesh from <paramref name="camera"/> and interpolates its uv and normals.</summary>
    internal static MeshTextureView SampleView(UvTexturedMesh mesh, GaussianSplatCamera camera)
    {
        if (mesh is null) throw new ArgumentNullException(nameof(mesh));
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        var atlas = mesh.Atlas;
        var raster = MeshRasterizer.Rasterize(atlas.Vertices, atlas.Faces, camera);
        var uv = raster.Interpolate(atlas.Uvs, 2, atlas.Faces);
        var normal = raster.Interpolate(mesh.Normals, 3, atlas.Faces);
        int pixels = camera.Width * camera.Height;
        var covered = new bool[pixels];
        var viewCosine = new double[pixels];
        var r = camera.CameraToWorld;
        for (int p = 0; p < pixels; p++)
        {
            if (raster.Face[p] < 0) continue;
            covered[p] = true;
            double nx = normal[3 * p], ny = normal[(3 * p) + 1], nz = normal[(3 * p) + 2];
            double length = Math.Sqrt(Math.Max((nx * nx) + (ny * ny) + (nz * nz), 1e-20));
            // rot_normal = normal @ pose[:3, :3]; its z is the dot with the OpenGL backward axis, the negated forward.
            viewCosine[p] = -((nx * r[2]) + (ny * r[5]) + (nz * r[8])) / length;
        }

        return new MeshTextureView(camera.Width, camera.Height, covered, uv, viewCosine);
    }

    /// <summary>
    /// The reference's bake: for each view, the pixels the mesh covers facing the camera (<c>viewcos &gt;
    /// threshold</c>) are splatted into the texture with <see cref="MipmapLinearGridPut"/>; a texel takes contributions
    /// only while its weight is below 0.1, so earlier views win; the sums are then normalised.
    /// </summary>
    /// <param name="views">Each view's mesh sample and its rendered image [H, W, 3].</param>
    /// <param name="textureSize">Texture resolution.</param>
    /// <param name="minResolution">Coarsest mipmap level. Reference: 256.</param>
    /// <param name="viewCosineThreshold">Reference: 0.5.</param>
    /// <returns>The texture [S, S, 3] and the mask of baked texels.</returns>
    internal static (double[] Albedo, bool[] Mask) Bake(
        IReadOnlyList<(MeshTextureView View, double[] Image)> views, int textureSize, int minResolution, double viewCosineThreshold)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        int texels = textureSize * textureSize;
        var albedo = new double[3 * texels];
        var count = new double[texels];
        foreach (var (view, image) in views)
        {
            int pixels = view.Width * view.Height;
            var coordinates = new List<double>();
            var values = new List<double>();
            for (int p = 0; p < pixels; p++)
            {
                if (!view.Covered[p] || !(view.ViewCosine[p] > viewCosineThreshold)) continue;
                double u = Clamp01(view.Uv[2 * p]), v = Clamp01(view.Uv[(2 * p) + 1]);
                // uvs[..., [1, 0]] * 2 - 1: (row, column) in [-1, 1].
                coordinates.Add((v * 2.0) - 1.0);
                coordinates.Add((u * 2.0) - 1.0);
                values.Add(image[3 * p]);
                values.Add(image[(3 * p) + 1]);
                values.Add(image[(3 * p) + 2]);
            }

            if (values.Count == 0) continue;
            var (viewAlbedo, viewCount) = MipmapLinearGridPut(
                textureSize, textureSize, coordinates.ToArray(), values.ToArray(), minResolution);
            for (int t = 0; t < texels; t++)
            {
                if (!(count[t] < 0.1)) continue;
                albedo[3 * t] += viewAlbedo[3 * t];
                albedo[(3 * t) + 1] += viewAlbedo[(3 * t) + 1];
                albedo[(3 * t) + 2] += viewAlbedo[(3 * t) + 2];
                count[t] += viewCount[t];
            }
        }

        var mask = new bool[texels];
        for (int t = 0; t < texels; t++)
        {
            if (!(count[t] > 0.0)) continue;
            mask[t] = true;
            albedo[3 * t] /= count[t];
            albedo[(3 * t) + 1] /= count[t];
            albedo[(3 * t) + 2] /= count[t];
        }

        return (albedo, mask);
    }

    /// <summary>
    /// The reference's <c>mipmap_linear_grid_put_2d(..., return_count=True)</c>: bilinear splatting at the full
    /// resolution, then at halved resolutions (while above <paramref name="minResolution"/>) upsampled to fill the
    /// texels still empty.
    /// </summary>
    /// <param name="height">Target height.</param>
    /// <param name="width">Target width.</param>
    /// <param name="coordinates">[N, 2] (row, column) in [-1, 1].</param>
    /// <param name="values">[N, 3].</param>
    /// <param name="minResolution">Coarsest level.</param>
    /// <returns>Weighted sums [H, W, 3] and weights [H, W].</returns>
    internal static (double[] Result, double[] Count) MipmapLinearGridPut(
        int height, int width, double[] coordinates, double[] values, int minResolution)
    {
        var result = new double[3 * height * width];
        var count = new double[height * width];
        int h = height, w = width;
        while (Math.Min(h, w) > minResolution)
        {
            bool anyEmpty = false;
            for (int t = 0; t < count.Length && !anyEmpty; t++)
                anyEmpty = count[t] == 0.0;
            if (!anyEmpty) break;
            var empty = new bool[count.Length];
            for (int t = 0; t < count.Length; t++)
                empty[t] = count[t] == 0.0;

            var (levelResult, levelCount) = LinearGridPut(h, w, coordinates, values);
            var upResult = ResizeBilinear(levelResult, h, w, 3, height, width);
            var upCount = ResizeBilinear(levelCount, h, w, 1, height, width);
            for (int t = 0; t < count.Length; t++)
            {
                if (!empty[t]) continue;
                result[3 * t] += upResult[3 * t];
                result[(3 * t) + 1] += upResult[(3 * t) + 1];
                result[(3 * t) + 2] += upResult[(3 * t) + 2];
                count[t] += upCount[t];
            }

            h /= 2;
            w /= 2;
        }

        return (result, count);
    }

    /// <summary>The reference's <c>linear_grid_put_2d(..., return_count=True)</c>.</summary>
    internal static (double[] Result, double[] Count) LinearGridPut(int height, int width, double[] coordinates, double[] values)
    {
        var result = new double[3 * height * width];
        var count = new double[height * width];
        int n = coordinates.Length / 2;
        for (int i = 0; i < n; i++)
        {
            double y = ((coordinates[2 * i] * 0.5) + 0.5) * (height - 1);
            double x = ((coordinates[(2 * i) + 1] * 0.5) + 0.5) * (width - 1);
            int y0 = Math.Min(Math.Max((int)Math.Floor(y), 0), height - 2);
            int x0 = Math.Min(Math.Max((int)Math.Floor(x), 0), width - 2);
            double fy = y - y0, fx = x - x0;
            Put(result, count, width, y0, x0, (1.0 - fy) * (1.0 - fx), values, i);
            Put(result, count, width, y0, x0 + 1, (1.0 - fy) * fx, values, i);
            Put(result, count, width, y0 + 1, x0, fy * (1.0 - fx), values, i);
            Put(result, count, width, y0 + 1, x0 + 1, fy * fx, values, i);
        }

        return (result, count);
    }

    private static void Put(double[] result, double[] count, int width, int y, int x, double weight, double[] values, int i)
    {
        int t = (y * width) + x;
        result[3 * t] += weight * values[3 * i];
        result[(3 * t) + 1] += weight * values[(3 * i) + 1];
        result[(3 * t) + 2] += weight * values[(3 * i) + 2];
        count[t] += weight;
    }

    /// <summary><c>F.interpolate(mode='bilinear', align_corners=False)</c> of an [H, W, C] array.</summary>
    internal static double[] ResizeBilinear(double[] source, int sourceHeight, int sourceWidth, int channels, int height, int width)
    {
        if (sourceHeight == height && sourceWidth == width)
            return (double[])source.Clone();
        var output = new double[height * width * channels];
        double scaleY = (double)sourceHeight / height, scaleX = (double)sourceWidth / width;
        for (int y = 0; y < height; y++)
        {
            double sy = Math.Max(0.0, ((y + 0.5) * scaleY) - 0.5);
            int y0 = Math.Min((int)sy, sourceHeight - 1);
            int y1 = Math.Min(y0 + 1, sourceHeight - 1);
            double fy = sy - y0;
            for (int x = 0; x < width; x++)
            {
                double sx = Math.Max(0.0, ((x + 0.5) * scaleX) - 0.5);
                int x0 = Math.Min((int)sx, sourceWidth - 1);
                int x1 = Math.Min(x0 + 1, sourceWidth - 1);
                double fx = sx - x0;
                for (int c = 0; c < channels; c++)
                {
                    double top = ((1.0 - fx) * source[(((y0 * sourceWidth) + x0) * channels) + c])
                        + (fx * source[(((y0 * sourceWidth) + x1) * channels) + c]);
                    double bottom = ((1.0 - fx) * source[(((y1 * sourceWidth) + x0) * channels) + c])
                        + (fx * source[(((y1 * sourceWidth) + x1) * channels) + c]);
                    output[(((y * width) + x) * channels) + c] = ((1.0 - fy) * top) + (fy * bottom);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// The reference's texture inpainting: texels within <paramref name="dilation"/> steps of the baked region
    /// (<c>binary_dilation</c>, 4-connected) copy the colour of the nearest texel (Euclidean) of the region's border
    /// band, the baked texels removed by <paramref name="erosion"/> steps of <c>binary_erosion</c>.
    /// </summary>
    /// <returns>The inpainted texture (a copy).</returns>
    internal static double[] Inpaint(double[] albedo, bool[] mask, int size, int dilation, int erosion)
    {
        if (albedo is null) throw new ArgumentNullException(nameof(albedo));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        var output = (double[])albedo.Clone();
        var dilated = mask;
        for (int i = 0; i < dilation; i++)
            dilated = Morph(dilated, size, dilate: true);
        var eroded = mask;
        for (int i = 0; i < erosion; i++)
            eroded = Morph(eroded, size, dilate: false);

        var search = new bool[mask.Length];
        bool anySearch = false;
        for (int t = 0; t < mask.Length; t++)
        {
            search[t] = mask[t] && !eroded[t];
            anySearch |= search[t];
        }

        if (!anySearch) return output;
        var nearest = NearestFeature(search, size);
        for (int t = 0; t < mask.Length; t++)
        {
            if (!dilated[t] || mask[t]) continue;
            int s = nearest[t];
            output[3 * t] = albedo[3 * s];
            output[(3 * t) + 1] = albedo[(3 * s) + 1];
            output[(3 * t) + 2] = albedo[(3 * s) + 2];
        }

        return output;
    }

    /// <summary>One step of scipy's <c>binary_dilation</c> / <c>binary_erosion</c> with the 4-connected cross and a
    /// zero border.</summary>
    private static bool[] Morph(bool[] mask, int size, bool dilate)
    {
        var output = new bool[mask.Length];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int t = (y * size) + x;
                bool up = y > 0 && mask[t - size], down = y < size - 1 && mask[t + size];
                bool left = x > 0 && mask[t - 1], right = x < size - 1 && mask[t + 1];
                output[t] = dilate
                    ? mask[t] || up || down || left || right
                    : mask[t] && up && down && left && right;
            }
        }

        return output;
    }

    /// <summary>
    /// For every texel the index of the nearest feature texel (exact Euclidean distance transform, Felzenszwalb and
    /// Huttenlocher), standing in for the reference's k-d tree query.
    /// </summary>
    private static int[] NearestFeature(bool[] feature, int size)
    {
        const double infinity = 1e20;
        // Column pass: the nearest feature row in each column.
        var columnRow = new int[feature.Length];
        var columnDistance = new double[feature.Length];
        for (int x = 0; x < size; x++)
        {
            int last = -1;
            for (int y = 0; y < size; y++)
            {
                int t = (y * size) + x;
                if (feature[t]) last = y;
                columnRow[t] = last;
                columnDistance[t] = last < 0 ? infinity : y - last;
            }

            last = -1;
            for (int y = size - 1; y >= 0; y--)
            {
                int t = (y * size) + x;
                if (feature[t]) last = y;
                if (last >= 0 && last - y < columnDistance[t])
                {
                    columnDistance[t] = last - y;
                    columnRow[t] = last;
                }
            }
        }

        // Row pass: the lower envelope of the parabolas (x - q)^2 + d(q)^2.
        var nearest = new int[feature.Length];
        var vertices = new int[size];
        var bounds = new double[size + 1];
        var f = new double[size];
        for (int y = 0; y < size; y++)
        {
            int k = -1;
            for (int q = 0; q < size; q++)
            {
                double d = columnDistance[(y * size) + q];
                f[q] = d >= infinity ? infinity : d * d;
                if (f[q] >= infinity) continue;
                if (k < 0)
                {
                    k = 0;
                    vertices[0] = q;
                    bounds[0] = double.NegativeInfinity;
                    bounds[1] = double.PositiveInfinity;
                    continue;
                }

                // bounds[0] is -infinity, so the loop stops at k = 0 at the latest.
                double s;
                while (true)
                {
                    int p = vertices[k];
                    s = ((f[q] + (q * (double)q)) - (f[p] + (p * (double)p))) / (2.0 * (q - p));
                    if (s <= bounds[k]) k--;
                    else break;
                }

                k++;
                vertices[k] = q;
                bounds[k] = s;
                bounds[k + 1] = double.PositiveInfinity;
            }

            if (k < 0)
            {
                for (int x = 0; x < size; x++) nearest[(y * size) + x] = -1;
                continue;
            }

            int j = 0;
            for (int x = 0; x < size; x++)
            {
                while (bounds[j + 1] < x) j++;
                int q = vertices[j];
                nearest[(y * size) + x] = (columnRow[(y * size) + q] * size) + q;
            }
        }

        return nearest;
    }

    /// <summary>The reference's <c>trunc_rev_sigmoid</c>: <c>logit(clamp(x, eps, 1 - eps))</c>.</summary>
    internal static double[] TruncatedInverseSigmoid(double[] values)
    {
        var output = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            double x = Math.Min(1.0 - InverseSigmoidEpsilon, Math.Max(InverseSigmoidEpsilon, values[i]));
            output[i] = Math.Log(x / (1.0 - x));
        }

        return output;
    }

    /// <summary><c>sigmoid</c> of every value.</summary>
    internal static double[] Sigmoid(double[] values)
    {
        var output = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
            output[i] = 1.0 / (1.0 + Math.Exp(-values[i]));
        return output;
    }

    /// <summary>
    /// The reference's mesh render: <c>sigmoid(texture(raw, uv))</c> where the mesh covers the pixel, the background
    /// elsewhere, clamped to [0, 1].
    /// </summary>
    /// <returns>[H, W, 3].</returns>
    internal static double[] Render(MeshTextureView view, double[] rawTexture, int textureSize, double background)
    {
        int pixels = view.Width * view.Height;
        var image = new double[3 * pixels];
        var raw = new double[3];
        for (int p = 0; p < pixels; p++)
        {
            if (!view.Covered[p])
            {
                image[3 * p] = background;
                image[(3 * p) + 1] = background;
                image[(3 * p) + 2] = background;
                continue;
            }

            SampleBilinear(rawTexture, textureSize, view.Uv[2 * p], view.Uv[(2 * p) + 1], raw);
            for (int c = 0; c < 3; c++)
                image[(3 * p) + c] = 1.0 / (1.0 + Math.Exp(-raw[c]));
        }

        return image;
    }

    /// <summary>
    /// Adds dL/d raw texture to <paramref name="rawGradient"/> for an image gradient of a <see cref="Render"/>.
    /// </summary>
    internal static void Backward(
        MeshTextureView view, double[] rawTexture, int textureSize, double[] imageGradient, double[] rawGradient)
    {
        int pixels = view.Width * view.Height;
        var raw = new double[3];
        var texel = new int[4];
        var weight = new double[4];
        for (int p = 0; p < pixels; p++)
        {
            if (!view.Covered[p]) continue;
            BilinearTaps(textureSize, view.Uv[2 * p], view.Uv[(2 * p) + 1], texel, weight);
            SampleTaps(rawTexture, texel, weight, raw);
            for (int c = 0; c < 3; c++)
            {
                double s = 1.0 / (1.0 + Math.Exp(-raw[c]));
                double g = imageGradient[(3 * p) + c] * s * (1.0 - s);
                if (g == 0.0) continue;
                for (int k = 0; k < 4; k++)
                    rawGradient[(3 * texel[k]) + c] += g * weight[k];
            }
        }
    }

    /// <summary>Bilinear lookup of an [S, S, 3] texture at (u, v) with wrap-around addressing (nvdiffrast's convention:
    /// texel centres at <c>(i + 0.5) / S</c>).</summary>
    internal static void SampleBilinear(double[] texture, int size, double u, double v, double[] output)
    {
        var texel = new int[4];
        var weight = new double[4];
        BilinearTaps(size, u, v, texel, weight);
        SampleTaps(texture, texel, weight, output);
    }

    private static void SampleTaps(double[] texture, int[] texel, double[] weight, double[] output)
    {
        for (int c = 0; c < 3; c++)
        {
            output[c] = (weight[0] * texture[(3 * texel[0]) + c]) + (weight[1] * texture[(3 * texel[1]) + c])
                + (weight[2] * texture[(3 * texel[2]) + c]) + (weight[3] * texture[(3 * texel[3]) + c]);
        }
    }

    private static void BilinearTaps(int size, double u, double v, int[] texel, double[] weight)
    {
        double x = (u * size) - 0.5, y = (v * size) - 0.5;
        double fx0 = Math.Floor(x), fy0 = Math.Floor(y);
        double fx = x - fx0, fy = y - fy0;
        int x0 = Wrap((int)fx0, size), x1 = Wrap((int)fx0 + 1, size);
        int y0 = Wrap((int)fy0, size), y1 = Wrap((int)fy0 + 1, size);
        texel[0] = (y0 * size) + x0;
        texel[1] = (y0 * size) + x1;
        texel[2] = (y1 * size) + x0;
        texel[3] = (y1 * size) + x1;
        weight[0] = (1.0 - fx) * (1.0 - fy);
        weight[1] = fx * (1.0 - fy);
        weight[2] = (1.0 - fx) * fy;
        weight[3] = fx * fy;
    }

    private static int Wrap(int i, int size)
    {
        int r = i % size;
        return r < 0 ? r + size : r;
    }

    private static double Clamp01(double x) => Math.Min(1.0, Math.Max(0.0, x));
}
