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
        int faceCount = atlas.Faces.Length / 3;
        // The position topology (mesh.f): the atlas faces through SourceVertex, so UV seams are not silhouettes.
        var edges = new Dictionary<long, List<int>>();
        for (int f = 0; f < faceCount; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = atlas.SourceVertex[atlas.Faces[(3 * f) + k]];
                int b = atlas.SourceVertex[atlas.Faces[(3 * f) + ((k + 1) % 3)]];
                long key = ((long)Math.Min(a, b) << 32) | (uint)Math.Max(a, b);
                if (!edges.TryGetValue(key, out var list))
                {
                    list = new List<int>(2);
                    edges[key] = list;
                }

                list.Add((3 * f) + k);
            }
        }

        OppositeFace = new int[3 * faceCount];
        for (int i = 0; i < OppositeFace.Length; i++) OppositeFace[i] = -1;
        foreach (var list in edges.Values)
        {
            // A manifold edge has exactly two sides; a boundary or non-manifold edge is always a silhouette.
            if (list.Count != 2) continue;
            OppositeFace[list[0]] = list[1] / 3;
            OppositeFace[list[1]] = list[0] / 3;
        }
    }

    /// <summary>The atlas (vertices split at seams, faces, texture coordinates).</summary>
    public UvAtlas Atlas { get; }

    /// <summary>Unit normals [V', 3] of the atlas vertices.</summary>
    public double[] Normals { get; }

    /// <summary>
    /// For edge k (vertex k to k + 1) of every face, the face across it in the position topology, or -1 for a boundary
    /// or non-manifold edge.
    /// </summary>
    public int[] OppositeFace { get; }
}

/// <summary>
/// The rasterized mesh seen from one camera: which pixels it covers, the texture coordinate there and the cosine
/// between the surface normal and the direction to the camera (the reference's <c>alpha</c>, <c>texc</c> and
/// <c>viewcos</c>), the uv derivatives that select the mip level, and the antialiasing blends of its silhouettes.
/// </summary>
internal sealed class MeshTextureView
{
    internal MeshTextureView(int width, int height, bool[] covered, double[] uv, double[] viewCosine, double[] uvDerivatives,
        AntialiasBlends antialias)
    {
        Width = width;
        Height = height;
        Covered = covered;
        Uv = uv;
        ViewCosine = viewCosine;
        UvDerivatives = uvDerivatives;
        Antialias = antialias;
    }

    /// <summary>Image width.</summary>
    public int Width { get; }

    /// <summary>Image height.</summary>
    public int Height { get; }

    /// <summary>Whether a triangle covers each pixel.</summary>
    public bool[] Covered { get; }

    /// <summary>Texture coordinates [H, W, 2] (u, v); zero where nothing is covered, as <c>dr.interpolate</c>.</summary>
    public double[] Uv { get; }

    /// <summary>Normal-to-camera cosine [H, W]; zero where nothing is covered.</summary>
    public double[] ViewCosine { get; }

    /// <summary>Pixel-space uv derivatives [H, W, 4]: du/dx, du/dy, dv/dx, dv/dy (nvdiffrast's <c>texc_db</c>).</summary>
    public double[] UvDerivatives { get; }

    /// <summary>The silhouette blends of <c>dr.antialias</c> for this view.</summary>
    public AntialiasBlends Antialias { get; }
}

/// <summary>
/// The pixel pairs <c>dr.antialias</c> blends (Laine et al., "Modular Primitives for High-Performance Differentiable
/// Rendering", 2020): <c>out[d] = in[d] + w (in[s] - in[d])</c> for every pair, all read from the input.
/// </summary>
internal sealed class AntialiasBlends
{
    internal AntialiasBlends(int[] destination, int[] source, double[] weight)
    {
        Destination = destination;
        Source = source;
        Weight = weight;
    }

    /// <summary>The pixel whose colour is blended.</summary>
    public int[] Destination { get; }

    /// <summary>The pixel across the silhouette edge it takes colour from.</summary>
    public int[] Source { get; }

    /// <summary>The blend weight in [0, 0.5]: the distance of the edge crossing from the pair's midpoint.</summary>
    public double[] Weight { get; }
}

/// <summary>
/// A view rendered at a super-sampled size and resized back (<c>mesh_renderer.py</c>'s <c>ssaa</c>): the view at the
/// render size, the resize taps, and the resized <c>alpha</c> and <c>viewcos</c>.
/// </summary>
internal sealed class SupersampledTextureView
{
    internal SupersampledTextureView(MeshTextureView view, int width, int height, ResizeTaps? taps, double[] alpha,
        double[] viewCosine)
    {
        View = view;
        Width = width;
        Height = height;
        Taps = taps;
        Alpha = alpha;
        ViewCosine = viewCosine;
    }

    /// <summary>The view at the render (super-sampled) size.</summary>
    public MeshTextureView View { get; }

    /// <summary>Output width.</summary>
    public int Width { get; }

    /// <summary>Output height.</summary>
    public int Height { get; }

    /// <summary>The bilinear taps from the render size to the output size; null when the sizes agree.</summary>
    public ResizeTaps? Taps { get; }

    /// <summary>The coverage resized to the output [H, W] (<c>scale_img_hwc(alpha)</c>).</summary>
    public double[] Alpha { get; }

    /// <summary>The view cosine resized to the output [H, W].</summary>
    public double[] ViewCosine { get; }
}

/// <summary>Four bilinear taps per output pixel of a resize, so a resize and its transpose share one definition.</summary>
internal sealed class ResizeTaps
{
    internal ResizeTaps(int sourcePixels, int[] index, double[] weight)
    {
        SourcePixels = sourcePixels;
        Index = index;
        Weight = weight;
    }

    /// <summary>The number of source pixels.</summary>
    public int SourcePixels { get; }

    /// <summary>Source pixel of each tap [P_out, 4].</summary>
    public int[] Index { get; }

    /// <summary>Weight of each tap [P_out, 4].</summary>
    public double[] Weight { get; }

    /// <summary>Resizes an [H, W, channels] image.</summary>
    public double[] Apply(double[] source, int channels)
    {
        int outputs = Index.Length / 4;
        var output = new double[outputs * channels];
        for (int p = 0; p < outputs; p++)
        {
            for (int k = 0; k < 4; k++)
            {
                int s = Index[(4 * p) + k];
                double w = Weight[(4 * p) + k];
                for (int c = 0; c < channels; c++)
                    output[(p * channels) + c] += w * source[(s * channels) + c];
            }
        }

        return output;
    }

    /// <summary>The transpose: scatters an output-size gradient back to the source size.</summary>
    public double[] Transpose(double[] gradient, int channels)
    {
        int outputs = Index.Length / 4;
        var result = new double[SourcePixels * channels];
        for (int p = 0; p < outputs; p++)
        {
            for (int k = 0; k < 4; k++)
            {
                int s = Index[(4 * p) + k];
                double w = Weight[(4 * p) + k];
                for (int c = 0; c < channels; c++)
                    result[(s * channels) + c] += w * gradient[(p * channels) + c];
            }
        }

        return result;
    }
}

/// <summary>
/// DreamGaussian's texture stage on fixed geometry: per-vertex normals, the multi-view back-projection into the UV
/// texture (<c>main.py save_model('geo+tex')</c>), the nearest-neighbour inpainting of its border, and the textured
/// renderer and its gradient with respect to the raw (pre-sigmoid) texture that <c>main2.py</c> optimises.
/// </summary>
/// <remarks>
/// Textures are [H, W, 3] row-major; texture coordinate v runs along the rows, as nvdiffrast's <c>dr.texture</c> and
/// the reference's <c>grid_put</c> index them. The renderer is <c>mesh_renderer.py</c>'s: <c>dr.texture</c> with
/// <c>filter_mode='linear-mipmap-linear'</c> (box-filtered mip pyramid, level of detail from the uv derivatives,
/// bilinear lookups with wrap-around addressing blended between two levels), <c>sigmoid</c>, <c>dr.antialias</c> over
/// the silhouette edges, compositing over the background, and the random super-sampling of <c>main2.py</c> (render at
/// <c>make_divisible(size * ssaa, 8)</c>, resize back with <c>scale_img_hwc</c>).
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

    /// <summary>
    /// Rasterizes the mesh from <paramref name="camera"/>, interpolates its uv (with derivatives) and normals, and finds
    /// the silhouette blends of <c>dr.antialias</c>.
    /// </summary>
    internal static MeshTextureView SampleView(UvTexturedMesh mesh, GaussianSplatCamera camera)
    {
        if (mesh is null) throw new ArgumentNullException(nameof(mesh));
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        var atlas = mesh.Atlas;
        var raster = MeshRasterizer.Rasterize(atlas.Vertices, atlas.Faces, camera);
        var uv = raster.Interpolate(atlas.Uvs, 2, atlas.Faces);
        var uvDerivatives = raster.InterpolateDerivatives(atlas.Uvs, 2, atlas.Faces);
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

        return new MeshTextureView(
            camera.Width, camera.Height, covered, uv, viewCosine, uvDerivatives, FindSilhouetteBlends(mesh, raster));
    }

    /// <summary>
    /// <c>dr.antialias</c>'s analysis: for every horizontally or vertically adjacent pixel pair showing different
    /// triangles, the nearer triangle's silhouette edge (a boundary edge, or one whose neighbour across it faces the other
    /// way or is not drawn) that crosses the segment between the pixel centres. The pixel in whose half the crossing lies
    /// takes the other pixel's colour, weighted by the crossing's distance from the midpoint.
    /// </summary>
    internal static AntialiasBlends FindSilhouetteBlends(UvTexturedMesh mesh, MeshRasterization raster)
    {
        var faces = mesh.Atlas.Faces;
        int width = raster.Width, height = raster.Height;
        var destination = new List<int>();
        var source = new List<int>();
        var weight = new List<double>();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int p = (y * width) + x;
                if (x + 1 < width) Pair(p, p + 1, x, y, x + 1, y);
                if (y + 1 < height) Pair(p, p + width, x, y, x, y + 1);
            }
        }

        return new AntialiasBlends(destination.ToArray(), source.ToArray(), weight.ToArray());

        void Pair(int p, int q, int px, int py, int qx, int qy)
        {
            int fp = raster.Face[p], fq = raster.Face[q];
            if (fp == fq) return;
            double dp = fp >= 0 ? raster.Depth[p] : double.PositiveInfinity;
            double dq = fq >= 0 ? raster.Depth[q] : double.PositiveInfinity;
            bool pNearer = dp <= dq;
            int face = pNearer ? fp : fq;
            int inside = pNearer ? p : q, outside = pNearer ? q : p;
            double ix = pNearer ? px : qx, iy = pNearer ? py : qy, ox = pNearer ? qx : px, oy = pNearer ? qy : py;
            int i0 = faces[3 * face], i1 = faces[(3 * face) + 1], i2 = faces[(3 * face) + 2];
            double winding = raster.SignedArea(i0, i1, i2);
            for (int k = 0; k < 3; k++)
            {
                if (!IsSilhouette(face, k, winding)) continue;
                int a = faces[(3 * face) + k], b = faces[(3 * face) + ((k + 1) % 3)];
                double ax = raster.ScreenX[a], ay = raster.ScreenY[a], bx = raster.ScreenX[b], by = raster.ScreenY[b];
                double ein = ((bx - ax) * (iy - ay)) - ((by - ay) * (ix - ax));
                double eout = ((bx - ax) * (oy - ay)) - ((by - ay) * (ox - ax));
                if (!(ein * eout < 0.0)) continue;
                double t = ein / (ein - eout);
                // The crossing must lie on the edge segment, not just its line.
                double cx = ix + (t * (ox - ix)), cy = iy + (t * (oy - iy));
                double ex = bx - ax, ey = by - ay, edgeLength2 = (ex * ex) + (ey * ey);
                if (edgeLength2 <= 0.0) continue;
                double along = (((cx - ax) * ex) + ((cy - ay) * ey)) / edgeLength2;
                if (along < 0.0 || along > 1.0) continue;
                if (t > 0.5)
                {
                    destination.Add(outside);
                    source.Add(inside);
                    weight.Add(t - 0.5);
                }
                else if (t < 0.5)
                {
                    destination.Add(inside);
                    source.Add(outside);
                    weight.Add(0.5 - t);
                }

                return;
            }
        }

        bool IsSilhouette(int face, int k, double winding)
        {
            int opposite = mesh.OppositeFace[(3 * face) + k];
            if (opposite < 0) return true;
            int j0 = faces[3 * opposite], j1 = faces[(3 * opposite) + 1], j2 = faces[(3 * opposite) + 2];
            if (!raster.IsDrawable(j0, j1, j2)) return true;
            double other = raster.SignedArea(j0, j1, j2);
            return (winding > 0.0) != (other > 0.0);
        }
    }

    /// <summary>main2.py's random super-sampling factor: <c>min(2, max(0.125, 2 * np.random.random()))</c>.</summary>
    internal static double DrawSupersampling(Random random) => Math.Min(2.0, Math.Max(0.125, 2.0 * random.NextDouble()));

    /// <summary>The render size for a super-sampling factor: <c>make_divisible(size * ssaa, 8)</c> (unchanged at 1).</summary>
    internal static int SupersampledSize(int size, double ssaa)
        => ssaa == 1.0 ? size : (int)(Math.Ceiling(size * ssaa / 8.0) * 8.0);

    /// <summary>
    /// Samples a view at the super-sampled size and prepares its resize back to <paramref name="width"/> x
    /// <paramref name="height"/>.
    /// </summary>
    /// <param name="mesh">The geometry.</param>
    /// <param name="cameraAt">The camera for a render width and height (same pose and field of view).</param>
    /// <param name="width">Output width.</param>
    /// <param name="height">Output height.</param>
    /// <param name="ssaa">The super-sampling factor.</param>
    internal static SupersampledTextureView SampleSupersampledView(
        UvTexturedMesh mesh, Func<int, int, GaussianSplatCamera> cameraAt, int width, int height, double ssaa)
    {
        if (cameraAt is null) throw new ArgumentNullException(nameof(cameraAt));
        int renderWidth = SupersampledSize(width, ssaa), renderHeight = SupersampledSize(height, ssaa);
        var view = SampleView(mesh, cameraAt(renderWidth, renderHeight));
        int pixels = renderWidth * renderHeight;
        var alpha = new double[pixels];
        for (int p = 0; p < pixels; p++) alpha[p] = view.Covered[p] ? 1.0 : 0.0;
        if (renderWidth == width && renderHeight == height)
            return new SupersampledTextureView(view, width, height, null, alpha, (double[])view.ViewCosine.Clone());

        var taps = ScaleImageTaps(renderHeight, renderWidth, height, width);
        return new SupersampledTextureView(view, width, height, taps, taps.Apply(alpha, 1), taps.Apply(view.ViewCosine, 1));
    }

    /// <summary>
    /// The taps of <c>scale_img_hwc</c>: minification (both sides shrink) is <c>F.interpolate(mode='bilinear')</c>
    /// (<c>align_corners=False</c>), magnification is bilinear with <c>align_corners=True</c>.
    /// </summary>
    internal static ResizeTaps ScaleImageTaps(int sourceHeight, int sourceWidth, int height, int width)
    {
        bool minify = sourceHeight > height && sourceWidth > width;
        var index = new int[4 * height * width];
        var weight = new double[4 * height * width];
        for (int y = 0; y < height; y++)
        {
            var (y0, y1, fy) = Source(y, sourceHeight, height, minify);
            for (int x = 0; x < width; x++)
            {
                var (x0, x1, fx) = Source(x, sourceWidth, width, minify);
                int p = (y * width) + x;
                index[4 * p] = (y0 * sourceWidth) + x0;
                index[(4 * p) + 1] = (y0 * sourceWidth) + x1;
                index[(4 * p) + 2] = (y1 * sourceWidth) + x0;
                index[(4 * p) + 3] = (y1 * sourceWidth) + x1;
                weight[4 * p] = (1.0 - fy) * (1.0 - fx);
                weight[(4 * p) + 1] = (1.0 - fy) * fx;
                weight[(4 * p) + 2] = fy * (1.0 - fx);
                weight[(4 * p) + 3] = fy * fx;
            }
        }

        return new ResizeTaps(sourceHeight * sourceWidth, index, weight);

        static (int Low, int High, double Fraction) Source(int i, int sourceSize, int size, bool alignCornersFalse)
        {
            double s = alignCornersFalse
                ? Math.Max(0.0, ((i + 0.5) * sourceSize / size) - 0.5)
                : (size > 1 ? (double)i * (sourceSize - 1) / (size - 1) : 0.0);
            int low = Math.Min((int)Math.Floor(s), sourceSize - 1);
            int high = Math.Min(low + 1, sourceSize - 1);
            return (low, high, s - low);
        }
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
    /// The mip pyramid nvdiffrast builds for <c>linear-mipmap-linear</c>: level 0 is the texture, each further level
    /// halves the size (down to 1 x 1) and averages 2 x 2 texels of the level below.
    /// </summary>
    /// <returns>The levels [S_l, S_l, 3] and their sizes.</returns>
    internal static (double[][] Levels, int[] Sizes) BuildMipmaps(double[] texture, int size)
    {
        var levels = new List<double[]> { texture };
        var sizes = new List<int> { size };
        int current = size;
        while (current > 1)
        {
            int next = Math.Max(1, current >> 1);
            var below = levels[levels.Count - 1];
            var level = new double[3 * next * next];
            for (int y = 0; y < next; y++)
            {
                for (int x = 0; x < next; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        double sum = below[(3 * (((2 * y) * current) + (2 * x))) + c]
                            + below[(3 * (((2 * y) * current) + (2 * x) + 1)) + c]
                            + below[(3 * ((((2 * y) + 1) * current) + (2 * x))) + c]
                            + below[(3 * ((((2 * y) + 1) * current) + (2 * x) + 1)) + c];
                        level[(3 * ((y * next) + x)) + c] = 0.25 * sum;
                    }
                }
            }

            levels.Add(level);
            sizes.Add(next);
            current = next;
        }

        return (levels.ToArray(), sizes.ToArray());
    }

    /// <summary>
    /// nvdiffrast's level of detail: with the uv derivatives in texels, the squared major axis of the footprint ellipse
    /// <c>l2b + sqrt(l2n)</c> (<c>l2b = (A + B) / 2</c>, <c>l2n = (A - B)^2 / 4 + C^2</c>) gives
    /// <c>0.5 log2(lenMajorSqr)</c>, clamped to the pyramid.
    /// </summary>
    internal static double MipLevel(double[] uvDerivatives, int pixel, int size, int maxLevel)
    {
        double dsdx = uvDerivatives[4 * pixel] * size, dsdy = uvDerivatives[(4 * pixel) + 1] * size;
        double dtdx = uvDerivatives[(4 * pixel) + 2] * size, dtdy = uvDerivatives[(4 * pixel) + 3] * size;
        double a = (dsdx * dsdx) + (dtdx * dtdx);
        double b = (dsdy * dsdy) + (dtdy * dtdy);
        double c = (dsdx * dsdy) + (dtdx * dtdy);
        double l2b = 0.5 * (a + b);
        double l2n = (0.25 * (a - b) * (a - b)) + (c * c);
        double lenMajorSqr = l2b + Math.Sqrt(l2n);
        if (!(lenMajorSqr > 0.0)) return 0.0;
        return Math.Min(maxLevel, Math.Max(0.0, 0.5 * Math.Log(lenMajorSqr, 2.0)));
    }

    /// <summary>
    /// The eight taps of a <c>linear-mipmap-linear</c> lookup: bilinear in levels floor(lod) and floor(lod) + 1, blended
    /// by the fraction.
    /// </summary>
    private static void TrilinearTaps(int[] sizes, double u, double v, double lod, int[] level, int[] texel, double[] weight)
    {
        int maxLevel = sizes.Length - 1;
        int l0 = Math.Min(maxLevel, (int)Math.Floor(lod));
        int l1 = Math.Min(maxLevel, l0 + 1);
        double f = l1 == l0 ? 0.0 : lod - l0;
        BilinearTaps(sizes[l0], u, v, texel, weight, 0);
        BilinearTaps(sizes[l1], u, v, texel, weight, 4);
        for (int k = 0; k < 4; k++)
        {
            level[k] = l0;
            weight[k] *= 1.0 - f;
            level[4 + k] = l1;
            weight[4 + k] *= f;
        }
    }

    /// <summary>
    /// The pre-antialiasing colours <c>sigmoid(dr.texture(raw, texc, texc_db, 'linear-mipmap-linear'))</c> [H, W, 3] of
    /// every pixel, the uncovered ones too (their uv and derivatives are zero, as <c>dr.interpolate</c> leaves them).
    /// </summary>
    private static double[] ShadeAlbedo(MeshTextureView view, double[][] levels, int[] sizes)
    {
        int pixels = view.Width * view.Height;
        var albedo = new double[3 * pixels];
        var level = new int[8];
        var texel = new int[8];
        var weight = new double[8];
        for (int p = 0; p < pixels; p++)
        {
            double lod = MipLevel(view.UvDerivatives, p, sizes[0], sizes.Length - 1);
            TrilinearTaps(sizes, view.Uv[2 * p], view.Uv[(2 * p) + 1], lod, level, texel, weight);
            for (int c = 0; c < 3; c++)
            {
                double raw = 0.0;
                for (int k = 0; k < 8; k++)
                    raw += weight[k] * levels[level[k]][(3 * texel[k]) + c];
                albedo[(3 * p) + c] = 1.0 / (1.0 + Math.Exp(-raw));
            }
        }

        return albedo;
    }

    /// <summary>
    /// The reference's mesh render at the view's size: the shaded albedo, <c>dr.antialias</c>, then
    /// <c>alpha * albedo + (1 - alpha) * bg</c>. (The final <c>clamp(0, 1)</c> is the identity: every step is a convex
    /// combination of sigmoids and the background.)
    /// </summary>
    /// <returns>[H, W, 3].</returns>
    internal static double[] Render(MeshTextureView view, double[] rawTexture, int textureSize, double background)
    {
        if (view is null) throw new ArgumentNullException(nameof(view));
        if (rawTexture is null) throw new ArgumentNullException(nameof(rawTexture));
        var (levels, sizes) = BuildMipmaps(rawTexture, textureSize);
        var albedo = ShadeAlbedo(view, levels, sizes);
        var image = (double[])albedo.Clone();
        var aa = view.Antialias;
        for (int i = 0; i < aa.Weight.Length; i++)
        {
            int d = aa.Destination[i], s = aa.Source[i];
            for (int c = 0; c < 3; c++)
                image[(3 * d) + c] += aa.Weight[i] * (albedo[(3 * s) + c] - albedo[(3 * d) + c]);
        }

        for (int p = 0; p < view.Covered.Length; p++)
        {
            if (view.Covered[p]) continue;
            image[3 * p] = background;
            image[(3 * p) + 1] = background;
            image[(3 * p) + 2] = background;
        }

        return image;
    }

    /// <summary>
    /// Adds dL/d raw texture to <paramref name="rawGradient"/> for an image gradient of a
    /// <see cref="Render(MeshTextureView, double[], int, double)"/>: through
    /// the compositing, the antialiasing blends, the sigmoid, the trilinear taps and the mip pyramid's box filters.
    /// </summary>
    internal static void Backward(
        MeshTextureView view, double[] rawTexture, int textureSize, double[] imageGradient, double[] rawGradient)
    {
        if (view is null) throw new ArgumentNullException(nameof(view));
        if (imageGradient is null) throw new ArgumentNullException(nameof(imageGradient));
        if (rawGradient is null) throw new ArgumentNullException(nameof(rawGradient));
        var (levels, sizes) = BuildMipmaps(rawTexture, textureSize);
        var albedo = ShadeAlbedo(view, levels, sizes);
        int pixels = view.Width * view.Height;
        var antialiased = new double[3 * pixels];
        for (int p = 0; p < pixels; p++)
        {
            if (!view.Covered[p]) continue;
            for (int c = 0; c < 3; c++)
                antialiased[(3 * p) + c] = imageGradient[(3 * p) + c];
        }

        var albedoGradient = (double[])antialiased.Clone();
        var aa = view.Antialias;
        for (int i = 0; i < aa.Weight.Length; i++)
        {
            int d = aa.Destination[i], s = aa.Source[i];
            for (int c = 0; c < 3; c++)
            {
                double g = aa.Weight[i] * antialiased[(3 * d) + c];
                albedoGradient[(3 * s) + c] += g;
                albedoGradient[(3 * d) + c] -= g;
            }
        }

        var levelGradient = new double[levels.Length][];
        for (int l = 0; l < levels.Length; l++) levelGradient[l] = new double[levels[l].Length];
        var level = new int[8];
        var texel = new int[8];
        var weight = new double[8];
        for (int p = 0; p < pixels; p++)
        {
            bool any = false;
            for (int c = 0; c < 3 && !any; c++) any = albedoGradient[(3 * p) + c] != 0.0;
            if (!any) continue;
            double lod = MipLevel(view.UvDerivatives, p, sizes[0], sizes.Length - 1);
            TrilinearTaps(sizes, view.Uv[2 * p], view.Uv[(2 * p) + 1], lod, level, texel, weight);
            for (int c = 0; c < 3; c++)
            {
                double s = albedo[(3 * p) + c];
                double g = albedoGradient[(3 * p) + c] * s * (1.0 - s);
                if (g == 0.0) continue;
                for (int k = 0; k < 8; k++)
                {
                    if (weight[k] != 0.0)
                        levelGradient[level[k]][(3 * texel[k]) + c] += g * weight[k];
                }
            }
        }

        // The transpose of the 2 x 2 box filters, coarsest level first.
        for (int l = levels.Length - 1; l >= 1; l--)
        {
            int size = sizes[l], below = sizes[l - 1];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        double g = 0.25 * levelGradient[l][(3 * ((y * size) + x)) + c];
                        if (g == 0.0) continue;
                        levelGradient[l - 1][(3 * (((2 * y) * below) + (2 * x))) + c] += g;
                        levelGradient[l - 1][(3 * (((2 * y) * below) + (2 * x) + 1)) + c] += g;
                        levelGradient[l - 1][(3 * ((((2 * y) + 1) * below) + (2 * x))) + c] += g;
                        levelGradient[l - 1][(3 * ((((2 * y) + 1) * below) + (2 * x) + 1)) + c] += g;
                    }
                }
            }
        }

        for (int i = 0; i < rawGradient.Length; i++)
            rawGradient[i] += levelGradient[0][i];
    }

    /// <summary>
    /// A super-sampled <see cref="Render(MeshTextureView, double[], int, double)"/> resized back to the output size
    /// (<c>scale_img_hwc(albedo)</c>).
    /// </summary>
    /// <returns>[H, W, 3] at the output size.</returns>
    internal static double[] Render(SupersampledTextureView view, double[] rawTexture, int textureSize, double background)
    {
        if (view is null) throw new ArgumentNullException(nameof(view));
        var image = Render(view.View, rawTexture, textureSize, background);
        return view.Taps is null ? image : view.Taps.Apply(image, 3);
    }

    /// <summary>The gradient of the super-sampled <see cref="Render(SupersampledTextureView, double[], int, double)"/>.</summary>
    internal static void Backward(
        SupersampledTextureView view, double[] rawTexture, int textureSize, double[] imageGradient, double[] rawGradient)
    {
        if (view is null) throw new ArgumentNullException(nameof(view));
        var gradient = view.Taps is null ? imageGradient : view.Taps.Transpose(imageGradient, 3);
        Backward(view.View, rawTexture, textureSize, gradient, rawGradient);
    }

    /// <summary>Bilinear lookup of an [S, S, 3] texture at (u, v) with wrap-around addressing (nvdiffrast's convention:
    /// texel centres at <c>(i + 0.5) / S</c>).</summary>
    internal static void SampleBilinear(double[] texture, int size, double u, double v, double[] output)
    {
        var texel = new int[4];
        var weight = new double[4];
        BilinearTaps(size, u, v, texel, weight);
        for (int c = 0; c < 3; c++)
        {
            output[c] = (weight[0] * texture[(3 * texel[0]) + c]) + (weight[1] * texture[(3 * texel[1]) + c])
                + (weight[2] * texture[(3 * texel[2]) + c]) + (weight[3] * texture[(3 * texel[3]) + c]);
        }
    }

    private static void BilinearTaps(int size, double u, double v, int[] texel, double[] weight, int offset = 0)
    {
        double x = (u * size) - 0.5, y = (v * size) - 0.5;
        double fx0 = Math.Floor(x), fy0 = Math.Floor(y);
        double fx = x - fx0, fy = y - fy0;
        int x0 = Wrap((int)fx0, size), x1 = Wrap((int)fx0 + 1, size);
        int y0 = Wrap((int)fy0, size), y1 = Wrap((int)fy0 + 1, size);
        texel[offset] = (y0 * size) + x0;
        texel[offset + 1] = (y0 * size) + x1;
        texel[offset + 2] = (y1 * size) + x0;
        texel[offset + 3] = (y1 * size) + x1;
        weight[offset] = (1.0 - fx) * (1.0 - fy);
        weight[offset + 1] = fx * (1.0 - fy);
        weight[offset + 2] = (1.0 - fx) * fy;
        weight[offset + 3] = fx * fy;
    }

    private static int Wrap(int i, int size)
    {
        int r = i % size;
        return r < 0 ? r + size : r;
    }

    private static double Clamp01(double x) => Math.Min(1.0, Math.Max(0.0, x));
}
