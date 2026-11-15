namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>Settings for <see cref="UvAtlasUnwrapper"/>.</summary>
internal sealed class UvAtlasOptions
{
    /// <summary>The texture resolution the atlas is packed for; the padding is measured in its texels. Default: 1024
    /// (DreamGaussian's <c>texture_size</c>).</summary>
    public int TextureResolution { get; set; } = 1024;

    /// <summary>Empty texels kept around every chart, so bilinear lookups do not bleed between charts. Default: 2.</summary>
    public int PaddingTexels { get; set; } = 2;
}

/// <summary>A UV atlas: the mesh with its vertices split along chart seams and a texture coordinate per vertex.</summary>
internal sealed class UvAtlas
{
    internal UvAtlas(double[] vertices, int[] faces, double[] uvs, int[] sourceVertex, int[] faceChart, int chartCount)
    {
        Vertices = vertices;
        Faces = faces;
        Uvs = uvs;
        SourceVertex = sourceVertex;
        FaceChart = faceChart;
        ChartCount = chartCount;
    }

    /// <summary>Vertex positions [V', 3]; a vertex on a seam appears once per chart.</summary>
    public double[] Vertices { get; }

    /// <summary>Triangles [F, 3] over the new vertices, in the input's order and winding.</summary>
    public int[] Faces { get; }

    /// <summary>Texture coordinates [V', 2] in [0, 1]; u along the texture's width, v along its height.</summary>
    public double[] Uvs { get; }

    /// <summary>For each new vertex, the input vertex it copies.</summary>
    public int[] SourceVertex { get; }

    /// <summary>The chart of each triangle.</summary>
    public int[] FaceChart { get; }

    /// <summary>The number of charts.</summary>
    public int ChartCount { get; }
}

/// <summary>
/// Unwraps a triangle mesh into a UV atlas of non-overlapping charts packed into the unit square: the job xatlas does
/// for DreamGaussian's <c>mesh.auto_uv()</c>.
/// </summary>
/// <remarks>
/// <para>
/// Charts are grown over edge-connected triangles that share a dominant normal direction (one of the six signed
/// axes) and are parameterised by orthographic projection onto the plane of that axis, which keeps every triangle's
/// orientation (no flips) and bounds the area distortion by <c>sqrt(3)</c>. A triangle joins a chart only if its
/// projection does not overlap the chart's projected triangles (an exact separating-axis test), so a chart is always
/// injective even where the surface folds back over itself along the axis. The charts are then packed into [0, 1]^2
/// with shelf packing, the largest uniform scale found by bisection, and <see cref="UvAtlasOptions.PaddingTexels"/>
/// of gutter.
/// </para>
/// <para>
/// xatlas segments charts by a cost on normal deviation, roundness and straightness and parameterises them with least
/// squares conformal maps, which gives fewer seams and less stretch. This unwrapper produces a valid atlas for the
/// same purpose (back-projection and texture refinement) with more charts; it does not reproduce xatlas's chart
/// boundaries or texel layout.
/// </para>
/// </remarks>
internal static class UvAtlasUnwrapper
{
    // Projection plane axes per signed dominant axis, ordered so a triangle facing the axis stays counter-clockwise.
    private static readonly int[][] PlaneAxes =
    {
        new[] { 1, 2 }, // +x: (y, z)
        new[] { 2, 1 }, // -x: (z, y)
        new[] { 2, 0 }, // +y: (z, x)
        new[] { 0, 2 }, // -y: (x, z)
        new[] { 0, 1 }, // +z: (x, y)
        new[] { 1, 0 }, // -z: (y, x)
    };

    /// <summary>Unwraps the mesh ([V, 3] vertices and [F, 3] faces, flattened).</summary>
    public static UvAtlas Unwrap(double[] vertices, int[] faces, UvAtlasOptions? options = null)
    {
        if (vertices is null) throw new ArgumentNullException(nameof(vertices));
        if (faces is null) throw new ArgumentNullException(nameof(faces));
        if (vertices.Length % 3 != 0) throw new ArgumentException("Vertices must be [V, 3].", nameof(vertices));
        if (faces.Length % 3 != 0) throw new ArgumentException("Faces must be [F, 3].", nameof(faces));
        var settings = options ?? new UvAtlasOptions();
        if (settings.TextureResolution < 1) throw new ArgumentOutOfRangeException(nameof(options), "TextureResolution must be positive.");
        if (settings.PaddingTexels < 0) throw new ArgumentOutOfRangeException(nameof(options), "PaddingTexels cannot be negative.");

        int faceCount = faces.Length / 3;
        int vertexCount = vertices.Length / 3;
        for (int i = 0; i < faces.Length; i++)
        {
            if (faces[i] < 0 || faces[i] >= vertexCount)
                throw new ArgumentException($"Face index {faces[i]} is out of range.", nameof(faces));
        }

        var label = new int[faceCount];
        var degenerate = new bool[faceCount];
        double edgeSum = 0.0;
        for (int f = 0; f < faceCount; f++)
        {
            Normal(vertices, faces, f, out double nx, out double ny, out double nz);
            double ax = Math.Abs(nx), ay = Math.Abs(ny), az = Math.Abs(nz);
            degenerate[f] = ax + ay + az == 0.0;
            if (ax >= ay && ax >= az) label[f] = nx >= 0.0 ? 0 : 1;
            else if (ay >= az) label[f] = ny >= 0.0 ? 2 : 3;
            else label[f] = nz >= 0.0 ? 4 : 5;
            edgeSum += Distance(vertices, faces[3 * f], faces[(3 * f) + 1]);
        }

        double cellSize = faceCount > 0 ? Math.Max(1e-12, 2.0 * edgeSum / faceCount) : 1.0;
        var adjacency = BuildAdjacency(faces, faceCount);

        // Chart growing.
        var chartOf = new int[faceCount];
        for (int f = 0; f < faceCount; f++) chartOf[f] = -1;
        var chartFaces = new List<List<int>>();
        var queue = new Queue<int>();
        for (int seed = 0; seed < faceCount; seed++)
        {
            if (chartOf[seed] >= 0) continue;
            int chart = chartFaces.Count;
            var members = new List<int>();
            chartFaces.Add(members);
            var axes = PlaneAxes[label[seed]];
            var hash = new Dictionary<long, List<int>>();
            Accept(seed);
            queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                int f = queue.Dequeue();
                foreach (int g in adjacency[f])
                {
                    if (chartOf[g] >= 0 || label[g] != label[seed]) continue;
                    if (!degenerate[g] && Overlaps(g)) continue;
                    Accept(g);
                    queue.Enqueue(g);
                }
            }

            void Accept(int f)
            {
                chartOf[f] = chart;
                members.Add(f);
                if (degenerate[f]) return;
                Bounds(f, out int x0, out int y0, out int x1, out int y1);
                for (int x = x0; x <= x1; x++)
                {
                    for (int y = y0; y <= y1; y++)
                    {
                        long key = ((long)x << 32) ^ (uint)y;
                        if (!hash.TryGetValue(key, out var list))
                        {
                            list = new List<int>();
                            hash[key] = list;
                        }

                        list.Add(f);
                    }
                }
            }

            bool Overlaps(int f)
            {
                Bounds(f, out int x0, out int y0, out int x1, out int y1);
                var tested = new HashSet<int>();
                var a = Project(vertices, faces, f, axes);
                for (int x = x0; x <= x1; x++)
                {
                    for (int y = y0; y <= y1; y++)
                    {
                        long key = ((long)x << 32) ^ (uint)y;
                        if (!hash.TryGetValue(key, out var list)) continue;
                        foreach (int other in list)
                        {
                            if (!tested.Add(other)) continue;
                            if (TrianglesOverlap(a, Project(vertices, faces, other, axes))) return true;
                        }
                    }
                }

                return false;
            }

            void Bounds(int f, out int x0, out int y0, out int x1, out int y1)
            {
                var p = Project(vertices, faces, f, axes);
                x0 = (int)Math.Floor(Math.Min(p[0], Math.Min(p[2], p[4])) / cellSize);
                x1 = (int)Math.Floor(Math.Max(p[0], Math.Max(p[2], p[4])) / cellSize);
                y0 = (int)Math.Floor(Math.Min(p[1], Math.Min(p[3], p[5])) / cellSize);
                y1 = (int)Math.Floor(Math.Max(p[1], Math.Max(p[3], p[5])) / cellSize);
            }
        }

        int chartCount = chartFaces.Count;
        // Chart-local 2D coordinates and bounds.
        var minU = new double[chartCount];
        var minV = new double[chartCount];
        var widthOf = new double[chartCount];
        var heightOf = new double[chartCount];
        for (int c = 0; c < chartCount; c++)
        {
            var axes = PlaneAxes[label[chartFaces[c][0]]];
            double u0 = double.MaxValue, v0 = double.MaxValue, u1 = double.MinValue, v1 = double.MinValue;
            foreach (int f in chartFaces[c])
            {
                for (int k = 0; k < 3; k++)
                {
                    int vertex = faces[(3 * f) + k];
                    double u = vertices[(3 * vertex) + axes[0]], v = vertices[(3 * vertex) + axes[1]];
                    u0 = Math.Min(u0, u);
                    v0 = Math.Min(v0, v);
                    u1 = Math.Max(u1, u);
                    v1 = Math.Max(v1, v);
                }
            }

            minU[c] = u0;
            minV[c] = v0;
            widthOf[c] = u1 - u0;
            heightOf[c] = v1 - v0;
        }

        double padding = (double)settings.PaddingTexels / settings.TextureResolution;
        var (offsetU, offsetV, scale) = Pack(widthOf, heightOf, padding);

        // Split vertices along chart seams.
        var newIndex = new Dictionary<long, int>();
        var newVertices = new List<double>();
        var uvs = new List<double>();
        var source = new List<int>();
        var newFaces = new int[faces.Length];
        for (int f = 0; f < faceCount; f++)
        {
            int c = chartOf[f];
            var axes = PlaneAxes[label[chartFaces[c][0]]];
            for (int k = 0; k < 3; k++)
            {
                int vertex = faces[(3 * f) + k];
                long key = ((long)c << 32) | (uint)vertex;
                if (!newIndex.TryGetValue(key, out int index))
                {
                    index = source.Count;
                    newIndex[key] = index;
                    source.Add(vertex);
                    newVertices.Add(vertices[3 * vertex]);
                    newVertices.Add(vertices[(3 * vertex) + 1]);
                    newVertices.Add(vertices[(3 * vertex) + 2]);
                    double u = offsetU[c] + ((vertices[(3 * vertex) + axes[0]] - minU[c]) * scale);
                    double v = offsetV[c] + ((vertices[(3 * vertex) + axes[1]] - minV[c]) * scale);
                    uvs.Add(Math.Min(1.0, Math.Max(0.0, u)));
                    uvs.Add(Math.Min(1.0, Math.Max(0.0, v)));
                }

                newFaces[(3 * f) + k] = index;
            }
        }

        return new UvAtlas(newVertices.ToArray(), newFaces, uvs.ToArray(), source.ToArray(), chartOf, chartCount);
    }

    /// <summary>
    /// Shelf-packs the chart rectangles into the unit square at the largest uniform scale bisection finds, each with
    /// <paramref name="padding"/> on every side.
    /// </summary>
    private static (double[] OffsetU, double[] OffsetV, double Scale) Pack(double[] widths, double[] heights, double padding)
    {
        int count = widths.Length;
        var order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int byHeight = heights[b].CompareTo(heights[a]);
            if (byHeight != 0) return byHeight;
            int byWidth = widths[b].CompareTo(widths[a]);
            return byWidth != 0 ? byWidth : a.CompareTo(b);
        });

        double largest = 0.0;
        for (int i = 0; i < count; i++) largest = Math.Max(largest, Math.Max(widths[i], heights[i]));
        if (count == 0 || 2.0 * padding * Math.Sqrt(count) >= 1.0)
            throw new ArgumentException("The padding is too large for this many charts; raise TextureResolution.");

        var offsetU = new double[count];
        var offsetV = new double[count];
        double lo = 0.0, hi = largest > 0.0 ? (1.0 - (2.0 * padding)) / largest : 1.0;
        if (!TryPack(order, widths, heights, padding, lo, offsetU, offsetV))
            throw new ArgumentException("The padding is too large for this many charts; raise TextureResolution.");
        for (int iteration = 0; iteration < 48; iteration++)
        {
            double mid = 0.5 * (lo + hi);
            if (TryPack(order, widths, heights, padding, mid, null, null)) lo = mid;
            else hi = mid;
        }

        TryPack(order, widths, heights, padding, lo, offsetU, offsetV);
        return (offsetU, offsetV, lo);
    }

    private static bool TryPack(int[] order, double[] widths, double[] heights, double padding, double scale,
        double[]? offsetU, double[]? offsetV)
    {
        double x = 0.0, y = 0.0, shelf = 0.0;
        foreach (int c in order)
        {
            double w = (widths[c] * scale) + (2.0 * padding);
            double h = (heights[c] * scale) + (2.0 * padding);
            if (w > 1.0 || h > 1.0) return false;
            if (x + w > 1.0)
            {
                y += shelf;
                x = 0.0;
                shelf = 0.0;
            }

            if (y + h > 1.0) return false;
            if (offsetU is not null && offsetV is not null)
            {
                offsetU[c] = x + padding;
                offsetV[c] = y + padding;
            }

            x += w;
            shelf = Math.Max(shelf, h);
        }

        return true;
    }

    private static List<int>[] BuildAdjacency(int[] faces, int faceCount)
    {
        var edges = new Dictionary<long, List<int>>();
        for (int f = 0; f < faceCount; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = faces[(3 * f) + k], b = faces[(3 * f) + ((k + 1) % 3)];
                long key = ((long)Math.Min(a, b) << 32) | (uint)Math.Max(a, b);
                if (!edges.TryGetValue(key, out var list))
                {
                    list = new List<int>(2);
                    edges[key] = list;
                }

                list.Add(f);
            }
        }

        var adjacency = new List<int>[faceCount];
        for (int f = 0; f < faceCount; f++) adjacency[f] = new List<int>(3);
        foreach (var list in edges.Values)
        {
            for (int i = 0; i < list.Count; i++)
            {
                for (int j = 0; j < list.Count; j++)
                {
                    if (i != j) adjacency[list[i]].Add(list[j]);
                }
            }
        }

        return adjacency;
    }

    private static void Normal(double[] v, int[] faces, int f, out double nx, out double ny, out double nz)
    {
        int a = faces[3 * f], b = faces[(3 * f) + 1], c = faces[(3 * f) + 2];
        double e1x = v[3 * b] - v[3 * a], e1y = v[(3 * b) + 1] - v[(3 * a) + 1], e1z = v[(3 * b) + 2] - v[(3 * a) + 2];
        double e2x = v[3 * c] - v[3 * a], e2y = v[(3 * c) + 1] - v[(3 * a) + 1], e2z = v[(3 * c) + 2] - v[(3 * a) + 2];
        nx = (e1y * e2z) - (e1z * e2y);
        ny = (e1z * e2x) - (e1x * e2z);
        nz = (e1x * e2y) - (e1y * e2x);
    }

    private static double Distance(double[] v, int a, int b)
    {
        double dx = v[3 * a] - v[3 * b], dy = v[(3 * a) + 1] - v[(3 * b) + 1], dz = v[(3 * a) + 2] - v[(3 * b) + 2];
        return Math.Sqrt((dx * dx) + (dy * dy) + (dz * dz));
    }

    private static double[] Project(double[] v, int[] faces, int f, int[] axes)
    {
        var p = new double[6];
        for (int k = 0; k < 3; k++)
        {
            int vertex = faces[(3 * f) + k];
            p[2 * k] = v[(3 * vertex) + axes[0]];
            p[(2 * k) + 1] = v[(3 * vertex) + axes[1]];
        }

        return p;
    }

    /// <summary>
    /// Whether two 2D triangles' interiors intersect (separating-axis test over the six edge normals); triangles that
    /// only touch along an edge or at a vertex do not overlap.
    /// </summary>
    internal static bool TrianglesOverlap(double[] a, double[] b)
    {
        double scale = 0.0;
        for (int i = 0; i < 6; i++) scale = Math.Max(scale, Math.Max(Math.Abs(a[i]), Math.Abs(b[i])));
        double tolerance = 1e-9 * Math.Max(scale, 1e-12);
        return !Separated(a, b, tolerance) && !Separated(b, a, tolerance);
    }

    private static bool Separated(double[] a, double[] b, double tolerance)
    {
        for (int e = 0; e < 3; e++)
        {
            double x0 = a[2 * e], y0 = a[(2 * e) + 1];
            double x1 = a[2 * ((e + 1) % 3)], y1 = a[(2 * ((e + 1) % 3)) + 1];
            double nx = y1 - y0, ny = x0 - x1;
            double length = Math.Sqrt((nx * nx) + (ny * ny));
            if (length == 0.0) continue;
            nx /= length;
            ny /= length;
            double minA = double.MaxValue, maxA = double.MinValue, minB = double.MaxValue, maxB = double.MinValue;
            for (int k = 0; k < 3; k++)
            {
                double pa = (a[2 * k] * nx) + (a[(2 * k) + 1] * ny);
                double pb = (b[2 * k] * nx) + (b[(2 * k) + 1] * ny);
                minA = Math.Min(minA, pa);
                maxA = Math.Max(maxA, pa);
                minB = Math.Min(minB, pb);
                maxB = Math.Max(maxB, pb);
            }

            if (maxA <= minB + tolerance || maxB <= minA + tolerance)
                return true;
        }

        return false;
    }
}
