namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>Settings for <see cref="UvAtlasUnwrapper"/>.</summary>
internal sealed class UvAtlasOptions
{
    /// <summary>The texture resolution the atlas is packed for; the padding is measured in its texels. Default: 1024
    /// (DreamGaussian's <c>texture_size</c>).</summary>
    public int TextureResolution { get; set; } = 1024;

    /// <summary>Empty texels kept around every chart, so bilinear lookups do not bleed between charts. Default: 2.</summary>
    public int PaddingTexels { get; set; } = 2;

    /// <summary>
    /// The half-angle of the normal cone a chart grows within: a triangle joins a chart while its normal is within this
    /// angle of the chart's area-weighted normal. Default: 60 degrees.
    /// </summary>
    public double MaxNormalDeviationDegrees { get; set; } = 60.0;
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
/// Unwraps a triangle mesh into a UV atlas of non-overlapping charts packed into the unit square, the pipeline xatlas
/// runs for DreamGaussian's <c>mesh.auto_uv()</c>: chart segmentation, a least squares conformal map per chart, and
/// packing.
/// </summary>
/// <remarks>
/// <para>
/// <b>Segmentation.</b> Charts grow over edge-connected triangles from the largest unassigned triangle, best candidate
/// first, while a candidate's normal stays within <see cref="UvAtlasOptions.MaxNormalDeviationDegrees"/> of the chart's
/// area-weighted normal (a normal cone, the normal-deviation term of xatlas's chart cost).
/// </para>
/// <para>
/// <b>Parameterisation.</b> Each chart is flattened with LSCM (Lévy, Petitjean, Ray and Maillot, "Least Squares
/// Conformal Maps for Automatic Texture Atlas Generation", SIGGRAPH 2002): with every triangle in its own orthonormal
/// frame and <c>W_j = z_{j+2} - z_{j+1}</c>, the conformal energy <c>sum_T |sum_j W_j U_j|^2 / (2 A_T)</c> is
/// minimised over complex coordinates <c>U = u + i v</c> with the two most distant vertices pinned, solved by conjugate
/// gradients on the normal equations. The chart is then scaled so its UV area equals its surface area (uniform texel
/// density). A chart whose map flips or overlaps itself is split with a narrower cone and re-flattened, down to single
/// triangles, as xatlas splits charts whose parameterisation is invalid.
/// </para>
/// <para>
/// <b>Packing.</b> Each chart is rotated to the minimum-area bounding rectangle of its convex hull (xatlas's
/// <c>rotateChartsToAxis</c>), the rectangles are shelf-packed into [0, 1]^2 at the largest uniform scale bisection
/// finds, and every chart keeps <see cref="UvAtlasOptions.PaddingTexels"/> of gutter. xatlas packs rasterised chart
/// shapes rather than rectangles, so its layout is tighter; texel density and chart shapes are the same idea.
/// </para>
/// </remarks>
internal static class UvAtlasUnwrapper
{
    private const int MaxSplitDepth = 6;

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
        if (!(settings.MaxNormalDeviationDegrees > 0.0 && settings.MaxNormalDeviationDegrees < 90.0))
            throw new ArgumentOutOfRangeException(nameof(options), "MaxNormalDeviationDegrees must be in (0, 90).");

        int faceCount = faces.Length / 3;
        int vertexCount = vertices.Length / 3;
        for (int i = 0; i < faces.Length; i++)
        {
            if (faces[i] < 0 || faces[i] >= vertexCount)
                throw new ArgumentException($"Face index {faces[i]} is out of range.", nameof(faces));
        }

        var mesh = new MeshData(vertices, faces);
        var charts = new List<Chart>();
        var all = new int[faceCount];
        for (int f = 0; f < faceCount; f++) all[f] = f;
        double cosLimit = Math.Cos(settings.MaxNormalDeviationDegrees * Math.PI / 180.0);
        foreach (var group in Segment(mesh, all, cosLimit))
            Build(mesh, group, cosLimit, 0, charts);

        int chartCount = charts.Count;
        var chartOf = new int[faceCount];
        var widths = new double[chartCount];
        var heights = new double[chartCount];
        for (int c = 0; c < chartCount; c++)
        {
            foreach (int f in charts[c].Faces) chartOf[f] = c;
            AlignToAxis(charts[c].Uv);
            double u0 = double.MaxValue, v0 = double.MaxValue, u1 = double.MinValue, v1 = double.MinValue;
            var uv = charts[c].Uv;
            for (int i = 0; i < uv.Length; i += 2)
            {
                u0 = Math.Min(u0, uv[i]);
                v0 = Math.Min(v0, uv[i + 1]);
                u1 = Math.Max(u1, uv[i]);
                v1 = Math.Max(v1, uv[i + 1]);
            }

            for (int i = 0; i < uv.Length; i += 2)
            {
                uv[i] -= u0;
                uv[i + 1] -= v0;
            }

            widths[c] = u1 - u0;
            heights[c] = v1 - v0;
        }

        double padding = (double)settings.PaddingTexels / settings.TextureResolution;
        var (offsetU, offsetV, scale) = Pack(widths, heights, padding);

        // Split vertices along chart seams.
        var newIndex = new Dictionary<long, int>();
        var newVertices = new List<double>();
        var uvs = new List<double>();
        var source = new List<int>();
        var newFaces = new int[faces.Length];
        for (int f = 0; f < faceCount; f++)
        {
            int c = chartOf[f];
            var chart = charts[c];
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
                    int local = chart.Local[vertex];
                    double u = offsetU[c] + (chart.Uv[2 * local] * scale);
                    double v = offsetV[c] + (chart.Uv[(2 * local) + 1] * scale);
                    uvs.Add(Math.Min(1.0, Math.Max(0.0, u)));
                    uvs.Add(Math.Min(1.0, Math.Max(0.0, v)));
                }

                newFaces[(3 * f) + k] = index;
            }
        }

        return new UvAtlas(newVertices.ToArray(), newFaces, uvs.ToArray(), source.ToArray(), chartOf, chartCount);
    }

    /// <summary>Flattens a group; a group whose map is not injective is split with a narrower cone.</summary>
    private static void Build(MeshData mesh, List<int> group, double cosLimit, int depth, List<Chart> charts)
    {
        var chart = Parameterize(mesh, group);
        if (chart is not null)
        {
            charts.Add(chart);
            return;
        }

        if (group.Count == 1 || depth >= MaxSplitDepth)
        {
            foreach (int f in group)
            {
                var single = Parameterize(mesh, new List<int> { f });
                if (single is null) throw new InvalidOperationException("A single triangle could not be flattened.");
                charts.Add(single);
            }

            return;
        }

        double narrower = Math.Cos(0.5 * Math.Acos(Math.Min(1.0, Math.Max(-1.0, cosLimit))));
        foreach (var part in Segment(mesh, group, narrower))
            Build(mesh, part, narrower, depth + 1, charts);
    }

    /// <summary>Normal-cone chart growing over the faces of <paramref name="subset"/>.</summary>
    internal static List<List<int>> Segment(MeshData mesh, IReadOnlyList<int> subset, double cosLimit)
    {
        int faceCount = mesh.FaceCount;
        var inSubset = new bool[faceCount];
        foreach (int f in subset) inSubset[f] = true;
        var taken = new bool[faceCount];
        var order = subset.OrderByDescending(f => mesh.Areas[f]).ThenBy(f => f).ToArray();
        var groups = new List<List<int>>();
        var normals = mesh.Normals;
        foreach (int seed in order)
        {
            if (taken[seed]) continue;
            var members = new List<int>();
            double sx = 0.0, sy = 0.0, sz = 0.0;
            var frontier = new SortedSet<(double Cost, int Face)>();
            var queued = new HashSet<int>();
            Take(seed);
            while (frontier.Count > 0)
            {
                var next = frontier.Min;
                frontier.Remove(next);
                queued.Remove(next.Face);
                if (taken[next.Face]) continue;
                // Re-evaluated against the chart's current normal; a rejected face can be queued again by a later member.
                if (Deviation(next.Face) > 1.0 - cosLimit) continue;
                Take(next.Face);
            }

            groups.Add(members);

            void Take(int f)
            {
                taken[f] = true;
                members.Add(f);
                sx += normals[3 * f] * mesh.Areas[f];
                sy += normals[(3 * f) + 1] * mesh.Areas[f];
                sz += normals[(3 * f) + 2] * mesh.Areas[f];
                foreach (int g in mesh.Adjacency[f])
                {
                    if (inSubset[g] && !taken[g] && queued.Add(g))
                        frontier.Add((Deviation(g), g));
                }
            }

            double Deviation(int f)
            {
                double length = Math.Sqrt((sx * sx) + (sy * sy) + (sz * sz));
                if (mesh.Areas[f] == 0.0 || length == 0.0) return 0.0;
                return 1.0 - (((normals[3 * f] * sx) + (normals[(3 * f) + 1] * sy) + (normals[(3 * f) + 2] * sz)) / length);
            }
        }

        return groups;
    }

    /// <summary>
    /// LSCM of one chart, scaled to its surface area; null when the map flips a triangle or overlaps itself.
    /// </summary>
    internal static Chart? Parameterize(MeshData mesh, List<int> group)
    {
        var faces = mesh.Faces;
        var positions = mesh.Vertices;
        var local = new Dictionary<int, int>();
        var globalOf = new List<int>();
        foreach (int f in group)
        {
            for (int k = 0; k < 3; k++)
            {
                int vertex = faces[(3 * f) + k];
                if (!local.ContainsKey(vertex))
                {
                    local[vertex] = globalOf.Count;
                    globalOf.Add(vertex);
                }
            }
        }

        int n = globalOf.Count;
        double nx = 0.0, ny = 0.0, nz = 0.0, area3 = 0.0;
        foreach (int f in group)
        {
            nx += mesh.Normals[3 * f] * mesh.Areas[f];
            ny += mesh.Normals[(3 * f) + 1] * mesh.Areas[f];
            nz += mesh.Normals[(3 * f) + 2] * mesh.Areas[f];
            area3 += mesh.Areas[f];
        }

        // An initial guess: orthographic projection onto the chart plane.
        double nl = Math.Sqrt((nx * nx) + (ny * ny) + (nz * nz));
        if (nl == 0.0)
        {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        }
        else
        {
            nx /= nl;
            ny /= nl;
            nz /= nl;
        }

        double ax, ay, az;
        if (Math.Abs(nx) <= Math.Abs(ny) && Math.Abs(nx) <= Math.Abs(nz)) { ax = 0.0; ay = -nz; az = ny; }
        else if (Math.Abs(ny) <= Math.Abs(nz)) { ax = -nz; ay = 0.0; az = nx; }
        else { ax = -ny; ay = nx; az = 0.0; }
        double al = Math.Sqrt((ax * ax) + (ay * ay) + (az * az));
        ax /= al;
        ay /= al;
        az /= al;
        double bx = (ny * az) - (nz * ay), by = (nz * ax) - (nx * az), bz = (nx * ay) - (ny * ax);
        var x = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            int g = globalOf[i];
            double px = positions[3 * g], py = positions[(3 * g) + 1], pz = positions[(3 * g) + 2];
            x[i] = (px * ax) + (py * ay) + (pz * az);
            x[n + i] = (px * bx) + (py * by) + (pz * bz);
        }

        // Pins: the two most distant vertices (two sweeps), at their 3D distance on the u axis.
        int pinA = Farthest(positions, globalOf, 0), pinB = Farthest(positions, globalOf, pinA);
        double pinDistance = Distance(positions, globalOf[pinA], globalOf[pinB]);
        if (n >= 3 && pinDistance > 0.0)
        {
            // Similarity taking the projected pins onto (0, 0) and (d, 0), applied to the guess.
            double dx = x[pinB] - x[pinA], dy = x[n + pinB] - x[n + pinA];
            double d2 = (dx * dx) + (dy * dy);
            double sr = 1.0, si = 0.0;
            if (d2 > 0.0)
            {
                sr = pinDistance * dx / d2;
                si = -pinDistance * dy / d2;
            }

            double ox = x[pinA], oy = x[n + pinA];
            for (int i = 0; i < n; i++)
            {
                double rx = x[i] - ox, ry = x[n + i] - oy;
                x[i] = (rx * sr) - (ry * si);
                x[n + i] = (rx * si) + (ry * sr);
            }

            x[pinA] = 0.0;
            x[n + pinA] = 0.0;
            x[pinB] = pinDistance;
            x[n + pinB] = 0.0;
            SolveLscm(mesh, group, local, n, x, pinA, pinB);
        }

        var uv = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            uv[2 * i] = x[i];
            uv[(2 * i) + 1] = x[n + i];
        }

        // Validity: no flipped (or collapsed) triangle and no two triangles overlapping.
        double area2 = 0.0;
        var triangles = new double[group.Count][];
        double edgeSum = 0.0;
        for (int t = 0; t < group.Count; t++)
        {
            int f = group[t];
            var tri = new double[6];
            for (int k = 0; k < 3; k++)
            {
                int l = local[faces[(3 * f) + k]];
                tri[2 * k] = uv[2 * l];
                tri[(2 * k) + 1] = uv[(2 * l) + 1];
            }

            triangles[t] = tri;
            double signed = 0.5 * (((tri[2] - tri[0]) * (tri[5] - tri[1])) - ((tri[4] - tri[0]) * (tri[3] - tri[1])));
            if (mesh.Areas[f] > 0.0 && !(signed > 1e-12 * mesh.Areas[f])) return null;
            area2 += Math.Max(0.0, signed);
            edgeSum += Math.Sqrt(((tri[2] - tri[0]) * (tri[2] - tri[0])) + ((tri[3] - tri[1]) * (tri[3] - tri[1])));
        }

        if (group.Count > 1 && SelfOverlaps(triangles, edgeSum / group.Count)) return null;

        if (area2 > 0.0 && area3 > 0.0)
        {
            double s = Math.Sqrt(area3 / area2);
            for (int i = 0; i < uv.Length; i++) uv[i] *= s;
        }

        return new Chart(group, local, uv);
    }

    /// <summary>
    /// Conjugate gradients on the normal equations (CGLS) of the LSCM system with the two pins fixed; <paramref name="x"/>
    /// holds [u; v] and is the starting point.
    /// </summary>
    private static void SolveLscm(MeshData mesh, List<int> group, Dictionary<int, int> local, int n, double[] x, int pinA, int pinB)
    {
        var faces = mesh.Faces;
        var p3 = mesh.Vertices;
        int rows = group.Count;
        var index = new int[3 * rows];
        var wr = new double[3 * rows];
        var wi = new double[3 * rows];
        for (int t = 0; t < rows; t++)
        {
            int f = group[t];
            int a = faces[3 * f], b = faces[(3 * f) + 1], c = faces[(3 * f) + 2];
            for (int k = 0; k < 3; k++) index[(3 * t) + k] = local[faces[(3 * f) + k]];
            double area = mesh.Areas[f];
            if (area <= 0.0) continue;
            // The triangle's own frame: z0 = 0, z1 = (|e1|, 0), z2 = (e2 . X, e2 . Y), counter-clockwise about its normal.
            double e1x = p3[3 * b] - p3[3 * a], e1y = p3[(3 * b) + 1] - p3[(3 * a) + 1], e1z = p3[(3 * b) + 2] - p3[(3 * a) + 2];
            double e2x = p3[3 * c] - p3[3 * a], e2y = p3[(3 * c) + 1] - p3[(3 * a) + 1], e2z = p3[(3 * c) + 2] - p3[(3 * a) + 2];
            double l1 = Math.Sqrt((e1x * e1x) + (e1y * e1y) + (e1z * e1z));
            double x2 = ((e2x * e1x) + (e2y * e1y) + (e2z * e1z)) / l1;
            double y2 = 2.0 * area / l1;
            double weight = 1.0 / Math.Sqrt(2.0 * area);
            // W_j = z_{j+2} - z_{j+1}.
            double[] zr = { 0.0, l1, x2 }, zi = { 0.0, 0.0, y2 };
            for (int j = 0; j < 3; j++)
            {
                int next = (j + 1) % 3, after = (j + 2) % 3;
                wr[(3 * t) + j] = (zr[after] - zr[next]) * weight;
                wi[(3 * t) + j] = (zi[after] - zi[next]) * weight;
            }
        }

        var free = new bool[n];
        for (int i = 0; i < n; i++) free[i] = i != pinA && i != pinB;

        void Apply(double[] vector, double[] re, double[] im)
        {
            for (int t = 0; t < rows; t++)
            {
                double sr = 0.0, si = 0.0;
                for (int j = 0; j < 3; j++)
                {
                    int i = index[(3 * t) + j];
                    double u = vector[i], v = vector[n + i];
                    sr += (wr[(3 * t) + j] * u) - (wi[(3 * t) + j] * v);
                    si += (wi[(3 * t) + j] * u) + (wr[(3 * t) + j] * v);
                }

                re[t] = sr;
                im[t] = si;
            }
        }

        void ApplyTranspose(double[] re, double[] im, double[] result)
        {
            Array.Clear(result, 0, result.Length);
            for (int t = 0; t < rows; t++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int i = index[(3 * t) + j];
                    if (!free[i]) continue;
                    result[i] += (wr[(3 * t) + j] * re[t]) + (wi[(3 * t) + j] * im[t]);
                    result[n + i] += (-wi[(3 * t) + j] * re[t]) + (wr[(3 * t) + j] * im[t]);
                }
            }
        }

        var rr = new double[rows];
        var ri = new double[rows];
        Apply(x, rr, ri);
        for (int t = 0; t < rows; t++)
        {
            rr[t] = -rr[t];
            ri[t] = -ri[t];
        }

        var s = new double[2 * n];
        ApplyTranspose(rr, ri, s);
        var p = (double[])s.Clone();
        double gamma = Dot(s, s), gamma0 = gamma;
        if (gamma0 == 0.0) return;
        var qr = new double[rows];
        var qi = new double[rows];
        int maxIterations = (4 * (2 * n)) + 100;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            Apply(p, qr, qi);
            double qq = Dot(qr, qr) + Dot(qi, qi);
            if (qq <= 0.0) break;
            double alpha = gamma / qq;
            for (int i = 0; i < x.Length; i++) x[i] += alpha * p[i];
            for (int t = 0; t < rows; t++)
            {
                rr[t] -= alpha * qr[t];
                ri[t] -= alpha * qi[t];
            }

            ApplyTranspose(rr, ri, s);
            double next = Dot(s, s);
            if (next <= 1e-24 * gamma0) break;
            double beta = next / gamma;
            for (int i = 0; i < p.Length; i++) p[i] = s[i] + (beta * p[i]);
            gamma = next;
        }
    }

    private static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static int Farthest(double[] positions, List<int> globalOf, int from)
    {
        int best = from;
        double bestDistance = -1.0;
        for (int i = 0; i < globalOf.Count; i++)
        {
            double d = Distance(positions, globalOf[from], globalOf[i]);
            if (d > bestDistance)
            {
                bestDistance = d;
                best = i;
            }
        }

        return best;
    }

    /// <summary>Whether any two of the triangles' interiors intersect, with a uniform grid over their bounds.</summary>
    private static bool SelfOverlaps(double[][] triangles, double meanEdge)
    {
        double cell = Math.Max(1e-12, 2.0 * meanEdge);
        var grid = new Dictionary<long, List<int>>();
        for (int t = 0; t < triangles.Length; t++)
        {
            var tri = triangles[t];
            int x0 = (int)Math.Floor(Math.Min(tri[0], Math.Min(tri[2], tri[4])) / cell);
            int x1 = (int)Math.Floor(Math.Max(tri[0], Math.Max(tri[2], tri[4])) / cell);
            int y0 = (int)Math.Floor(Math.Min(tri[1], Math.Min(tri[3], tri[5])) / cell);
            int y1 = (int)Math.Floor(Math.Max(tri[1], Math.Max(tri[3], tri[5])) / cell);
            var tested = new HashSet<int>();
            for (int x = x0; x <= x1; x++)
            {
                for (int y = y0; y <= y1; y++)
                {
                    long key = ((long)x << 32) ^ (uint)y;
                    if (!grid.TryGetValue(key, out var list))
                    {
                        list = new List<int>();
                        grid[key] = list;
                    }

                    foreach (int other in list)
                    {
                        if (tested.Add(other) && TrianglesOverlap(tri, triangles[other])) return true;
                    }

                    list.Add(t);
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Rotates a chart to the minimum-area bounding rectangle of its convex hull (one hull edge on an axis), the
    /// longer side along u.
    /// </summary>
    internal static void AlignToAxis(double[] uv)
    {
        var hull = ConvexHull(uv);
        if (hull.Count < 2) return;
        double bestArea = double.MaxValue, bestCos = 1.0, bestSin = 0.0, bestWidth = 0.0, bestHeight = 0.0;
        for (int e = 0; e < hull.Count; e++)
        {
            int a = hull[e], b = hull[(e + 1) % hull.Count];
            double dx = uv[2 * b] - uv[2 * a], dy = uv[(2 * b) + 1] - uv[(2 * a) + 1];
            double length = Math.Sqrt((dx * dx) + (dy * dy));
            if (length == 0.0) continue;
            double c = dx / length, s = dy / length;
            double u0 = double.MaxValue, u1 = double.MinValue, v0 = double.MaxValue, v1 = double.MinValue;
            foreach (int h in hull)
            {
                double u = (uv[2 * h] * c) + (uv[(2 * h) + 1] * s);
                double v = (-uv[2 * h] * s) + (uv[(2 * h) + 1] * c);
                u0 = Math.Min(u0, u);
                u1 = Math.Max(u1, u);
                v0 = Math.Min(v0, v);
                v1 = Math.Max(v1, v);
            }

            double area = (u1 - u0) * (v1 - v0);
            if (area < bestArea - 1e-15)
            {
                bestArea = area;
                bestCos = c;
                bestSin = s;
                bestWidth = u1 - u0;
                bestHeight = v1 - v0;
            }
        }

        // Rotating by -theta puts the edge on u; a further quarter turn keeps the longer side along u. Both are
        // rotations, so the chart's orientation is kept.
        if (bestHeight > bestWidth)
        {
            double c = bestCos, s = bestSin;
            bestCos = -s;
            bestSin = c;
        }

        for (int i = 0; i < uv.Length; i += 2)
        {
            double u = uv[i], v = uv[i + 1];
            uv[i] = (u * bestCos) + (v * bestSin);
            uv[i + 1] = (-u * bestSin) + (v * bestCos);
        }
    }

    /// <summary>Andrew's monotone chain; the hull's point indices, counter-clockwise.</summary>
    private static List<int> ConvexHull(double[] uv)
    {
        int count = uv.Length / 2;
        var order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int byU = uv[2 * a].CompareTo(uv[2 * b]);
            return byU != 0 ? byU : uv[(2 * a) + 1].CompareTo(uv[(2 * b) + 1]);
        });

        double Cross(int o, int a, int b)
            => ((uv[2 * a] - uv[2 * o]) * (uv[(2 * b) + 1] - uv[(2 * o) + 1]))
               - ((uv[(2 * a) + 1] - uv[(2 * o) + 1]) * (uv[2 * b] - uv[2 * o]));

        var hull = new List<int>();
        foreach (int p in order)
        {
            while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0.0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }

        int lower = hull.Count + 1;
        for (int i = count - 2; i >= 0; i--)
        {
            int p = order[i];
            while (hull.Count >= lower && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0.0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }

        if (hull.Count > 1) hull.RemoveAt(hull.Count - 1);
        return hull;
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

    private static double Distance(double[] v, int a, int b)
    {
        double dx = v[3 * a] - v[3 * b], dy = v[(3 * a) + 1] - v[(3 * b) + 1], dz = v[(3 * a) + 2] - v[(3 * b) + 2];
        return Math.Sqrt((dx * dx) + (dy * dy) + (dz * dz));
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

    /// <summary>The mesh with its per-face unit normals, areas and edge adjacency.</summary>
    internal sealed class MeshData
    {
        public MeshData(double[] vertices, int[] faces)
        {
            Vertices = vertices;
            Faces = faces;
            FaceCount = faces.Length / 3;
            Normals = new double[3 * FaceCount];
            Areas = new double[FaceCount];
            for (int f = 0; f < FaceCount; f++)
            {
                int a = faces[3 * f], b = faces[(3 * f) + 1], c = faces[(3 * f) + 2];
                double e1x = vertices[3 * b] - vertices[3 * a], e1y = vertices[(3 * b) + 1] - vertices[(3 * a) + 1];
                double e1z = vertices[(3 * b) + 2] - vertices[(3 * a) + 2];
                double e2x = vertices[3 * c] - vertices[3 * a], e2y = vertices[(3 * c) + 1] - vertices[(3 * a) + 1];
                double e2z = vertices[(3 * c) + 2] - vertices[(3 * a) + 2];
                double nx = (e1y * e2z) - (e1z * e2y), ny = (e1z * e2x) - (e1x * e2z), nz = (e1x * e2y) - (e1y * e2x);
                double length = Math.Sqrt((nx * nx) + (ny * ny) + (nz * nz));
                Areas[f] = 0.5 * length;
                if (length > 0.0)
                {
                    Normals[3 * f] = nx / length;
                    Normals[(3 * f) + 1] = ny / length;
                    Normals[(3 * f) + 2] = nz / length;
                }
            }

            Adjacency = BuildAdjacency(faces, FaceCount);
        }

        public double[] Vertices { get; }
        public int[] Faces { get; }
        public int FaceCount { get; }
        public double[] Normals { get; }
        public double[] Areas { get; }
        public List<int>[] Adjacency { get; }

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
                        if (i != j && list[i] != list[j]) adjacency[list[i]].Add(list[j]);
                    }
                }
            }

            return adjacency;
        }
    }

    /// <summary>A flattened chart: its faces, the local index of each input vertex and the UVs [n, 2] per local vertex.</summary>
    internal sealed class Chart
    {
        public Chart(List<int> faces, Dictionary<int, int> local, double[] uv)
        {
            Faces = faces;
            Local = local;
            Uv = uv;
        }

        public List<int> Faces { get; }
        public Dictionary<int, int> Local { get; }
        public double[] Uv { get; }
    }
}
