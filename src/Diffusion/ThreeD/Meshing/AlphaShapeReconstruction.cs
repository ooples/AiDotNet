namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// The regularised alpha shape of a point set (Edelsbrunner &amp; Muecke, "Three-Dimensional Alpha Shapes", ACM TOG 1994):
/// the boundary of the union of the Delaunay tetrahedra whose circumscribed sphere has a radius below alpha.
/// </summary>
/// <remarks>
/// <para>
/// The Delaunay tetrahedralisation is built incrementally (Bowyer 1981; Watson 1981): each point, inserted in Morton
/// order so that the point-location walk stays short, removes the tetrahedra whose circumsphere contains it (a
/// connected cavity found by breadth-first search from the tetrahedron that contains the point) and is joined to the
/// cavity's boundary faces. When round-off leaves the cavity not star-shaped from the point, the offending neighbour
/// is added to the cavity, so every new tetrahedron is positively oriented. A large enclosing tetrahedron starts the
/// construction and is discarded at the end.
/// </para>
/// <para>
/// The alpha complex here is regularised (its dangling faces and edges are dropped, as in CGAL's REGULARIZED mode and
/// Open3D's alpha shape): the triangles are the faces of kept tetrahedra that are not shared with another kept
/// tetrahedron, each wound counter-clockwise seen from outside its tetrahedron.
/// </para>
/// </remarks>
internal static class AlphaShapeReconstruction
{
    private static readonly int[,] FaceOpposite =
    {
        { 1, 2, 3 }, { 0, 3, 2 }, { 0, 1, 3 }, { 0, 2, 1 }
    };

    internal static MeshSurface Reconstruct(double[] points, double alpha)
    {
        if (!(alpha > 0.0))
            throw new ArgumentOutOfRangeException(nameof(alpha), "The alpha radius must be positive.");

        var delaunay = Delaunay.Build(points);
        int count = points.Length / 3;
        double alphaSquared = double.IsPositiveInfinity(alpha) ? double.PositiveInfinity : alpha * alpha;
        int tetCount = delaunay.TetCount;
        var kept = new bool[tetCount];
        for (int t = 0; t < tetCount; t++)
            kept[t] = delaunay.IsAlive(t) && !delaunay.TouchesSuper(t) && delaunay.RadiusSquared(t) < alphaSquared;

        var triangles = new List<int>();
        for (int t = 0; t < tetCount; t++)
        {
            if (!kept[t])
                continue;
            for (int f = 0; f < 4; f++)
            {
                int neighbor = delaunay.Neighbor(t, f);
                if (neighbor >= 0 && kept[neighbor])
                    continue;
                for (int k = 0; k < 3; k++)
                    triangles.Add(delaunay.Vertex(t, FaceOpposite[f, k]));
            }
        }

        // Keep only the samples the triangles use.
        var map = new int[count];
        for (int i = 0; i < count; i++)
            map[i] = -1;
        var coordinates = new List<double>();
        for (int i = 0; i < triangles.Count; i++)
        {
            int v = triangles[i];
            if (map[v] < 0)
            {
                map[v] = coordinates.Count / 3;
                coordinates.Add(points[3 * v]);
                coordinates.Add(points[(3 * v) + 1]);
                coordinates.Add(points[(3 * v) + 2]);
            }

            triangles[i] = map[v];
        }

        var surface = new MeshSurface(coordinates.ToArray(), triangles.ToArray(), null);
        surface.RepairDegenerateNormals();
        return surface;
    }

    /// <summary>
    /// An incremental Bowyer-Watson Delaunay tetrahedralisation with neighbour links.
    /// </summary>
    internal sealed class Delaunay
    {
        private readonly double[] _points;
        private readonly int _inputCount;
        private readonly List<int> _vertices = new List<int>();
        private readonly List<int> _neighbors = new List<int>();
        private readonly List<double> _spheres = new List<double>();
        private readonly List<bool> _alive = new List<bool>();
        private int _lastTet;

        private Delaunay(double[] points, int inputCount)
        {
            _points = points;
            _inputCount = inputCount;
        }

        internal int TetCount => _alive.Count;

        internal bool IsAlive(int t) => _alive[t];

        internal int Vertex(int t, int corner) => _vertices[(4 * t) + corner];

        internal int Neighbor(int t, int face) => _neighbors[(4 * t) + face];

        internal double RadiusSquared(int t) => _spheres[(4 * t) + 3];

        internal bool TouchesSuper(int t)
        {
            for (int c = 0; c < 4; c++)
            {
                if (Vertex(t, c) >= _inputCount)
                    return true;
            }

            return false;
        }

        internal static Delaunay Build(double[] input)
        {
            int count = input.Length / 3;
            if (count < 4)
                throw new ArgumentException("A tetrahedralisation needs at least four points.", nameof(input));

            var (center, size) = ScreenedPoissonReconstruction.BoundingCube(input, 1.0);
            // The four corners of a tetrahedron well outside the bounding box.
            double s = 20.0 * size;
            var points = new double[input.Length + 12];
            Array.Copy(input, points, input.Length);
            double[][] super =
            {
                new[] { center[0] + s, center[1] + s, center[2] + s },
                new[] { center[0] - s, center[1] - s, center[2] + s },
                new[] { center[0] - s, center[1] + s, center[2] - s },
                new[] { center[0] + s, center[1] - s, center[2] - s }
            };
            for (int k = 0; k < 4; k++)
                Array.Copy(super[k], 0, points, input.Length + (3 * k), 3);

            var delaunay = new Delaunay(points, count);
            int a = count, b = count + 1, c = count + 2, d = count + 3;
            if (delaunay.Orient(a, b, c, d) < 0.0)
            {
                int swap = c;
                c = d;
                d = swap;
            }

            delaunay.AddTet(a, b, c, d);
            double duplicateSquared = 1e-20 * size * size;
            foreach (int p in MortonOrder(input, center, size))
                delaunay.Insert(p, duplicateSquared);
            return delaunay;
        }

        private static int[] MortonOrder(double[] input, double[] center, double size)
        {
            int count = input.Length / 3;
            var keys = new long[count];
            var order = new int[count];
            for (int i = 0; i < count; i++)
            {
                long key = 0;
                var q = new long[3];
                for (int axis = 0; axis < 3; axis++)
                {
                    double unit = ((input[(3 * i) + axis] - center[axis]) / size) + 0.5;
                    q[axis] = Math.Max(0L, Math.Min(1023L, (long)(unit * 1023.0)));
                }

                for (int bit = 9; bit >= 0; bit--)
                {
                    for (int axis = 0; axis < 3; axis++)
                        key = (key << 1) | ((q[axis] >> bit) & 1L);
                }

                keys[i] = key;
                order[i] = i;
            }

            Array.Sort(keys, order);
            return order;
        }

        private void Insert(int p, double duplicateSquared)
        {
            int start = Locate(p);
            for (int c = 0; c < 4; c++)
            {
                if (DistanceSquared(Vertex(start, c), p) <= duplicateSquared)
                    return; // A duplicate of an existing vertex adds nothing.
            }

            // Cavity: the connected set of tetrahedra whose circumsphere contains p.
            var bad = new HashSet<int> { start };
            var stack = new Stack<int>();
            stack.Push(start);
            while (stack.Count > 0)
            {
                int t = stack.Pop();
                for (int f = 0; f < 4; f++)
                {
                    int n = Neighbor(t, f);
                    if (n >= 0 && !bad.Contains(n) && InSphere(n, p))
                    {
                        bad.Add(n);
                        stack.Push(n);
                    }
                }
            }

            // Grow the cavity until every boundary face sees p from its inner side (star-shaped).
            List<(int Tet, int Face)> boundary;
            while (true)
            {
                boundary = new List<(int, int)>();
                int offending = -1;
                foreach (int t in bad)
                {
                    for (int f = 0; f < 4; f++)
                    {
                        int n = Neighbor(t, f);
                        if (n >= 0 && bad.Contains(n))
                            continue;
                        boundary.Add((t, f));
                        if (offending < 0 && OrientReplaced(t, f, p) <= 0.0 && n >= 0)
                            offending = n;
                    }
                }

                if (offending < 0)
                    break;
                bad.Add(offending);
            }

            var newTets = new List<int>(boundary.Count);
            var edgeToTet = new Dictionary<long, (int Tet, int Face)>();
            long stride = _points.Length / 3;
            foreach (var (t, f) in boundary)
            {
                int outer = Neighbor(t, f);
                var v = new int[4];
                for (int c = 0; c < 4; c++)
                    v[c] = c == f ? p : Vertex(t, c);
                int created = AddTet(v[0], v[1], v[2], v[3]);
                newTets.Add(created);
                SetNeighbor(created, f, outer);
                if (outer >= 0)
                {
                    for (int g = 0; g < 4; g++)
                    {
                        if (Neighbor(outer, g) == t)
                            SetNeighbor(outer, g, created);
                    }
                }

                // The faces through p: opposite each corner c != f, made of p and the other two old corners.
                for (int c = 0; c < 4; c++)
                {
                    if (c == f)
                        continue;
                    int u = -1, w = -1;
                    for (int e = 0; e < 4; e++)
                    {
                        if (e == f || e == c)
                            continue;
                        if (u < 0) u = v[e]; else w = v[e];
                    }

                    long key = Math.Min(u, w) * stride + Math.Max(u, w);
                    if (edgeToTet.TryGetValue(key, out var other))
                    {
                        SetNeighbor(created, c, other.Tet);
                        SetNeighbor(other.Tet, other.Face, created);
                        edgeToTet.Remove(key);
                    }
                    else
                    {
                        edgeToTet[key] = (created, c);
                    }
                }
            }

            foreach (int t in bad)
                _alive[t] = false;
            _lastTet = newTets[0];
        }

        /// <summary>
        /// A tetrahedron containing p, by a visibility walk from the last created one; falls back to any live
        /// tetrahedron whose circumsphere contains p when the walk does not settle.
        /// </summary>
        private int Locate(int p)
        {
            int t = _lastTet;
            int limit = 4 * TetCount;
            for (int step = 0; step < limit; step++)
            {
                int next = -1;
                for (int f = 0; f < 4; f++)
                {
                    if (OrientReplaced(t, f, p) < 0.0)
                    {
                        next = Neighbor(t, f);
                        break;
                    }
                }

                if (next < 0)
                    return t;
                t = next;
            }

            for (int candidate = 0; candidate < TetCount; candidate++)
            {
                if (_alive[candidate] && InSphere(candidate, p))
                    return candidate;
            }

            throw new InvalidOperationException("Point location failed in the Delaunay tetrahedralisation.");
        }

        private int AddTet(int a, int b, int c, int d)
        {
            int t = _alive.Count;
            _vertices.Add(a);
            _vertices.Add(b);
            _vertices.Add(c);
            _vertices.Add(d);
            for (int k = 0; k < 4; k++)
                _neighbors.Add(-1);
            _alive.Add(true);
            var (x, y, z, r2) = Circumsphere(a, b, c, d);
            _spheres.Add(x);
            _spheres.Add(y);
            _spheres.Add(z);
            _spheres.Add(r2);
            return t;
        }

        private void SetNeighbor(int t, int face, int neighbor) => _neighbors[(4 * t) + face] = neighbor;

        private bool InSphere(int t, int p)
        {
            double dx = _points[3 * p] - _spheres[4 * t];
            double dy = _points[(3 * p) + 1] - _spheres[(4 * t) + 1];
            double dz = _points[(3 * p) + 2] - _spheres[(4 * t) + 2];
            return (dx * dx) + (dy * dy) + (dz * dz) < _spheres[(4 * t) + 3];
        }

        private double OrientReplaced(int t, int face, int p)
        {
            var v = new int[4];
            for (int c = 0; c < 4; c++)
                v[c] = c == face ? p : Vertex(t, c);
            return Orient(v[0], v[1], v[2], v[3]);
        }

        /// <summary>(b - a) . ((c - a) x (d - a)): positive for a positively oriented tetrahedron.</summary>
        internal double Orient(int a, int b, int c, int d)
        {
            double bx = _points[3 * b] - _points[3 * a], by = _points[(3 * b) + 1] - _points[(3 * a) + 1], bz = _points[(3 * b) + 2] - _points[(3 * a) + 2];
            double cx = _points[3 * c] - _points[3 * a], cy = _points[(3 * c) + 1] - _points[(3 * a) + 1], cz = _points[(3 * c) + 2] - _points[(3 * a) + 2];
            double dx = _points[3 * d] - _points[3 * a], dy = _points[(3 * d) + 1] - _points[(3 * a) + 1], dz = _points[(3 * d) + 2] - _points[(3 * a) + 2];
            return (bx * ((cy * dz) - (cz * dy))) + (by * ((cz * dx) - (cx * dz))) + (bz * ((cx * dy) - (cy * dx)));
        }

        private (double X, double Y, double Z, double RadiusSquared) Circumsphere(int a, int b, int c, int d)
        {
            double ax = _points[3 * a], ay = _points[(3 * a) + 1], az = _points[(3 * a) + 2];
            double bx = _points[3 * b] - ax, by = _points[(3 * b) + 1] - ay, bz = _points[(3 * b) + 2] - az;
            double cx = _points[3 * c] - ax, cy = _points[(3 * c) + 1] - ay, cz = _points[(3 * c) + 2] - az;
            double dx = _points[3 * d] - ax, dy = _points[(3 * d) + 1] - ay, dz = _points[(3 * d) + 2] - az;
            double b2 = (bx * bx) + (by * by) + (bz * bz);
            double c2 = (cx * cx) + (cy * cy) + (cz * cz);
            double d2 = (dx * dx) + (dy * dy) + (dz * dz);
            double det = 2.0 * ((bx * ((cy * dz) - (cz * dy))) + (by * ((cz * dx) - (cx * dz))) + (bz * ((cx * dy) - (cy * dx))));
            if (det == 0.0)
                return (ax, ay, az, double.PositiveInfinity);

            // Solve [b; c; d] x = 0.5 [|b|^2; |c|^2; |d|^2] by Cramer's rule (det already carries the factor 2).
            double x = ((b2 * ((cy * dz) - (cz * dy))) - (by * ((c2 * dz) - (cz * d2))) + (bz * ((c2 * dy) - (cy * d2)))) / det;
            double y = ((bx * ((c2 * dz) - (cz * d2))) - (b2 * ((cx * dz) - (cz * dx))) + (bz * ((cx * d2) - (c2 * dx)))) / det;
            double z = ((bx * ((cy * d2) - (c2 * dy))) - (by * ((cx * d2) - (c2 * dx))) + (b2 * ((cx * dy) - (cy * dx)))) / det;
            return (ax + x, ay + y, az + z, (x * x) + (y * y) + (z * z));
        }

        private double DistanceSquared(int i, int j)
        {
            double dx = _points[3 * i] - _points[3 * j], dy = _points[(3 * i) + 1] - _points[(3 * j) + 1], dz = _points[(3 * i) + 2] - _points[(3 * j) + 2];
            return (dx * dx) + (dy * dy) + (dz * dz);
        }
    }
}
