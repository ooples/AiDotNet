namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// The ball-pivoting algorithm (Bernardini, Mittleman, Rushmeier, Silva &amp; Taubin, "The Ball-Pivoting Algorithm for
/// Surface Reconstruction", IEEE TVCG 1999).
/// </summary>
/// <remarks>
/// <para>
/// A ball of radius rho touching three samples, with no other sample inside it, spans a triangle. Starting from such a
/// seed triangle, the ball is pivoted around each edge of the advancing front (keeping contact with the edge's two
/// end points) until it touches another sample; that sample and the edge form the next triangle. The front is a set of
/// directed edges, each the free edge of one triangle; a new triangle whose edge meets an opposite front edge glues
/// the two together (the paper's join and glue operations). An edge whose pivot finds no valid sample becomes a
/// boundary edge. When the front is exhausted, a new seed is searched among the unused samples.
/// </para>
/// <para>
/// With several radii (the paper's section 5), each larger radius first re-pivots the boundary edges left by the
/// previous one, then seeds again, which closes holes where the sampling was too sparse for the smaller ball.
/// </para>
/// <para>
/// Triangles are only accepted when consistent with the oriented sample normals (the paper's condition on the seed
/// and pivot triangles), the ball centre is always placed on the outward side, and every triangle is wound
/// counter-clockwise seen from outside. Edges stay manifold: an edge is never used by more than two triangles, and
/// two triangles share an edge only with opposite directions.
/// </para>
/// </remarks>
internal static class BallPivotingReconstruction
{
    private sealed class FrontEdge
    {
        internal FrontEdge(int a, int b, int opposite, double cx, double cy, double cz)
        {
            A = a;
            B = b;
            Opposite = opposite;
            CenterX = cx;
            CenterY = cy;
            CenterZ = cz;
        }

        internal int A { get; }

        internal int B { get; }

        internal int Opposite { get; }

        internal double CenterX { get; }

        internal double CenterY { get; }

        internal double CenterZ { get; }

        internal bool Boundary { get; set; }
    }

    private sealed class State
    {
        internal State(double[] points, double[] normals)
        {
            Points = points;
            Normals = normals;
            Count = points.Length / 3;
            Used = new bool[Count];
            FrontDegree = new int[Count];
        }

        internal double[] Points { get; }

        internal double[] Normals { get; }

        internal int Count { get; }

        internal bool[] Used { get; }

        internal int[] FrontDegree { get; }

        internal List<int> Triangles { get; } = new List<int>();

        internal Dictionary<long, int> EdgeUse { get; } = new Dictionary<long, int>();

        internal Dictionary<long, FrontEdge> Front { get; } = new Dictionary<long, FrontEdge>();

        internal Queue<long> Active { get; } = new Queue<long>();

        internal long Directed(int a, int b) => ((long)a * Count) + b;

        internal long Undirected(int a, int b) => a < b ? Directed(a, b) : Directed(b, a);
    }

    internal static MeshSurface Reconstruct(double[] points, double[] normals, double[] radii)
    {
        if (radii.Length == 0)
            throw new ArgumentException("Ball pivoting needs at least one radius.", nameof(radii));
        for (int r = 0; r < radii.Length; r++)
        {
            if (!(radii[r] > 0.0) || double.IsInfinity(radii[r]))
                throw new ArgumentOutOfRangeException(nameof(radii), "Every ball radius must be positive and finite.");
            if (r > 0 && radii[r] <= radii[r - 1])
                throw new ArgumentException("Ball radii must be strictly increasing.", nameof(radii));
        }

        var state = new State(points, normals);
        var nearby = new List<int>();
        for (int r = 0; r < radii.Length; r++)
        {
            double rho = radii[r];
            var grid = new PointGrid(points, 2.0 * rho);
            if (r > 0)
            {
                foreach (var entry in state.Front)
                {
                    if (entry.Value.Boundary)
                    {
                        entry.Value.Boundary = false;
                        state.Active.Enqueue(entry.Key);
                    }
                }
            }

            int seedCursor = 0;
            while (true)
            {
                ExpandFront(state, grid, rho, nearby);
                if (!FindSeed(state, grid, rho, ref seedCursor, nearby))
                    break;
            }
        }

        return Compact(state);
    }

    private static void ExpandFront(State state, PointGrid grid, double rho, List<int> nearby)
    {
        while (state.Active.Count > 0)
        {
            long key = state.Active.Dequeue();
            if (!state.Front.TryGetValue(key, out var edge) || edge.Boundary)
                continue;

            if (Pivot(state, grid, edge, rho, nearby, out int k, out double cx, out double cy, out double cz)
                && CanJoin(state, edge.A, edge.B, k))
            {
                AddTriangle(state, edge.B, edge.A, k, cx, cy, cz);
            }
            else
            {
                edge.Boundary = true;
            }
        }
    }

    /// <summary>
    /// Rotates the ball of the front edge (a, b) about the edge, away from the edge's triangle, and returns the first
    /// sample it touches with a valid, empty ball on the outward side of the new triangle (b, a, k).
    /// </summary>
    private static bool Pivot(
        State state, PointGrid grid, FrontEdge edge, double rho, List<int> nearby,
        out int hit, out double hitX, out double hitY, out double hitZ)
    {
        hit = -1;
        hitX = hitY = hitZ = 0.0;
        var p = state.Points;
        int a = edge.A, b = edge.B;
        double mx = 0.5 * (p[3 * a] + p[3 * b]), my = 0.5 * (p[(3 * a) + 1] + p[(3 * b) + 1]), mz = 0.5 * (p[(3 * a) + 2] + p[(3 * b) + 2]);
        double ex = p[3 * b] - p[3 * a], ey = p[(3 * b) + 1] - p[(3 * a) + 1], ez = p[(3 * b) + 2] - p[(3 * a) + 2];
        double edgeLength = Math.Sqrt((ex * ex) + (ey * ey) + (ez * ez));
        if (edgeLength == 0.0)
            return false;
        ex /= edgeLength;
        ey /= edgeLength;
        ez /= edgeLength;

        double ux = edge.CenterX - mx, uy = edge.CenterY - my, uz = edge.CenterZ - mz;
        double pivotRadius = Math.Sqrt((ux * ux) + (uy * uy) + (uz * uz));
        grid.QueryRadius(mx, my, mz, pivotRadius + rho, nearby);

        double bestAngle = double.MaxValue;
        foreach (int k in nearby)
        {
            if (k == a || k == b || k == edge.Opposite)
                continue;
            if (!BallCenter(state, b, a, k, rho, out double cx, out double cy, out double cz))
                continue;

            double vx = cx - mx, vy = cy - my, vz = cz - mz;
            double cross = (ex * ((uy * vz) - (uz * vy))) + (ey * ((uz * vx) - (ux * vz))) + (ez * ((ux * vy) - (uy * vx)));
            double dot = (ux * vx) + (uy * vy) + (uz * vz);
            double angle = Math.Atan2(cross, dot);
            if (angle < 0.0)
                angle += 2.0 * Math.PI;
            if (angle < bestAngle || (angle == bestAngle && k < hit))
            {
                bestAngle = angle;
                hit = k;
                hitX = cx;
                hitY = cy;
                hitZ = cz;
            }
        }

        return hit >= 0 && IsEmpty(state, grid, hitX, hitY, hitZ, rho, a, b, hit, nearby);
    }

    /// <summary>
    /// The centre of the radius-<paramref name="rho"/> ball through (i, j, k) on the side of the triangle's normal
    /// (j - i) x (k - i), when it exists and the triangle agrees with the sample normals.
    /// </summary>
    private static bool BallCenter(State state, int i, int j, int k, double rho, out double cx, out double cy, out double cz)
    {
        cx = cy = cz = 0.0;
        var p = state.Points;
        var n = state.Normals;
        double ax = p[3 * i] - p[3 * k], ay = p[(3 * i) + 1] - p[(3 * k) + 1], az = p[(3 * i) + 2] - p[(3 * k) + 2];
        double bx = p[3 * j] - p[3 * k], by = p[(3 * j) + 1] - p[(3 * k) + 1], bz = p[(3 * j) + 2] - p[(3 * k) + 2];
        // a x b = (i - k) x (j - k) = (j - i) x (k - i): the triangle normal.
        double nx = (ay * bz) - (az * by), ny = (az * bx) - (ax * bz), nz = (ax * by) - (ay * bx);
        double normalSquared = (nx * nx) + (ny * ny) + (nz * nz);
        double aSquared = (ax * ax) + (ay * ay) + (az * az), bSquared = (bx * bx) + (by * by) + (bz * bz);
        if (normalSquared <= 1e-24 * aSquared * bSquared)
            return false;

        double normalLength = Math.Sqrt(normalSquared);
        double ux = nx / normalLength, uy = ny / normalLength, uz = nz / normalLength;
        // The triangle must face the way the samples do (n_k, and the edge's two samples on average).
        double dotK = (ux * n[3 * k]) + (uy * n[(3 * k) + 1]) + (uz * n[(3 * k) + 2]);
        double dotEdge = (ux * (n[3 * i] + n[3 * j])) + (uy * (n[(3 * i) + 1] + n[(3 * j) + 1]))
            + (uz * (n[(3 * i) + 2] + n[(3 * j) + 2]));
        if (dotK <= 0.0 || dotEdge <= 0.0)
            return false;

        // Circumcentre: k + ((|a|^2 b - |b|^2 a) x (a x b)) / (2 |a x b|^2).
        double wx = (aSquared * bx) - (bSquared * ax), wy = (aSquared * by) - (bSquared * ay), wz = (aSquared * bz) - (bSquared * az);
        double ox = ((wy * nz) - (wz * ny)) / (2.0 * normalSquared);
        double oy = ((wz * nx) - (wx * nz)) / (2.0 * normalSquared);
        double oz = ((wx * ny) - (wy * nx)) / (2.0 * normalSquared);
        double circumSquared = (ox * ox) + (oy * oy) + (oz * oz);
        double heightSquared = (rho * rho) - circumSquared;
        if (heightSquared < 0.0)
            return false;

        double height = Math.Sqrt(heightSquared);
        cx = p[3 * k] + ox + (height * ux);
        cy = p[(3 * k) + 1] + oy + (height * uy);
        cz = p[(3 * k) + 2] + oz + (height * uz);
        return true;
    }

    private static bool IsEmpty(
        State state, PointGrid grid, double cx, double cy, double cz, double rho, int i, int j, int k, List<int> scratch)
    {
        // A sample counts as inside when it is clearly inside: the three touching samples and round-off are excluded.
        double limit = rho * rho * (1.0 - 1e-9);
        grid.QueryRadius(cx, cy, cz, rho, scratch);
        foreach (int s in scratch)
        {
            if (s != i && s != j && s != k && grid.DistanceSquared(s, cx, cy, cz) < limit)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Whether the triangle (b, a, k) can join the mesh across the front edge (a, b) without making an edge
    /// non-manifold or inconsistently oriented, and without using an interior vertex.
    /// </summary>
    private static bool CanJoin(State state, int a, int b, int k)
    {
        if (state.Used[k] && state.FrontDegree[k] == 0)
            return false;

        return EdgeAvailable(state, a, k) && EdgeAvailable(state, k, b);
    }

    /// <summary>
    /// Whether the directed edge (u, v) can be added: it is new, or it closes against the front edge (v, u).
    /// </summary>
    private static bool EdgeAvailable(State state, int u, int v)
    {
        int use = state.EdgeUse.TryGetValue(state.Undirected(u, v), out int n) ? n : 0;
        if (use == 0)
            return true;
        return use == 1 && state.Front.ContainsKey(state.Directed(v, u));
    }

    /// <summary>
    /// Adds the triangle (i, j, k) (counter-clockwise seen from outside) and updates the front: each of its directed
    /// edges either glues against the opposite front edge or becomes a new active front edge.
    /// </summary>
    private static void AddTriangle(State state, int i, int j, int k, double cx, double cy, double cz)
    {
        state.Triangles.Add(i);
        state.Triangles.Add(j);
        state.Triangles.Add(k);
        state.Used[i] = state.Used[j] = state.Used[k] = true;

        AddEdge(state, i, j, k, cx, cy, cz);
        AddEdge(state, j, k, i, cx, cy, cz);
        AddEdge(state, k, i, j, cx, cy, cz);
    }

    private static void AddEdge(State state, int u, int v, int opposite, double cx, double cy, double cz)
    {
        long undirected = state.Undirected(u, v);
        state.EdgeUse[undirected] = state.EdgeUse.TryGetValue(undirected, out int n) ? n + 1 : 1;

        long reverse = state.Directed(v, u);
        if (state.Front.Remove(reverse))
        {
            // Glue: the edge is now interior.
            state.FrontDegree[u]--;
            state.FrontDegree[v]--;
            return;
        }

        long key = state.Directed(u, v);
        state.Front[key] = new FrontEdge(u, v, opposite, cx, cy, cz);
        state.FrontDegree[u]++;
        state.FrontDegree[v]++;
        state.Active.Enqueue(key);
    }

    /// <summary>
    /// Searches the unused samples, from <paramref name="cursor"/> on, for a seed triangle: a sample sigma and two
    /// unused neighbours whose ball of radius rho is empty and agrees with their normals.
    /// </summary>
    private static bool FindSeed(State state, PointGrid grid, double rho, ref int cursor, List<int> nearby)
    {
        var p = state.Points;
        var n = state.Normals;
        var candidates = new List<int>();
        var scratch = new List<int>();
        for (; cursor < state.Count; cursor++)
        {
            int sigma = cursor;
            if (state.Used[sigma])
                continue;

            double sx = p[3 * sigma], sy = p[(3 * sigma) + 1], sz = p[(3 * sigma) + 2];
            grid.QueryRadius(sx, sy, sz, 2.0 * rho, nearby);
            candidates.Clear();
            foreach (int q in nearby)
            {
                if (q != sigma && !state.Used[q])
                    candidates.Add(q);
            }

            // Nearest first (ties by index), so the seed is as small as the data allows.
            candidates.Sort((x, y) =>
            {
                int byDistance = grid.DistanceSquared(x, sx, sy, sz).CompareTo(grid.DistanceSquared(y, sx, sy, sz));
                return byDistance != 0 ? byDistance : x.CompareTo(y);
            });

            for (int u = 0; u < candidates.Count; u++)
            {
                for (int w = u + 1; w < candidates.Count; w++)
                {
                    int first = candidates[u], second = candidates[w];
                    // Wind the triangle so its normal agrees with sigma's.
                    double ax = p[3 * first] - sx, ay = p[(3 * first) + 1] - sy, az = p[(3 * first) + 2] - sz;
                    double bx = p[3 * second] - sx, by = p[(3 * second) + 1] - sy, bz = p[(3 * second) + 2] - sz;
                    double nx = (ay * bz) - (az * by), ny = (az * bx) - (ax * bz), nz = (ax * by) - (ay * bx);
                    if ((nx * n[3 * sigma]) + (ny * n[(3 * sigma) + 1]) + (nz * n[(3 * sigma) + 2]) < 0.0)
                    {
                        int swap = first;
                        first = second;
                        second = swap;
                    }

                    if (!BallCenter(state, sigma, first, second, rho, out double cx, out double cy, out double cz))
                        continue;
                    if (!IsEmpty(state, grid, cx, cy, cz, rho, sigma, first, second, scratch))
                        continue;

                    AddTriangle(state, sigma, first, second, cx, cy, cz);
                    cursor++;
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Drops unused samples and renumbers the triangles onto the used ones.</summary>
    private static MeshSurface Compact(State state)
    {
        var map = new int[state.Count];
        int used = 0;
        for (int i = 0; i < state.Count; i++)
            map[i] = state.Used[i] ? used++ : -1;

        var coordinates = new double[3 * used];
        var normals = new double[3 * used];
        for (int i = 0; i < state.Count; i++)
        {
            int m = map[i];
            if (m < 0)
                continue;
            for (int axis = 0; axis < 3; axis++)
            {
                coordinates[(3 * m) + axis] = state.Points[(3 * i) + axis];
                normals[(3 * m) + axis] = state.Normals[(3 * i) + axis];
            }
        }

        var triangles = new int[state.Triangles.Count];
        for (int t = 0; t < triangles.Length; t++)
            triangles[t] = map[state.Triangles[t]];
        return new MeshSurface(coordinates, triangles, normals);
    }
}
