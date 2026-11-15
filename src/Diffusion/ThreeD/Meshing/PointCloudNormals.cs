namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Oriented normal estimation for an unorganised point cloud (Hoppe, DeRose, Duchamp, McDonald &amp; Stuetzle,
/// "Surface Reconstruction from Unorganized Points", SIGGRAPH 1992, sections 3.2 and 3.3).
/// </summary>
/// <remarks>
/// <para>
/// Each point gets a tangent plane fitted to its k nearest neighbours: the plane's centre is the neighbourhood
/// centroid and its normal is the eigenvector of the neighbourhood covariance with the smallest eigenvalue. The sign
/// of each normal is then made consistent by propagating along a minimum spanning tree of the Riemannian graph (the
/// symmetric k-nearest-neighbour graph) with edge cost <c>1 - |n_i . n_j|</c>, so orientation flows between nearly
/// parallel planes first. Each connected component is rooted at its point of largest z, whose normal is forced to
/// +z, which points the normals outward for a closed surface.
/// </para>
/// </remarks>
internal static class PointCloudNormals
{
    /// <summary>
    /// Estimates unit outward normals and tangent-plane centres.
    /// </summary>
    /// <param name="points">(x, y, z) per point.</param>
    /// <param name="neighborCount">The neighbourhood size k (including the point itself).</param>
    /// <param name="grid">A neighbour grid over <paramref name="points"/>.</param>
    internal static (double[] Normals, double[] Centers) Estimate(double[] points, int neighborCount, PointGrid grid)
    {
        int count = points.Length / 3;
        if (count < 3)
            throw new ArgumentException("Normal estimation needs at least three points.", nameof(points));
        if (neighborCount < 3)
            throw new ArgumentOutOfRangeException(nameof(neighborCount), "The neighbourhood needs at least three points.");

        int k = Math.Min(neighborCount, count);
        var normals = new double[points.Length];
        var centers = new double[points.Length];
        var neighbors = new int[count][];
        var covariance = new double[3, 3];

        for (int i = 0; i < count; i++)
        {
            double x = points[3 * i], y = points[(3 * i) + 1], z = points[(3 * i) + 2];
            int[] near = grid.Nearest(x, y, z, k, -1, out _);
            neighbors[i] = near;

            double cx = 0.0, cy = 0.0, cz = 0.0;
            foreach (int j in near)
            {
                cx += points[3 * j];
                cy += points[(3 * j) + 1];
                cz += points[(3 * j) + 2];
            }

            cx /= near.Length;
            cy /= near.Length;
            cz /= near.Length;
            centers[3 * i] = cx;
            centers[(3 * i) + 1] = cy;
            centers[(3 * i) + 2] = cz;

            Array.Clear(covariance, 0, 9);
            foreach (int j in near)
            {
                double dx = points[3 * j] - cx, dy = points[(3 * j) + 1] - cy, dz = points[(3 * j) + 2] - cz;
                covariance[0, 0] += dx * dx;
                covariance[0, 1] += dx * dy;
                covariance[0, 2] += dx * dz;
                covariance[1, 1] += dy * dy;
                covariance[1, 2] += dy * dz;
                covariance[2, 2] += dz * dz;
            }

            covariance[1, 0] = covariance[0, 1];
            covariance[2, 0] = covariance[0, 2];
            covariance[2, 1] = covariance[1, 2];
            var (nx, ny, nz) = SmallestEigenvector(covariance);
            normals[3 * i] = nx;
            normals[(3 * i) + 1] = ny;
            normals[(3 * i) + 2] = nz;
        }

        Orient(points, normals, neighbors);
        return (normals, centers);
    }

    /// <summary>
    /// Makes the normal signs consistent along a minimum spanning tree of the symmetric neighbour graph (Prim's
    /// algorithm with cost 1 - |n_i . n_j|), rooting each component at its highest point with normal +z.
    /// </summary>
    internal static void Orient(double[] points, double[] normals, int[][] neighbors)
    {
        int count = points.Length / 3;
        var adjacency = new List<int>[count];
        for (int i = 0; i < count; i++)
            adjacency[i] = new List<int>();
        for (int i = 0; i < count; i++)
        {
            foreach (int j in neighbors[i])
            {
                if (j == i)
                    continue;
                if (!adjacency[i].Contains(j))
                    adjacency[i].Add(j);
                if (!adjacency[j].Contains(i))
                    adjacency[j].Add(i);
            }
        }

        // Points in order of decreasing z, so each new component starts at its highest unvisited point.
        var byHeight = new int[count];
        var heights = new double[count];
        for (int i = 0; i < count; i++)
        {
            byHeight[i] = i;
            heights[i] = -points[(3 * i) + 2];
        }

        Array.Sort(heights, byHeight);

        var inTree = new bool[count];
        var heap = new MinHeap(count);
        foreach (int root in byHeight)
        {
            if (inTree[root])
                continue;

            if (normals[(3 * root) + 2] < 0.0)
                Flip(normals, root);

            heap.Push(0.0, root, root);
            while (heap.Count > 0)
            {
                var (_, vertex, parent) = heap.Pop();
                if (inTree[vertex])
                    continue;

                if (vertex != parent && Dot(normals, vertex, parent) < 0.0)
                    Flip(normals, vertex);
                inTree[vertex] = true;

                foreach (int next in adjacency[vertex])
                {
                    if (!inTree[next])
                        heap.Push(1.0 - Math.Abs(Dot(normals, vertex, next)), next, vertex);
                }
            }
        }
    }

    /// <summary>
    /// The unit eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (cyclic Jacobi rotations).
    /// </summary>
    internal static (double X, double Y, double Z) SmallestEigenvector(double[,] matrix)
    {
        var a = (double[,])matrix.Clone();
        var v = new double[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = Math.Abs(a[0, 1]) + Math.Abs(a[0, 2]) + Math.Abs(a[1, 2]);
            double scale = Math.Abs(a[0, 0]) + Math.Abs(a[1, 1]) + Math.Abs(a[2, 2]);
            if (off <= 1e-15 * Math.Max(scale, double.Epsilon))
                break;

            for (int p = 0; p < 2; p++)
            {
                for (int q = p + 1; q < 3; q++)
                {
                    if (a[p, q] == 0.0)
                        continue;

                    double theta = (a[q, q] - a[p, p]) / (2.0 * a[p, q]);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt((theta * theta) + 1.0));
                    if (theta == 0.0)
                        t = 1.0;
                    double c = 1.0 / Math.Sqrt((t * t) + 1.0), s = t * c;
                    for (int r = 0; r < 3; r++)
                    {
                        double arp = a[r, p], arq = a[r, q];
                        a[r, p] = (c * arp) - (s * arq);
                        a[r, q] = (s * arp) + (c * arq);
                    }

                    for (int r = 0; r < 3; r++)
                    {
                        double apr = a[p, r], aqr = a[q, r];
                        a[p, r] = (c * apr) - (s * aqr);
                        a[q, r] = (s * apr) + (c * aqr);
                    }

                    for (int r = 0; r < 3; r++)
                    {
                        double vrp = v[r, p], vrq = v[r, q];
                        v[r, p] = (c * vrp) - (s * vrq);
                        v[r, q] = (s * vrp) + (c * vrq);
                    }
                }
            }
        }

        int smallest = 0;
        for (int i = 1; i < 3; i++)
        {
            if (a[i, i] < a[smallest, smallest])
                smallest = i;
        }

        double x = v[0, smallest], y = v[1, smallest], z = v[2, smallest];
        double length = Math.Sqrt((x * x) + (y * y) + (z * z));
        return length > 0.0 ? (x / length, y / length, z / length) : (0.0, 0.0, 1.0);
    }

    private static double Dot(double[] normals, int i, int j)
    {
        return (normals[3 * i] * normals[3 * j]) + (normals[(3 * i) + 1] * normals[(3 * j) + 1])
            + (normals[(3 * i) + 2] * normals[(3 * j) + 2]);
    }

    private static void Flip(double[] normals, int i)
    {
        normals[3 * i] = -normals[3 * i];
        normals[(3 * i) + 1] = -normals[(3 * i) + 1];
        normals[(3 * i) + 2] = -normals[(3 * i) + 2];
    }

    /// <summary>A binary min-heap of (cost, vertex, parent); ties break on the lower vertex for determinism.</summary>
    private sealed class MinHeap
    {
        private readonly List<(double Cost, int Vertex, int Parent)> _items;

        internal MinHeap(int capacity)
        {
            _items = new List<(double, int, int)>(capacity);
        }

        internal int Count => _items.Count;

        internal void Push(double cost, int vertex, int parent)
        {
            _items.Add((cost, vertex, parent));
            int i = _items.Count - 1;
            while (i > 0)
            {
                int up = (i - 1) / 2;
                if (!Less(_items[i], _items[up]))
                    break;
                (_items[i], _items[up]) = (_items[up], _items[i]);
                i = up;
            }
        }

        internal (double Cost, int Vertex, int Parent) Pop()
        {
            var top = _items[0];
            int last = _items.Count - 1;
            _items[0] = _items[last];
            _items.RemoveAt(last);
            int i = 0;
            while (true)
            {
                int left = (2 * i) + 1, right = left + 1, smallest = i;
                if (left < _items.Count && Less(_items[left], _items[smallest]))
                    smallest = left;
                if (right < _items.Count && Less(_items[right], _items[smallest]))
                    smallest = right;
                if (smallest == i)
                    break;
                (_items[i], _items[smallest]) = (_items[smallest], _items[i]);
                i = smallest;
            }

            return top;
        }

        private static bool Less((double Cost, int Vertex, int Parent) a, (double Cost, int Vertex, int Parent) b)
        {
            return a.Cost < b.Cost || (a.Cost == b.Cost && a.Vertex < b.Vertex);
        }
    }
}
