namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// A uniform bucket grid over a point set for radius and k-nearest-neighbour queries.
/// </summary>
/// <remarks>
/// Points are bucketed by cell and stored cell by cell (counting sort), so a query visits only the cells its search
/// cube overlaps. The cell size is the requested size, enlarged when needed so the grid has at most about four cells
/// per point; a larger cell only costs time, never correctness.
/// </remarks>
internal sealed class PointGrid
{
    private readonly double[] _points;
    private readonly double _minX;
    private readonly double _minY;
    private readonly double _minZ;
    private readonly double _cell;
    private readonly int _gx;
    private readonly int _gy;
    private readonly int _gz;
    private readonly int[] _cellStart;
    private readonly int[] _order;

    internal PointGrid(double[] points, double cellSize)
    {
        if (points.Length % 3 != 0)
            throw new ArgumentException("Points must come in (x, y, z) triples.", nameof(points));

        _points = points;
        int count = points.Length / 3;
        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
        for (int i = 0; i < count; i++)
        {
            minX = Math.Min(minX, points[3 * i]);
            minY = Math.Min(minY, points[(3 * i) + 1]);
            minZ = Math.Min(minZ, points[(3 * i) + 2]);
            maxX = Math.Max(maxX, points[3 * i]);
            maxY = Math.Max(maxY, points[(3 * i) + 1]);
            maxZ = Math.Max(maxZ, points[(3 * i) + 2]);
        }

        if (count == 0)
        {
            minX = minY = minZ = 0.0;
            maxX = maxY = maxZ = 0.0;
        }

        double extent = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
        double cell = cellSize > 0.0 && !double.IsInfinity(cellSize) ? cellSize : Math.Max(extent, 1e-12);
        cell = Math.Max(cell, Math.Max(extent, 1e-12) * 1e-6);
        long maxCells = Math.Max(64L, 4L * count);
        while (CellCount(maxX - minX, cell) * CellCount(maxY - minY, cell) * CellCount(maxZ - minZ, cell) > maxCells)
            cell *= 1.25;

        _minX = minX;
        _minY = minY;
        _minZ = minZ;
        _cell = cell;
        _gx = (int)CellCount(maxX - minX, cell);
        _gy = (int)CellCount(maxY - minY, cell);
        _gz = (int)CellCount(maxZ - minZ, cell);

        var cellOf = new int[count];
        _cellStart = new int[(_gx * _gy * _gz) + 1];
        for (int i = 0; i < count; i++)
        {
            int c = CellIndex(Clamp(Coordinate(points[3 * i], _minX), _gx), Clamp(Coordinate(points[(3 * i) + 1], _minY), _gy),
                Clamp(Coordinate(points[(3 * i) + 2], _minZ), _gz));
            cellOf[i] = c;
            _cellStart[c + 1]++;
        }

        for (int c = 0; c < _cellStart.Length - 1; c++)
            _cellStart[c + 1] += _cellStart[c];

        _order = new int[count];
        var fill = new int[_cellStart.Length - 1];
        for (int i = 0; i < count; i++)
        {
            int c = cellOf[i];
            _order[_cellStart[c] + fill[c]] = i;
            fill[c]++;
        }
    }

    internal int Count => _points.Length / 3;

    /// <summary>
    /// Appends to <paramref name="result"/> every point within <paramref name="radius"/> of (x, y, z).
    /// </summary>
    internal void QueryRadius(double x, double y, double z, double radius, List<int> result)
    {
        result.Clear();
        double radiusSquared = radius * radius;
        int x0 = Clamp(Coordinate(x - radius, _minX), _gx), x1 = Clamp(Coordinate(x + radius, _minX), _gx);
        int y0 = Clamp(Coordinate(y - radius, _minY), _gy), y1 = Clamp(Coordinate(y + radius, _minY), _gy);
        int z0 = Clamp(Coordinate(z - radius, _minZ), _gz), z1 = Clamp(Coordinate(z + radius, _minZ), _gz);
        for (int cx = x0; cx <= x1; cx++)
        {
            for (int cy = y0; cy <= y1; cy++)
            {
                for (int cz = z0; cz <= z1; cz++)
                {
                    int c = CellIndex(cx, cy, cz);
                    for (int k = _cellStart[c]; k < _cellStart[c + 1]; k++)
                    {
                        int i = _order[k];
                        if (DistanceSquared(i, x, y, z) <= radiusSquared)
                            result.Add(i);
                    }
                }
            }
        }
    }

    /// <summary>
    /// The <paramref name="k"/> points nearest to (x, y, z), nearest first, skipping <paramref name="exclude"/>
    /// (pass -1 to keep every point). Fewer are returned when the set is smaller.
    /// </summary>
    internal int[] Nearest(double x, double y, double z, int k, int exclude, out double[] distancesSquared)
    {
        int available = Count - (exclude >= 0 && exclude < Count ? 1 : 0);
        int wanted = Math.Max(0, Math.Min(k, available));
        var best = new int[wanted];
        var bestDistance = new double[wanted];
        int found = 0;
        if (wanted == 0)
        {
            distancesSquared = bestDistance;
            return best;
        }

        int qx = Clamp(Coordinate(x, _minX), _gx), qy = Clamp(Coordinate(y, _minY), _gy), qz = Clamp(Coordinate(z, _minZ), _gz);
        int maxRing = Math.Max(_gx, Math.Max(_gy, _gz));
        for (int ring = 0; ring <= maxRing; ring++)
        {
            for (int cx = qx - ring; cx <= qx + ring; cx++)
            {
                if (cx < 0 || cx >= _gx)
                    continue;
                for (int cy = qy - ring; cy <= qy + ring; cy++)
                {
                    if (cy < 0 || cy >= _gy)
                        continue;
                    for (int cz = qz - ring; cz <= qz + ring; cz++)
                    {
                        if (cz < 0 || cz >= _gz)
                            continue;
                        // Only the shell of this ring; the inner cells were visited by earlier rings.
                        if (Math.Abs(cx - qx) != ring && Math.Abs(cy - qy) != ring && Math.Abs(cz - qz) != ring)
                            continue;

                        int c = CellIndex(cx, cy, cz);
                        for (int s = _cellStart[c]; s < _cellStart[c + 1]; s++)
                        {
                            int i = _order[s];
                            if (i == exclude)
                                continue;

                            double d = DistanceSquared(i, x, y, z);
                            if (found == wanted && d >= bestDistance[wanted - 1])
                                continue;

                            // Insertion into the sorted best list; ties keep the lower index first (deterministic).
                            int position = found < wanted ? found : wanted - 1;
                            while (position > 0
                                && (bestDistance[position - 1] > d || (bestDistance[position - 1] == d && best[position - 1] > i)))
                            {
                                bestDistance[position] = bestDistance[position - 1];
                                best[position] = best[position - 1];
                                position--;
                            }

                            bestDistance[position] = d;
                            best[position] = i;
                            if (found < wanted)
                                found++;
                        }
                    }
                }
            }

            // Every point outside the searched block is at least ring * cell away from the query.
            if (found == wanted)
            {
                double reach = ring * _cell;
                if (bestDistance[wanted - 1] <= reach * reach)
                    break;
            }
        }

        distancesSquared = bestDistance;
        return best;
    }

    internal double DistanceSquared(int i, double x, double y, double z)
    {
        double dx = _points[3 * i] - x, dy = _points[(3 * i) + 1] - y, dz = _points[(3 * i) + 2] - z;
        return (dx * dx) + (dy * dy) + (dz * dz);
    }

    private static long CellCount(double extent, double cell) => Math.Max(1L, (long)Math.Floor(extent / cell) + 1);

    private int Coordinate(double value, double min) => (int)Math.Floor((value - min) / _cell);

    private static int Clamp(int value, int count) => value < 0 ? 0 : value >= count ? count - 1 : value;

    private int CellIndex(int x, int y, int z) => (((x * _gy) + y) * _gz) + z;
}
