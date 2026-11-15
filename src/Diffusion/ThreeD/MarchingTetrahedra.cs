namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Iso-surface extraction from a regular scalar grid by marching tetrahedra.
/// </summary>
/// <remarks>
/// Every grid cell is split into six tetrahedra around its main diagonal (a conforming split, so neighbouring
/// cells agree on their shared faces). A tetrahedron the surface {value = threshold} crosses contributes one
/// triangle (one corner on its own side) or two (two corners on each side), with vertices linearly interpolated
/// along the crossed edges.
/// </remarks>
internal static class MarchingTetrahedra
{
    private static readonly int[][] CubeCorners =
    {
        new[] { 0, 0, 0 }, new[] { 1, 0, 0 }, new[] { 1, 1, 0 }, new[] { 0, 1, 0 },
        new[] { 0, 0, 1 }, new[] { 1, 0, 1 }, new[] { 1, 1, 1 }, new[] { 0, 1, 1 }
    };

    private static readonly int[][] CubeTetrahedra =
    {
        new[] { 0, 5, 1, 6 }, new[] { 0, 1, 2, 6 }, new[] { 0, 2, 3, 6 },
        new[] { 0, 3, 7, 6 }, new[] { 0, 7, 4, 6 }, new[] { 0, 4, 5, 6 }
    };

    /// <summary>
    /// Extracts the surface where <paramref name="grid"/> crosses <paramref name="threshold"/>; grid index i maps to
    /// the coordinate <c>origin + i * step</c> on every axis. Values above the threshold are inside.
    /// </summary>
    /// <returns>Vertex coordinates (x, y, z per vertex) and triangle vertex indices (three per triangle).</returns>
    internal static (List<double> Coordinates, List<int> Triangles) Extract(
        double[,,] grid, double threshold, double origin, double step)
    {
        var coordinates = new List<double>();
        var triangles = new List<int>();
        int nx = grid.GetLength(0), ny = grid.GetLength(1), nz = grid.GetLength(2);
        var values = new double[4];
        // One coordinate buffer per tetrahedron corner, refilled in place: Interpolate copies what it keeps.
        var points = new[] { new double[3], new double[3], new double[3], new double[3] };
        var crossing = new List<double[]>(4);
        var crossingIndex = new int[4];
        // Twice-area squared below which a crossing triangle counts as collapsed (all three crossings on one point
        // or one line, e.g. a surface through grid vertices); such faces carry no surface and are dropped.
        double minimumNormalSquared = 1e-24 * step * step * step * step;

        for (int x = 0; x < nx - 1; x++)
        {
            for (int y = 0; y < ny - 1; y++)
            {
                for (int z = 0; z < nz - 1; z++)
                {
                    foreach (var tetrahedron in CubeTetrahedra)
                    {
                        int insideCount = 0;
                        for (int k = 0; k < 4; k++)
                        {
                            var corner = CubeCorners[tetrahedron[k]];
                            int cx = x + corner[0], cy = y + corner[1], cz = z + corner[2];
                            values[k] = grid[cx, cy, cz];
                            points[k][0] = origin + (cx * step);
                            points[k][1] = origin + (cy * step);
                            points[k][2] = origin + (cz * step);
                            if (values[k] > threshold)
                                insideCount++;
                        }

                        if (insideCount == 0 || insideCount == 4)
                            continue;

                        crossing.Clear();
                        if (insideCount == 2)
                        {
                            // Quad between the two inside (a, b) and two outside (c, d) corners, walked as a cycle.
                            int a = -1, b = -1, c = -1, d = -1;
                            for (int k = 0; k < 4; k++)
                            {
                                if (values[k] > threshold)
                                {
                                    if (a < 0) a = k; else b = k;
                                }
                                else
                                {
                                    if (c < 0) c = k; else d = k;
                                }
                            }

                            crossing.Add(Interpolate(points[a], values[a], points[c], values[c], threshold));
                            crossing.Add(Interpolate(points[a], values[a], points[d], values[d], threshold));
                            crossing.Add(Interpolate(points[b], values[b], points[d], values[d], threshold));
                            crossing.Add(Interpolate(points[b], values[b], points[c], values[c], threshold));
                        }
                        else
                        {
                            // One corner on its own side: a triangle across its three edges.
                            bool loneInside = insideCount == 1;
                            int lone = 0;
                            for (int k = 0; k < 4; k++)
                            {
                                if ((values[k] > threshold) == loneInside)
                                    lone = k;
                            }

                            for (int k = 0; k < 4; k++)
                            {
                                if (k != lone)
                                    crossing.Add(Interpolate(points[lone], values[lone], points[k], values[k], threshold));
                            }
                        }

                        // Each triangle (one, or two for a quad) is emitted on its own: a collapsed one is dropped,
                        // and each is wound counter-clockwise seen from outside, so its normal points from the inside
                        // corners to the outside ones (the direction the field decreases through the surface).
                        // Crossing vertices are stored only once a kept triangle uses them.
                        var outward = InsideToOutside(values, points, threshold);
                        for (int k = 0; k < crossing.Count; k++)
                            crossingIndex[k] = -1;
                        EmitTriangle(crossing, crossingIndex, 0, 1, 2, outward, minimumNormalSquared, coordinates, triangles);
                        if (crossing.Count == 4)
                            EmitTriangle(crossing, crossingIndex, 0, 2, 3, outward, minimumNormalSquared, coordinates, triangles);
                    }
                }
            }
        }

        return (coordinates, triangles);
    }

    /// <summary>
    /// The direction from the mean of the inside corners to the mean of the outside corners.
    /// </summary>
    private static double[] InsideToOutside(double[] values, double[][] points, double threshold)
    {
        var inside = new double[3];
        var outside = new double[3];
        int insideCount = 0, outsideCount = 0;
        for (int k = 0; k < 4; k++)
        {
            bool isInside = values[k] > threshold;
            var target = isInside ? inside : outside;
            for (int axis = 0; axis < 3; axis++)
                target[axis] += points[k][axis];
            if (isInside) insideCount++; else outsideCount++;
        }

        return new[]
        {
            (outside[0] / outsideCount) - (inside[0] / insideCount),
            (outside[1] / outsideCount) - (inside[1] / insideCount),
            (outside[2] / outsideCount) - (inside[2] / insideCount)
        };
    }

    /// <summary>
    /// Appends the crossing triangle (i0, i1, i2) wound so its normal (p1 - p0) x (p2 - p0) points along
    /// <paramref name="outward"/>, unless it has (numerically) zero area.
    /// </summary>
    private static void EmitTriangle(
        List<double[]> crossing, int[] crossingIndex, int i0, int i1, int i2, double[] outward,
        double minimumNormalSquared, List<double> coordinates, List<int> triangles)
    {
        var p0 = crossing[i0];
        var p1 = crossing[i1];
        var p2 = crossing[i2];
        double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
        double vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
        double nx = (uy * vz) - (uz * vy);
        double ny = (uz * vx) - (ux * vz);
        double nz = (ux * vy) - (uy * vx);
        if ((nx * nx) + (ny * ny) + (nz * nz) <= minimumNormalSquared)
            return;

        bool facesOutward = (nx * outward[0]) + (ny * outward[1]) + (nz * outward[2]) >= 0.0;
        triangles.Add(VertexIndex(crossing, crossingIndex, i0, coordinates));
        triangles.Add(VertexIndex(crossing, crossingIndex, facesOutward ? i1 : i2, coordinates));
        triangles.Add(VertexIndex(crossing, crossingIndex, facesOutward ? i2 : i1, coordinates));
    }

    private static int VertexIndex(List<double[]> crossing, int[] crossingIndex, int i, List<double> coordinates)
    {
        if (crossingIndex[i] < 0)
        {
            crossingIndex[i] = coordinates.Count / 3;
            coordinates.AddRange(crossing[i]);
        }

        return crossingIndex[i];
    }

    private static double[] Interpolate(double[] p0, double v0, double[] p1, double v1, double threshold)
    {
        double t = v1 == v0 ? 0.5 : (threshold - v0) / (v1 - v0);
        return new[] { p0[0] + (t * (p1[0] - p0[0])), p0[1] + (t * (p1[1] - p0[1])), p0[2] + (t * (p1[2] - p0[2])) };
    }
}
