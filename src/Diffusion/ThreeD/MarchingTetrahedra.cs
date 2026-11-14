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
        var points = new double[4][];
        var crossing = new List<double[]>(4);

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
                            points[k] = new[] { origin + (cx * step), origin + (cy * step), origin + (cz * step) };
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

                        // Wind every triangle counter-clockwise seen from outside, so its normal points from the
                        // inside corners to the outside ones (the direction the field decreases through the surface).
                        if (!FacesOutward(crossing, values, points, threshold))
                            crossing.Reverse();

                        int baseIndex = coordinates.Count / 3;
                        foreach (var point in crossing)
                            coordinates.AddRange(point);
                        triangles.Add(baseIndex);
                        triangles.Add(baseIndex + 1);
                        triangles.Add(baseIndex + 2);
                        if (crossing.Count == 4)
                        {
                            triangles.Add(baseIndex);
                            triangles.Add(baseIndex + 2);
                            triangles.Add(baseIndex + 3);
                        }
                    }
                }
            }
        }

        return (coordinates, triangles);
    }

    /// <summary>
    /// True when the normal (p1 - p0) x (p2 - p0) of the first crossing triangle points from the mean of the inside
    /// corners towards the mean of the outside corners.
    /// </summary>
    private static bool FacesOutward(List<double[]> crossing, double[] values, double[][] points, double threshold)
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

        var p0 = crossing[0];
        var p1 = crossing[1];
        var p2 = crossing[2];
        double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
        double vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
        double nx = (uy * vz) - (uz * vy);
        double ny = (uz * vx) - (ux * vz);
        double nz = (ux * vy) - (uy * vx);
        double dx = (outside[0] / outsideCount) - (inside[0] / insideCount);
        double dy = (outside[1] / outsideCount) - (inside[1] / insideCount);
        double dz = (outside[2] / outsideCount) - (inside[2] / insideCount);
        return (nx * dx) + (ny * dy) + (nz * dz) >= 0.0;
    }

    private static double[] Interpolate(double[] p0, double v0, double[] p1, double v1, double threshold)
    {
        double t = v1 == v0 ? 0.5 : (threshold - v0) / (v1 - v0);
        return new[] { p0[0] + (t * (p1[0] - p0[0])), p0[1] + (t * (p1[1] - p0[1])), p0[2] + (t * (p1[2] - p0[2])) };
    }
}
