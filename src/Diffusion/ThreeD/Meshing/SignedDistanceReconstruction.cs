namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Surface reconstruction by marching cubes over the signed distance to the nearest tangent plane (Hoppe, DeRose,
/// Duchamp, McDonald &amp; Stuetzle, "Surface Reconstruction from Unorganized Points", SIGGRAPH 1992, section 3.4).
/// </summary>
/// <remarks>
/// <para>
/// Each sample i carries an oriented tangent plane (centre o_i, outward unit normal n_i). At a grid point x the signed
/// distance is <c>f(x) = (x - o_i) . n_i</c> for the plane whose centre is nearest to x. When the projection
/// <c>z = x - f(x) n_i</c> lies farther than rho + delta from every sample, x is not near the sampled surface and f is
/// left undefined; marching cubes skips the cells touching undefined points, so holes in the data stay holes rather
/// than being bridged. The surface is the zero level set, with f negative inside.
/// </para>
/// </remarks>
internal static class SignedDistanceReconstruction
{
    internal static MeshSurface Reconstruct(
        double[] points, double[] normals, double[] centers, double supportRadius, int resolution)
    {
        if (resolution < 2)
            throw new ArgumentOutOfRangeException(nameof(resolution), "The grid needs at least two cells.");
        if (!(supportRadius > 0.0) || double.IsInfinity(supportRadius))
            throw new ArgumentOutOfRangeException(nameof(supportRadius), "The support radius must be positive.");

        var (_, size) = ScreenedPoissonReconstruction.BoundingCube(points, 1.0);
        double spacing = size / resolution;
        // One support radius of margin around the box, so the surface closes inside the grid.
        int margin = (int)Math.Ceiling(supportRadius / spacing) + 1;
        var min = new double[3];
        var counts = new int[3];
        for (int axis = 0; axis < 3; axis++)
        {
            double low = double.MaxValue, high = double.MinValue;
            for (int i = axis; i < points.Length; i += 3)
            {
                low = Math.Min(low, points[i]);
                high = Math.Max(high, points[i]);
            }

            min[axis] = low - (margin * spacing);
            counts[axis] = (int)Math.Ceiling((high - low) / spacing) + (2 * margin) + 1;
        }

        double cellHint = spacing;
        var centerGrid = new PointGrid(centers, cellHint);
        var sampleGrid = new PointGrid(points, cellHint);
        int nx = counts[0], ny = counts[1], nz = counts[2];
        var values = new double[checked(nx * ny * nz)];
        double supportSquared = supportRadius * supportRadius;
        System.Threading.Tasks.Parallel.For(0, nx, ix =>
        {
            for (int iy = 0; iy < ny; iy++)
            {
                for (int iz = 0; iz < nz; iz++)
                {
                    double x = min[0] + (ix * spacing), y = min[1] + (iy * spacing), z = min[2] + (iz * spacing);
                    int[] nearest = centerGrid.Nearest(x, y, z, 1, -1, out _);
                    int i = nearest[0];
                    double f = ((x - centers[3 * i]) * normals[3 * i]) + ((y - centers[(3 * i) + 1]) * normals[(3 * i) + 1])
                        + ((z - centers[(3 * i) + 2]) * normals[(3 * i) + 2]);
                    double px = x - (f * normals[3 * i]), py = y - (f * normals[(3 * i) + 1]), pz = z - (f * normals[(3 * i) + 2]);
                    sampleGrid.Nearest(px, py, pz, 1, -1, out double[] distanceSquared);
                    values[(((ix * ny) + iy) * nz) + iz] = distanceSquared[0] <= supportSquared ? f : double.NaN;
                }
            }
        });

        return MarchingCubes.ExtractSurface(values, nx, ny, nz, 0.0, IsoSurfaceInterior.BelowIsoLevel,
            min[0], min[1], min[2], spacing);
    }
}
