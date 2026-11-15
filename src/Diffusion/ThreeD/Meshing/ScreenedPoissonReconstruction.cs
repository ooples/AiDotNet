using System.Threading.Tasks;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Screened Poisson surface reconstruction (Kazhdan &amp; Hoppe, "Screened Poisson Surface Reconstruction", ACM TOG
/// 2013).
/// </summary>
/// <remarks>
/// <para>
/// The surface is the level set of an indicator function chi whose gradient best matches the field V of the inward
/// normals of the oriented samples, while chi is pulled to zero at the samples themselves (eq. 4):
/// </para>
/// <code>
/// E(chi) = integral |grad chi - V|^2 + (alpha * Area(P) / |P|) * sum_p chi(p)^2
/// </code>
/// <para>
/// The points are mapped into the unit cube (the bounding box scaled by <see cref="SurfaceReconstructionOptions.PoissonScale"/>),
/// chi lives on the nodes of a 2^depth grid with trilinear elements, the gradient term is discretised on the grid edges
/// (V splatted onto the edge midpoints, each sample weighted by its area estimate), and the screening term evaluates
/// chi at the samples by trilinear interpolation. The normal equations
/// <c>(D^T D + (beta / h) B^T B) chi = h D^T V</c> are symmetric positive definite and are solved by Jacobi-preconditioned
/// conjugate gradients, cascading from a coarse depth to the target depth as in the paper's multigrid (section 5).
/// The iso-value is the mean of chi at the samples (section 4.3 of Kazhdan et al. 2006, kept in the 2013 paper) and the
/// surface is extracted by marching cubes.
/// </para>
/// <para>
/// Differences from the reference implementation: the grid is dense rather than an adaptive octree, and the elements
/// are trilinear (degree 1) rather than the degree-2 B-splines of PoissonRecon. Both change cost and smoothness, not the
/// energy that is minimised. Sample areas come from the k-nearest-neighbour density, pi * d_k^2 / (k + 1).
/// </para>
/// </remarks>
internal static class ScreenedPoissonReconstruction
{
    internal static MeshSurface Reconstruct(
        double[] points, double[] normals, double[] areas, SurfaceReconstructionOptions options)
    {
        int count = points.Length / 3;
        if (count < 4)
            throw new ArgumentException("Poisson reconstruction needs at least four points.", nameof(points));
        if (options.PoissonDepth < 2 || options.PoissonDepth > 10)
            throw new ArgumentOutOfRangeException(nameof(options), "PoissonDepth must be between 2 and 10.");
        if (!(options.PoissonScale >= 1.0))
            throw new ArgumentOutOfRangeException(nameof(options), "PoissonScale must be at least 1.");
        if (options.PoissonPointWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(options), "PoissonPointWeight must be non-negative.");

        var (center, size) = BoundingCube(points, options.PoissonScale);
        // Samples in unit-cube coordinates [0, 1]^3 and areas in unit-cube area units.
        var unit = new double[points.Length];
        double totalArea = 0.0;
        var unitAreas = new double[count];
        for (int i = 0; i < count; i++)
        {
            for (int axis = 0; axis < 3; axis++)
                unit[(3 * i) + axis] = ((points[(3 * i) + axis] - center[axis]) / size) + 0.5;
            unitAreas[i] = areas[i] / (size * size);
            totalArea += unitAreas[i];
        }

        double beta = options.PoissonPointWeight * totalArea / count;
        int minimumDepth = Math.Max(2, Math.Min(options.PoissonMinimumDepth, options.PoissonDepth));
        float[]? chi = null;
        int previousResolution = 0;
        for (int depth = minimumDepth; depth <= options.PoissonDepth; depth++)
        {
            int resolution = 1 << depth;
            var initial = chi is null ? new float[Cube(resolution + 1)] : Prolong(chi, previousResolution);
            chi = SolveLevel(unit, normals, unitAreas, beta, resolution, initial, options);
            previousResolution = resolution;
        }

        if (chi is null)
            throw new InvalidOperationException("The Poisson solve produced no indicator function.");

        int finest = previousResolution;
        int nodes = finest + 1;
        double h = 1.0 / finest;
        double iso = 0.0;
        for (int i = 0; i < count; i++)
            iso += Interpolate(chi, nodes, unit[3 * i] / h, unit[(3 * i) + 1] / h, unit[(3 * i) + 2] / h);
        iso /= count;

        var values = new double[chi.Length];
        for (int i = 0; i < chi.Length; i++)
            values[i] = chi[i];
        double spacing = size / finest;
        return MarchingCubes.ExtractSurface(values, nodes, nodes, nodes, iso, IsoSurfaceInterior.AboveIsoLevel,
            center[0] - (size / 2.0), center[1] - (size / 2.0), center[2] - (size / 2.0), spacing);
    }

    /// <summary>The centre and side of the cube that holds the bounding box scaled by <paramref name="scale"/>.</summary>
    internal static (double[] Center, double Size) BoundingCube(double[] points, double scale)
    {
        var min = new[] { double.MaxValue, double.MaxValue, double.MaxValue };
        var max = new[] { double.MinValue, double.MinValue, double.MinValue };
        for (int i = 0; i < points.Length; i += 3)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                min[axis] = Math.Min(min[axis], points[i + axis]);
                max[axis] = Math.Max(max[axis], points[i + axis]);
            }
        }

        double extent = Math.Max(max[0] - min[0], Math.Max(max[1] - min[1], max[2] - min[2]));
        if (!(extent > 0.0) || double.IsInfinity(extent))
            throw new ArgumentException("The points must span a non-empty, finite volume.", nameof(points));

        var center = new[] { 0.5 * (min[0] + max[0]), 0.5 * (min[1] + max[1]), 0.5 * (min[2] + max[2]) };
        return (center, extent * scale);
    }

    /// <summary>
    /// Assembles and solves one level of the screened Poisson system on a grid of <paramref name="resolution"/> cells.
    /// </summary>
    private static float[] SolveLevel(
        double[] unit, double[] normals, double[] areas, double beta, int resolution, float[] chi,
        SurfaceReconstructionOptions options)
    {
        int nodes = resolution + 1;
        int total = Cube(nodes);
        double h = 1.0 / resolution;
        double gamma = beta / h;
        int count = areas.Length;

        // Right-hand side h D^T V, with V = -sum_p A_p N_p delta_p (the indicator is 1 inside, so its gradient is the
        // inward normal) splatted onto the edge midpoints: h V_e = -A_p N_p,axis w_e(p) / h^2.
        var rhs = new double[total];
        for (int i = 0; i < count; i++)
        {
            double gx = unit[3 * i] / h, gy = unit[(3 * i) + 1] / h, gz = unit[(3 * i) + 2] / h;
            double scaleFactor = -areas[i] / (h * h);
            for (int axis = 0; axis < 3; axis++)
            {
                double component = scaleFactor * normals[(3 * i) + axis];
                if (component == 0.0)
                    continue;

                // Edge midpoints along this axis sit at half-integer coordinates on it.
                double ex = axis == 0 ? gx - 0.5 : gx, ey = axis == 1 ? gy - 0.5 : gy, ez = axis == 2 ? gz - 0.5 : gz;
                int lx = axis == 0 ? resolution - 1 : resolution, ly = axis == 1 ? resolution - 1 : resolution;
                int lz = axis == 2 ? resolution - 1 : resolution;
                int x0 = ClampBase(ex, lx), y0 = ClampBase(ey, ly), z0 = ClampBase(ez, lz);
                double tx = Clamp01(ex - x0), ty = Clamp01(ey - y0), tz = Clamp01(ez - z0);
                for (int c = 0; c < 8; c++)
                {
                    int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
                    double w = (dx == 1 ? tx : 1.0 - tx) * (dy == 1 ? ty : 1.0 - ty) * (dz == 1 ? tz : 1.0 - tz);
                    if (w == 0.0)
                        continue;

                    int ix = Math.Min(x0 + dx, lx), iy = Math.Min(y0 + dy, ly), iz = Math.Min(z0 + dz, lz);
                    int low = Index(ix, iy, iz, nodes);
                    int high = Index(ix + (axis == 0 ? 1 : 0), iy + (axis == 1 ? 1 : 0), iz + (axis == 2 ? 1 : 0), nodes);
                    double u = component * w;
                    rhs[low] -= u;
                    rhs[high] += u;
                }
            }
        }

        // Screening: chi(p) = sum_c w_c chi[node_c], the trilinear interpolation over the sample's cell.
        var sampleBase = new int[count];
        var sampleWeights = new double[count * 8];
        for (int i = 0; i < count; i++)
        {
            double gx = unit[3 * i] / h, gy = unit[(3 * i) + 1] / h, gz = unit[(3 * i) + 2] / h;
            int x0 = ClampBase(gx, resolution), y0 = ClampBase(gy, resolution), z0 = ClampBase(gz, resolution);
            double tx = Clamp01(gx - x0), ty = Clamp01(gy - y0), tz = Clamp01(gz - z0);
            sampleBase[i] = Index(x0, y0, z0, nodes);
            for (int c = 0; c < 8; c++)
            {
                int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
                sampleWeights[(8 * i) + c] = (dx == 1 ? tx : 1.0 - tx) * (dy == 1 ? ty : 1.0 - ty) * (dz == 1 ? tz : 1.0 - tz);
            }
        }

        var cornerOffset = new int[8];
        for (int c = 0; c < 8; c++)
            cornerOffset[c] = Index(c & 1, (c >> 1) & 1, (c >> 2) & 1, nodes);

        var diagonal = new float[total];
        Parallel.For(0, nodes, x =>
        {
            for (int y = 0; y < nodes; y++)
            {
                for (int z = 0; z < nodes; z++)
                {
                    int degree = (x > 0 ? 1 : 0) + (x < nodes - 1 ? 1 : 0) + (y > 0 ? 1 : 0) + (y < nodes - 1 ? 1 : 0)
                        + (z > 0 ? 1 : 0) + (z < nodes - 1 ? 1 : 0);
                    diagonal[Index(x, y, z, nodes)] = degree;
                }
            }
        });
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 8; c++)
            {
                double w = sampleWeights[(8 * i) + c];
                diagonal[sampleBase[i] + cornerOffset[c]] += (float)(gamma * w * w);
            }
        }

        var residual = new float[total];
        var direction = new float[total];
        var product = new float[total];
        var sampleValues = new double[count];

        Apply(chi, product, nodes, gamma, sampleBase, sampleWeights, cornerOffset, sampleValues);
        for (int i = 0; i < total; i++)
            residual[i] = (float)(rhs[i] - product[i]);
        double rhsNorm = Math.Sqrt(SumOfSquares(rhs));
        rhs = Array.Empty<double>();
        if (rhsNorm == 0.0)
            return chi;

        for (int i = 0; i < total; i++)
            direction[i] = residual[i] / diagonal[i];
        double rz = PreconditionedNorm(residual, diagonal, nodes);
        for (int iteration = 0; iteration < options.PoissonSolverIterations; iteration++)
        {
            Apply(direction, product, nodes, gamma, sampleBase, sampleWeights, cornerOffset, sampleValues);
            double curvature = Dot(direction, product, nodes);
            if (!(curvature > 0.0))
                break;

            double step = rz / curvature;
            float stepF = (float)step;
            Parallel.For(0, nodes, x =>
            {
                int start = x * nodes * nodes, end = start + (nodes * nodes);
                for (int i = start; i < end; i++)
                {
                    chi[i] += stepF * direction[i];
                    residual[i] -= stepF * product[i];
                }
            });

            double residualNorm = Math.Sqrt(Dot(residual, residual, nodes));
            if (residualNorm <= options.PoissonSolverTolerance * rhsNorm)
                break;

            double rzNext = PreconditionedNorm(residual, diagonal, nodes);
            float momentum = (float)(rzNext / rz);
            rz = rzNext;
            Parallel.For(0, nodes, x =>
            {
                int start = x * nodes * nodes, end = start + (nodes * nodes);
                for (int i = start; i < end; i++)
                    direction[i] = (residual[i] / diagonal[i]) + (momentum * direction[i]);
            });
        }

        return chi;
    }

    /// <summary>
    /// output = (D^T D + gamma B^T B) input: the 7-point graph Laplacian with natural (Neumann) boundaries plus the
    /// screening term over the samples.
    /// </summary>
    private static void Apply(
        float[] input, float[] output, int nodes, double gamma, int[] sampleBase, double[] sampleWeights,
        int[] cornerOffset, double[] sampleValues)
    {
        int strideX = nodes * nodes;
        Parallel.For(0, nodes, x =>
        {
            for (int y = 0; y < nodes; y++)
            {
                for (int z = 0; z < nodes; z++)
                {
                    int i = (x * strideX) + (y * nodes) + z;
                    double center = input[i];
                    double sum = 0.0;
                    if (x > 0) sum += center - input[i - strideX];
                    if (x < nodes - 1) sum += center - input[i + strideX];
                    if (y > 0) sum += center - input[i - nodes];
                    if (y < nodes - 1) sum += center - input[i + nodes];
                    if (z > 0) sum += center - input[i - 1];
                    if (z < nodes - 1) sum += center - input[i + 1];
                    output[i] = (float)sum;
                }
            }
        });

        if (gamma == 0.0)
            return;

        for (int p = 0; p < sampleBase.Length; p++)
        {
            double value = 0.0;
            for (int c = 0; c < 8; c++)
                value += sampleWeights[(8 * p) + c] * input[sampleBase[p] + cornerOffset[c]];
            sampleValues[p] = value;
        }

        for (int p = 0; p < sampleBase.Length; p++)
        {
            double scaled = gamma * sampleValues[p];
            for (int c = 0; c < 8; c++)
                output[sampleBase[p] + cornerOffset[c]] += (float)(scaled * sampleWeights[(8 * p) + c]);
        }
    }

    /// <summary>Trilinear prolongation of a node grid of <paramref name="coarseResolution"/> cells to twice the cells.</summary>
    private static float[] Prolong(float[] coarse, int coarseResolution)
    {
        int coarseNodes = coarseResolution + 1;
        int fineNodes = (2 * coarseResolution) + 1;
        var fine = new float[Cube(fineNodes)];
        Parallel.For(0, fineNodes, x =>
        {
            for (int y = 0; y < fineNodes; y++)
            {
                for (int z = 0; z < fineNodes; z++)
                    fine[Index(x, y, z, fineNodes)] = (float)Interpolate(coarse, coarseNodes, x * 0.5, y * 0.5, z * 0.5);
            }
        });
        return fine;
    }

    /// <summary>Trilinear interpolation of a node grid at grid coordinates (gx, gy, gz).</summary>
    internal static double Interpolate(float[] values, int nodes, double gx, double gy, double gz)
    {
        int resolution = nodes - 1;
        int x0 = ClampBase(gx, resolution), y0 = ClampBase(gy, resolution), z0 = ClampBase(gz, resolution);
        double tx = Clamp01(gx - x0), ty = Clamp01(gy - y0), tz = Clamp01(gz - z0);
        double result = 0.0;
        for (int c = 0; c < 8; c++)
        {
            int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
            double w = (dx == 1 ? tx : 1.0 - tx) * (dy == 1 ? ty : 1.0 - ty) * (dz == 1 ? tz : 1.0 - tz);
            if (w != 0.0)
                result += w * values[Index(x0 + dx, y0 + dy, z0 + dz, nodes)];
        }

        return result;
    }

    /// <summary>The lower lattice index of a cell containing coordinate g, for a lattice of cells [0, cells).</summary>
    private static int ClampBase(double g, int cells)
    {
        int i = (int)Math.Floor(g);
        return i < 0 ? 0 : i > cells - 1 ? cells - 1 : i;
    }

    private static double Clamp01(double t) => t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;

    private static int Index(int x, int y, int z, int nodes) => (((x * nodes) + y) * nodes) + z;

    private static int Cube(int n) => checked(n * n * n);

    private static double SumOfSquares(double[] values)
    {
        double sum = 0.0;
        foreach (double v in values)
            sum += v * v;
        return sum;
    }

    /// <summary>A deterministic parallel dot product: per-slice partial sums added in slice order.</summary>
    private static double Dot(float[] a, float[] b, int nodes)
    {
        var partial = new double[nodes];
        Parallel.For(0, nodes, x =>
        {
            int start = x * nodes * nodes, end = start + (nodes * nodes);
            double sum = 0.0;
            for (int i = start; i < end; i++)
                sum += (double)a[i] * b[i];
            partial[x] = sum;
        });
        double total = 0.0;
        foreach (double p in partial)
            total += p;
        return total;
    }

    private static double PreconditionedNorm(float[] residual, float[] diagonal, int nodes)
    {
        var partial = new double[nodes];
        Parallel.For(0, nodes, x =>
        {
            int start = x * nodes * nodes, end = start + (nodes * nodes);
            double sum = 0.0;
            for (int i = start; i < end; i++)
                sum += (double)residual[i] * residual[i] / diagonal[i];
            partial[x] = sum;
        });
        double total = 0.0;
        foreach (double p in partial)
            total += p;
        return total;
    }
}
