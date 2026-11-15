using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Interfaces;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Meshing;

/// <summary>
/// Reference tests for the shared meshing components: marching cubes (Lorensen &amp; Cline 1987), screened Poisson
/// (Kazhdan &amp; Hoppe 2013), ball pivoting (Bernardini et al. 1999), Hoppe et al. (1992)'s normals and signed
/// distance, the Delaunay alpha shape, and DreamGaussian's Gaussian density extraction (Tang et al. 2024).
/// </summary>
public class MeshingComponentTests
{
    // ------------------------------------------------------------------ helpers

    private static Tensor<double> SphereField(int n, double radius, double cx = 0, double cy = 0, double cz = 0)
    {
        double step = 2.0 / (n - 1);
        var values = new double[n * n * n];
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++)
            {
                for (int z = 0; z < n; z++)
                {
                    double px = -1 + (x * step) - cx, py = -1 + (y * step) - cy, pz = -1 + (z * step) - cz;
                    values[(((x * n) + y) * n) + z] = Math.Sqrt((px * px) + (py * py) + (pz * pz)) - radius;
                }
            }
        }

        return new Tensor<double>(new[] { n, n, n }, new Vector<double>(values));
    }

    /// <summary>Points on a Fibonacci sphere and their outward normals.</summary>
    private static (double[] Points, double[] Normals) FibonacciSphere(int count, double radius, double ox = 0, double oy = 0, double oz = 0)
    {
        var points = new double[3 * count];
        var normals = new double[3 * count];
        double golden = Math.PI * (1.0 + Math.Sqrt(5.0));
        for (int i = 0; i < count; i++)
        {
            double phi = Math.Acos(1.0 - (2.0 * (i + 0.5) / count));
            double theta = golden * (i + 0.5);
            double nx = Math.Cos(theta) * Math.Sin(phi), ny = Math.Sin(theta) * Math.Sin(phi), nz = Math.Cos(phi);
            normals[3 * i] = nx;
            normals[(3 * i) + 1] = ny;
            normals[(3 * i) + 2] = nz;
            points[3 * i] = ox + (radius * nx);
            points[(3 * i) + 1] = oy + (radius * ny);
            points[(3 * i) + 2] = oz + (radius * nz);
        }

        return (points, normals);
    }

    private static Tensor<double> AsTensor(double[] values, int columns) =>
        new Tensor<double>(new[] { values.Length / columns, columns }, new Vector<double>(values));

    private static MeshSurface ToSurface(Mesh3D<double> mesh)
    {
        double[] coordinates = mesh.Vertices.ToArray();
        int faces = mesh.Faces.GetLength(0);
        var triangles = new int[3 * faces];
        for (int f = 0; f < faces; f++)
        {
            for (int k = 0; k < 3; k++)
                triangles[(3 * f) + k] = mesh.Faces[f, k];
        }

        return new MeshSurface(coordinates, triangles, mesh.Normals?.ToArray());
    }

    private static void AssertClosedAndOriented(MeshSurface surface)
    {
        Assert.True(surface.TriangleCount > 0);
        var (unmatched, nonManifold) = surface.EdgeDefects();
        Assert.Equal(0, unmatched);
        Assert.Equal(0, nonManifold);
        Assert.True(surface.SignedVolume() > 0.0);
    }

    private static double MaxRadiusError(MeshSurface surface, double radius, double ox = 0, double oy = 0, double oz = 0)
    {
        double worst = 0.0;
        for (int v = 0; v < surface.VertexCount; v++)
        {
            double x = surface.Coordinates[3 * v] - ox, y = surface.Coordinates[(3 * v) + 1] - oy, z = surface.Coordinates[(3 * v) + 2] - oz;
            worst = Math.Max(worst, Math.Abs(Math.Sqrt((x * x) + (y * y) + (z * z)) - radius));
        }

        return worst;
    }

    // ------------------------------------------------------------------ marching cubes tables

    [Fact]
    public void MarchingCubesTable_SingleCornerCase_MatchesLorensenCline()
    {
        // Corner 0 inside crosses edges 0, 3 and 8 (Bourke's edgeTable[1] = 0x109, triTable[1] = {0, 8, 3});
        // the triangle here is wound to face away from corner 0.
        Assert.Equal(0x109, MarchingCubes.EdgeTable[1]);
        Assert.Equal(new[] { 0, 3, 8 }, MarchingCubes.TriangleTable[1]);
        Assert.Empty(MarchingCubes.TriangleTable[0]);
        Assert.Empty(MarchingCubes.TriangleTable[255]);
        Assert.Equal(0, MarchingCubes.EdgeTable[0]);
        Assert.Equal(0, MarchingCubes.EdgeTable[255]);
    }

    [Fact]
    public void MarchingCubesTable_EveryCaseUsesExactlyItsCrossedEdges()
    {
        for (int mask = 0; mask < 256; mask++)
        {
            int used = 0;
            foreach (int e in MarchingCubes.TriangleTable[mask])
                used |= 1 << e;
            Assert.Equal(MarchingCubes.EdgeTable[mask], used);
            Assert.Equal(0, MarchingCubes.TriangleTable[mask].Length % 3);

            // Within one case every triangle edge between two crossings is used once in each direction, except the
            // edges on the cell faces, which are used once (they are shared with the neighbouring cell).
            var directed = new Dictionary<(int, int), int>();
            var triangles = MarchingCubes.TriangleTable[mask];
            for (int t = 0; t < triangles.Length; t += 3)
            {
                for (int k = 0; k < 3; k++)
                {
                    var key = (triangles[t + k], triangles[t + ((k + 1) % 3)]);
                    directed[key] = directed.TryGetValue(key, out int c) ? c + 1 : 1;
                }
            }

            Assert.All(directed.Values, count => Assert.Equal(1, count));
        }
    }

    [Fact]
    public void MarchingCubesTable_ComplementCasesShareCrossedEdges()
    {
        for (int mask = 0; mask < 256; mask++)
            Assert.Equal(MarchingCubes.EdgeTable[mask], MarchingCubes.EdgeTable[255 - mask]);
    }

    // ------------------------------------------------------------------ marching cubes extraction

    [Fact]
    public void MarchingCubes_SphereSdf_IsWatertightOnTheRadiusWithRadialNormals()
    {
        const int n = 25;
        const double radius = 0.6;
        double spacing = 2.0 / (n - 1);
        var mesh = MarchingCubes.ExtractInCube(SphereField(n, radius), 0.0, IsoSurfaceInterior.BelowIsoLevel);
        var surface = ToSurface(mesh);

        AssertClosedAndOriented(surface);
        Assert.True(MaxRadiusError(surface, radius) < spacing, "vertex error must be below the grid spacing");
        double exact = 4.0 / 3.0 * Math.PI * radius * radius * radius;
        Assert.InRange(surface.SignedVolume(), 0.95 * exact, 1.0 * exact);

        Assert.NotNull(mesh.Normals);
        var normals = surface.Normals ?? Array.Empty<double>();
        for (int v = 0; v < surface.VertexCount; v++)
        {
            double x = surface.Coordinates[3 * v], y = surface.Coordinates[(3 * v) + 1], z = surface.Coordinates[(3 * v) + 2];
            double r = Math.Sqrt((x * x) + (y * y) + (z * z));
            double dot = ((normals[3 * v] * x) + (normals[(3 * v) + 1] * y) + (normals[(3 * v) + 2] * z)) / r;
            Assert.True(dot > 0.99, $"normal {v} is not radial: {dot}");
        }
    }

    [Fact]
    public void MarchingCubes_DensityConventionMatchesNegatedSdf()
    {
        const int n = 13;
        var sdf = SphereField(n, 0.5, 0.1, -0.05, 0.02);
        var density = new Tensor<double>(new[] { n, n, n }, new Vector<double>(sdf.ToArray().Select(v => -v).ToArray()));
        var a = ToSurface(MarchingCubes.ExtractInCube(sdf, 0.0, IsoSurfaceInterior.BelowIsoLevel));
        var b = ToSurface(MarchingCubes.ExtractInCube(density, 0.0, IsoSurfaceInterior.AboveIsoLevel));

        Assert.Equal(a.Coordinates.Length, b.Coordinates.Length);
        for (int i = 0; i < a.Coordinates.Length; i++)
            Assert.Equal(a.Coordinates[i], b.Coordinates[i], 12);
        Assert.Equal(a.Triangles, b.Triangles);
        AssertClosedAndOriented(b);
    }

    [Fact]
    public void MarchingCubes_LinearField_InterpolatesThePlaneExactly()
    {
        // f = x - 0.3 on the unit cell: the surface is the plane x = 0.3, two triangles facing +x (outside, f > 0).
        var values = new double[8];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                    values[(((x * 2) + y) * 2) + z] = x - 0.3;
        var mesh = MarchingCubes.Extract(new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(values)),
            0.0, IsoSurfaceInterior.BelowIsoLevel, 0, 0, 0, 1.0);
        var surface = ToSurface(mesh);

        Assert.Equal(2, surface.TriangleCount);
        Assert.Equal(4, surface.VertexCount);
        for (int v = 0; v < 4; v++)
            Assert.Equal(0.3, surface.Coordinates[3 * v], 12);
        for (int f = 0; f < 2; f++)
        {
            var (nx, _, _) = surface.FaceCross(surface.Triangles[3 * f], surface.Triangles[(3 * f) + 1], surface.Triangles[(3 * f) + 2]);
            Assert.True(nx > 0.0);
        }
    }

    [Fact]
    public void MarchingCubes_RandomFieldInsideAnOutsideShell_IsClosed()
    {
        // Ambiguous faces abound in a random field; the face rule must keep neighbouring cells crack-free.
        const int n = 12;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        var values = new double[n * n * n];
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                for (int z = 0; z < n; z++)
                {
                    bool border = x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1;
                    values[(((x * n) + y) * n) + z] = border ? -1.0 : (2.0 * random.NextDouble()) - 1.0;
                }

        var surface = MarchingCubes.ExtractSurface(values, n, n, n, 0.0, IsoSurfaceInterior.AboveIsoLevel, 0, 0, 0, 1.0);
        AssertClosedAndOriented(surface);
    }

    [Fact]
    public void MarchingCubes_UndefinedSamplesAreSkipped()
    {
        const int n = 15;
        var full = SphereField(n, 0.6).ToArray();
        var holed = (double[])full.Clone();
        for (int i = 0; i < holed.Length / 2; i++)
            holed[i] = double.NaN;

        var complete = MarchingCubes.ExtractSurface(full, n, n, n, 0.0, IsoSurfaceInterior.BelowIsoLevel, -1, -1, -1, 2.0 / (n - 1));
        var partial = MarchingCubes.ExtractSurface(holed, n, n, n, 0.0, IsoSurfaceInterior.BelowIsoLevel, -1, -1, -1, 2.0 / (n - 1));

        Assert.True(partial.TriangleCount > 0 && partial.TriangleCount < complete.TriangleCount);
        Assert.All(partial.Coordinates, c => Assert.False(double.IsNaN(c)));
        Assert.True(partial.EdgeDefects().UnmatchedEdges > 0, "skipping cells opens the surface");
    }

    [Fact]
    public void MarchingCubes_FloatMatchesDouble()
    {
        var field = SphereField(11, 0.55);
        var asDouble = MarchingCubes.ExtractInCube(field, 0.0, IsoSurfaceInterior.BelowIsoLevel);
        var asFloat = MarchingCubes.ExtractInCube(field.Cast<float>(), 0.0, IsoSurfaceInterior.BelowIsoLevel);

        Assert.Equal(asDouble.Faces, asFloat.Faces);
        var d = asDouble.Vertices.ToArray();
        var f = asFloat.Vertices.ToArray();
        for (int i = 0; i < d.Length; i++)
            Assert.Equal(d[i], f[i], 5);
    }

    // ------------------------------------------------------------------ normals

    [Fact]
    public void Normals_OnSphere_PointOutward()
    {
        var (points, _) = FibonacciSphere(500, 0.7, 0.2, 0.1, -0.3);
        var normals = PointCloudSurfaceReconstruction.EstimateNormals(AsTensor(points, 3), 10).ToArray();
        for (int i = 0; i < 500; i++)
        {
            double rx = points[3 * i] - 0.2, ry = points[(3 * i) + 1] - 0.1, rz = points[(3 * i) + 2] + 0.3;
            double dot = ((normals[3 * i] * rx) + (normals[(3 * i) + 1] * ry) + (normals[(3 * i) + 2] * rz)) / 0.7;
            Assert.True(dot > 0.95, $"normal {i}: {dot}");
        }
    }

    [Fact]
    public void Normals_SmallestEigenvectorOfKnownMatrix()
    {
        // diag(3, 1, 2) rotated: the smallest eigenvalue's eigenvector is the rotated y axis.
        double c = Math.Cos(0.4), s = Math.Sin(0.4);
        double[,] rotation = { { c, -s, 0 }, { s, c, 0 }, { 0, 0, 1 } };
        double[] eigen = { 3, 1, 2 };
        var matrix = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    matrix[i, j] += rotation[i, k] * eigen[k] * rotation[j, k];

        var (x, y, z) = PointCloudNormals.SmallestEigenvector(matrix);
        Assert.Equal(1.0, Math.Abs((x * -s) + (y * c)), 10);
        Assert.Equal(0.0, z, 10);
    }

    // ------------------------------------------------------------------ point cloud reconstruction

    private static SurfaceReconstructionOptions SmallPoisson() =>
        new SurfaceReconstructionOptions { PoissonDepth = 5, PoissonMinimumDepth = 3 };

    [Fact]
    public void Poisson_SphereWithNormals_IsWatertightOnTheRadius()
    {
        var (points, normals) = FibonacciSphere(2000, 0.5, 0.1, -0.2, 0.3);
        var mesh = PointCloudSurfaceReconstruction.Reconstruct(
            AsTensor(points, 3), SurfaceReconstructionMethod.Poisson, SmallPoisson(), AsTensor(normals, 3));
        var surface = ToSurface(mesh);

        AssertClosedAndOriented(surface);
        double h = 1.1 / 32.0;
        Assert.True(MaxRadiusError(surface, 0.5, 0.1, -0.2, 0.3) < 0.5 * h);
        double exact = 4.0 / 3.0 * Math.PI * 0.125;
        Assert.InRange(surface.SignedVolume(), 0.97 * exact, 1.03 * exact);
    }

    [Fact]
    public void Poisson_EstimatedNormals_DeterministicAndInputDependent()
    {
        var (big, _) = FibonacciSphere(1500, 0.5);
        var (small, _) = FibonacciSphere(1500, 0.3);
        var first = ToSurface(PointCloudSurfaceReconstruction.Reconstruct(AsTensor(big, 3), SurfaceReconstructionMethod.Poisson, SmallPoisson()));
        var second = ToSurface(PointCloudSurfaceReconstruction.Reconstruct(AsTensor(big, 3), SurfaceReconstructionMethod.Poisson, SmallPoisson()));
        var other = ToSurface(PointCloudSurfaceReconstruction.Reconstruct(AsTensor(small, 3), SurfaceReconstructionMethod.Poisson, SmallPoisson()));

        AssertClosedAndOriented(first);
        Assert.Equal(first.Coordinates, second.Coordinates);
        Assert.Equal(first.Triangles, second.Triangles);
        Assert.True(MaxRadiusError(first, 0.5) < 1.1 / 32.0);
        Assert.True(MaxRadiusError(other, 0.3) < 0.66 / 32.0);
        Assert.True(other.SignedVolume() < 0.5 * first.SignedVolume());
    }

    [Fact]
    public void BallPivoting_Sphere_InterpolatesEverySampleInAClosedSurface()
    {
        var (points, normals) = FibonacciSphere(800, 0.5);
        var mesh = PointCloudSurfaceReconstruction.Reconstruct(
            AsTensor(points, 3), SurfaceReconstructionMethod.BallPivoting, null, AsTensor(normals, 3));
        var surface = ToSurface(mesh);

        AssertClosedAndOriented(surface);
        Assert.Equal(800, surface.VertexCount);
        // A closed genus-0 triangulation of V vertices has 2V - 4 triangles.
        Assert.Equal((2 * 800) - 4, surface.TriangleCount);
        Assert.True(MaxRadiusError(surface, 0.5) < 1e-12, "ball pivoting interpolates the samples");
    }

    [Fact]
    public void SignedDistanceMarchingCubes_Sphere_IsClosedNearTheRadius()
    {
        var (points, _) = FibonacciSphere(1500, 0.5);
        var options = new SurfaceReconstructionOptions { SignedDistanceResolution = 32 };
        var surface = ToSurface(PointCloudSurfaceReconstruction.Reconstruct(
            AsTensor(points, 3), SurfaceReconstructionMethod.MarchingCubes, options));

        AssertClosedAndOriented(surface);
        Assert.True(MaxRadiusError(surface, 0.5) < 2.0 / 32.0);
    }

    [Fact]
    public void Delaunay_EveryTetrahedronHasAnEmptyCircumsphere()
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var points = new double[3 * 200];
        for (int i = 0; i < points.Length; i++)
            points[i] = (2.0 * random.NextDouble()) - 1.0;

        var delaunay = AlphaShapeReconstruction.Delaunay.Build(points);
        int checkedTets = 0;
        double volume = 0.0;
        for (int t = 0; t < delaunay.TetCount; t++)
        {
            if (!delaunay.IsAlive(t) || delaunay.TouchesSuper(t))
                continue;
            checkedTets++;
            var v = new int[4];
            for (int c = 0; c < 4; c++)
                v[c] = delaunay.Vertex(t, c);
            double orient = delaunay.Orient(v[0], v[1], v[2], v[3]);
            Assert.True(orient > 0.0);
            volume += orient / 6.0;

            // Circumcentre by brute force from the four vertices.
            var (cx, cy, cz, r2) = Circumsphere(points, v);
            for (int p = 0; p < 200; p++)
            {
                if (Array.IndexOf(v, p) >= 0)
                    continue;
                double dx = points[3 * p] - cx, dy = points[(3 * p) + 1] - cy, dz = points[(3 * p) + 2] - cz;
                Assert.True((dx * dx) + (dy * dy) + (dz * dz) >= r2 * (1 - 1e-9), $"point {p} inside tet {t}");
            }
        }

        Assert.True(checkedTets > 500);
        Assert.True(volume > 0.0 && volume < 8.0);
    }

    private static (double X, double Y, double Z, double R2) Circumsphere(double[] p, int[] v)
    {
        // Solve 2 (b - a) . x = |b|^2 - |a|^2 etc. with Gaussian elimination.
        var m = new double[3, 4];
        for (int r = 0; r < 3; r++)
        {
            double sa = 0, sb = 0;
            for (int k = 0; k < 3; k++)
            {
                m[r, k] = 2.0 * (p[(3 * v[r + 1]) + k] - p[(3 * v[0]) + k]);
                sa += p[(3 * v[0]) + k] * p[(3 * v[0]) + k];
                sb += p[(3 * v[r + 1]) + k] * p[(3 * v[r + 1]) + k];
            }

            m[r, 3] = sb - sa;
        }

        for (int col = 0; col < 3; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < 3; r++)
                if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col])) pivot = r;
            for (int k = 0; k < 4; k++)
                (m[col, k], m[pivot, k]) = (m[pivot, k], m[col, k]);
            for (int r = 0; r < 3; r++)
            {
                if (r == col) continue;
                double f = m[r, col] / m[col, col];
                for (int k = 0; k < 4; k++)
                    m[r, k] -= f * m[col, k];
            }
        }

        double x = m[0, 3] / m[0, 0], y = m[1, 3] / m[1, 1], z = m[2, 3] / m[2, 2];
        double dx = x - p[3 * v[0]], dy = y - p[(3 * v[0]) + 1], dz = z - p[(3 * v[0]) + 2];
        return (x, y, z, (dx * dx) + (dy * dy) + (dz * dz));
    }

    [Fact]
    public void AlphaShape_SolidBall_BoundaryEnclosesTheKeptTetrahedra()
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        var list = new List<double>();
        while (list.Count < 3 * 400)
        {
            double x = (2.0 * random.NextDouble()) - 1.0, y = (2.0 * random.NextDouble()) - 1.0, z = (2.0 * random.NextDouble()) - 1.0;
            if ((x * x) + (y * y) + (z * z) < 1.0)
            {
                list.Add(x);
                list.Add(y);
                list.Add(z);
            }
        }

        double[] points = list.ToArray();
        var delaunay = AlphaShapeReconstruction.Delaunay.Build(points);
        foreach (double alpha in new[] { double.PositiveInfinity, 0.4 })
        {
            double keptVolume = 0.0;
            for (int t = 0; t < delaunay.TetCount; t++)
            {
                if (delaunay.IsAlive(t) && !delaunay.TouchesSuper(t) && delaunay.RadiusSquared(t) < alpha * alpha)
                    keptVolume += delaunay.Orient(delaunay.Vertex(t, 0), delaunay.Vertex(t, 1), delaunay.Vertex(t, 2), delaunay.Vertex(t, 3)) / 6.0;
            }

            var surface = AlphaShapeReconstruction.Reconstruct(points, alpha);
            Assert.Equal(0, surface.EdgeDefects().UnmatchedEdges);
            Assert.Equal(keptVolume, surface.SignedVolume(), 9);
        }

        // A smaller alpha carves the shape: less volume than the hull.
        Assert.True(AlphaShapeReconstruction.Reconstruct(points, 0.25).SignedVolume()
            < AlphaShapeReconstruction.Reconstruct(points, double.PositiveInfinity).SignedVolume());
    }

    [Fact]
    public void Reconstruct_RejectsBatchedClouds()
    {
        var cloud = new Tensor<double>(new[] { 2, 10, 3 });
        Assert.Throws<ArgumentException>(() =>
            PointCloudSurfaceReconstruction.Reconstruct(cloud, SurfaceReconstructionMethod.Poisson));
    }

    [Fact]
    public void ThreeDDiffusionBase_PointCloudToMesh_UsesTheRealReconstruction()
    {
        var (points, _) = FibonacciSphere(1500, 0.5);
        var model = new PointEModel<double> { SurfaceReconstruction = SmallPoisson() };
        var cloud = new Tensor<double>(new[] { 1, 1500, 3 }, new Vector<double>(points));
        var surface = ToSurface(model.PointCloudToMesh(cloud, SurfaceReconstructionMethod.Poisson));

        AssertClosedAndOriented(surface);
        Assert.True(MaxRadiusError(surface, 0.5) < 1.1 / 32.0);
    }

    // ------------------------------------------------------------------ Gaussians

    private static GaussianCloud<double> Cloud(double[] positions, double[] scales, double[] rotations, double[] opacities)
    {
        int n = opacities.Length;
        return new GaussianCloud<double>(
            new Tensor<double>(new[] { n, 3 }, new Vector<double>(positions)),
            new Tensor<double>(new[] { n, 3 }, new Vector<double>(scales)),
            new Tensor<double>(new[] { n, 4 }, new Vector<double>(rotations)),
            new Tensor<double>(new[] { n }, new Vector<double>(opacities)));
    }

    [Fact]
    public void GaussianDensity_MatchesTheHandWrittenSum()
    {
        // Two anisotropic Gaussians, one rotated 30 degrees about z; a third below the opacity cull.
        double half = Math.PI / 12.0;
        var cloud = Cloud(
            new[] { 0.0, 0.0, 0.0, 0.3, 0.1, -0.2, 0.5, 0.5, 0.5 },
            new[] { 0.2, 0.1, 0.15, 0.05, 0.25, 0.1, 0.3, 0.3, 0.3 },
            new[] { 1.0, 0, 0, 0, Math.Cos(half), 0, 0, Math.Sin(half), 1, 0, 0, 0 },
            new[] { 0.8, 0.6, 0.004 });
        var options = new GaussianMeshExtractionOptions { Resolution = 16, BlockCount = 4, RelaxRatio = 10.0 };
        var grid = GaussianMeshExtractor.ExtractDensityGrid(cloud, options);
        double[] density = grid.Density.ToArray();

        // Hand reference: d(x) = sum alpha exp(-1/2 (x - mu)^T R S^-2 R^T (x - mu)), culled Gaussian excluded.
        double c = Math.Cos(2 * half), s = Math.Sin(2 * half);
        foreach (var (i, j, k) in new[] { (3, 7, 9), (8, 8, 8), (12, 2, 5), (0, 15, 15) })
        {
            double x = grid.OriginX + (i * grid.Spacing), y = grid.OriginY + (j * grid.Spacing), z = grid.OriginZ + (k * grid.Spacing);
            double first = 0.8 * Math.Exp(-0.5 * ((x * x / 0.04) + (y * y / 0.01) + (z * z / 0.0225)));
            double dx = x - 0.3, dy = y - 0.1, dz = z + 0.2;
            double u = (c * dx) + (s * dy), v = (-s * dx) + (c * dy); // R^T (x - mu)
            double second = 0.6 * Math.Exp(-0.5 * ((u * u / 0.0025) + (v * v / 0.0625) + (dz * dz / 0.01)));
            Assert.Equal(first + second, density[(((i * 16) + j) * 16) + k], 10);
        }

        // The grid spans the normalised [-1, 1] cube: centre of the kept centres, largest side mapped to 1.8.
        double scale = 1.8 / 0.3;
        Assert.Equal(0.15 - (1.0 / scale), grid.OriginX, 12);
        Assert.Equal(2.0 / 15.0 / scale, grid.Spacing, 12);
    }

    [Fact]
    public void GaussianMesh_SingleIsotropicGaussian_IsTheAnalyticLevelSphere()
    {
        // alpha exp(-r^2 / (2 sigma^2)) = tau  =>  r = sigma sqrt(2 ln(alpha / tau)).
        const double sigma = 0.3, alpha = 0.9, tau = 0.5;
        var cloud = Cloud(new[] { 0.2, -0.1, 0.4 }, new[] { sigma, sigma, sigma }, new[] { 1.0, 0, 0, 0 }, new[] { alpha });
        var options = new GaussianMeshExtractionOptions { Resolution = 32, BlockCount = 4, DensityThreshold = tau };
        var surface = ToSurface(GaussianMeshExtractor.ExtractMesh(cloud, options));

        double radius = sigma * Math.Sqrt(2.0 * Math.Log(alpha / tau));
        AssertClosedAndOriented(surface);
        Assert.True(MaxRadiusError(surface, radius, 0.2, -0.1, 0.4) < 2.0 / 31.0);

        // DreamGaussian's default threshold of 1 is above this Gaussian's peak: no surface.
        var none = GaussianMeshExtractor.ExtractMesh(cloud, new GaussianMeshExtractionOptions { Resolution = 32, BlockCount = 4 });
        Assert.Equal(0, none.Faces.GetLength(0));
    }

    [Fact]
    public void GaussianSplatting_ExportsActivatedGaussians()
    {
        var splatting = new GaussianSplatting<double>();
        var cloud = splatting.ToGaussianCloud();
        Assert.Equal(splatting.GaussianCount, cloud.Count);
        var diagnostics = splatting.GetGaussianDiagnostics();
        double[] opacities = cloud.Opacities.ToArray();
        double[] positions = cloud.Positions.ToArray();
        for (int i = 0; i < cloud.Count; i++)
        {
            Assert.Equal(diagnostics[i].Opacity, opacities[i], 12);
            Assert.Equal(diagnostics[i].PosX, positions[3 * i], 12);
        }

        Assert.All(cloud.Scales.ToArray(), s => Assert.True(s >= 1e-6));
    }
}
