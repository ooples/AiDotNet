using AiDotNet.Clustering.DistanceMetrics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Deep math-correctness integration tests for clustering distance metrics.
/// Tests verify exact hand-calculated values, metric properties (symmetry, triangle
/// inequality, identity), cross-metric relationships, and edge cases for Euclidean,
/// Manhattan, Cosine, Chebyshev, Minkowski, and Mahalanobis distances.
/// </summary>
public class ClusteringDistanceMetricsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;

    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++)
            v[i] = values[i];
        return v;
    }

    private static Matrix<double> MakeMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var m = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = data[i, j];
        return m;
    }

    // ─── Euclidean Distance Tests ────────────────────────────────────────────

    [Fact]
    public void Euclidean_PythagoreanTriple_345()
    {
        // dist((0,0), (3,4)) = sqrt(9+16) = 5
        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var eucl = new EuclideanDistance<double>();
        Assert.Equal(5.0, eucl.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Euclidean_IdenticalVectors_ReturnsZero()
    {
        var a = Vec(3, 7, -2);
        var eucl = new EuclideanDistance<double>();
        Assert.Equal(0.0, eucl.Compute(a, a), Tolerance);
    }

    [Fact]
    public void Euclidean_Symmetric()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 5, 6);

        var eucl = new EuclideanDistance<double>();
        Assert.Equal(eucl.Compute(a, b), eucl.Compute(b, a), Tolerance);
    }

    [Fact]
    public void Euclidean_TriangleInequality()
    {
        var a = Vec(0, 0);
        var b = Vec(3, 4);
        var c = Vec(6, 0);

        var eucl = new EuclideanDistance<double>();
        double ab = eucl.Compute(a, b);
        double bc = eucl.Compute(b, c);
        double ac = eucl.Compute(a, c);

        Assert.True(ac <= ab + bc + 1e-10,
            $"Triangle inequality violated: d(a,c)={ac} > d(a,b)+d(b,c)={ab + bc}");
    }

    [Fact]
    public void Euclidean_Squared_EqualsSquareOfDistance()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var eucl = new EuclideanDistance<double>();
        double dist = eucl.Compute(a, b);
        double distSq = eucl.ComputeSquared(a, b);

        Assert.Equal(dist * dist, distSq, Tolerance);
    }

    [Fact]
    public void Euclidean_HandCalculated_3D()
    {
        // dist((1,2,3), (4,6,3)) = sqrt((3)^2 + (4)^2 + 0^2) = sqrt(9+16) = 5
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var eucl = new EuclideanDistance<double>();
        Assert.Equal(5.0, eucl.Compute(a, b), Tolerance);
    }

    // ─── Manhattan Distance Tests ────────────────────────────────────────────

    [Fact]
    public void Manhattan_HandCalculated()
    {
        // |3-0| + |4-0| = 7
        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var manh = new ManhattanDistance<double>();
        Assert.Equal(7.0, manh.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Manhattan_AlwaysGreaterOrEqualToEuclidean()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var manh = new ManhattanDistance<double>();
        var eucl = new EuclideanDistance<double>();

        double manhDist = manh.Compute(a, b);
        double euclDist = eucl.Compute(a, b);

        Assert.True(manhDist >= euclDist - 1e-10,
            $"Manhattan should be >= Euclidean. Manhattan={manhDist}, Euclidean={euclDist}");
    }

    [Fact]
    public void Manhattan_Symmetric()
    {
        var a = Vec(1, -2, 3);
        var b = Vec(-4, 5, 6);

        var manh = new ManhattanDistance<double>();
        Assert.Equal(manh.Compute(a, b), manh.Compute(b, a), Tolerance);
    }

    [Fact]
    public void Manhattan_IdenticalVectors_ReturnsZero()
    {
        var a = Vec(5, -3, 7);
        var manh = new ManhattanDistance<double>();
        Assert.Equal(0.0, manh.Compute(a, a), Tolerance);
    }

    // ─── Chebyshev Distance Tests ────────────────────────────────────────────

    [Fact]
    public void Chebyshev_HandCalculated()
    {
        // max(|3-0|, |4-0|) = max(3, 4) = 4
        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var cheb = new ChebyshevDistance<double>();
        Assert.Equal(4.0, cheb.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Chebyshev_AlwaysLessOrEqualToManhattan()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var cheb = new ChebyshevDistance<double>();
        var manh = new ManhattanDistance<double>();

        double chebDist = cheb.Compute(a, b);
        double manhDist = manh.Compute(a, b);

        Assert.True(chebDist <= manhDist + 1e-10,
            $"Chebyshev should be <= Manhattan. Chebyshev={chebDist}, Manhattan={manhDist}");
    }

    [Fact]
    public void Chebyshev_AlwaysLessOrEqualToEuclidean()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var cheb = new ChebyshevDistance<double>();
        var eucl = new EuclideanDistance<double>();

        double chebDist = cheb.Compute(a, b);
        double euclDist = eucl.Compute(a, b);

        Assert.True(chebDist <= euclDist + 1e-10,
            $"Chebyshev should be <= Euclidean. Chebyshev={chebDist}, Euclidean={euclDist}");
    }

    [Fact]
    public void Chebyshev_SingleDimension_EqualsAbsDiff()
    {
        var a = Vec(3.0);
        var b = Vec(-7.0);

        var cheb = new ChebyshevDistance<double>();
        Assert.Equal(10.0, cheb.Compute(a, b), Tolerance);
    }

    // ─── Cosine Distance Tests ───────────────────────────────────────────────

    [Fact]
    public void Cosine_ParallelVectors_ReturnsZero()
    {
        // Same direction → similarity = 1, distance = 0
        var a = Vec(1, 2, 3);
        var b = Vec(2, 4, 6);

        var cosine = new CosineDistance<double>();
        Assert.Equal(0.0, cosine.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Cosine_PerpendicularVectors_ReturnsOne()
    {
        // 90 degrees → similarity = 0, distance = 1
        var a = Vec(1, 0);
        var b = Vec(0, 1);

        var cosine = new CosineDistance<double>();
        Assert.Equal(1.0, cosine.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Cosine_OppositeVectors_ReturnsTwo()
    {
        // 180 degrees → similarity = -1, distance = 2
        var a = Vec(1, 0);
        var b = Vec(-1, 0);

        var cosine = new CosineDistance<double>();
        Assert.Equal(2.0, cosine.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Cosine_MagnitudeIndependent()
    {
        // Cosine distance should be the same regardless of vector magnitude
        var a = Vec(1, 2, 3);
        var b = Vec(4, 5, 6);
        var aScaled = Vec(100, 200, 300);
        var bScaled = Vec(400, 500, 600);

        var cosine = new CosineDistance<double>();
        double original = cosine.Compute(a, b);
        double scaled = cosine.Compute(aScaled, bScaled);

        Assert.Equal(original, scaled, Tolerance);
    }

    [Fact]
    public void Cosine_HandCalculated_45Degrees()
    {
        // cos(45°) = 1/sqrt(2) ≈ 0.7071
        // Vectors: (1,0) and (1,1)
        // dot = 1, ||a||=1, ||b||=sqrt(2)
        // similarity = 1/sqrt(2)
        // distance = 1 - 1/sqrt(2) ≈ 0.2929
        var a = Vec(1, 0);
        var b = Vec(1, 1);

        var cosine = new CosineDistance<double>();
        double expected = 1.0 - 1.0 / Math.Sqrt(2);
        Assert.Equal(expected, cosine.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Cosine_Similarity_PlusDistance_EqualsOne()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 5, 6);

        var cosine = new CosineDistance<double>();
        double distance = cosine.Compute(a, b);
        double similarity = cosine.ComputeSimilarity(a, b);

        Assert.Equal(1.0, distance + similarity, Tolerance);
    }

    [Fact]
    public void Cosine_ZeroVector_ReturnsOne()
    {
        var a = Vec(0, 0, 0);
        var b = Vec(1, 2, 3);

        var cosine = new CosineDistance<double>();
        Assert.Equal(1.0, cosine.Compute(a, b), Tolerance);
    }

    // ─── Minkowski Distance Tests ────────────────────────────────────────────

    [Fact]
    public void Minkowski_P1_MatchesManhattan()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var mink = new MinkowskiDistance<double>(p: 1.0);
        var manh = new ManhattanDistance<double>();

        Assert.Equal(manh.Compute(a, b), mink.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Minkowski_P2_MatchesEuclidean()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var mink = new MinkowskiDistance<double>(p: 2.0);
        var eucl = new EuclideanDistance<double>();

        Assert.Equal(eucl.Compute(a, b), mink.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Minkowski_LargeP_ApproachesChebyshev()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var minkLargeP = new MinkowskiDistance<double>(p: 100.0);
        var cheb = new ChebyshevDistance<double>();

        double minkDist = minkLargeP.Compute(a, b);
        double chebDist = cheb.Compute(a, b);

        // For large p, Minkowski should be very close to Chebyshev
        Assert.Equal(chebDist, minkDist, 0.01);
    }

    [Fact]
    public void Minkowski_P3_HandCalculated()
    {
        // L3 distance: (|1-4|^3 + |2-6|^3 + |3-3|^3)^(1/3)
        // = (27 + 64 + 0)^(1/3) = 91^(1/3) ≈ 4.4979
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        var mink = new MinkowskiDistance<double>(p: 3.0);
        double expected = Math.Pow(91, 1.0 / 3.0);
        Assert.Equal(expected, mink.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Minkowski_InvalidP_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new MinkowskiDistance<double>(p: 0.5));
    }

    [Fact]
    public void Minkowski_IncreasingP_DistanceDecreases()
    {
        // For any two points: L1 >= L2 >= ... >= Linf
        var a = Vec(1, 2, 3);
        var b = Vec(4, 6, 3);

        double prevDist = double.MaxValue;
        foreach (double p in new[] { 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0 })
        {
            var mink = new MinkowskiDistance<double>(p: p);
            double dist = mink.Compute(a, b);

            Assert.True(dist <= prevDist + 1e-10,
                $"Minkowski distance should decrease with increasing p. p={p}, dist={dist}, prev={prevDist}");
            prevDist = dist;
        }
    }

    // ─── Mahalanobis Distance Tests ──────────────────────────────────────────

    [Fact]
    public void Mahalanobis_IdentityCovariance_MatchesEuclidean()
    {
        // With identity inverse covariance, Mahalanobis = Euclidean
        var identity = new Matrix<double>(2, 2);
        identity[0, 0] = 1; identity[0, 1] = 0;
        identity[1, 0] = 0; identity[1, 1] = 1;

        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var maha = new MahalanobisDistance<double>(identity);
        var eucl = new EuclideanDistance<double>();

        Assert.Equal(eucl.Compute(a, b), maha.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Mahalanobis_NoCovariance_FallsBackToEuclidean()
    {
        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var maha = new MahalanobisDistance<double>(); // No covariance matrix
        var eucl = new EuclideanDistance<double>();

        Assert.Equal(eucl.Compute(a, b), maha.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Mahalanobis_ScaledCovariance_HandCalculated()
    {
        // Inverse covariance = diag(4, 1) means x-dimension is "4x more important"
        // d^2 = (3-0)^2*4 + (4-0)^2*1 = 36 + 16 = 52
        // d = sqrt(52) ≈ 7.2111
        var invCov = new Matrix<double>(2, 2);
        invCov[0, 0] = 4; invCov[0, 1] = 0;
        invCov[1, 0] = 0; invCov[1, 1] = 1;

        var a = Vec(0, 0);
        var b = Vec(3, 4);

        var maha = new MahalanobisDistance<double>(invCov);
        Assert.Equal(Math.Sqrt(52), maha.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Mahalanobis_Symmetric()
    {
        var invCov = new Matrix<double>(2, 2);
        invCov[0, 0] = 2; invCov[0, 1] = 0.5;
        invCov[1, 0] = 0.5; invCov[1, 1] = 3;

        var a = Vec(1, 2);
        var b = Vec(4, 6);

        var maha = new MahalanobisDistance<double>(invCov);
        Assert.Equal(maha.Compute(a, b), maha.Compute(b, a), Tolerance);
    }

    // ─── Pairwise Distance Tests ─────────────────────────────────────────────

    [Fact]
    public void Euclidean_PairwiseMatrix_Symmetric()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 3, 4 }, { 6, 0 } });

        var eucl = new EuclideanDistance<double>();
        var pairwise = eucl.ComputePairwise(data);

        // Should be symmetric
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(pairwise[i, j], pairwise[j, i], Tolerance);

        // Diagonal should be zero
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, pairwise[i, i], Tolerance);

        // Known values
        Assert.Equal(5.0, pairwise[0, 1], Tolerance); // (0,0)→(3,4) = 5
        Assert.Equal(6.0, pairwise[0, 2], Tolerance); // (0,0)→(6,0) = 6
    }

    [Fact]
    public void Euclidean_ComputeToAll_HandCalculated()
    {
        var point = Vec(0, 0);
        var data = MakeMatrix(new double[,] { { 3, 4 }, { 6, 0 }, { 0, 5 } });

        var eucl = new EuclideanDistance<double>();
        var distances = eucl.ComputeToAll(point, data);

        Assert.Equal(5.0, distances[0], Tolerance); // (0,0)→(3,4) = 5
        Assert.Equal(6.0, distances[1], Tolerance); // (0,0)→(6,0) = 6
        Assert.Equal(5.0, distances[2], Tolerance); // (0,0)→(0,5) = 5
    }

    [Fact]
    public void PairwiseTwoMatrices_HandCalculated()
    {
        var x = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 } });
        var y = MakeMatrix(new double[,] { { 3, 4 }, { 0, 1 } });

        var eucl = new EuclideanDistance<double>();
        var pairwise = eucl.ComputePairwise(x, y);

        // pairwise[i,j] = dist(x[i], y[j])
        Assert.Equal(5.0, pairwise[0, 0], Tolerance);  // (0,0)→(3,4) = 5
        Assert.Equal(1.0, pairwise[0, 1], Tolerance);  // (0,0)→(0,1) = 1
        Assert.Equal(Math.Sqrt(4 + 16), pairwise[1, 0], Tolerance); // (1,0)→(3,4) = sqrt(20)
        Assert.Equal(Math.Sqrt(1 + 1), pairwise[1, 1], Tolerance);  // (1,0)→(0,1) = sqrt(2)
    }

    // ─── Cross-Metric Ordering Tests ─────────────────────────────────────────

    [Fact]
    public void AllMetrics_IdenticalPoints_ReturnZero()
    {
        var a = Vec(3, 7, -2, 5);

        var eucl = new EuclideanDistance<double>();
        var manh = new ManhattanDistance<double>();
        var cheb = new ChebyshevDistance<double>();
        var mink3 = new MinkowskiDistance<double>(p: 3.0);

        Assert.Equal(0.0, eucl.Compute(a, a), Tolerance);
        Assert.Equal(0.0, manh.Compute(a, a), Tolerance);
        Assert.Equal(0.0, cheb.Compute(a, a), Tolerance);
        Assert.Equal(0.0, mink3.Compute(a, a), Tolerance);
    }

    [Fact]
    public void MetricOrdering_Chebyshev_LE_Euclidean_LE_Manhattan()
    {
        // For any pair: Chebyshev <= Euclidean <= Manhattan
        var a = Vec(1, 5, 3, -2);
        var b = Vec(4, 1, -1, 7);

        var eucl = new EuclideanDistance<double>();
        var manh = new ManhattanDistance<double>();
        var cheb = new ChebyshevDistance<double>();

        double euclD = eucl.Compute(a, b);
        double manhD = manh.Compute(a, b);
        double chebD = cheb.Compute(a, b);

        Assert.True(chebD <= euclD + 1e-10, $"Chebyshev({chebD}) should be <= Euclidean({euclD})");
        Assert.True(euclD <= manhD + 1e-10, $"Euclidean({euclD}) should be <= Manhattan({manhD})");
    }

    // ─── Vector Length Mismatch Tests ────────────────────────────────────────

    [Fact]
    public void AllMetrics_DifferentLengths_ThrowsArgumentException()
    {
        var a = Vec(1, 2, 3);
        var b = Vec(4, 5);

        Assert.Throws<ArgumentException>(() => new EuclideanDistance<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new ManhattanDistance<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new ChebyshevDistance<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new CosineDistance<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MinkowskiDistance<double>().Compute(a, b));
    }

    // ─── Non-Negativity Tests ────────────────────────────────────────────────

    [Fact]
    public void AllMetrics_AlwaysNonNegative()
    {
        var pairs = new[]
        {
            (Vec(0, 0), Vec(1, 1)),
            (Vec(-5, 3), Vec(2, -7)),
            (Vec(100, 200, 300), Vec(-100, -200, -300)),
        };

        var eucl = new EuclideanDistance<double>();
        var manh = new ManhattanDistance<double>();
        var cheb = new ChebyshevDistance<double>();
        var cosine = new CosineDistance<double>();

        foreach (var (a, b) in pairs)
        {
            Assert.True(eucl.Compute(a, b) >= 0, $"Euclidean should be non-negative for {a}, {b}");
            Assert.True(manh.Compute(a, b) >= 0, $"Manhattan should be non-negative for {a}, {b}");
            Assert.True(cheb.Compute(a, b) >= 0, $"Chebyshev should be non-negative for {a}, {b}");
            Assert.True(cosine.Compute(a, b) >= -1e-10, $"Cosine should be non-negative for {a}, {b}");
        }
    }

    // ─── High-Dimensional Test ───────────────────────────────────────────────

    [Fact]
    public void Euclidean_HighDimensional_Correct()
    {
        // In d dimensions, dist((1,1,...,1), (0,0,...,0)) = sqrt(d)
        int d = 100;
        var a = new Vector<double>(d);
        var b = new Vector<double>(d);
        for (int i = 0; i < d; i++)
        {
            a[i] = 1.0;
            b[i] = 0.0;
        }

        var eucl = new EuclideanDistance<double>();
        Assert.Equal(Math.Sqrt(d), eucl.Compute(a, b), Tolerance);
    }

    [Fact]
    public void Manhattan_HighDimensional_Correct()
    {
        // In d dimensions, L1 dist((1,1,...,1), (0,0,...,0)) = d
        int d = 100;
        var a = new Vector<double>(d);
        var b = new Vector<double>(d);
        for (int i = 0; i < d; i++)
        {
            a[i] = 1.0;
            b[i] = 0.0;
        }

        var manh = new ManhattanDistance<double>();
        Assert.Equal((double)d, manh.Compute(a, b), Tolerance);
    }

    // ─── Cosine Distance Edge Cases ──────────────────────────────────────────

    [Fact]
    public void Cosine_NearlyParallel_SmallDistance()
    {
        var a = Vec(1, 0, 0);
        var b = Vec(1, 0.001, 0);

        var cosine = new CosineDistance<double>();
        double dist = cosine.Compute(a, b);

        Assert.True(dist < 0.001, $"Nearly parallel vectors should have very small cosine distance. Got {dist}");
    }

    [Fact]
    public void Cosine_RangeIsZeroToTwo()
    {
        // Cosine distance ranges from 0 (identical direction) to 2 (opposite direction)
        var pairs = new[]
        {
            (Vec(1, 0), Vec(1, 0)),     // Same = 0
            (Vec(1, 0), Vec(0, 1)),     // Perpendicular = 1
            (Vec(1, 0), Vec(-1, 0)),    // Opposite = 2
            (Vec(1, 2, 3), Vec(-3, -2, -1)), // Some angle
        };

        var cosine = new CosineDistance<double>();
        foreach (var (a, b) in pairs)
        {
            double dist = cosine.Compute(a, b);
            Assert.InRange(dist, -1e-10, 2.0 + 1e-10);
        }
    }
}
