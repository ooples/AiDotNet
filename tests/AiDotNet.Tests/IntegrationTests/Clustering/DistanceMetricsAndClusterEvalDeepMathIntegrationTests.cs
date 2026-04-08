using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Deep math-correctness integration tests for clustering distance metrics
/// (Euclidean, Manhattan, Chebyshev, Cosine, Minkowski) and cluster evaluation
/// metrics (Silhouette, Davies-Bouldin). Verifies hand-calculated values,
/// metric axioms, and mathematical identities.
/// </summary>
public class DistanceMetricsAndClusterEvalDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;

    #region Euclidean Distance

    [Fact]
    public void Euclidean_HandCalculated_3D()
    {
        // d((1,2,3),(4,6,3)) = sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9+16+0) = 5
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 3 });

        double dist = euclidean.Compute(a, b);
        Assert.Equal(5.0, dist, Tolerance);
    }

    [Fact]
    public void Euclidean_HandCalculated_2D()
    {
        // d((0,0),(3,4)) = sqrt(9+16) = 5
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 3, 4 });

        double dist = euclidean.Compute(a, b);
        Assert.Equal(5.0, dist, Tolerance);
    }

    [Fact]
    public void Euclidean_IdenticalVectors_IsZero()
    {
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3, 4 });

        double dist = euclidean.Compute(a, a);
        Assert.Equal(0.0, dist, Tolerance);
    }

    [Fact]
    public void Euclidean_Symmetric()
    {
        // d(a,b) = d(b,a)
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double distAB = euclidean.Compute(a, b);
        double distBA = euclidean.Compute(b, a);
        Assert.Equal(distAB, distBA, Tolerance);
    }

    [Fact]
    public void Euclidean_TriangleInequality()
    {
        // d(a,c) <= d(a,b) + d(b,c)
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 1, 1 });
        var c = new Vector<double>(new double[] { 3, 0 });

        double dAC = euclidean.Compute(a, c);
        double dAB = euclidean.Compute(a, b);
        double dBC = euclidean.Compute(b, c);

        Assert.True(dAC <= dAB + dBC + Tolerance,
            $"Triangle inequality violated: d(a,c)={dAC} > d(a,b)+d(b,c)={dAB + dBC}");
    }

    [Fact]
    public void Euclidean_NonNegative()
    {
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { -5, 3, -1 });
        var b = new Vector<double>(new double[] { 2, -4, 6 });

        double dist = euclidean.Compute(a, b);
        Assert.True(dist >= 0, $"Distance should be non-negative, got {dist}");
    }

    [Fact]
    public void Euclidean_Squared_IsSquareOfDistance()
    {
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 3 });

        double dist = euclidean.Compute(a, b);
        double distSquared = euclidean.ComputeSquared(a, b);
        Assert.Equal(dist * dist, distSquared, Tolerance);
    }

    [Fact]
    public void Euclidean_UnitVector_Equals1()
    {
        // d(origin, unit vector) = 1
        var euclidean = new EuclideanDistance<double>();
        var origin = new Vector<double>(new double[] { 0, 0, 0 });
        var unit = new Vector<double>(new double[] { 1.0 / Math.Sqrt(3), 1.0 / Math.Sqrt(3), 1.0 / Math.Sqrt(3) });

        double dist = euclidean.Compute(origin, unit);
        Assert.Equal(1.0, dist, LooseTolerance);
    }

    [Fact]
    public void Euclidean_MismatchedLengths_Throws()
    {
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2 });
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        Assert.Throws<ArgumentException>(() => euclidean.Compute(a, b));
    }

    #endregion

    #region Manhattan Distance

    [Fact]
    public void Manhattan_HandCalculated_2D()
    {
        // d((0,0),(3,4)) = |3| + |4| = 7
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 3, 4 });

        double dist = manhattan.Compute(a, b);
        Assert.Equal(7.0, dist, Tolerance);
    }

    [Fact]
    public void Manhattan_HandCalculated_3D()
    {
        // d((1,2,3),(4,6,1)) = |3| + |4| + |2| = 9
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 1 });

        double dist = manhattan.Compute(a, b);
        Assert.Equal(9.0, dist, Tolerance);
    }

    [Fact]
    public void Manhattan_IdenticalVectors_IsZero()
    {
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });

        double dist = manhattan.Compute(a, a);
        Assert.Equal(0.0, dist, Tolerance);
    }

    [Fact]
    public void Manhattan_Symmetric()
    {
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { -1, 3, 7 });
        var b = new Vector<double>(new double[] { 4, -2, 1 });

        Assert.Equal(manhattan.Compute(a, b), manhattan.Compute(b, a), Tolerance);
    }

    [Fact]
    public void Manhattan_TriangleInequality()
    {
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 1, 1 });
        var c = new Vector<double>(new double[] { 3, 0 });

        double dAC = manhattan.Compute(a, c);
        double dAB = manhattan.Compute(a, b);
        double dBC = manhattan.Compute(b, c);

        Assert.True(dAC <= dAB + dBC + Tolerance);
    }

    [Fact]
    public void Manhattan_GreaterThanOrEqualEuclidean()
    {
        // Manhattan >= Euclidean always (by Cauchy-Schwarz inequality)
        var manhattan = new ManhattanDistance<double>();
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dMan = manhattan.Compute(a, b);
        double dEuc = euclidean.Compute(a, b);

        Assert.True(dMan >= dEuc - Tolerance,
            $"Manhattan ({dMan}) should be >= Euclidean ({dEuc})");
    }

    #endregion

    #region Chebyshev Distance

    [Fact]
    public void Chebyshev_HandCalculated_2D()
    {
        // d((0,0),(3,4)) = max(|3|, |4|) = 4
        var chebyshev = new ChebyshevDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 3, 4 });

        double dist = chebyshev.Compute(a, b);
        Assert.Equal(4.0, dist, Tolerance);
    }

    [Fact]
    public void Chebyshev_HandCalculated_3D()
    {
        // d((1,2,3),(4,6,1)) = max(|3|, |4|, |2|) = 4
        var chebyshev = new ChebyshevDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 1 });

        double dist = chebyshev.Compute(a, b);
        Assert.Equal(4.0, dist, Tolerance);
    }

    [Fact]
    public void Chebyshev_IdenticalVectors_IsZero()
    {
        var chebyshev = new ChebyshevDistance<double>();
        var a = new Vector<double>(new double[] { 5, -3, 7 });

        Assert.Equal(0.0, chebyshev.Compute(a, a), Tolerance);
    }

    [Fact]
    public void Chebyshev_LessThanOrEqualManhattan()
    {
        // Chebyshev <= Manhattan always
        var chebyshev = new ChebyshevDistance<double>();
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dCheb = chebyshev.Compute(a, b);
        double dMan = manhattan.Compute(a, b);

        Assert.True(dCheb <= dMan + Tolerance,
            $"Chebyshev ({dCheb}) should be <= Manhattan ({dMan})");
    }

    [Fact]
    public void Chebyshev_LessThanOrEqualEuclidean()
    {
        // Chebyshev <= Euclidean always
        var chebyshev = new ChebyshevDistance<double>();
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dCheb = chebyshev.Compute(a, b);
        double dEuc = euclidean.Compute(a, b);

        Assert.True(dCheb <= dEuc + Tolerance,
            $"Chebyshev ({dCheb}) should be <= Euclidean ({dEuc})");
    }

    [Fact]
    public void Chebyshev_EmptyVectors_IsZero()
    {
        var chebyshev = new ChebyshevDistance<double>();
        var a = new Vector<double>(0);
        var b = new Vector<double>(0);

        Assert.Equal(0.0, chebyshev.Compute(a, b), Tolerance);
    }

    #endregion

    #region Cosine Distance

    [Fact]
    public void Cosine_IdenticalVectors_IsZero()
    {
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });

        double dist = cosine.Compute(a, a);
        Assert.Equal(0.0, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_OrthogonalVectors_IsOne()
    {
        // cos(90 degrees) = 0, distance = 1 - 0 = 1
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 0 });
        var b = new Vector<double>(new double[] { 0, 1 });

        double dist = cosine.Compute(a, b);
        Assert.Equal(1.0, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_OppositeVectors_IsTwo()
    {
        // cos(180 degrees) = -1, distance = 1 - (-1) = 2
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 0 });
        var b = new Vector<double>(new double[] { -1, 0 });

        double dist = cosine.Compute(a, b);
        Assert.Equal(2.0, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_ParallelVectors_IsZero()
    {
        // Vectors in same direction: distance = 0
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 2, 4, 6 });

        double dist = cosine.Compute(a, b);
        Assert.Equal(0.0, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_ScaleInvariant()
    {
        // d(a, b) = d(c*a, b) for any c > 0
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 5, 6 });
        var scaledA = new Vector<double>(new double[] { 10, 20, 30 });

        double dist1 = cosine.Compute(a, b);
        double dist2 = cosine.Compute(scaledA, b);

        Assert.Equal(dist1, dist2, LooseTolerance);
    }

    [Fact]
    public void Cosine_HandCalculated_45Degrees()
    {
        // a=(1,0), b=(1,1): cos = 1/sqrt(2), distance = 1 - 1/sqrt(2)
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 0 });
        var b = new Vector<double>(new double[] { 1, 1 });

        double expected = 1.0 - 1.0 / Math.Sqrt(2);
        double dist = cosine.Compute(a, b);
        Assert.Equal(expected, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_Bounded_ZeroToTwo()
    {
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, -2, 3, -4 });
        var b = new Vector<double>(new double[] { -3, 2, 1, -5 });

        double dist = cosine.Compute(a, b);
        Assert.True(dist >= -Tolerance && dist <= 2.0 + Tolerance,
            $"Cosine distance {dist} should be in [0, 2]");
    }

    [Fact]
    public void Cosine_ZeroVector_ReturnsOne()
    {
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 0, 0, 0 });
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        double dist = cosine.Compute(a, b);
        Assert.Equal(1.0, dist, LooseTolerance);
    }

    [Fact]
    public void Cosine_Similarity_PlusDistance_IsOne()
    {
        var cosine = new CosineDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 5, 6 });

        double dist = cosine.Compute(a, b);
        double sim = cosine.ComputeSimilarity(a, b);

        Assert.Equal(1.0, dist + sim, LooseTolerance);
    }

    #endregion

    #region Minkowski Distance

    [Fact]
    public void Minkowski_P1_EqualsManhattan()
    {
        var minkowski = new MinkowskiDistance<double>(1.0);
        var manhattan = new ManhattanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dMinkowski = minkowski.Compute(a, b);
        double dManhattan = manhattan.Compute(a, b);

        Assert.Equal(dManhattan, dMinkowski, Tolerance);
    }

    [Fact]
    public void Minkowski_P2_EqualsEuclidean()
    {
        var minkowski = new MinkowskiDistance<double>(2.0);
        var euclidean = new EuclideanDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dMinkowski = minkowski.Compute(a, b);
        double dEuclidean = euclidean.Compute(a, b);

        Assert.Equal(dEuclidean, dMinkowski, Tolerance);
    }

    [Fact]
    public void Minkowski_HighP_ApproachesChebyshev()
    {
        // As p -> inf, Minkowski -> Chebyshev
        var minkowski = new MinkowskiDistance<double>(50.0);
        var chebyshev = new ChebyshevDistance<double>();
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        double dMinkowski = minkowski.Compute(a, b);
        double dChebyshev = chebyshev.Compute(a, b);

        Assert.Equal(dChebyshev, dMinkowski, 0.1);
    }

    [Fact]
    public void Minkowski_HandCalculated_P3()
    {
        // d_3((0,0),(3,4)) = (3^3 + 4^3)^(1/3) = (27+64)^(1/3) = 91^(1/3)
        var minkowski = new MinkowskiDistance<double>(3.0);
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 3, 4 });

        double expected = Math.Pow(91, 1.0 / 3.0);
        double dist = minkowski.Compute(a, b);
        Assert.Equal(expected, dist, LooseTolerance);
    }

    [Fact]
    public void Minkowski_InvalidP_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MinkowskiDistance<double>(0.5));
    }

    [Fact]
    public void Minkowski_IdenticalVectors_IsZero()
    {
        var minkowski = new MinkowskiDistance<double>(3.0);
        var a = new Vector<double>(new double[] { 1, 2, 3 });

        Assert.Equal(0.0, minkowski.Compute(a, a), Tolerance);
    }

    [Fact]
    public void Minkowski_Symmetric()
    {
        var minkowski = new MinkowskiDistance<double>(3.0);
        var a = new Vector<double>(new double[] { 1, 2, 3 });
        var b = new Vector<double>(new double[] { 4, 6, 8 });

        Assert.Equal(minkowski.Compute(a, b), minkowski.Compute(b, a), Tolerance);
    }

    [Fact]
    public void Minkowski_IncreasingP_DecreasingDistance()
    {
        // For any vectors, d_p1 >= d_p2 when p1 < p2 (not always true for raw Minkowski,
        // but for the specific case where the max diff != all diffs equal)
        // Actually the general Lp norm relationship is: L_inf <= L_p <= n^(1/p) * L_inf
        // and L_p >= L_q when p < q for the same vector
        // For distances: L1 >= L2 >= ... >= L_inf
        var a = new Vector<double>(new double[] { 0, 0, 0 });
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        var d1 = new MinkowskiDistance<double>(1.0).Compute(a, b);
        var d2 = new MinkowskiDistance<double>(2.0).Compute(a, b);
        var d3 = new MinkowskiDistance<double>(3.0).Compute(a, b);
        var d10 = new MinkowskiDistance<double>(10.0).Compute(a, b);

        Assert.True(d1 >= d2 - Tolerance, $"L1 ({d1}) should be >= L2 ({d2})");
        Assert.True(d2 >= d3 - Tolerance, $"L2 ({d2}) should be >= L3 ({d3})");
        Assert.True(d3 >= d10 - Tolerance, $"L3 ({d3}) should be >= L10 ({d10})");
    }

    #endregion

    #region Pairwise Distance Computation

    [Fact]
    public void Euclidean_Pairwise_SymmetricMatrix()
    {
        var euclidean = new EuclideanDistance<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 3, 4 },
            { 1, 1 }
        });

        var pairwise = euclidean.ComputePairwise(data);

        // Should be symmetric
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(pairwise[i, j], pairwise[j, i], Tolerance);

        // Diagonal should be zero
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, pairwise[i, i], Tolerance);

        // d((0,0),(3,4)) = 5
        Assert.Equal(5.0, pairwise[0, 1], Tolerance);
    }

    [Fact]
    public void Euclidean_ComputeToAll_CorrectDistances()
    {
        var euclidean = new EuclideanDistance<double>();
        var point = new Vector<double>(new double[] { 0, 0 });
        var data = new Matrix<double>(new double[,]
        {
            { 3, 4 },
            { 0, 5 },
            { 1, 0 }
        });

        var distances = euclidean.ComputeToAll(point, data);

        Assert.Equal(5.0, distances[0], Tolerance); // sqrt(9+16)
        Assert.Equal(5.0, distances[1], Tolerance); // sqrt(0+25)
        Assert.Equal(1.0, distances[2], Tolerance); // sqrt(1+0)
    }

    #endregion

    #region Metric Ordering Invariants

    [Fact]
    public void AllDistances_CorrectOrdering()
    {
        // For any point pair: Chebyshev <= Euclidean <= Manhattan
        var chebyshev = new ChebyshevDistance<double>();
        var euclidean = new EuclideanDistance<double>();
        var manhattan = new ManhattanDistance<double>();

        var vectors = new[]
        {
            (new Vector<double>(new double[] { 1, 2, 3 }), new Vector<double>(new double[] { 4, 6, 8 })),
            (new Vector<double>(new double[] { 0, 0, 0 }), new Vector<double>(new double[] { 1, 1, 1 })),
            (new Vector<double>(new double[] { -1, 5, -3 }), new Vector<double>(new double[] { 2, -1, 4 })),
        };

        foreach (var (a, b) in vectors)
        {
            double dCheb = chebyshev.Compute(a, b);
            double dEuc = euclidean.Compute(a, b);
            double dMan = manhattan.Compute(a, b);

            Assert.True(dCheb <= dEuc + Tolerance,
                $"Chebyshev ({dCheb}) should be <= Euclidean ({dEuc})");
            Assert.True(dEuc <= dMan + Tolerance,
                $"Euclidean ({dEuc}) should be <= Manhattan ({dMan})");
        }
    }

    #endregion

    #region Silhouette Score

    [Fact]
    public void Silhouette_PerfectClusters_NearOne()
    {
        // Two well-separated clusters: points at (0,0),(1,0) in cluster 0
        //                              points at (10,0),(11,0) in cluster 1
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 10, 0 },
            { 11, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = silhouette.Compute(data, labels);

        // Should be close to 1 for well-separated clusters
        Assert.True(score > 0.8, $"Silhouette score {score} should be > 0.8 for well-separated clusters");
    }

    [Fact]
    public void Silhouette_SingleCluster_ReturnsZero()
    {
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 2, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0 });

        double score = silhouette.Compute(data, labels);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void Silhouette_Bounded_Minus1To1()
    {
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 5, 0 },
            { 6, 0 },
            { 3, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1, 0 });

        double score = silhouette.Compute(data, labels);
        Assert.True(score >= -1.0 - Tolerance && score <= 1.0 + Tolerance,
            $"Silhouette score {score} should be in [-1, 1]");
    }

    [Fact]
    public void Silhouette_HandCalculated_SimpleCase()
    {
        // 4 points on 1D line: 0, 1, 5, 6
        // Cluster 0: {0, 1}, Cluster 1: {5, 6}
        // For point 0: a(0)=|0-1|=1, b(0)=mean(|0-5|,|0-6|)=5.5, s(0)=(5.5-1)/5.5 = 4.5/5.5
        // For point 1: a(1)=|1-0|=1, b(1)=mean(|1-5|,|1-6|)=4.5, s(1)=(4.5-1)/4.5 = 3.5/4.5
        // For point 5: a(5)=|5-6|=1, b(5)=mean(|5-0|,|5-1|)=4.5, s(5)=(4.5-1)/4.5 = 3.5/4.5
        // For point 6: a(6)=|6-5|=1, b(6)=mean(|6-0|,|6-1|)=5.5, s(6)=(5.5-1)/5.5 = 4.5/5.5
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0 },
            { 1 },
            { 5 },
            { 6 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = silhouette.Compute(data, labels);

        double s0 = 4.5 / 5.5;
        double s1 = 3.5 / 4.5;
        double s2 = 3.5 / 4.5;
        double s3 = 4.5 / 5.5;
        double expected = (s0 + s1 + s2 + s3) / 4.0;

        Assert.Equal(expected, score, LooseTolerance);
    }

    [Fact]
    public void Silhouette_SinglePoint_ReturnsZero()
    {
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0 }
        });
        var labels = new Vector<double>(new double[] { 0 });

        double score = silhouette.Compute(data, labels);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void Silhouette_PerSampleScores_AverageEqualsOverall()
    {
        var silhouette = new SilhouetteScore<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 5, 0 },
            { 6, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double overallScore = silhouette.Compute(data, labels);
        double[] sampleScores = silhouette.ComputeSampleScores(data, labels);

        double avgSampleScore = sampleScores.Average();
        Assert.Equal(overallScore, avgSampleScore, LooseTolerance);
    }

    #endregion

    #region Davies-Bouldin Index

    [Fact]
    public void DaviesBouldin_PerfectClusters_LowScore()
    {
        // Well-separated clusters should have low DB index
        var db = new DaviesBouldinIndex<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 0, 1 },
            { 100, 100 },
            { 101, 100 },
            { 100, 101 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });

        double score = db.Compute(data, labels);

        Assert.True(score < 0.5, $"DB index {score} should be < 0.5 for well-separated clusters");
    }

    [Fact]
    public void DaviesBouldin_SingleCluster_ReturnsZero()
    {
        var db = new DaviesBouldinIndex<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 2, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0 });

        double score = db.Compute(data, labels);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void DaviesBouldin_NonNegative()
    {
        var db = new DaviesBouldinIndex<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 5, 0 },
            { 6, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = db.Compute(data, labels);
        Assert.True(score >= -Tolerance, $"DB index {score} should be non-negative");
    }

    [Fact]
    public void DaviesBouldin_HandCalculated_Simple()
    {
        // Cluster 0: (0, 0), (2, 0) -> centroid (1, 0), scatter = mean(|1-0|, |1-2|) = mean(1, 1) = 1
        // Cluster 1: (10, 0), (12, 0) -> centroid (11, 0), scatter = 1
        // Centroid distance: |11-1| = 10
        // R(0,1) = (1 + 1) / 10 = 0.2
        // R(1,0) = (1 + 1) / 10 = 0.2
        // DB = (max(R(0,j)) + max(R(1,j))) / 2 = (0.2 + 0.2) / 2 = 0.2
        var db = new DaviesBouldinIndex<double>();
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 2, 0 },
            { 10, 0 },
            { 12, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = db.Compute(data, labels);
        Assert.Equal(0.2, score, LooseTolerance);
    }

    [Fact]
    public void DaviesBouldin_OverlappingClusters_HigherScore()
    {
        var db = new DaviesBouldinIndex<double>();

        // Well-separated
        var dataSep = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 100, 0 },
            { 101, 0 }
        });
        var labelsSep = new Vector<double>(new double[] { 0, 0, 1, 1 });

        // Overlapping
        var dataOverlap = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 2, 0 },
            { 3, 0 }
        });
        var labelsOverlap = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double scoreSep = db.Compute(dataSep, labelsSep);
        double scoreOverlap = db.Compute(dataOverlap, labelsOverlap);

        Assert.True(scoreSep < scoreOverlap,
            $"Separated ({scoreSep}) should have lower DB than overlapping ({scoreOverlap})");
    }

    #endregion

    #region Cross-Metric Consistency

    [Fact]
    public void GoodClustering_HighSilhouette_LowDaviesBouldin()
    {
        // Well-separated clusters should give high silhouette and low DB
        var silhouette = new SilhouetteScore<double>();
        var db = new DaviesBouldinIndex<double>();

        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 0.5, 0.5 },
            { 50, 50 },
            { 51, 50 },
            { 50.5, 50.5 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });

        double sil = silhouette.Compute(data, labels);
        double dbScore = db.Compute(data, labels);

        Assert.True(sil > 0.9, $"Silhouette {sil} should be > 0.9 for well-separated clusters");
        Assert.True(dbScore < 0.1, $"DB index {dbScore} should be < 0.1 for well-separated clusters");
    }

    [Fact]
    public void BadClustering_LowSilhouette_HighDaviesBouldin()
    {
        // Poorly separated clusters should give lower silhouette and higher DB
        var silhouette = new SilhouetteScore<double>();
        var db = new DaviesBouldinIndex<double>();

        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 2, 0 },
            { 3, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double sil = silhouette.Compute(data, labels);
        double dbScore = db.Compute(data, labels);

        // Compare with well-separated version
        var dataSep = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 100, 0 },
            { 101, 0 }
        });
        var labelsSep = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double silSep = silhouette.Compute(dataSep, labelsSep);
        double dbSep = db.Compute(dataSep, labelsSep);

        Assert.True(sil < silSep,
            $"Poor silhouette ({sil}) should be < good silhouette ({silSep})");
        Assert.True(dbScore > dbSep,
            $"Poor DB ({dbScore}) should be > good DB ({dbSep})");
    }

    #endregion

    #region Distance Metric with Non-Euclidean Clustering

    [Fact]
    public void Silhouette_WithManhattanDistance()
    {
        var manhattan = new ManhattanDistance<double>();
        var silhouette = new SilhouetteScore<double>(manhattan);

        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 10, 0 },
            { 11, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = silhouette.Compute(data, labels);
        Assert.True(score > 0.7, $"Manhattan silhouette {score} should be > 0.7");
    }

    [Fact]
    public void DaviesBouldin_WithManhattanDistance()
    {
        var manhattan = new ManhattanDistance<double>();
        var db = new DaviesBouldinIndex<double>(manhattan);

        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 1, 0 },
            { 10, 0 },
            { 11, 0 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        double score = db.Compute(data, labels);
        Assert.True(score >= 0, $"DB score with Manhattan should be non-negative, got {score}");
        Assert.True(score < 1.0, $"DB score {score} should be < 1 for well-separated clusters");
    }

    #endregion
}
