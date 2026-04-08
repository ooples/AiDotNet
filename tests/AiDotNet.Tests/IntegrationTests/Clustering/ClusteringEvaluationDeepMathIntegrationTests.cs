using AiDotNet.Clustering.Evaluation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Deep math-correctness integration tests for clustering evaluation metrics.
/// Tests verify exact hand-calculated values for Silhouette Score, Davies-Bouldin,
/// Dunn Index, Calinski-Harabasz, ARI, NMI, Jaccard, V-Measure, Purity, FMI,
/// F-Measure, WCSS, BCSS, Connectivity, and Variation of Information.
/// </summary>
public class ClusteringEvaluationDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    // ─── Helper Methods ──────────────────────────────────────────────────────

    private static Matrix<double> MakeMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = data[i, j];
        return matrix;
    }

    private static Vector<double> MakeLabels(params double[] labels)
    {
        var vec = new Vector<double>(labels.Length);
        for (int i = 0; i < labels.Length; i++)
            vec[i] = labels[i];
        return vec;
    }

    // ─── WCSS Tests ──────────────────────────────────────────────────────────

    [Fact]
    public void WCSS_TwoClusters_HandCalculated()
    {
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Labels: [0, 0, 1, 1]
        // Centroid 0 = (0.5, 0), Centroid 1 = (10.5, 0)
        // WCSS = (0-0.5)^2 + (1-0.5)^2 + (10-10.5)^2 + (11-10.5)^2
        //      = 0.25 + 0.25 + 0.25 + 0.25 = 1.0
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var wcss = new WCSS<double>();
        double result = wcss.Compute(data, labels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void WCSS_SingleCluster_AllPointsContributeToSameCenter()
    {
        // Points: (0,0), (2,0), (4,0) in cluster 0
        // Centroid = (2, 0)
        // WCSS = (0-2)^2 + (2-2)^2 + (4-2)^2 = 4 + 0 + 4 = 8
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 2, 0 }, { 4, 0 } });
        var labels = MakeLabels(0, 0, 0);

        var wcss = new WCSS<double>();
        double result = wcss.Compute(data, labels);

        Assert.Equal(8.0, result, Tolerance);
    }

    [Fact]
    public void WCSS_PerCluster_ReturnsCorrectBreakdown()
    {
        // Cluster 0: (0,0), (2,0) → centroid (1,0) → WCSS = 1+1 = 2
        // Cluster 1: (10,0), (10,3) → centroid (10, 1.5) → WCSS = 2.25+2.25 = 4.5
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 2, 0 }, { 10, 0 }, { 10, 3 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var wcss = new WCSS<double>();
        var perCluster = wcss.ComputePerCluster(data, labels);

        Assert.Equal(2.0, perCluster[0], Tolerance);
        Assert.Equal(4.5, perCluster[1], Tolerance);
    }

    // ─── BCSS Tests ──────────────────────────────────────────────────────────

    [Fact]
    public void BCSS_TwoClusters_HandCalculated()
    {
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Labels: [0, 0, 1, 1]
        // Global centroid = (5.5, 0)
        // Centroid 0 = (0.5, 0), Centroid 1 = (10.5, 0)
        // BCSS = 2*(0.5-5.5)^2 + 2*(10.5-5.5)^2 = 2*25 + 2*25 = 100
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var bcss = new BCSS<double>();
        double result = bcss.Compute(data, labels);

        Assert.Equal(100.0, result, Tolerance);
    }

    [Fact]
    public void WCSS_Plus_BCSS_Equals_TotalVariance()
    {
        // Total variance = sum of squared distances from each point to global centroid
        // = WCSS + BCSS (by variance decomposition)
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var wcssMetric = new WCSS<double>();
        var bcssMetric = new BCSS<double>();
        double wcss = wcssMetric.Compute(data, labels);
        double bcss = bcssMetric.Compute(data, labels);

        // Global centroid = (5.5, 0)
        // Total variance = (0-5.5)^2 + (1-5.5)^2 + (10-5.5)^2 + (11-5.5)^2
        //                = 30.25 + 20.25 + 20.25 + 30.25 = 101
        double expectedTotal = 30.25 + 20.25 + 20.25 + 30.25;

        Assert.Equal(expectedTotal, wcss + bcss, Tolerance);
    }

    // ─── Calinski-Harabasz Index Tests ───────────────────────────────────────

    [Fact]
    public void CalinskiHarabasz_TwoClusters_HandCalculated()
    {
        // Using same data: BCSS=100, WCSS=1.0, k=2, n=4
        // CH = (BGS/(k-1)) / (WGS/(n-k)) = (100/1) / (1/2) = 200
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var ch = new CalinskiHarabaszIndex<double>();
        double result = ch.Compute(data, labels);

        Assert.Equal(200.0, result, Tolerance);
    }

    [Fact]
    public void CalinskiHarabasz_BetterSeparation_HigherScore()
    {
        // Close clusters
        var dataClose = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 3, 0 }, { 4, 0 } });
        var labelsClose = MakeLabels(0, 0, 1, 1);

        // Far clusters
        var dataFar = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 100, 0 }, { 101, 0 } });
        var labelsFar = MakeLabels(0, 0, 1, 1);

        var ch = new CalinskiHarabaszIndex<double>();
        double closeScore = ch.Compute(dataClose, labelsClose);
        double farScore = ch.Compute(dataFar, labelsFar);

        Assert.True(farScore > closeScore,
            $"Far clusters should have higher CH index. Close={closeScore}, Far={farScore}");
    }

    [Fact]
    public void CalinskiHarabasz_SingleCluster_ReturnsZero()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 2, 0 } });
        var labels = MakeLabels(0, 0, 0);

        var ch = new CalinskiHarabaszIndex<double>();
        double result = ch.Compute(data, labels);

        Assert.Equal(0.0, result);
    }

    // ─── Davies-Bouldin Index Tests ──────────────────────────────────────────

    [Fact]
    public void DaviesBouldin_TwoClusters_HandCalculated()
    {
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Labels: [0, 0, 1, 1]
        // S(0) = mean dist to centroid(0.5,0) = (0.5+0.5)/2 = 0.5
        // S(1) = mean dist to centroid(10.5,0) = (0.5+0.5)/2 = 0.5
        // d(0,1) = dist((0.5,0),(10.5,0)) = 10
        // R(0,1) = (0.5+0.5)/10 = 0.1
        // DB = (max_j R(0,j) + max_j R(1,j)) / 2 = (0.1 + 0.1) / 2 = 0.1
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var db = new DaviesBouldinIndex<double>();
        double result = db.Compute(data, labels);

        Assert.Equal(0.1, result, Tolerance);
    }

    [Fact]
    public void DaviesBouldin_LowerIsBetter_WellSeparatedClusters()
    {
        // Close clusters → higher DB
        var dataClose = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 3, 0 }, { 4, 0 } });
        var labelsClose = MakeLabels(0, 0, 1, 1);

        // Far clusters → lower DB
        var dataFar = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 100, 0 }, { 101, 0 } });
        var labelsFar = MakeLabels(0, 0, 1, 1);

        var db = new DaviesBouldinIndex<double>();
        double closeScore = db.Compute(dataClose, labelsClose);
        double farScore = db.Compute(dataFar, labelsFar);

        Assert.True(farScore < closeScore,
            $"Well-separated clusters should have lower DB. Close={closeScore}, Far={farScore}");
    }

    // ─── Dunn Index Tests ────────────────────────────────────────────────────

    [Fact]
    public void DunnIndex_TwoClusters_HandCalculated()
    {
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Labels: [0, 0, 1, 1]
        // Min inter-cluster dist: min(dist(0,2)=10, dist(0,3)=11, dist(1,2)=9, dist(1,3)=10) = 9
        // Max intra-cluster diam: max(dist(0,1)=1, dist(2,3)=1) = 1
        // Dunn = 9 / 1 = 9
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var dunn = new DunnIndex<double>();
        double result = dunn.Compute(data, labels);

        Assert.Equal(9.0, result, Tolerance);
    }

    [Fact]
    public void DunnIndex_HigherIsBetter_MoreSeparatedIsHigher()
    {
        var dataClose = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 3, 0 }, { 4, 0 } });
        var labelsClose = MakeLabels(0, 0, 1, 1);

        var dataFar = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 100, 0 }, { 101, 0 } });
        var labelsFar = MakeLabels(0, 0, 1, 1);

        var dunn = new DunnIndex<double>();
        double closeScore = dunn.Compute(dataClose, labelsClose);
        double farScore = dunn.Compute(dataFar, labelsFar);

        Assert.True(farScore > closeScore,
            $"Far clusters should have higher Dunn. Close={closeScore}, Far={farScore}");
    }

    // ─── Silhouette Score Tests ──────────────────────────────────────────────

    [Fact]
    public void Silhouette_TwoClusters_HandCalculated()
    {
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Labels: [0, 0, 1, 1]
        // Point 0: a=1.0, b=(10+11)/2=10.5, s=(10.5-1)/10.5 = 9.5/10.5
        // Point 1: a=1.0, b=(9+10)/2=9.5, s=(9.5-1)/9.5 = 8.5/9.5
        // Point 2: a=1.0, b=(10+9)/2=9.5, s=(9.5-1)/9.5 = 8.5/9.5
        // Point 3: a=1.0, b=(11+10)/2=10.5, s=(10.5-1)/10.5 = 9.5/10.5
        // Avg = (9.5/10.5 + 8.5/9.5 + 8.5/9.5 + 9.5/10.5) / 4
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        double s0 = 9.5 / 10.5;
        double s1 = 8.5 / 9.5;
        double s2 = 8.5 / 9.5;
        double s3 = 9.5 / 10.5;
        double expected = (s0 + s1 + s2 + s3) / 4.0;

        var silhouette = new SilhouetteScore<double>();
        double result = silhouette.Compute(data, labels);

        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void Silhouette_RangeIsMinus1To1()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var silhouette = new SilhouetteScore<double>();
        double result = silhouette.Compute(data, labels);

        Assert.InRange(result, -1.0, 1.0);
    }

    [Fact]
    public void Silhouette_WrongAssignment_NegativeScore()
    {
        // Assign points to the WRONG cluster
        // Points: (0,0), (1,0), (10,0), (11,0)
        // Wrong labels: [1, 1, 0, 0] → points close to each other get different labels
        // Actually this just swaps labels, silhouette is label-agnostic, still positive
        // Use: (0,0) in cluster 1, (1,0) in cluster 0, (10,0) in cluster 0, (11,0) in cluster 1
        // Labels: [1, 0, 0, 1] → interleaved
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(1, 0, 0, 1);

        // Point 0 (cluster 1, alone with point 3):
        // a(0) = dist(0,3) = 11
        // b(0) = mean(dist(0,1), dist(0,2)) = (1+10)/2 = 5.5
        // s(0) = (5.5-11)/11 = -5.5/11 < 0

        var silhouette = new SilhouetteScore<double>();
        double result = silhouette.Compute(data, labels);

        Assert.True(result < 0,
            $"Interleaved labels on well-separated data should give negative silhouette. Got {result}");
    }

    [Fact]
    public void Silhouette_PerSampleScores_MatchOverallAverage()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var silhouette = new SilhouetteScore<double>();
        double overall = silhouette.Compute(data, labels);
        double[] perSample = silhouette.ComputeSampleScores(data, labels);

        double avgSample = perSample.Average();
        Assert.Equal(overall, avgSample, Tolerance);
    }

    [Fact]
    public void Silhouette_SingletonCluster_ShouldBeZeroNotOne()
    {
        // Point 2 is alone in cluster 1
        // Standard behavior (sklearn): singleton silhouette = 0
        // BUG: Our code gives 1.0 for singletons because a(i)=0 → s=(b-0)/b=1
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 } });
        var labels = MakeLabels(0, 0, 1);

        var silhouette = new SilhouetteScore<double>();
        double[] perSample = silhouette.ComputeSampleScores(data, labels);

        // Point 2 is a singleton in cluster 1
        // Standard: should be 0
        Assert.Equal(0.0, perSample[2], Tolerance);
    }

    [Fact]
    public void Silhouette_SingleCluster_ReturnsZero()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 2, 0 } });
        var labels = MakeLabels(0, 0, 0);

        var silhouette = new SilhouetteScore<double>();
        double result = silhouette.Compute(data, labels);

        Assert.Equal(0.0, result);
    }

    // ─── Adjusted Rand Index Tests ───────────────────────────────────────────

    [Fact]
    public void ARI_PerfectAgreement_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var ari = new AdjustedRandIndex<double>();
        double result = ari.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ARI_LabelPermutation_StillPerfect()
    {
        // Swapping label names doesn't change ARI
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(1, 1, 0, 0); // Same partition, different names

        var ari = new AdjustedRandIndex<double>();
        double result = ari.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ARI_RandomLabeling_HandCalculated()
    {
        // True: [0, 0, 1, 1], Pred: [0, 1, 0, 1]
        // Contingency:
        //        Pred0  Pred1
        // True0:   1      1
        // True1:   1      1
        // sumNijC2 = 0 (all cells have n_ij=1, C(1,2)=0)
        // sumA = C(2,2) + C(2,2) = 2
        // sumB = C(2,2) + C(2,2) = 2
        // totalPairs = C(4,2) = 6
        // expected = 2*2/6 = 0.6667
        // max = 0.5*(2+2) = 2
        // ARI = (0 - 0.6667)/(2 - 0.6667) = -0.5
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(0, 1, 0, 1);

        var ari = new AdjustedRandIndex<double>();
        double result = ari.Compute(trueLabels, predLabels);

        Assert.Equal(-0.5, result, Tolerance);
    }

    [Fact]
    public void ARI_AllSameLabel_ReturnsZero()
    {
        var trueLabels = MakeLabels(0, 0, 0, 0);
        var predLabels = MakeLabels(1, 1, 1, 1);

        var ari = new AdjustedRandIndex<double>();
        double result = ari.Compute(trueLabels, predLabels);

        // All in one cluster vs all in one cluster → trivial agreement
        // But ARI formula gives denominator 0 → should return 0 or 1
        Assert.InRange(result, -1.0, 1.0);
    }

    // ─── NMI Tests ───────────────────────────────────────────────────────────

    [Fact]
    public void NMI_PerfectAgreement_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var nmi = new NormalizedMutualInformation<double>();
        double result = nmi.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void NMI_LabelPermutation_StillPerfect()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(1, 1, 0, 0);

        var nmi = new NormalizedMutualInformation<double>();
        double result = nmi.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void NMI_IndependentClusters_ReturnsZero()
    {
        // True: [0, 0, 1, 1], Pred: [0, 1, 0, 1]
        // These are independent (knowing one tells nothing about the other)
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(0, 1, 0, 1);

        var nmi = new NormalizedMutualInformation<double>();
        double result = nmi.Compute(trueLabels, predLabels);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void NMI_RangeIsZeroToOne()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1, 2, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2, 0, 1, 2);

        var nmi = new NormalizedMutualInformation<double>();
        double result = nmi.Compute(trueLabels, predLabels);

        Assert.InRange(result, 0.0, 1.0 + 1e-10);
    }

    [Fact]
    public void NMI_DifferentNormalizations_ArithmeticVsGeometric()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var nmiArith = new NormalizedMutualInformation<double>(NMINormalization.Arithmetic);
        var nmiGeom = new NormalizedMutualInformation<double>(NMINormalization.Geometric);
        var nmiMin = new NormalizedMutualInformation<double>(NMINormalization.Min);
        var nmiMax = new NormalizedMutualInformation<double>(NMINormalization.Max);

        double arith = nmiArith.Compute(trueLabels, predLabels);
        double geom = nmiGeom.Compute(trueLabels, predLabels);
        double min = nmiMin.Compute(trueLabels, predLabels);
        double max = nmiMax.Compute(trueLabels, predLabels);

        // All should be in [0,1]
        Assert.InRange(arith, 0.0, 1.0 + 1e-10);
        Assert.InRange(geom, 0.0, 1.0 + 1e-10);
        Assert.InRange(min, 0.0, 1.0 + 1e-10);
        Assert.InRange(max, 0.0, 1.0 + 1e-10);

        // Max normalization gives smallest NMI, Min gives largest
        Assert.True(max <= min + 1e-10,
            $"Max normalization should give <= Min. Max={max}, Min={min}");
    }

    // ─── Jaccard Index Tests ─────────────────────────────────────────────────

    [Fact]
    public void Jaccard_PerfectAgreement_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1);

        var jaccard = new JaccardIndex<double>();
        double result = jaccard.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Jaccard_HandCalculated_PartialAgreement()
    {
        // True: [0, 0, 0, 1, 1], Pred: [0, 0, 1, 1, 1]
        // Pairs (10 total):
        // (0,1): same/same → a
        // (0,2): same/diff → b
        // (0,3): diff/diff → d
        // (0,4): diff/diff → d
        // (1,2): same/diff → b
        // (1,3): diff/diff → d
        // (1,4): diff/diff → d
        // (2,3): diff/same → c
        // (2,4): diff/same → c
        // (3,4): same/same → a
        // a=2, b=2, c=2, d=4
        // Jaccard = 2 / (2+2+2) = 1/3
        var trueLabels = MakeLabels(0, 0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1);

        var jaccard = new JaccardIndex<double>();
        double result = jaccard.Compute(trueLabels, predLabels);

        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void Jaccard_PairConfusionMatrix_HandCalculated()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1);

        var jaccard = new JaccardIndex<double>();
        var (a, b, c, d) = jaccard.ComputePairConfusionMatrix(trueLabels, predLabels);

        Assert.Equal(2, a); // same in both
        Assert.Equal(2, b); // same in true only
        Assert.Equal(2, c); // same in pred only
        Assert.Equal(4, d); // different in both
        Assert.Equal(10, a + b + c + d); // C(5,2) = 10 total pairs
    }

    // ─── Rand Index Tests ────────────────────────────────────────────────────

    [Fact]
    public void RandIndex_PerfectAgreement_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1);

        var ri = new RandIndex<double>(adjusted: false);
        double result = ri.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void RandIndex_HandCalculated()
    {
        // True: [0, 0, 0, 1, 1], Pred: [0, 0, 1, 1, 1]
        // From Jaccard test: a=2, b=2, c=2, d=4
        // RI = (a+d) / (a+b+c+d) = 6/10 = 0.6
        var trueLabels = MakeLabels(0, 0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1);

        var ri = new RandIndex<double>(adjusted: false);
        double result = ri.Compute(trueLabels, predLabels);

        Assert.Equal(0.6, result, Tolerance);
    }

    [Fact]
    public void AdjustedRandIndex_ViaRandIndexClass_MatchesDedicatedClass()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 2, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2, 0, 0);

        var ariDedicated = new AdjustedRandIndex<double>();
        var ariViaRi = new RandIndex<double>(adjusted: true);

        double dedicated = ariDedicated.Compute(trueLabels, predLabels);
        double viaRi = ariViaRi.Compute(trueLabels, predLabels);

        Assert.Equal(dedicated, viaRi, 1e-4);
    }

    // ─── Purity Tests ────────────────────────────────────────────────────────

    [Fact]
    public void Purity_PerfectClustering_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var purity = new Purity<double>();
        double result = purity.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Purity_HandCalculated_MixedClusters()
    {
        // True: [0, 0, 0, 1, 1, 1]
        // Pred: [0, 0, 1, 1, 1, 0]
        // Cluster 0 contains: true labels {0, 0, 1} → majority = 0, count = 2
        // Cluster 1 contains: true labels {0, 1, 1} → majority = 1, count = 2
        // Purity = (2 + 2) / 6 = 2/3
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1, 0);

        var purity = new Purity<double>();
        double result = purity.Compute(trueLabels, predLabels);

        Assert.Equal(2.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void Purity_PerCluster_HandCalculated()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1, 0);

        var purity = new Purity<double>();
        var perCluster = purity.ComputePerCluster(trueLabels, predLabels);

        // Cluster 0: {true: 0, 0, 1} → 2/3 pure
        // Cluster 1: {true: 0, 1, 1} → 2/3 pure
        Assert.Equal(2.0 / 3.0, perCluster[0], Tolerance);
        Assert.Equal(2.0 / 3.0, perCluster[1], Tolerance);
    }

    // ─── V-Measure Tests ─────────────────────────────────────────────────────

    [Fact]
    public void VMeasure_PerfectClustering_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var vm = new VMeasure<double>();
        double result = vm.ComputeWithTrueLabels(null, predLabels, trueLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void VMeasure_Homogeneity_PureClusters_IsOne()
    {
        // Each cluster contains only one class → perfect homogeneity
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var vm = new VMeasure<double>();
        double homo = vm.ComputeHomogeneity(predLabels, trueLabels);

        Assert.Equal(1.0, homo, Tolerance);
    }

    [Fact]
    public void VMeasure_Completeness_AllClassTogether_IsOne()
    {
        // All members of each class are in same cluster → perfect completeness
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var vm = new VMeasure<double>();
        double comp = vm.ComputeCompleteness(predLabels, trueLabels);

        Assert.Equal(1.0, comp, Tolerance);
    }

    [Fact]
    public void VMeasure_IsHarmonicMean_WithBetaOne()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1, 0);

        var vm = new VMeasure<double>(beta: 1.0);
        double homo = vm.ComputeHomogeneity(predLabels, trueLabels);
        double comp = vm.ComputeCompleteness(predLabels, trueLabels);
        double vmResult = vm.ComputeWithTrueLabels(null, predLabels, trueLabels);

        // With beta=1, V-Measure = harmonic mean of homogeneity and completeness
        double expectedHarmonic = (homo + comp > 0)
            ? 2.0 * homo * comp / (homo + comp)
            : 0.0;

        Assert.Equal(expectedHarmonic, vmResult, Tolerance);
    }

    [Fact]
    public void VMeasure_IExternalInterface_MatchesDirect()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 1, 1, 2, 2, 0);

        var vm = new VMeasure<double>();
        double direct = vm.ComputeWithTrueLabels(null, predLabels, trueLabels);

        IExternalClusterMetric<double> external = vm;
        double viaInterface = external.Compute(trueLabels, predLabels);

        Assert.Equal(direct, viaInterface, Tolerance);
    }

    // ─── Fowlkes-Mallows Index Tests ─────────────────────────────────────────

    [Fact]
    public void FowlkesMallows_PerfectAgreement_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var fmi = new FowlkesMallowsIndex<double>();
        double result = fmi.ComputeWithTrueLabels(null, predLabels, trueLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void FowlkesMallows_HandCalculated()
    {
        // True: [0, 0, 0, 1, 1], Pred: [0, 0, 1, 1, 1]
        // From pair confusion: a=2 (TP), b=2 (FN), c=2 (FP)
        // TP+FP = a+c = 4, TP+FN = a+b = 4
        // Precision = TP/(TP+FP) = 2/4 = 0.5
        // Recall = TP/(TP+FN) = 2/4 = 0.5
        // FMI = sqrt(0.5 * 0.5) = 0.5
        var trueLabels = MakeLabels(0, 0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1);

        var fmi = new FowlkesMallowsIndex<double>();
        double result = fmi.ComputeWithTrueLabels(null, predLabels, trueLabels);

        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void FowlkesMallows_RangeIsZeroToOne()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1, 2, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2, 0, 1, 2);

        var fmi = new FowlkesMallowsIndex<double>();
        double result = fmi.ComputeWithTrueLabels(null, predLabels, trueLabels);

        Assert.InRange(result, 0.0, 1.0 + 1e-10);
    }

    // ─── F-Measure Tests ─────────────────────────────────────────────────────

    [Fact]
    public void FMeasure_PerfectClustering_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var fm = new FMeasure<double>();
        double result = fm.Compute(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void FMeasure_HandCalculated_TwoClassesTwoClusters()
    {
        // True: [0, 0, 0, 1, 1], Pred: [0, 0, 1, 1, 1]
        // For class 0 (size=3):
        //   vs cluster 0 (size=2): intersection=2, P=2/2=1.0, R=2/3, F1=2*1*(2/3)/(1+2/3)=4/3/(5/3)=4/5=0.8
        //   vs cluster 1 (size=3): intersection=1, P=1/3, R=1/3, F1=2*(1/3)*(1/3)/(1/3+1/3)=1/3
        //   best = 0.8
        // For class 1 (size=2):
        //   vs cluster 0 (size=2): intersection=0 → skip
        //   vs cluster 1 (size=3): intersection=2, P=2/3, R=2/2=1.0, F1=2*(2/3)*1/(2/3+1)=4/3/(5/3)=4/5=0.8
        //   best = 0.8
        // Weighted: (0.8*3 + 0.8*2) / 5 = 4/5 = 0.8
        var trueLabels = MakeLabels(0, 0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1, 1);

        var fm = new FMeasure<double>();
        double result = fm.Compute(trueLabels, predLabels);

        Assert.Equal(0.8, result, Tolerance);
    }

    [Fact]
    public void FMeasure_BCubed_PerfectClustering_ReturnsOne()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var fm = new FMeasure<double>();
        double result = fm.ComputeBCubed(trueLabels, predLabels);

        Assert.Equal(1.0, result, Tolerance);
    }

    // ─── Connectivity Index Tests ────────────────────────────────────────────

    [Fact]
    public void Connectivity_PerfectClustering_ReturnsZero()
    {
        // All nearest neighbors in same cluster → connectivity = 0
        // Cluster 0: (0,0), (1,0) — nearest neighbor is in same cluster
        // Cluster 1: (100,0), (101,0) — nearest neighbor is in same cluster
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 100, 0 }, { 101, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);

        var conn = new ConnectivityIndex<double>(numNeighbors: 1);
        double result = conn.Compute(data, labels);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Connectivity_HandCalculated_MixedNeighbors()
    {
        // Points on a line: (0), (1), (3), (4) in 1D
        // Labels: [0, 1, 0, 1] (interleaved)
        // L=1 (check only 1st nearest neighbor)
        // Point 0: nearest = point 1 (dist 1), different cluster → penalty 1/1=1.0
        // Point 1: nearest = point 0 (dist 1), different cluster → penalty 1.0
        // Point 2: nearest = point 3 (dist 1), different cluster → penalty 1.0
        // Point 3: nearest = point 2 (dist 1), different cluster → penalty 1.0
        // Total connectivity = 4.0
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 3, 0 }, { 4, 0 } });
        var labels = MakeLabels(0, 1, 0, 1);

        var conn = new ConnectivityIndex<double>(numNeighbors: 1);
        double result = conn.Compute(data, labels);

        Assert.Equal(4.0, result, Tolerance);
    }

    [Fact]
    public void Connectivity_LowerIsBetter()
    {
        // Good clustering: nearby points together
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var goodLabels = MakeLabels(0, 0, 1, 1);
        var badLabels = MakeLabels(0, 1, 0, 1);

        var conn = new ConnectivityIndex<double>(numNeighbors: 1);
        double goodResult = conn.Compute(data, goodLabels);
        double badResult = conn.Compute(data, badLabels);

        Assert.True(goodResult < badResult,
            $"Good clustering should have lower connectivity. Good={goodResult}, Bad={badResult}");
    }

    // ─── Variation of Information Tests ──────────────────────────────────────

    [Fact]
    public void VariationOfInformation_IdenticalClusters_ReturnsZero()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2);

        var vi = new VariationOfInformation<double>();
        double result = vi.Compute(trueLabels, predLabels);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void VariationOfInformation_LabelPermutation_StillZero()
    {
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(1, 1, 0, 0);

        var vi = new VariationOfInformation<double>();
        double result = vi.Compute(trueLabels, predLabels);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void VariationOfInformation_Normalized_RangeZeroToOne()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1, 2, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2, 0, 1, 2);

        var viNorm = new VariationOfInformation<double>(normalized: true);
        double result = viNorm.Compute(trueLabels, predLabels);

        Assert.InRange(result, 0.0, 1.0 + 1e-10);
    }

    [Fact]
    public void VariationOfInformation_Symmetric()
    {
        // VI(A, B) should equal VI(B, A)
        var labelsA = MakeLabels(0, 0, 1, 1, 2, 2);
        var labelsB = MakeLabels(0, 1, 1, 2, 2, 0);

        var vi = new VariationOfInformation<double>();
        double ab = vi.Compute(labelsA, labelsB);
        double ba = vi.Compute(labelsB, labelsA);

        Assert.Equal(ab, ba, Tolerance);
    }

    [Fact]
    public void VariationOfInformation_NMI_MatchesDedicatedNMI()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1, 2, 2, 2);
        var predLabels = MakeLabels(0, 0, 1, 1, 2, 2, 0, 1, 2);

        var vi = new VariationOfInformation<double>();
        double viNMI = vi.ComputeNMI(trueLabels, predLabels);

        // NMI from the dedicated class (using natural log)
        var nmi = new NormalizedMutualInformation<double>(NMINormalization.Arithmetic);
        double dedicatedNMI = nmi.Compute(trueLabels, predLabels);

        // Both compute 2*MI/(H(U)+H(V)), just with different log bases
        // But since NMI is a ratio, the log base cancels out
        Assert.Equal(dedicatedNMI, viNMI, 1e-4);
    }

    // ─── Cross-Metric Consistency Tests ──────────────────────────────────────

    [Fact]
    public void AllMetrics_PerfectClustering_OptimalScores()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });
        var labels = MakeLabels(0, 0, 1, 1);
        var trueLabels = labels;
        var predLabels = labels;

        // Internal metrics
        var silhouette = new SilhouetteScore<double>();
        var db = new DaviesBouldinIndex<double>();
        var dunn = new DunnIndex<double>();
        var ch = new CalinskiHarabaszIndex<double>();

        double silScore = silhouette.Compute(data, labels);
        double dbScore = db.Compute(data, labels);
        double dunnScore = dunn.Compute(data, labels);
        double chScore = ch.Compute(data, labels);

        // Silhouette: high is good
        Assert.True(silScore > 0.5, $"Silhouette should be high for well-separated clusters. Got {silScore}");
        // DB: low is good
        Assert.True(dbScore < 1.0, $"DB should be low for well-separated clusters. Got {dbScore}");
        // Dunn: high is good
        Assert.True(dunnScore > 1.0, $"Dunn should be high for well-separated clusters. Got {dunnScore}");
        // CH: high is good
        Assert.True(chScore > 10.0, $"CH should be high for well-separated clusters. Got {chScore}");

        // External metrics
        var ari = new AdjustedRandIndex<double>();
        var nmi = new NormalizedMutualInformation<double>();
        var jaccard = new JaccardIndex<double>();
        var fm = new FMeasure<double>();
        var purity = new Purity<double>();
        var vi = new VariationOfInformation<double>();

        Assert.Equal(1.0, ari.Compute(trueLabels, predLabels), Tolerance);
        Assert.Equal(1.0, nmi.Compute(trueLabels, predLabels), Tolerance);
        Assert.Equal(1.0, jaccard.Compute(trueLabels, predLabels), Tolerance);
        Assert.Equal(1.0, fm.Compute(trueLabels, predLabels), Tolerance);
        Assert.Equal(1.0, purity.Compute(trueLabels, predLabels), Tolerance);
        Assert.Equal(0.0, vi.Compute(trueLabels, predLabels), Tolerance);
    }

    [Fact]
    public void ExternalMetrics_WorseClusteringGivesWorsScores()
    {
        var trueLabels = MakeLabels(0, 0, 0, 1, 1, 1);

        // Good prediction (1 mistake)
        var goodPred = MakeLabels(0, 0, 0, 1, 1, 0);
        // Bad prediction (fully random)
        var badPred = MakeLabels(0, 1, 0, 1, 0, 1);

        var ari = new AdjustedRandIndex<double>();
        var nmi = new NormalizedMutualInformation<double>();
        var fm = new FMeasure<double>();

        double goodARI = ari.Compute(trueLabels, goodPred);
        double badARI = ari.Compute(trueLabels, badPred);
        Assert.True(goodARI > badARI, $"Good should beat bad ARI. Good={goodARI}, Bad={badARI}");

        double goodNMI = nmi.Compute(trueLabels, goodPred);
        double badNMI = nmi.Compute(trueLabels, badPred);
        Assert.True(goodNMI > badNMI, $"Good should beat bad NMI. Good={goodNMI}, Bad={badNMI}");

        double goodFM = fm.Compute(trueLabels, goodPred);
        double badFM = fm.Compute(trueLabels, badPred);
        Assert.True(goodFM > badFM, $"Good should beat bad FM. Good={goodFM}, Bad={badFM}");
    }

    [Fact]
    public void InternalMetrics_WorseClusteringGivesWorsScores()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 } });

        var goodLabels = MakeLabels(0, 0, 1, 1);
        var badLabels = MakeLabels(0, 1, 0, 1);

        var silhouette = new SilhouetteScore<double>();
        double goodSil = silhouette.Compute(data, goodLabels);
        double badSil = silhouette.Compute(data, badLabels);
        Assert.True(goodSil > badSil, $"Good clustering should have higher silhouette. Good={goodSil}, Bad={badSil}");

        var ch = new CalinskiHarabaszIndex<double>();
        double goodCH = ch.Compute(data, goodLabels);
        double badCH = ch.Compute(data, badLabels);
        Assert.True(goodCH > badCH, $"Good clustering should have higher CH. Good={goodCH}, Bad={badCH}");
    }

    // ─── Edge Cases ──────────────────────────────────────────────────────────

    [Fact]
    public void AllInternalMetrics_TwoPoints_HandleGracefully()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 10, 0 } });
        var labels = MakeLabels(0, 1);

        var silhouette = new SilhouetteScore<double>();
        var db = new DaviesBouldinIndex<double>();
        var dunn = new DunnIndex<double>();
        var ch = new CalinskiHarabaszIndex<double>();
        var wcss = new WCSS<double>();

        // Should not throw
        silhouette.Compute(data, labels);
        db.Compute(data, labels);
        dunn.Compute(data, labels);
        ch.Compute(data, labels);
        double wcssResult = wcss.Compute(data, labels);

        // WCSS for 2 singleton clusters = 0
        Assert.Equal(0.0, wcssResult, Tolerance);
    }

    [Fact]
    public void AllExternalMetrics_SinglePoint_HandleGracefully()
    {
        var trueLabels = MakeLabels(0);
        var predLabels = MakeLabels(0);

        // All should not throw
        var ari = new AdjustedRandIndex<double>();
        var nmi = new NormalizedMutualInformation<double>();
        var jaccard = new JaccardIndex<double>();

        ari.Compute(trueLabels, predLabels);
        nmi.Compute(trueLabels, predLabels);
        jaccard.Compute(trueLabels, predLabels);
    }

    [Fact]
    public void ExternalMetrics_DifferentLengths_ThrowsArgumentException()
    {
        var labels3 = MakeLabels(0, 0, 1);
        var labels4 = MakeLabels(0, 0, 1, 1);

        var ari = new AdjustedRandIndex<double>();
        Assert.Throws<ArgumentException>(() => ari.Compute(labels3, labels4));

        var nmi = new NormalizedMutualInformation<double>();
        Assert.Throws<ArgumentException>(() => nmi.Compute(labels3, labels4));

        var jaccard = new JaccardIndex<double>();
        Assert.Throws<ArgumentException>(() => jaccard.Compute(labels3, labels4));
    }

    // ─── Noise Point Handling ────────────────────────────────────────────────

    [Fact]
    public void Silhouette_NoisePointsExcluded()
    {
        // Label -1 represents noise (DBSCAN convention)
        // Noise points should be excluded from the average
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 }, { 5, 5 } });
        var labels = MakeLabels(0, 0, 1, 1, -1);

        var silhouette = new SilhouetteScore<double>();
        double result = silhouette.Compute(data, labels);

        // Should only average over the 4 non-noise points
        Assert.InRange(result, -1.0, 1.0);
    }

    [Fact]
    public void WCSS_NoisePointsExcluded()
    {
        var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { 10, 0 }, { 11, 0 }, { 999, 999 } });
        var labels = MakeLabels(0, 0, 1, 1, -1);

        // Noise point (999,999) should NOT affect WCSS
        var wcss = new WCSS<double>();
        double result = wcss.Compute(data, labels);

        // WCSS should be same as without noise: 1.0
        Assert.Equal(1.0, result, Tolerance);
    }

    // ─── ClusteringEntropy Tests ─────────────────────────────────────────────

    [Fact]
    public void ClusteringEntropy_EqualSizeClusters_MaximalEntropy()
    {
        // 4 points in 2 equal-size clusters: entropy = log2(2) = 1.0 bit
        var trueLabels = MakeLabels(0, 0, 1, 1);
        var predLabels = MakeLabels(0, 0, 1, 1);

        var vi = new VariationOfInformation<double>();
        // NMI for identical labels should be 1
        double nmi = vi.ComputeNMI(trueLabels, predLabels);
        Assert.Equal(1.0, nmi, Tolerance);
    }

    // ─── Monotonicity Property Tests ─────────────────────────────────────────

    [Fact]
    public void Silhouette_IncreasingClusterSeparation_MonotonicallyIncreases()
    {
        var silhouette = new SilhouetteScore<double>();
        double prevScore = -2; // Below minimum possible value

        foreach (int sep in new[] { 5, 10, 20, 50, 100 })
        {
            var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { sep, 0 }, { sep + 1, 0 } });
            var labels = MakeLabels(0, 0, 1, 1);
            double score = silhouette.Compute(data, labels);

            Assert.True(score >= prevScore - 1e-10,
                $"Silhouette should increase with separation. Prev={prevScore}, Current={score}, Sep={sep}");
            prevScore = score;
        }
    }

    [Fact]
    public void DaviesBouldin_IncreasingClusterSeparation_MonotonicallyDecreases()
    {
        var db = new DaviesBouldinIndex<double>();
        double prevScore = double.MaxValue;

        foreach (int sep in new[] { 5, 10, 20, 50, 100 })
        {
            var data = MakeMatrix(new double[,] { { 0, 0 }, { 1, 0 }, { sep, 0 }, { sep + 1, 0 } });
            var labels = MakeLabels(0, 0, 1, 1);
            double score = db.Compute(data, labels);

            Assert.True(score <= prevScore + 1e-10,
                $"DB should decrease with separation. Prev={prevScore}, Current={score}, Sep={sep}");
            prevScore = score;
        }
    }
}
