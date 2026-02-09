using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class ClusteringAdditionalMetricsIntegrationTests
{
    [Fact]
    public void WcssAndBcss_ComputeExpectedValues()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 2, spacing: 10.0);
        var wcss = new WCSS<double>();

        double total = wcss.Compute(dataset.Data, dataset.Labels);

        Assert.Equal(0.08, total, 1e-6);

        var perCluster = wcss.ComputePerCluster(dataset.Data, dataset.Labels);
        Assert.Equal(2, perCluster.Count);
        Assert.Equal(0.04, perCluster[0], 1e-6);
        Assert.Equal(0.04, perCluster[1], 1e-6);

        var bcss = new BCSS<double>();
        double bcssValue = bcss.Compute(dataset.Data, dataset.Labels);
        Assert.Equal(200.0, bcssValue, 1e-6);
    }

    [Fact]
    public void PurityAndFMeasure_ReturnPerfectForIdenticalLabels()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 3);
        var purity = new Purity<double>();

        double purityScore = purity.Compute(dataset.Labels, dataset.Labels);
        Assert.Equal(1.0, purityScore, 1e-6);

        var perCluster = purity.ComputePerCluster(dataset.Labels, dataset.Labels);
        Assert.Equal(2, perCluster.Count);
        Assert.Equal(1.0, perCluster[0], 1e-6);
        Assert.Equal(1.0, perCluster[1], 1e-6);

        var fmeasure = new FMeasure<double>();
        Assert.Equal(1.0, fmeasure.Compute(dataset.Labels, dataset.Labels), 1e-6);

        var matrix = fmeasure.ComputeMatrix(dataset.Labels, dataset.Labels);
        Assert.True(matrix[(0, 0)] >= 1.0 - 1e-6);
        Assert.True(matrix[(1, 1)] >= 1.0 - 1e-6);
        Assert.Equal(1.0, fmeasure.ComputeBCubed(dataset.Labels, dataset.Labels), 1e-6);
    }

    [Fact]
    public void EntropyAndVariationOfInformation_ReportExpectedRanges()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 3);

        var entropy = new ClusteringEntropy<double>();
        double entropyValue = entropy.Compute(dataset.Labels, dataset.Labels);
        Assert.Equal(0.0, entropyValue, 1e-6);

        var perCluster = entropy.ComputePerCluster(dataset.Labels, dataset.Labels);
        Assert.Equal(2, perCluster.Count);
        Assert.Equal(0.0, perCluster[0], 1e-6);
        Assert.Equal(0.0, perCluster[1], 1e-6);

        double normalized = entropy.ComputeNormalized(dataset.Labels, dataset.Labels);
        Assert.Equal(0.0, normalized, 1e-6);

        var conditionalEntropy = new ConditionalEntropy<double>();
        Assert.Equal(0.0, conditionalEntropy.Compute(dataset.Labels, dataset.Labels), 1e-6);

        var homogeneity = new Homogeneity<double>();
        Assert.Equal(1.0, homogeneity.Compute(dataset.Labels, dataset.Labels), 1e-6);

        var completeness = new Completeness<double>();
        Assert.Equal(1.0, completeness.Compute(dataset.Labels, dataset.Labels), 1e-6);

        var vi = new VariationOfInformation<double>();
        double viValue = vi.Compute(dataset.Labels, dataset.Labels);
        Assert.Equal(0.0, viValue, 1e-6);

        double nmi = vi.ComputeNMI(dataset.Labels, dataset.Labels);
        Assert.Equal(1.0, nmi, 1e-6);

        double ami = vi.ComputeAMI(dataset.Labels, dataset.Labels);
        Assert.Equal(1.0, ami, 1e-6);

        var adjustedMi = new AdjustedMutualInformation<double>();
        Assert.Equal(1.0, adjustedMi.Compute(dataset.Labels, dataset.Labels), 1e-6);

        var singleCluster = new Vector<double>(dataset.Labels.Length);
        for (int i = 0; i < singleCluster.Length; i++)
        {
            singleCluster[i] = 0.0;
        }

        double viMismatch = vi.Compute(dataset.Labels, singleCluster);
        Assert.True(viMismatch >= 0.0);

        var normalizedVi = new VariationOfInformation<double>(normalized: true);
        double viNormalizedValue = normalizedVi.Compute(dataset.Labels, singleCluster);
        Assert.True(viNormalizedValue >= 0.0 && viNormalizedValue <= 1.0 + 1e-6);
    }

    [Fact]
    public void ClusterMetrics_EvaluateAndToString_ReturnsValues()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 4);
        var kmeans = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 30,
            NumInitializations = 1,
            Seed = 12
        });

        var labels = kmeans.FitPredict(dataset.Data);
        var metrics = new ClusterMetrics<double>();
        var scores = metrics.Evaluate(dataset.Data, labels, dataset.Labels);

        Assert.True(scores.HasExternalMetrics);
        Assert.InRange(scores.Silhouette, -1.0, 1.0);
        Assert.True(scores.DaviesBouldin >= 0.0);
        Assert.True(scores.CalinskiHarabasz > 0.0);
        Assert.True(scores.AdjustedRandIndex.HasValue);
        Assert.True(scores.NormalizedMutualInformation.HasValue);

        var summary = scores.ToString();
        Assert.Contains("Clustering Evaluation Metrics", summary);
    }

    [Fact]
    public void ClusterMetrics_EvaluateExternal_ReturnsExternalMetrics()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 3);
        var metrics = new ClusterMetrics<double>();
        var scores = metrics.EvaluateExternal(dataset.Labels, dataset.Labels);

        Assert.True(scores.HasExternalMetrics);
        Assert.Equal(1.0, scores.AdjustedRandIndex.Value, 1e-6);
        Assert.Equal(1.0, scores.NormalizedMutualInformation.Value, 1e-6);
    }

    [Fact]
    public void ExternalClusterMetricBase_UtilityMethods_ReturnExpectedValues()
    {
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predictedLabels = new Vector<double>(new[] { 0.0, 1.0, 1.0, 1.0 });

        var probe = new ExternalMetricProbe();
        var (contingency, trueCounts, predCounts) = probe.BuildTable(trueLabels, predictedLabels);

        Assert.Equal(2, trueCounts.Count);
        Assert.Equal(2, predCounts.Count);
        Assert.True(contingency.Count > 0);

        double entropy = probe.EntropyFromCounts(trueCounts, trueLabels.Length);
        Assert.Equal(1.0, entropy, 1e-6);

        double labelEntropy = probe.Entropy(trueLabels);
        Assert.Equal(entropy, labelEntropy, 1e-6);

        double mutualInfo = probe.MutualInfo(contingency, trueCounts, predCounts, trueLabels.Length);
        Assert.True(mutualInfo >= 0.0);

        var pairCounts = probe.PairCounts(trueLabels, predictedLabels);
        Assert.Equal(1L, pairCounts.A);
        Assert.Equal(1L, pairCounts.B);
        Assert.Equal(2L, pairCounts.C);
        Assert.Equal(2L, pairCounts.D);

        Assert.Equal(10.0, probe.BinomialPublic(5, 2), 1e-6);
    }

    private sealed class ExternalMetricProbe : ExternalClusterMetricBase<double>
    {
        public override double Compute(Vector<double> trueLabels, Vector<double> predictedLabels)
        {
            return 0.0;
        }

        public (Dictionary<(int True, int Pred), int> Contingency,
            Dictionary<int, int> TrueCounts,
            Dictionary<int, int> PredCounts)
            BuildTable(Vector<double> trueLabels, Vector<double> predictedLabels)
        {
            return BuildContingencyTable(trueLabels, predictedLabels);
        }

        public double EntropyFromCounts(Dictionary<int, int> counts, int total)
        {
            return ComputeEntropyFromCounts(counts, total);
        }

        public double Entropy(Vector<double> labels)
        {
            return ComputeEntropy(labels);
        }

        public double MutualInfo(
            Dictionary<(int True, int Pred), int> contingency,
            Dictionary<int, int> trueCounts,
            Dictionary<int, int> predCounts,
            int total)
        {
            return ComputeMutualInformation(contingency, trueCounts, predCounts, total);
        }

        public (long A, long B, long C, long D) PairCounts(
            Vector<double> trueLabels,
            Vector<double> predictedLabels)
        {
            return ComputePairConfusionMatrix(trueLabels, predictedLabels);
        }

        public double BinomialPublic(int n, int k)
        {
            return Binomial(n, k);
        }
    }
}
