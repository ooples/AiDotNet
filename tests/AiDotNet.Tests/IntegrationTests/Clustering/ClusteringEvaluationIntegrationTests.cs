using AiDotNet.Clustering.AutoML;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class ClusteringEvaluationIntegrationTests
{
    [Fact]
    public void ElbowMethod_ComputesElbowAndWcss()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var method = new ElbowMethod<double>(randomState: 42);

        var result = method.Compute(dataset.Data, minClusters: 1, maxClusters: 4);

        Assert.Equal(result.KValues.Length, result.WCSSValues.Length);
        Assert.Equal(result.KValues.Length, result.ImprovementRates.Length);
        Assert.InRange(result.ElbowK, 1, 4);

        for (int i = 1; i < result.WCSSValues.Length; i++)
        {
            Assert.True(result.WCSSValues[i] <= result.WCSSValues[i - 1] + 1e-8);
        }
    }

    [Fact]
    public void GapStatistic_ComputesOptimalKAndValues()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var method = new GapStatistic<double>(numReferences: 3, randomState: 7);

        var result = method.Compute(dataset.Data, minClusters: 1, maxClusters: 4);

        Assert.Equal(result.KValues.Length, result.GapValues.Length);
        Assert.Equal(result.KValues.Length, result.StandardErrors.Length);
        Assert.Equal(result.KValues.Length, result.WCSSValues.Length);
        Assert.Equal(result.KValues.Length, result.ReferenceWCSSValues.Length);
        Assert.InRange(result.OptimalK, 1, 4);

        for (int i = 0; i < result.GapValues.Length; i++)
        {
            Assert.True(!double.IsNaN(result.GapValues[i]) && !double.IsInfinity(result.GapValues[i]));
            Assert.True(result.StandardErrors[i] >= 0);
        }
    }

    [Fact]
    public void StabilityValidation_EvaluateRange_ReturnsResults()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var validator = new StabilityValidation<double>(numSubsamples: 5, subsampleFraction: 0.75, randomState: 19);

        var analysis = validator.EvaluateRange(dataset.Data, minClusters: 2, maxClusters: 3);

        Assert.Equal(2, analysis.Results.Length);
        Assert.InRange(analysis.OptimalK, 2, 3);

        foreach (var result in analysis.Results)
        {
            Assert.True(result.NumComparisons > 0);
            Assert.Equal(result.NumComparisons, result.AllScores.Length);
            Assert.True(result.AverageStability >= -1.0 && result.AverageStability <= 1.0);
        }
    }

    [Fact]
    public void BootstrapValidation_Evaluate_ReturnsConfidenceIntervals()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var validator = new BootstrapValidation<double>(numBootstraps: 5, randomState: 11);

        var result = validator.Evaluate(dataset.Data, numClusters: 2, confidenceLevel: 0.9);

        Assert.Equal(5, result.NumBootstraps);
        Assert.Equal(0.9, result.ConfidenceLevel, 1e-6);
        Assert.True(result.Silhouette.NumSamples > 0);
        Assert.True(result.Silhouette.Lower <= result.Silhouette.Mean);
        Assert.True(result.Silhouette.Mean <= result.Silhouette.Upper);
    }

    [Fact]
    public void BootstrapValidation_ComputeAssignmentConfidence_ReturnsScoresInRange()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var validator = new BootstrapValidation<double>(numBootstraps: 5, randomState: 11);

        var confidence = validator.ComputeAssignmentConfidence(dataset.Data, numClusters: 2);

        Assert.Equal(dataset.Data.Rows, confidence.Length);
        for (int i = 0; i < confidence.Length; i++)
        {
            Assert.True(confidence[i] >= 0.0 && confidence[i] <= 1.0);
        }
    }

    [Fact]
    public void ClusteringEvaluator_EvaluateAll_ReturnsMetrics()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var kmeans = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 5
        });

        var labels = kmeans.FitPredict(dataset.Data);
        var evaluator = new ClusteringEvaluator<double>();

        var result = evaluator.EvaluateAll(dataset.Data, labels, dataset.Labels);

        Assert.True(result.InternalMetrics.ContainsKey("Silhouette Score"));
        Assert.True(result.InternalMetrics.ContainsKey("Davies-Bouldin Index"));
        Assert.True(result.InternalMetrics.ContainsKey("Calinski-Harabasz Index"));
        Assert.True(result.ExternalMetrics.ContainsKey("Adjusted Rand Index"));
        Assert.True(result.ExternalMetrics.ContainsKey("Normalized Mutual Information"));
        Assert.Equal(dataset.Data.Rows, result.NumPoints);
        Assert.Equal(result.NumClusters, result.ClusterSizes.Length);
    }

    [Fact]
    public void ClusteringEvaluator_CompareClusterings_RanksResults()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var k2 = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 5
        });

        var k3 = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 3,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 5
        });

        var labels2 = k2.FitPredict(dataset.Data);
        var labels3 = k3.FitPredict(dataset.Data);

        var evaluator = new ClusteringEvaluator<double>();
        var results = evaluator.CompareClusterings(
            dataset.Data,
            new List<Vector<double>> { labels2, labels3 },
            new List<string> { "K2", "K3" });

        Assert.Equal(2, results.Count);
        Assert.True(results[0].CompositeScore >= results[1].CompositeScore);

        var names = new HashSet<string>();
        foreach (var result in results)
        {
            names.Add(result.AlgorithmName);
        }
        Assert.Contains("K2", names);
        Assert.Contains("K3", names);
    }

    [Fact]
    public void ClusteringEvaluator_FindOptimalK_ReturnsRecommendations()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var evaluator = new ClusteringEvaluator<double>();

        var analysis = evaluator.FindOptimalK(
            dataset.Data,
            k =>
            {
                var model = new KMeans<double>(new KMeansOptions<double>
                {
                    NumClusters = k,
                    MaxIterations = 30,
                    NumInitializations = 1,
                    RandomState = 8
                });

                return model.FitPredict(dataset.Data);
            },
            (min: 2, max: 3));

        Assert.Equal(2, analysis.Results.Count);
        Assert.InRange(analysis.ConsensusK, 2, 3);
        Assert.True(analysis.Recommendations.Count > 0);
    }

    [Fact]
    public void ClusteringAutoML_Fit_ReturnsBestResult()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var options = new ClusteringAutoMLOptions
        {
            TryKMeans = true,
            TryDBSCAN = false,
            TryGaussianMixture = false,
            TryFuzzyCMeans = false,
            TryAutoK = false,
            KRange = new[] { 2, 3 },
            RandomSeed = 21,
            MaxIterationsPerTrial = 50
        };

        var automl = new ClusteringAutoML<double>(options);
        var result = automl.Fit(dataset.Data);

        Assert.Equal(2, result.TotalTrials);
        Assert.True(result.SuccessfulTrials > 0);
        var best = ClusteringTestHelpers.RequireNotNull(result.BestResult, "BestResult");
        Assert.True(best.Evaluation.NumClusters >= 2);
    }

    [Fact]
    public void ClusteringGridSearch_Search_ReturnsBestResult()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var gridSearch = new ClusteringGridSearch<double>();
        var grid = new Dictionary<string, object[]>
        {
            ["NumClusters"] = new object[] { 2, 3 },
            ["InitMethod"] = new object[] { KMeansInitMethod.Random, KMeansInitMethod.KMeansPlusPlus }
        };

        GridSearchResult<double> result = gridSearch.Search(
            dataset.Data,
            parameters =>
            {
                var options = new KMeansOptions<double>
                {
                    NumClusters = (int)parameters["NumClusters"],
                    InitMethod = (KMeansInitMethod)parameters["InitMethod"],
                    MaxIterations = 30,
                    NumInitializations = 1,
                    RandomState = 42
                };

                return new KMeans<double>(options);
            },
            grid);

        Assert.Equal(4, result.TotalCombinations);
        Assert.True(result.SuccessfulTrials > 0);
        _ = ClusteringTestHelpers.RequireNotNull(result.BestResult, "BestResult");
        Assert.Equal("Silhouette Score", result.PrimaryMetric);
    }

    [Fact]
    public void ClusteringGridSearch_SearchCV_ReturnsBestResult()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var gridSearch = new ClusteringGridSearch<double>();
        var grid = new Dictionary<string, object[]>
        {
            ["NumClusters"] = new object[] { 2, 3 },
            ["InitMethod"] = new object[] { KMeansInitMethod.KMeansPlusPlus }
        };

        GridSearchCVResult<double> result = gridSearch.SearchCV(
            dataset.Data,
            parameters =>
            {
                var options = new KMeansOptions<double>
                {
                    NumClusters = (int)parameters["NumClusters"],
                    InitMethod = (KMeansInitMethod)parameters["InitMethod"],
                    MaxIterations = 30,
                    NumInitializations = 1,
                    RandomState = 42
                };

                return new KMeans<double>(options);
            },
            grid,
            numFolds: 3);

        Assert.Equal(2, result.TotalCombinations);
        Assert.True(result.SuccessfulTrials > 0);
        var best = ClusteringTestHelpers.RequireNotNull(result.BestResult, "BestResult");
        Assert.True(best.FoldScores.Length > 0);
        Assert.Equal(3, result.NumFolds);
    }
}
