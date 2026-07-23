using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Asserts that building a clustering model auto-evaluates it with cluster-validity indices, and that
/// ConfigureClusterMetric / ConfigureExternalClusterMetric add a custom index to that set.
/// </summary>
/// <remarks>
/// Before this wiring, ConfigureClusterMetric stored a private field no consumer read, so the metric
/// was silently dropped and no clustering model produced any evaluation. These tests pin the observable
/// behaviour: AiModelResult.ClusteringEvaluation is populated for any clustering model, a configured
/// custom metric appears in it, and external metrics run only when integer ground-truth labels exist.
/// </remarks>
public class ClusteringAutoEvaluationBuildTests
{
    private static (KMeans<double> model, InMemoryDataLoader<double, Matrix<double>, Vector<double>> loader, ClusteringDataset data)
        BuildKMeansSetup()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var model = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2, Seed = 42 });
        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(dataset.Data, dataset.Labels);
        return (model, loader, dataset);
    }

    [Fact(Timeout = 120000)]
    public async Task Build_ClusteringModel_AutoEvaluatesWithDefaultInternalIndices()
    {
        var (model, loader, _) = BuildKMeansSetup();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .BuildAsync();

        Assert.NotNull(result.ClusteringEvaluation);
        // The default internal indices run for any clustering model, with no configuration.
        Assert.Contains("Silhouette Score", result.ClusteringEvaluation!.InternalMetrics.Keys);
        Assert.Contains("Davies-Bouldin Index", result.ClusteringEvaluation.InternalMetrics.Keys);
        Assert.Equal(2, result.ClusteringEvaluation.NumClusters);
        // Same values are mirrored, by name, into ConfiguredMetrics.
        Assert.Contains("Silhouette Score", result.ConfiguredMetrics.Keys);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureClusterMetric_AddsCustomIndexToTheDefaultSet()
    {
        var (model, loader, _) = BuildKMeansSetup();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureClusterMetric(new WCSS<double>())
            .BuildAsync();

        Assert.NotNull(result.ClusteringEvaluation);
        // The configured custom metric appears alongside the defaults, not replacing them.
        Assert.Contains(new WCSS<double>().Name, result.ClusteringEvaluation!.InternalMetrics.Keys);
        Assert.Contains("Silhouette Score", result.ClusteringEvaluation.InternalMetrics.Keys);
    }

    [Fact(Timeout = 120000)]
    public async Task Build_WithIntegerGroundTruth_ComputesExternalIndices()
    {
        var (model, loader, _) = BuildKMeansSetup();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureExternalClusterMetric(new AdjustedRandIndex<double>())
            .BuildAsync();

        Assert.NotNull(result.ClusteringEvaluation);
        // The two blobs are integer-labelled ground truth, so external metrics run and the configured
        // one is present.
        Assert.NotEmpty(result.ClusteringEvaluation!.ExternalMetrics);
        Assert.Contains(
            result.ClusteringEvaluation.ExternalMetrics.Keys,
            k => k.Contains("Rand"));
    }
}
