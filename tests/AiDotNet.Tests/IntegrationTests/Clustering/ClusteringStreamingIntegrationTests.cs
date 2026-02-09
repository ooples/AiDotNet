using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Streaming;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class ClusteringStreamingIntegrationTests
{
    [Fact]
    public void StreamingMiniBatchKMeans_TrainAndPredict_Works()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 6);
        var options = new MiniBatchKMeansOptions<double>
        {
            NumClusters = 2,
            BatchSize = 4,
            MaxIterations = 20,
            MaxNoImprovement = 3,
            Seed = 42
        };

        var model = new MiniBatchKMeans<double>(options);
        model.Train(dataset.Data);

        Assert.True(model.IsTrained);
        var centers = ClusteringTestHelpers.RequireNotNull(model.ClusterCenters, "ClusterCenters");
        Assert.Equal(2, centers.Rows);
        Assert.Equal(dataset.Data.Columns, centers.Columns);

        var labels = ClusteringTestHelpers.RequireNotNull(model.Labels, "Labels");
        Assert.Equal(dataset.Data.Rows, labels.Length);
        ClusteringTestHelpers.AssertAllAssigned(labels);

        var predicted = model.Predict(dataset.Data);
        Assert.Equal(dataset.Data.Rows, predicted.Length);
    }

    [Fact]
    public void StreamingMiniBatchKMeans_PartialFit_UpdatesCenters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 4);
        var options = new MiniBatchKMeansOptions<double>
        {
            NumClusters = 2,
            BatchSize = 4,
            MaxIterations = 10,
            MaxNoImprovement = 2,
            Seed = 7
        };

        var model = new MiniBatchKMeans<double>(options);
        model.Train(dataset.Data);

        var centersBefore = ClusteringTestHelpers.RequireNotNull(model.ClusterCenters, "ClusterCenters");

        var newData = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 4, spacing: 12.0).Data;
        model.PartialFit(newData);

        var centersAfter = ClusteringTestHelpers.RequireNotNull(model.ClusterCenters, "ClusterCenters");
        Assert.Equal(centersBefore.Rows, centersAfter.Rows);
        Assert.Equal(centersBefore.Columns, centersAfter.Columns);
        Assert.True(!double.IsNaN(centersAfter[0, 0]));
        Assert.True(!double.IsInfinity(centersAfter[0, 0]));
    }

    [Fact]
    public void StreamingMiniBatchKMeans_FitPredict_ReturnsLabels()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 3);
        var model = new MiniBatchKMeans<double>(new MiniBatchKMeansOptions<double>
        {
            NumClusters = 2,
            BatchSize = 3,
            MaxIterations = 10,
            MaxNoImprovement = 2,
            Seed = 5
        });

        var labels = model.FitPredict(dataset.Data);

        Assert.Equal(dataset.Data.Rows, labels.Length);
        ClusteringTestHelpers.AssertAllAssigned(labels);
    }
}
