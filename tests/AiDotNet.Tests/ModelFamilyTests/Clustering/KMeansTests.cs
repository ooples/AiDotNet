using AiDotNet.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class KMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new KMeans<double>(new KMeansOptions<double> { NumClusters = NumClusters });

    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new KMeans<double>(new KMeansOptions<double> { NumClusters = 1 });

    /// <summary>
    /// Plain Euclidean distance trains on a dedicated dense kernel; any other metric, including a subclass of
    /// <see cref="EuclideanDistance{T}"/>, keeps the generic path. Both must find the same clustering.
    /// </summary>
    [Fact]
    public void Train_EuclideanFastPathMatchesGenericMetricPath()
    {
        var rand = RandomHelper.CreateSeededRandom(11);
        const int rows = 600, columns = 5;
        var x = new Matrix<double>(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                x[i, j] = (i % 4) * 3.0 * (j % 2 == 0 ? 1 : -1) + rand.NextDouble() * 4 - 2;

        KMeans<double> Fit(IDistanceMetric<double>? metric)
        {
            var model = new KMeans<double>(new KMeansOptions<double>
            {
                NumClusters = 4, NumInitializations = 3, Seed = 7, DistanceMetric = metric
            });
            model.Train(x, new Vector<double>(rows));
            return model;
        }

        var fast = Fit(null);
        var generic = Fit(new SubclassedEuclidean());

        Assert.NotNull(fast.Labels);
        Assert.NotNull(generic.Labels);
        Assert.NotNull(fast.ClusterCenters);
        Assert.NotNull(generic.ClusterCenters);
        var fastLabels = fast.Labels ?? new Vector<double>(0);
        var genericLabels = generic.Labels ?? new Vector<double>(0);
        var fastCenters = fast.ClusterCenters ?? new Matrix<double>(0, 0);
        var genericCenters = generic.ClusterCenters ?? new Matrix<double>(0, 0);

        Assert.Equal(generic.NumIterations, fast.NumIterations);
        for (int i = 0; i < rows; i++)
            Assert.Equal(genericLabels[i], fastLabels[i]);
        for (int k = 0; k < 4; k++)
            for (int j = 0; j < columns; j++)
                Assert.Equal(genericCenters[k, j], fastCenters[k, j], 9);
        Assert.Equal(generic.Inertia, fast.Inertia, 6);

        var probe = new Matrix<double>(3, columns);
        for (int j = 0; j < columns; j++) { probe[0, j] = 0.5; probe[1, j] = -4 + j; probe[2, j] = 6 - j; }
        var fastPredicted = fast.Predict(probe);
        var genericPredicted = generic.Predict(probe);
        var fastDistances = fast.Transform(probe);
        var genericDistances = generic.Transform(probe);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(genericPredicted[i], fastPredicted[i]);
            for (int k = 0; k < 4; k++)
                Assert.Equal(genericDistances[i, k], fastDistances[i, k], 9);
        }
    }

    /// <summary>
    /// The labels a fit returns are those of the centers it returns: each training row's label is its nearest center,
    /// so Predict on the training data reproduces Labels. A fit stopped by the center-shift tolerance (MaxIterations
    /// and Tolerance loose enough that labels are still moving) must still relabel by the final centers.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Train_LabelsAreThoseOfTheReturnedCenters(bool genericMetric)
    {
        const int rows = 400, columns = 3;
        var x = new Matrix<double>(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                x[i, j] = ((i * 37 + j * 101 + i * j * 13) % 97) / 97.0;

        var model = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 5, NumInitializations = 1, Seed = 2, MaxIterations = 3, Tolerance = 0.5,
            DistanceMetric = genericMetric ? new SubclassedEuclidean() : null
        });
        model.Train(x, new Vector<double>(rows));

        var labels = model.Labels ?? new Vector<double>(0);
        var predicted = model.Predict(x);
        Assert.Equal(rows, labels.Length);
        for (int i = 0; i < rows; i++)
            Assert.Equal(predicted[i], labels[i]);
    }

    /// <summary>
    /// Greedy k-means++ (the default) keeps the best of several drawn candidates per center, so its seeding leaves less
    /// squared distance than classic k-means++ (one candidate). On uniform data no seeding finds a perfect clustering, so
    /// the quality of each draw shows; one Lloyd step makes the inertia a measure of the seeding. Summed over twenty
    /// seeds, on both training paths.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Train_GreedySeedingLeavesLessInertiaThanClassic(bool genericMetric)
    {
        const int rows = 1000;
        var rand = RandomHelper.CreateSeededRandom(5);
        var x = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            x[i, 0] = rand.NextDouble() * 10;
            x[i, 1] = rand.NextDouble() * 10;
        }

        double TotalInertia(int? localTrials)
        {
            double total = 0;
            for (int seed = 1; seed <= 20; seed++)
            {
                var model = new KMeans<double>(new KMeansOptions<double>
                {
                    NumClusters = 8, NumInitializations = 1, Seed = seed, MaxIterations = 1,
                    KMeansPlusPlusLocalTrials = localTrials,
                    DistanceMetric = genericMetric ? new SubclassedEuclidean() : null
                });
                model.Train(x, new Vector<double>(rows));
                total += model.Inertia;
            }

            return total;
        }

        double greedy = TotalInertia(null);
        double classic = TotalInertia(1);
        Assert.True(greedy < classic, $"Greedy seeding left {greedy} total inertia; classic left {classic}.");
    }

    [Fact]
    public void Train_RejectsFewerThanOneLocalTrial()
    {
        var model = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2, KMeansPlusPlusLocalTrials = 0 });
        var x = new Matrix<double>(4, 1);
        for (int i = 0; i < 4; i++) x[i, 0] = i;

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Train(x, new Vector<double>(4)));
    }

    /// <summary>Euclidean distance under another type, which routes training through the generic metric path.</summary>
    private sealed class SubclassedEuclidean : EuclideanDistance<double>
    {
    }
}
