using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for clustering models. Tests deep mathematical invariants
/// that any correctly implemented clustering algorithm must satisfy.
/// </summary>
public abstract class ClusteringModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 90;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 2;
    protected virtual int NumClusters => 3;

    // =====================================================
    // MATHEMATICAL INVARIANT: Correct Number of Distinct Clusters
    // On well-separated data with K=3 blobs, the model should produce
    // exactly K (or close to K) distinct cluster labels. If it produces 1,
    // it collapsed. If it produces N, it didn't cluster at all.
    // =====================================================

    [Fact]
    public void DistinctClusters_ShouldBeReasonable()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        var distinctClusters = new HashSet<int>();
        for (int i = 0; i < assignments.Length; i++)
        {
            if (!double.IsNaN(assignments[i]) && !double.IsInfinity(assignments[i]))
                distinctClusters.Add((int)Math.Round(assignments[i]));
        }

        Assert.True(distinctClusters.Count >= 2,
            $"Only {distinctClusters.Count} distinct clusters found (expected ~{NumClusters}). " +
            "Model may have collapsed to a single cluster.");
        Assert.True(distinctClusters.Count <= TrainSamples / 2,
            $"{distinctClusters.Count} distinct clusters found — model may not be clustering at all.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Adjusted Rand Index > 0
    // The ground truth labels are known (from data generation).
    // ARI > 0 means the clustering is better than random assignment.
    // ARI = 0 means random; ARI < 0 means anti-correlated.
    // =====================================================

    [Fact]
    public void AdjustedRandIndex_ShouldBePositive()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(assignments))
        {
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(trainY, assignments);
            Assert.True(ari > 0.0,
                $"ARI = {ari:F4} — clustering is no better than random on well-separated blobs. " +
                "The algorithm may not be finding the true cluster structure.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Identical Points → Same Cluster
    // Duplicating a point must give the same cluster assignment.
    // Violation indicates non-deterministic assignment or input-order dependence.
    // =====================================================

    [Fact]
    public void IdenticalPoints_ShouldGetSameCluster()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);

        var duplicateX = new Matrix<double>(2, Features);
        for (int j = 0; j < Features; j++)
        {
            duplicateX[0, j] = trainX[0, j];
            duplicateX[1, j] = trainX[0, j];
        }

        var assignments = model.Predict(duplicateX);
        Assert.Equal(assignments[0], assignments[1]);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Cluster Compactness
    // Points within the same ground-truth cluster should (mostly) get
    // the same predicted label. Average intra-cluster purity should be > 0.6.
    // =====================================================

    [Fact]
    public void IntraClusterPurity_ShouldBeHigh()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (!ModelTestHelpers.AllFinite(assignments))
            return;

        // For each ground-truth cluster, find the most common predicted label
        double totalPurity = 0;
        int numGroups = 0;
        for (int c = 0; c < NumClusters; c++)
        {
            var labelCounts = new Dictionary<int, int>();
            int groupSize = 0;
            for (int i = 0; i < trainY.Length; i++)
            {
                if ((int)Math.Round(trainY[i]) == c)
                {
                    int predicted = (int)Math.Round(assignments[i]);
                    labelCounts[predicted] = labelCounts.GetValueOrDefault(predicted) + 1;
                    groupSize++;
                }
            }
            if (groupSize > 0 && labelCounts.Count > 0)
            {
                double maxFraction = (double)labelCounts.Values.Max() / groupSize;
                totalPurity += maxFraction;
                numGroups++;
            }
        }

        if (numGroups > 0)
        {
            double avgPurity = totalPurity / numGroups;
            Assert.True(avgPurity > 0.6,
                $"Average intra-cluster purity = {avgPurity:F4} (should be >0.6). " +
                "Clusters don't correspond well to ground truth on well-separated blobs.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Points Near Cluster Centers
    // For centroid-based methods, the mean of points assigned to a cluster
    // should be close to the center used for generation.
    // This catches centroid initialization bugs.
    // =====================================================

    [Fact]
    public void ClusterMeans_ShouldBeNearGenerationCenters()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        // Well-separated clusters at 0, 10, 20 (spacing = 10, std = 0.5)
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (!ModelTestHelpers.AllFinite(assignments))
            return;

        // Compute mean of each predicted cluster
        var distinctLabels = new HashSet<int>();
        for (int i = 0; i < assignments.Length; i++)
            distinctLabels.Add((int)Math.Round(assignments[i]));

        foreach (var label in distinctLabels)
        {
            double[] meanFeatures = new double[Features];
            int count = 0;
            for (int i = 0; i < assignments.Length; i++)
            {
                if ((int)Math.Round(assignments[i]) == label)
                {
                    for (int j = 0; j < Features; j++)
                        meanFeatures[j] += trainX[i, j];
                    count++;
                }
            }

            if (count > 0)
            {
                for (int j = 0; j < Features; j++)
                    meanFeatures[j] /= count;

                // Each cluster mean should be near one of the generation centers (0, 10, 20)
                double minDist = double.MaxValue;
                for (int c = 0; c < NumClusters; c++)
                {
                    double dist = 0;
                    for (int j = 0; j < Features; j++)
                    {
                        double diff = meanFeatures[j] - c * 10.0;
                        dist += diff * diff;
                    }
                    minDist = Math.Min(minDist, Math.Sqrt(dist));
                }

                Assert.True(minDist < 5.0,
                    $"Cluster label={label} mean is {minDist:F2} away from nearest generation center. " +
                    "Expected <5.0 for well-separated blobs with std=0.5.");
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scale Invariance
    // Scaling all features by same constant should not change assignments
    // (for distance-based methods with uniform scaling).
    // =====================================================

    [Fact]
    public void UniformScaling_ShouldPreserveAssignments()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng1);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng2);

        // Scale all features by 100
        var scaledX = new Matrix<double>(TrainSamples, Features);
        for (int i = 0; i < TrainSamples; i++)
            for (int j = 0; j < Features; j++)
                scaledX[i, j] = trainX2[i, j] * 100.0;

        model1.Train(trainX1, trainY1);
        model2.Train(scaledX, trainY2);

        var assign1 = model1.Predict(trainX1);
        var assign2 = model2.Predict(scaledX);

        if (ModelTestHelpers.AllFinite(assign1) && ModelTestHelpers.AllFinite(assign2))
        {
            // The actual label values may differ, but the clustering structure should be similar.
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(assign1, assign2);
            Assert.True(ari > 0.5,
                $"ARI between original and 100x-scaled = {ari:F4}. " +
                "Uniform scaling shouldn't change cluster structure.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Output Shape, Clone, Metadata
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var a1 = model.Predict(trainX);
        var a2 = model.Predict(trainX);

        for (int i = 0; i < a1.Length; i++)
            Assert.Equal(a1[i], a2[i]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateClusterData(TestSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.Equal(TestSamples, model.Predict(testX).Length);
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalAssignments()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();
        var a1 = model.Predict(trainX);
        var a2 = cloned.Predict(trainX);

        for (int i = 0; i < a1.Length; i++)
            Assert.Equal(a1[i], a2[i]);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0,
            "Trained clustering model should have parameters (e.g., centroids).");
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact]
    public void Builder_PredictionsShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateClusterData(TrainSamples, NumClusters, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateClusterData(TestSamples, NumClusters, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var assignments = result.Predict(testX);
        Assert.True(ModelTestHelpers.AllFinite(assignments),
            "Builder pipeline cluster assignments contain NaN or Infinity.");
    }
}
