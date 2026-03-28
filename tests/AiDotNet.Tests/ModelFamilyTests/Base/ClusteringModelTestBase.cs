using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for clustering models. Tests deep mathematical invariants
/// that any correctly implemented clustering algorithm must satisfy.
/// Modeled after the Classification gold standard with domain-specific invariants.
/// </summary>
public abstract class ClusteringModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 90;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 3;
    protected virtual int NumClusters => 3;

    /// <summary>
    /// Whether this model exposes flat parameter vectors. Some clustering models
    /// (density-based like DBSCAN, hierarchical) don't have flat parameters.
    /// </summary>
    protected virtual bool HasFlatParameters => true;

    /// <summary>
    /// Override for models that generate data differently.
    /// </summary>
    protected virtual (Matrix<double> X, Vector<double> Y) GenerateData(
        int samples, int nClusters, int features, Random rng)
        => ModelTestHelpers.GenerateClusterData(samples, nClusters, features, rng);

    // =====================================================
    // MATHEMATICAL INVARIANT: Adjusted Rand Index > 0
    // On well-separated blobs, ANY clustering algorithm should
    // produce assignments correlated with ground truth (ARI > 0).
    // ARI = 0 means random; ARI < 0 means anti-correlated.
    // =====================================================

    [Fact]
    public void AdjustedRandIndex_ShouldBePositive()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(assignments))
        {
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(trainY, assignments);
            Assert.True(ari > 0.0,
                $"ARI = {ari:F4} — clustering is no better than random on well-separated blobs.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: High Purity on Separable Data
    // On perfectly separable blobs (spacing=10, std=0.5), purity
    // should be > 80%. This is the clustering equivalent of
    // Classification's "Accuracy_ShouldBeHigh_OnPerfectlySeparableData".
    // =====================================================

    [Fact]
    public void IntraClusterPurity_ShouldBeHigh_OnSeparableData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (!ModelTestHelpers.AllFinite(assignments)) return;

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
                totalPurity += (double)labelCounts.Values.Max() / groupSize;
                numGroups++;
            }
        }

        if (numGroups > 0)
        {
            double avgPurity = totalPurity / numGroups;
            Assert.True(avgPurity > 0.8,
                $"Average intra-cluster purity = {avgPurity:F4} (should be >0.8 on perfectly separable data).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data → Better or Equal Clustering
    // Doubling training data should not make ARI worse.
    // Mirrors Classification's "MoreData_ShouldNotDegrade_Accuracy".
    // =====================================================

    [Fact]
    public void MoreData_ShouldNotDegrade_ClusterQuality()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var (trainX1, trainY1) = GenerateData(30, NumClusters, Features, rng1);

        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model2 = CreateModel();
        var (trainX2, trainY2) = GenerateData(120, NumClusters, Features, rng2);

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, trainY2);

        var assign1 = model1.Predict(trainX1);
        var assign2 = model2.Predict(trainX2);

        if (ModelTestHelpers.AllFinite(assign1) && ModelTestHelpers.AllFinite(assign2))
        {
            double ari1 = ModelTestHelpers.CalculateAdjustedRandIndex(trainY1, assign1);
            double ari2 = ModelTestHelpers.CalculateAdjustedRandIndex(trainY2, assign2);

            Assert.True(ari2 >= ari1 - 0.2,
                $"4x more data made ARI worse: ARI(30)={ari1:F4}, ARI(120)={ari2:F4}.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Irrelevant Feature Immunity
    // Adding a pure noise feature should not significantly change
    // cluster structure. Mirrors Classification's IrrelevantFeature test.
    // =====================================================

    [Fact]
    public void IrrelevantFeature_ShouldNotDegrade_ClusterQuality()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX_real, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng1);

        // Add noise feature with SAME scale as real features (range 0-30 for 3 clusters at 0,10,20).
        // Using ModelTestHelpers.AddNoiseFeature (*100) would dominate Euclidean distance.
        var rngNoise = ModelTestHelpers.CreateSeededRandom(77);
        var trainX_noisy = new Matrix<double>(TrainSamples, Features + 1);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                trainX_noisy[i, j] = trainX_real[i, j];
            trainX_noisy[i, Features] = rngNoise.NextDouble() * 5.0; // smaller than inter-cluster distance
        }

        model1.Train(trainX_real, trainY);
        model2.Train(trainX_noisy, trainY);

        var assign1 = model1.Predict(trainX_real);
        var assign2 = model2.Predict(trainX_noisy);

        if (ModelTestHelpers.AllFinite(assign1) && ModelTestHelpers.AllFinite(assign2))
        {
            double ari1 = ModelTestHelpers.CalculateAdjustedRandIndex(trainY, assign1);
            double ari2 = ModelTestHelpers.CalculateAdjustedRandIndex(trainY, assign2);

            Assert.True(ari2 >= ari1 - 0.5,
                $"Adding noise feature degraded clustering: ARI_clean={ari1:F4}, ARI_noisy={ari2:F4}.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance
    // Shifting all points by a constant vector should produce the
    // SAME cluster assignments. Cluster structure is relative, not absolute.
    // =====================================================

    [Fact]
    public void TranslationEquivariance_ShiftingPoints_PreservesAssignments()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = GenerateData(TrainSamples, NumClusters, Features, rng1);
        var (trainX2, trainY2) = GenerateData(TrainSamples, NumClusters, Features, rng2);

        // Shift all points by (1000, 1000, ...)
        var shiftedX = new Matrix<double>(TrainSamples, Features);
        for (int i = 0; i < TrainSamples; i++)
            for (int j = 0; j < Features; j++)
                shiftedX[i, j] = trainX2[i, j] + 1000.0;

        model1.Train(trainX1, trainY1);
        model2.Train(shiftedX, trainY2);

        var assign1 = model1.Predict(trainX1);
        var assign2 = model2.Predict(shiftedX);

        if (ModelTestHelpers.AllFinite(assign1) && ModelTestHelpers.AllFinite(assign2))
        {
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(assign1, assign2);
            Assert.True(ari > 0.8,
                $"Translation equivariance violated: ARI between original and shifted = {ari:F4}. " +
                "Shifting all points by constant should preserve cluster structure.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Uniform Scaling Equivariance
    // Scaling all features by same constant should preserve assignments.
    // =====================================================

    [Fact]
    public void UniformScaling_ShouldPreserveAssignments()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = GenerateData(TrainSamples, NumClusters, Features, rng1);
        var (trainX2, trainY2) = GenerateData(TrainSamples, NumClusters, Features, rng2);

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
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(assign1, assign2);
            Assert.True(ari > 0.5,
                $"ARI between original and 100x-scaled = {ari:F4}. " +
                "Uniform scaling shouldn't change cluster structure.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Cluster Means Near Centers
    // The mean of points in each cluster should be near a generation center.
    // =====================================================

    [Fact]
    public void ClusterMeans_ShouldBeNearGenerationCenters()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        if (!ModelTestHelpers.AllFinite(assignments)) return;

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
                    $"Cluster label={label} mean is {minDist:F2} away from nearest generation center.");
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Correct Number of Distinct Clusters
    // =====================================================

    [Fact]
    public void DistinctClusters_ShouldBeReasonable()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        var distinctClusters = new HashSet<int>();
        for (int i = 0; i < assignments.Length; i++)
        {
            if (!double.IsNaN(assignments[i]) && !double.IsInfinity(assignments[i]))
                distinctClusters.Add((int)Math.Round(assignments[i]));
        }

        Assert.True(distinctClusters.Count >= 2,
            $"Only {distinctClusters.Count} distinct clusters found. Model collapsed to single cluster.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Identical Points → Same Cluster
    // =====================================================

    [Fact]
    public void IdenticalPoints_ShouldGetSameCluster()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

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
    // MATHEMATICAL INVARIANT: Predictions Are Finite
    // =====================================================

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        var assignments = model.Predict(trainX);

        for (int i = 0; i < assignments.Length; i++)
        {
            Assert.False(double.IsNaN(assignments[i]), $"Assignment[{i}] is NaN.");
            Assert.False(double.IsInfinity(assignments[i]), $"Assignment[{i}] is Infinity.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Output Shape, Clone, Metadata, Parameters
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

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
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);
        var (testX, _) = GenerateData(TestSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.Equal(TestSamples, model.Predict(testX).Length);
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalAssignments()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

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
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        if (!HasFlatParameters) return;

        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);

        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0,
            "Trained clustering model should have parameters (e.g., centroids).");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Single Cluster Data
    // When all points are near one center, the model should assign
    // (almost) all points to the same cluster.
    // =====================================================

    /// <summary>
    /// Creates a model configured for single-cluster detection. For k-based models
    /// (KMeans, MiniBatchKMeans), this should return the model with k=1.
    /// Default: returns the standard model (correct for auto-k algorithms like DBSCAN, MeanShift).
    /// </summary>
    protected virtual IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => CreateModel();

    [Fact]
    public void SingleClusterData_ShouldAssignSameCluster()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateSingleClusterModel();

        // All points clustered tightly around one center
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = 5.0 + ModelTestHelpers.NextGaussian(rng) * 0.01;
            y[i] = 0;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            // Count distinct assignments
            var distinctClusters = new HashSet<int>();
            for (int i = 0; i < predictions.Length; i++)
                distinctClusters.Add((int)Math.Round(predictions[i]));

            Assert.True(distinctClusters.Count <= 2,
                $"Single-cluster data produced {distinctClusters.Count} clusters. " +
                "Expected 1 (or at most 2 with noise) for tightly grouped data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Permutation Invariance
    // Shuffling row order should produce equivalent cluster assignments.
    // =====================================================

    [Fact]
    public void PermutationInvariance_ShuffledRows_SameAssignments()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (x1, y1) = GenerateData(TrainSamples, NumClusters, Features, rng1);
        var (x2, y2) = GenerateData(TrainSamples, NumClusters, Features, rng2);

        model1.Train(x1, y1);
        model2.Train(x2, y2);

        var pred1 = model1.Predict(x1);
        var pred2 = model2.Predict(x2);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            // Same data, same seed → should produce same assignments
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(pred1, pred2);
            Assert.True(ari > 0.8,
                $"ARI = {ari:F4} between identical data runs. " +
                "Clustering should be deterministic for same input.");
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);
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
    public void Builder_ClusteringShouldBeatRandom()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = GenerateData(TrainSamples, NumClusters, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var assignments = result.Predict(trainX);
        if (ModelTestHelpers.AllFinite(assignments))
        {
            double ari = ModelTestHelpers.CalculateAdjustedRandIndex(trainY, assignments);
            Assert.True(ari > 0.0,
                $"Builder pipeline ARI = {ari:F4} — clustering through builder should beat random.");
        }
    }
}
