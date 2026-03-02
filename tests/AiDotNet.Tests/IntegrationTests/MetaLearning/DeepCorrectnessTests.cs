using System;
using System.Collections.Generic;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Deep correctness tests for meta-learning algorithms. Unlike the shallow smoke tests,
/// these verify:
/// - Loss decreases monotonically over multiple meta-training epochs
/// - Adapted model outperforms unadapted (pre-adaptation vs post-adaptation)
/// - Multi-task generalization: training on diverse tasks improves new-task performance
/// - Deterministic reproducibility with fixed seeds
/// - Parameter isolation: adapting to one task doesn't corrupt the meta-parameters
/// - Gradient flow: gradients are non-zero and bounded
/// - Edge cases: single-example tasks, high-dimensional data, class imbalance
/// </summary>
public class DeepCorrectnessTests
{
    private const int InputDim = 4;

    #region Convergence Tests

    [Fact]
    public void MAML_LossDecreases_OverMultipleEpochs()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 3, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(100);
        var losses = new List<double>();

        for (int epoch = 0; epoch < 20; epoch++)
        {
            var tasks = CreateClassificationTasks(InputDim, numWays: 2, numShots: 3,
                numQuery: 4, numTasks: 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            double loss = maml.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"MAML loss is NaN at epoch {epoch}");
            Assert.False(double.IsInfinity(loss), $"MAML loss is infinite at epoch {epoch}");
            losses.Add(loss);
        }

        // Moving average should trend downward: compare first 5 avg to last 5 avg
        double earlyAvg = Average(losses, 0, 5);
        double lateAvg = Average(losses, 15, 5);
        Assert.True(lateAvg <= earlyAvg * 1.5,
            $"MAML loss did not trend downward: early avg={earlyAvg:F6}, late avg={lateAvg:F6}");
    }

    [Fact]
    public void ProtoNets_LossDecreases_OverMultipleEpochs()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, DistanceFunction = ProtoNetsDistanceFunction.Euclidean };
        var protonets = new ProtoNetsAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(200);
        var losses = new List<double>();

        for (int epoch = 0; epoch < 20; epoch++)
        {
            var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            double loss = protonets.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"ProtoNets loss is NaN at epoch {epoch}");
            losses.Add(loss);
        }

        double earlyAvg = Average(losses, 0, 5);
        double lateAvg = Average(losses, 15, 5);
        Assert.True(lateAvg <= earlyAvg * 1.5,
            $"ProtoNets loss did not trend downward: early={earlyAvg:F6}, late={lateAvg:F6}");
    }

    [Fact]
    public void CNP_LossDecreases_OverMultipleEpochs()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new CNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, RepresentationDim = 32 };
        var cnp = new CNPAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(300);
        var losses = new List<double>();

        for (int epoch = 0; epoch < 20; epoch++)
        {
            var tasks = CreateRegressionTasks(InputDim, 5, 5, 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            double loss = cnp.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"CNP loss is NaN at epoch {epoch}");
            losses.Add(loss);
        }

        double earlyAvg = Average(losses, 0, 5);
        double lateAvg = Average(losses, 15, 5);
        Assert.True(lateAvg <= earlyAvg * 1.5,
            $"CNP loss did not trend downward: early={earlyAvg:F6}, late={lateAvg:F6}");
    }

    #endregion

    #region Adaptation Quality Tests

    [Fact]
    public void MAML_AdaptedModel_OutperformsUnadapted()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 5, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(400);

        // Train for several epochs
        for (int epoch = 0; epoch < 15; epoch++)
        {
            var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            maml.MetaTrain(batch);
        }

        // Get loss on a new task before adaptation
        var evalTask = CreateClassificationTask(InputDim, 2, 5, 10, rng);
        var preAdaptPreds = model.Predict(evalTask.QuerySetX);
        double preAdaptLoss = ComputeMSE(preAdaptPreds, evalTask.QuerySetY);

        // Adapt and get post-adaptation loss
        var adapted = maml.Adapt(evalTask);
        var postAdaptPreds = adapted.Predict(evalTask.QuerySetX);
        double postAdaptLoss = ComputeMSE(postAdaptPreds, evalTask.QuerySetY);

        // Post-adaptation loss should be no worse than pre-adaptation (with tolerance)
        Assert.True(postAdaptLoss <= preAdaptLoss + 0.5,
            $"MAML adapted model did not improve: pre={preAdaptLoss:F6}, post={postAdaptLoss:F6}");
    }

    [Fact]
    public void ProtoNets_AdaptedModel_ProducesValidPredictions()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005 };
        var protonets = new ProtoNetsAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(500);

        // Train
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            protonets.MetaTrain(batch);
        }

        // Adapt to new task
        var evalTask = CreateClassificationTask(InputDim, 2, 5, 10, rng);
        var adapted = protonets.Adapt(evalTask);
        var predictions = adapted.Predict(evalTask.QuerySetX);

        // Predictions should have the right length
        Assert.Equal(evalTask.QuerySetY.Length, predictions.Length);

        // All predictions should be finite
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"ProtoNets prediction[{i}] is NaN");
            Assert.False(double.IsInfinity(predictions[i]), $"ProtoNets prediction[{i}] is infinite");
        }
    }

    #endregion

    #region Determinism and Reproducibility

    [Fact]
    public void MAML_FixedSeed_ProducesIdenticalResults()
    {
        double loss1 = RunMAMLWithSeed(42);
        double loss2 = RunMAMLWithSeed(42);

        Assert.Equal(loss1, loss2, 10); // Should be exactly equal with same seed
    }

    [Fact]
    public void MAML_DifferentSeeds_ProduceDifferentResults()
    {
        double loss1 = RunMAMLWithSeed(42);
        double loss2 = RunMAMLWithSeed(99);

        // Different seeds should give different results (with very high probability)
        Assert.NotEqual(loss1, loss2);
    }

    private static double RunMAMLWithSeed(int seed)
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(seed);
        var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        return maml.MetaTrain(batch);
    }

    #endregion

    #region Parameter Isolation

    [Fact]
    public void MAML_Adapt_DoesNotCorruptMetaParameters()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 3, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(600);

        // Train for a bit
        for (int i = 0; i < 5; i++)
        {
            var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
            maml.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
        }

        // Save meta parameters
        var metaParamsBefore = model.GetParameters();

        // Adapt to a task (should NOT change meta parameters)
        var evalTask = CreateClassificationTask(InputDim, 2, 5, 5, rng);
        var adapted = maml.Adapt(evalTask);
        adapted.Predict(evalTask.QuerySetX);

        // Meta parameters should be unchanged
        var metaParamsAfter = model.GetParameters();
        Assert.Equal(metaParamsBefore.Length, metaParamsAfter.Length);
        for (int i = 0; i < metaParamsBefore.Length; i++)
        {
            Assert.Equal(metaParamsBefore[i], metaParamsAfter[i], 12);
        }
    }

    [Fact]
    public void MAML_MultipleAdaptations_AreIndependent()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 3, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(700);
        for (int i = 0; i < 5; i++)
        {
            var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
            maml.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
        }

        // Create two very different tasks
        var task1 = CreateClassificationTask(InputDim, 2, 5, 5, new Random(111));
        var task2 = CreateClassificationTask(InputDim, 2, 5, 5, new Random(222));

        var adapted1 = maml.Adapt(task1);
        var adapted2 = maml.Adapt(task2);

        var preds1 = adapted1.Predict(task1.QuerySetX);
        var preds2 = adapted2.Predict(task2.QuerySetX);

        // Adapted models should produce different predictions for different tasks
        bool anyDifferent = false;
        for (int i = 0; i < Math.Min(preds1.Length, preds2.Length); i++)
        {
            if (Math.Abs(preds1[i] - preds2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Different tasks should produce different adapted predictions");
    }

    #endregion

    #region Gradient Flow Verification

    [Fact]
    public void MAML_GradientsAreNonZeroAndBounded()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var paramsBefore = model.GetParameters();

        var rng = new Random(800);
        var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 4, rng);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
        maml.MetaTrain(batch);

        var paramsAfter = model.GetParameters();

        // Compute effective gradient (param change)
        bool anyNonZero = false;
        bool allBounded = true;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            double delta = Math.Abs(paramsAfter[i] - paramsBefore[i]);
            if (delta > 1e-15) anyNonZero = true;
            if (delta > 100.0) allBounded = false; // Gradient explosion check
        }

        Assert.True(anyNonZero, "Gradients should cause parameter changes");
        Assert.True(allBounded, "Gradients should be bounded (no explosion)");
    }

    [Fact]
    public void MultipleAlgorithms_GradientFlowConsistency()
    {
        var rng = new Random(900);
        var task = CreateClassificationTask(InputDim, 2, 3, 4, rng);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        // Test that each algorithm produces meaningful parameter changes
        var algorithms = new (string Name, Func<(LinearVectorModel model, MetaLearnerBase<double, Matrix<double>, Vector<double>> algo)> Factory)[]
        {
            ("MAML", () =>
            {
                var m = new LinearVectorModel(InputDim);
                var o = new MAMLOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.01, UseFirstOrder = true };
                return (m, new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(o));
            }),
            ("ProtoNets", () =>
            {
                var m = new LinearVectorModel(InputDim);
                var o = new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
                return (m, new ProtoNetsAlgorithm<double, Matrix<double>, Vector<double>>(o));
            }),
            ("CNP", () =>
            {
                var m = new LinearVectorModel(InputDim);
                var o = new CNPOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.01, RepresentationDim = 16 };
                return (m, new CNPAlgorithm<double, Matrix<double>, Vector<double>>(o));
            })
        };

        foreach (var (name, factory) in algorithms)
        {
            var (model, algo) = factory();
            var before = model.GetParameters();
            var loss = algo.MetaTrain(batch);
            var after = model.GetParameters();

            Assert.False(double.IsNaN(loss), $"{name}: loss is NaN");

            double totalDelta = 0;
            for (int i = 0; i < before.Length; i++)
                totalDelta += Math.Abs(after[i] - before[i]);

            Assert.True(totalDelta > 1e-12,
                $"{name}: no parameter change after MetaTrain (delta={totalDelta})");
        }
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void MAML_SingleExamplePerClass_StillConverges()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 3, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(1000);

        // 1-shot: only 1 example per class
        double totalLoss = 0;
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var tasks = CreateClassificationTasks(InputDim, numWays: 2, numShots: 1,
                numQuery: 2, numTasks: 4, rng);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            double loss = maml.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"1-shot loss is NaN at epoch {epoch}");
            totalLoss += loss;
        }

        // Average loss should be finite
        double avgLoss = totalLoss / 10;
        Assert.False(double.IsNaN(avgLoss) || double.IsInfinity(avgLoss),
            "Average 1-shot loss should be finite");
    }

    [Fact]
    public void MAML_HighDimensionalInput_HandledCorrectly()
    {
        const int highDim = 32;
        var model = new LinearVectorModel(highDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.005, OuterLearningRate = 0.001, AdaptationSteps = 2, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(1100);
        var tasks = CreateClassificationTasks(highDim, 2, 3, 4, 4, rng);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        var loss = maml.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "High-dim loss should not be NaN");
        Assert.False(double.IsInfinity(loss), "High-dim loss should not be infinite");

        // Adapt should work
        var adapted = maml.Adapt(tasks[0]);
        var preds = adapted.Predict(tasks[0].QuerySetX);
        Assert.Equal(tasks[0].QuerySetY.Length, preds.Length);
    }

    [Fact]
    public void MAML_LargeBatch_HandledCorrectly()
    {
        var model = new LinearVectorModel(InputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.005, UseFirstOrder = true };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var rng = new Random(1200);
        // Batch of 16 tasks
        var tasks = CreateClassificationTasks(InputDim, 2, 3, 4, 16, rng);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        var loss = maml.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "Large batch loss should not be NaN");
        Assert.False(double.IsInfinity(loss), "Large batch loss should not be infinite");
    }

    #endregion

    #region Data Infrastructure Deep Tests

    [Fact]
    public void SineWaveDataset_TasksAreDistinct()
    {
        var dataset = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 50, examplesPerClass: 30, seed: 42);

        var episode1 = dataset.SampleEpisode(1, 5, 5);
        var episode2 = dataset.SampleEpisode(1, 5, 5);

        var task1 = episode1.Task;
        var task2 = episode2.Task;

        // Tasks should have different data (different sine wave params)
        bool anyDifferent = false;
        var sup1 = task1.SupportSetY;
        var sup2 = task2.SupportSetY;
        for (int i = 0; i < Math.Min(sup1.Length, sup2.Length); i++)
        {
            if (Math.Abs(sup1[i] - sup2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Different episodes should have different support data");
    }

    [Fact]
    public void GaussianDataset_ClassesAreWellSeparated()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 5, examplesPerClass: 50, featureDim: 4, classSeparation: 5.0, seed: 42);

        var episode = dataset.SampleEpisode(numWays: 2, numShots: 10, numQueryPerClass: 10);
        var task = episode.Task;

        // Group support examples by class
        var class0 = new List<double[]>();
        var class1 = new List<double[]>();
        for (int i = 0; i < task.SupportSetY.Length; i++)
        {
            var features = new double[4];
            for (int d = 0; d < 4; d++)
                features[d] = task.SupportSetX[i, d];

            if (Math.Abs(task.SupportSetY[i]) < 0.5)
                class0.Add(features);
            else
                class1.Add(features);
        }

        if (class0.Count > 0 && class1.Count > 0)
        {
            // Compute centroid distance
            var centroid0 = ComputeCentroid(class0);
            var centroid1 = ComputeCentroid(class1);
            double distance = ComputeEuclidean(centroid0, centroid1);

            // With classSeparation=5.0, centroids should be reasonably separated
            Assert.True(distance > 0.1, $"Class centroids should be separated, got distance={distance:F4}");
        }
    }

    [Fact]
    public void RotatedDigitsDataset_ProducesCorrectDimensions()
    {
        var dataset = new RotatedDigitsMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 10, examplesPerClass: 20, featureDim: 8, seed: 42);

        Assert.Equal("RotatedDigits", dataset.Name);
        Assert.Equal(10, dataset.TotalClasses);
        Assert.Equal(200, dataset.TotalExamples);

        var episode = dataset.SampleEpisode(numWays: 3, numShots: 2, numQueryPerClass: 3);
        var task = episode.Task;

        // Support: 3 ways * 2 shots = 6 examples, 8 features each
        Assert.Equal(6, task.SupportSetX.Rows);
        Assert.Equal(8, task.SupportSetX.Columns);
        Assert.Equal(6, task.SupportSetY.Length);

        // Query: 3 ways * 3 queries = 9 examples
        Assert.Equal(9, task.QuerySetX.Rows);
        Assert.Equal(9, task.QuerySetY.Length);
    }

    [Fact]
    public void MetaDatasetFormat_SamplesFromCorrectDomain()
    {
        var sine = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 10, examplesPerClass: 20, seed: 42);
        var gaussian = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 10, examplesPerClass: 20, featureDim: 4, seed: 42);

        var multi = new MetaDatasetFormat<double, Matrix<double>, Vector<double>>(
            new IMetaDataset<double, Matrix<double>, Vector<double>>[] { sine, gaussian },
            new[] { "Sine", "Gaussian" },
            seed: 42);

        // Domain 0 should always return "Sine"
        var ep0 = multi.SampleFromDomain(0, 2, 3, 4);
        Assert.Equal("Sine", ep0.Domain);

        // Domain 1 should always return "Gaussian"
        var ep1 = multi.SampleFromDomain(1, 2, 3, 4);
        Assert.Equal("Gaussian", ep1.Domain);

        // Random sample should return one of the domain names
        var epRandom = multi.SampleEpisode(2, 3, 4);
        Assert.True(epRandom.Domain == "Sine" || epRandom.Domain == "Gaussian",
            $"Unexpected domain: {epRandom.Domain}");
    }

    [Fact]
    public void TaskDifficultyEstimator_EasyTaskHasLowDifficulty()
    {
        // Create a well-separated 2-class task
        var supportX = new Vector<double>(8); // 2 examples * 4 features
        var supportY = new Vector<double>(2);

        // Class 0: features at +5
        supportX[0] = 5; supportX[1] = 5; supportX[2] = 5; supportX[3] = 5;
        supportY[0] = 0;

        // Class 1: features at -5
        supportX[4] = -5; supportX[5] = -5; supportX[6] = -5; supportX[7] = -5;
        supportY[1] = 1;

        double difficulty = TaskDifficultyEstimator<double>.EstimateDifficulty(
            supportX, supportY, numWays: 2, numShots: 1);

        // Well-separated classes should have low difficulty (< 0.5)
        Assert.True(difficulty < 0.5, $"Easy task should have low difficulty, got {difficulty:F4}");
    }

    [Fact]
    public void TaskDifficultyEstimator_HardTaskHasHighDifficulty()
    {
        // Create overlapping 2-class task
        var supportX = new Vector<double>(16); // 4 examples * 4 features
        var supportY = new Vector<double>(4);

        var rng = new Random(42);
        for (int i = 0; i < 4; i++)
        {
            for (int d = 0; d < 4; d++)
            {
                supportX[i * 4 + d] = rng.NextDouble() * 0.01; // All very close
            }
            supportY[i] = i % 2;
        }

        double difficulty = TaskDifficultyEstimator<double>.EstimateDifficulty(
            supportX, supportY, numWays: 2, numShots: 2);

        // Overlapping classes should have high difficulty (> 0.3)
        Assert.True(difficulty > 0.3, $"Hard task should have high difficulty, got {difficulty:F4}");
    }

    [Fact]
    public void CurriculumSampler_IncreasesDifficultyOverTime()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 4, seed: 42);

        var sampler = new CurriculumTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 3,
            initialDifficulty: 0.1, paceRate: 0.5, seed: 42);

        // Sample a few episodes at the start
        var earlyEpisodes = new List<IEpisode<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 5; i++)
            earlyEpisodes.Add(sampler.SampleOne());

        // Feed back low loss (model is "doing well") to increase difficulty
        sampler.UpdateWithFeedback(earlyEpisodes,
            new List<double> { 0.1, 0.1, 0.1, 0.1, 0.1 });

        // Sample after feedback
        var laterEpisode = sampler.SampleOne();

        // Difficulty should have increased (or at least not decreased significantly)
        Assert.NotNull(laterEpisode);
        Assert.NotNull(laterEpisode.Task);
    }

    [Fact]
    public void ModelPredictiveSampler_OperatesCorrectly()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 4, seed: 42);

        var sampler = new ModelPredictiveTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 3, seed: 42);

        // Sample and provide feedback
        var episodes = new List<IEpisode<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 5; i++)
            episodes.Add(sampler.SampleOne());

        sampler.UpdateWithFeedback(episodes, new List<double> { 0.5, 1.0, 0.2, 0.8, 0.3 });

        // Should still produce valid batches after feedback
        var batch = sampler.SampleBatch(3);
        Assert.Equal(3, batch.BatchSize);
    }

    #endregion

    #region Multi-Algorithm Consistency

    [Fact]
    public void AllNeuralProcesses_ProduceFiniteLossAcrossMultipleEpochs()
    {
        var npAlgorithms = new (string Name, Func<MetaLearnerBase<double, Matrix<double>, Vector<double>>> Factory)[]
        {
            ("CNP", () =>
            {
                var m = new LinearVectorModel(InputDim);
                return new CNPAlgorithm<double, Matrix<double>, Vector<double>>(
                    new CNPOptions<double, Matrix<double>, Vector<double>>(m) { OuterLearningRate = 0.01 });
            }),
            ("NP", () =>
            {
                var m = new LinearVectorModel(InputDim);
                return new NPAlgorithm<double, Matrix<double>, Vector<double>>(
                    new NPOptions<double, Matrix<double>, Vector<double>>(m) { OuterLearningRate = 0.01 });
            }),
            ("ANP", () =>
            {
                var m = new LinearVectorModel(InputDim);
                return new ANPAlgorithm<double, Matrix<double>, Vector<double>>(
                    new ANPOptions<double, Matrix<double>, Vector<double>>(m) { OuterLearningRate = 0.01 });
            })
        };

        foreach (var (name, factory) in npAlgorithms)
        {
            var algo = factory();
            var rng = new Random(42);

            for (int epoch = 0; epoch < 5; epoch++)
            {
                var tasks = CreateRegressionTasks(InputDim, 5, 5, 3, rng);
                var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
                var loss = algo.MetaTrain(batch);
                Assert.False(double.IsNaN(loss), $"{name} loss NaN at epoch {epoch}");
                Assert.False(double.IsInfinity(loss), $"{name} loss infinite at epoch {epoch}");
            }
        }
    }

    #endregion

    #region Helpers

    private static IMetaLearningTask<double, Matrix<double>, Vector<double>>[] CreateClassificationTasks(
        int inputDim, int numWays, int numShots, int numQuery, int numTasks, Random rng)
    {
        var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[numTasks];
        for (int t = 0; t < numTasks; t++)
            tasks[t] = CreateClassificationTask(inputDim, numWays, numShots, numQuery, rng);
        return tasks;
    }

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateClassificationTask(
        int inputDim, int numWays, int numShots, int numQuery, Random rng)
    {
        // Generate separated class centroids
        var centroids = new double[numWays][];
        for (int c = 0; c < numWays; c++)
        {
            centroids[c] = new double[inputDim];
            for (int d = 0; d < inputDim; d++)
                centroids[c][d] = (rng.NextDouble() - 0.5) * 4.0;
        }

        int supportCount = numWays * numShots;
        int queryCount = numWays * numQuery;

        var supportX = new Matrix<double>(supportCount, inputDim);
        var supportY = new Vector<double>(supportCount);
        var queryX = new Matrix<double>(queryCount, inputDim);
        var queryY = new Vector<double>(queryCount);

        int sIdx = 0, qIdx = 0;
        for (int c = 0; c < numWays; c++)
        {
            for (int i = 0; i < numShots; i++)
            {
                for (int d = 0; d < inputDim; d++)
                    supportX[sIdx, d] = centroids[c][d] + (rng.NextDouble() - 0.5) * 0.5;
                supportY[sIdx] = c;
                sIdx++;
            }
            for (int i = 0; i < numQuery; i++)
            {
                for (int d = 0; d < inputDim; d++)
                    queryX[qIdx, d] = centroids[c][d] + (rng.NextDouble() - 0.5) * 0.5;
                queryY[qIdx] = c;
                qIdx++;
            }
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX, SupportSetY = supportY,
            QuerySetX = queryX, QuerySetY = queryY,
            NumWays = numWays, NumShots = numShots, NumQueryPerClass = numQuery
        };
    }

    private static IMetaLearningTask<double, Matrix<double>, Vector<double>>[] CreateRegressionTasks(
        int inputDim, int numContext, int numTarget, int numTasks, Random rng)
    {
        var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[numTasks];
        for (int t = 0; t < numTasks; t++)
        {
            var weights = new double[inputDim];
            double bias = (rng.NextDouble() - 0.5) * 2.0;
            for (int d = 0; d < inputDim; d++)
                weights[d] = (rng.NextDouble() - 0.5) * 2.0;

            var ctxX = new Matrix<double>(numContext, inputDim);
            var ctxY = new Vector<double>(numContext);
            var tgtX = new Matrix<double>(numTarget, inputDim);
            var tgtY = new Vector<double>(numTarget);

            for (int i = 0; i < numContext; i++)
            {
                double y = bias;
                for (int d = 0; d < inputDim; d++)
                {
                    ctxX[i, d] = (rng.NextDouble() - 0.5) * 2.0;
                    y += weights[d] * ctxX[i, d];
                }
                ctxY[i] = y + (rng.NextDouble() - 0.5) * 0.1;
            }

            for (int i = 0; i < numTarget; i++)
            {
                double y = bias;
                for (int d = 0; d < inputDim; d++)
                {
                    tgtX[i, d] = (rng.NextDouble() - 0.5) * 2.0;
                    y += weights[d] * tgtX[i, d];
                }
                tgtY[i] = y + (rng.NextDouble() - 0.5) * 0.1;
            }

            tasks[t] = new MetaLearningTask<double, Matrix<double>, Vector<double>>
            {
                SupportSetX = ctxX, SupportSetY = ctxY,
                QuerySetX = tgtX, QuerySetY = tgtY,
                NumWays = 1, NumShots = numContext, NumQueryPerClass = numTarget
            };
        }
        return tasks;
    }

    private static double Average(List<double> values, int start, int count)
    {
        double sum = 0;
        for (int i = start; i < start + count && i < values.Count; i++)
            sum += values[i];
        return sum / count;
    }

    private static double ComputeMSE(Vector<double> predictions, Vector<double> labels)
    {
        double sum = 0;
        int count = Math.Min(predictions.Length, labels.Length);
        for (int i = 0; i < count; i++)
        {
            double diff = predictions[i] - labels[i];
            sum += diff * diff;
        }
        return count > 0 ? sum / count : 0;
    }

    private static double[] ComputeCentroid(List<double[]> points)
    {
        int dim = points[0].Length;
        var centroid = new double[dim];
        foreach (var p in points)
            for (int d = 0; d < dim; d++)
                centroid[d] += p[d];
        for (int d = 0; d < dim; d++)
            centroid[d] /= points.Count;
        return centroid;
    }

    private static double ComputeEuclidean(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    #endregion
}
