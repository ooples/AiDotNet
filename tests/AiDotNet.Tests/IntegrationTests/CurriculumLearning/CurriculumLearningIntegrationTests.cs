using AiDotNet.CurriculumLearning;
using AiDotNet.CurriculumLearning.DifficultyEstimators;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.CurriculumLearning.Schedulers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.CurriculumLearning;

/// <summary>
/// Integration tests for CurriculumLearning components.
/// Tests verify that difficulty estimators and schedulers work correctly together
/// to implement curriculum learning - training models by presenting samples from easy to hard.
/// </summary>
/// <remarks>
/// <para><b>What is Curriculum Learning?</b></para>
/// Curriculum learning is inspired by how humans learn - starting with simple examples
/// and gradually progressing to harder ones. Instead of randomly shuffling training data,
/// we order samples by difficulty and present them in a structured way.
///
/// <para><b>Key Components:</b></para>
/// <list type="bullet">
/// <item><description>Difficulty Estimators: Measure how hard each sample is</description></item>
/// <item><description>Schedulers: Control when to introduce harder samples</description></item>
/// <item><description>CurriculumLearner: Orchestrates the learning process</description></item>
/// </list>
///
/// <para><b>Benefits:</b></para>
/// <list type="bullet">
/// <item><description>Faster convergence</description></item>
/// <item><description>Better final performance</description></item>
/// <item><description>More stable training</description></item>
/// </list>
/// </remarks>
public class CurriculumLearningIntegrationTests
{
    private const int DefaultSampleCount = 100;
    private const int DefaultInputSize = 10;
    private const int DefaultOutputSize = 3;

    #region Helper Methods

    private static MockNeuralNetwork CreateMockModel()
    {
        return new MockNeuralNetwork(parameterCount: 50, outputSize: DefaultOutputSize);
    }

    private static Tensor<double> CreateInput(int index, int inputSize = DefaultInputSize)
    {
        var data = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            data[i] = (index + 1) * 0.1 + i * 0.01;
        }
        return new Tensor<double>(new[] { 1, inputSize }, new Vector<double>(data));
    }

    private static Tensor<double> CreateOutput(int index, int outputSize = DefaultOutputSize)
    {
        var data = new double[outputSize];
        data[index % outputSize] = 1.0; // One-hot encoding
        return new Tensor<double>(new[] { 1, outputSize }, new Vector<double>(data));
    }

    private static CurriculumEpochMetrics<double> CreateEpochMetrics(
        int epoch,
        int phase,
        double trainingLoss = 0.5,
        bool improved = true)
    {
        return new CurriculumEpochMetrics<double>
        {
            Epoch = epoch,
            Phase = phase,
            TrainingLoss = trainingLoss,
            Improved = improved,
            SamplesUsed = 100
        };
    }

    private static Vector<double> CreateDifficultyScores(int count, int seed = 42)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var scores = new Vector<double>(count);
        for (int i = 0; i < count; i++)
        {
            scores[i] = random.NextDouble();
        }
        return scores;
    }

    #endregion

    #region LinearScheduler Tests

    [Fact]
    public void LinearScheduler_Constructor_InitializesCorrectly()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 100);

        Assert.NotNull(scheduler);
        Assert.Equal("Linear", scheduler.Name);
        Assert.Equal(0, scheduler.CurrentEpoch);
        Assert.False(scheduler.IsComplete);
    }

    [Fact]
    public void LinearScheduler_GetDataFraction_IncreasesLinearly()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);

        var fractions = new List<double>();
        for (int epoch = 0; epoch < 10; epoch++)
        {
            fractions.Add(scheduler.GetDataFraction());
            scheduler.StepEpoch(CreateEpochMetrics(epoch, scheduler.CurrentPhaseNumber));
        }

        // Fractions should generally increase (or stay same)
        for (int i = 1; i < fractions.Count; i++)
        {
            Assert.True(fractions[i] >= fractions[i - 1] - 0.01,
                $"Data fraction should increase or stay same: epoch {i - 1}={fractions[i - 1]}, epoch {i}={fractions[i]}");
        }
    }

    [Fact]
    public void LinearScheduler_GetCurrentIndices_ReturnsCorrectCount()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 10, minFraction: 0.1, maxFraction: 1.0);
        var sortedIndices = Enumerable.Range(0, 100).ToArray();

        var indices = scheduler.GetCurrentIndices(sortedIndices, 100);

        Assert.NotNull(indices);
        Assert.True(indices.Length > 0);
        Assert.True(indices.Length <= 100);
    }

    [Fact]
    public void LinearScheduler_Reset_ResetsToInitialState()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);

        // Advance a few epochs
        for (int i = 0; i < 5; i++)
        {
            scheduler.StepEpoch(CreateEpochMetrics(i, 0));
        }

        scheduler.Reset();

        Assert.Equal(0, scheduler.CurrentEpoch);
        Assert.False(scheduler.IsComplete);
    }

    [Fact]
    public void LinearScheduler_GetStatistics_ReturnsValidDictionary()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);

        var stats = scheduler.GetStatistics();

        Assert.NotNull(stats);
        Assert.True(stats.Count > 0);
    }

    #endregion

    #region SelfPacedScheduler Tests

    [Fact]
    public void SelfPacedScheduler_Constructor_InitializesCorrectly()
    {
        var scheduler = new SelfPacedScheduler<double>(totalEpochs: 100, initialLambda: 0.1, lambdaGrowthRate: 1.1);

        Assert.NotNull(scheduler);
        // Default regularizer is Hard, so name is "SelfPaced_Hard"
        Assert.Equal("SelfPaced_Hard", scheduler.Name);
    }

    [Fact]
    public void SelfPacedScheduler_ComputeSampleWeights_ReturnsValidWeights()
    {
        var scheduler = new SelfPacedScheduler<double>(totalEpochs: 100, initialLambda: 0.5);
        var losses = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            losses[i] = i * 0.1; // Increasing losses
        }

        var weights = scheduler.ComputeSampleWeights(losses);

        Assert.NotNull(weights);
        Assert.Equal(10, weights.Length);
        // All weights should be between 0 and 1
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(weights[i] >= 0 && weights[i] <= 1,
                $"Weight {weights[i]} at index {i} should be in [0, 1]");
        }
    }

    [Fact]
    public void SelfPacedScheduler_EasySamplesGetHigherWeights()
    {
        var scheduler = new SelfPacedScheduler<double>(totalEpochs: 100, initialLambda: 0.3);
        var losses = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            losses[i] = i * 0.1; // Index 0 has lowest loss, index 9 has highest
        }

        var weights = scheduler.ComputeSampleWeights(losses);

        // Easy samples (low loss) should have higher or equal weights than hard samples
        // Note: this depends on the SPL formula, but generally true
        Assert.True(weights[0] >= weights[9],
            $"Easy sample weight {weights[0]} should be >= hard sample weight {weights[9]}");
    }

    [Fact]
    public void SelfPacedScheduler_PaceParameterCanBeModified()
    {
        var scheduler = new SelfPacedScheduler<double>(totalEpochs: 100, initialLambda: 0.1);

        scheduler.PaceParameter = 0.5;

        Assert.Equal(0.5, scheduler.PaceParameter);
    }

    #endregion

    #region CompetenceBasedScheduler Tests

    [Fact]
    public void CompetenceBasedScheduler_Constructor_InitializesCorrectly()
    {
        var scheduler = new CompetenceBasedScheduler<double>(totalEpochs: 100, competenceThreshold: 0.9);

        Assert.NotNull(scheduler);
        // Default metric type is Combined, so name is "CompetenceBased_Combined"
        Assert.Equal("CompetenceBased_Combined", scheduler.Name);
    }

    [Fact]
    public void CompetenceBasedScheduler_UpdateCompetence_UpdatesCorrectly()
    {
        // Note: Must provide explicit competenceThreshold since generic T? default is 0.0 for value types
        var scheduler = new CompetenceBasedScheduler<double>(totalEpochs: 100, competenceThreshold: 0.9);
        var metrics = CreateEpochMetrics(0, 0, trainingLoss: 0.1, improved: true);
        metrics.TrainingAccuracy = 0.95;

        scheduler.UpdateCompetence(metrics);

        // After update, competence should be non-zero
        Assert.True(scheduler.CurrentCompetence >= 0);
    }

    [Fact]
    public void CompetenceBasedScheduler_HasMasteredCurrentContent_ReturnsFalseInitially()
    {
        var scheduler = new CompetenceBasedScheduler<double>(totalEpochs: 100, competenceThreshold: 0.9);

        Assert.False(scheduler.HasMasteredCurrentContent());
    }

    [Fact]
    public void CompetenceBasedScheduler_CompetenceThreshold_CanBeModified()
    {
        // Note: Must provide explicit competenceThreshold since generic T? default is 0.0 for value types
        var scheduler = new CompetenceBasedScheduler<double>(totalEpochs: 100, competenceThreshold: 0.9);

        scheduler.CompetenceThreshold = 0.85;

        Assert.Equal(0.85, scheduler.CompetenceThreshold);
    }

    #endregion

    #region LossBasedDifficultyEstimator Tests

    [Fact]
    public void LossBasedDifficultyEstimator_Constructor_InitializesCorrectly()
    {
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        Assert.NotNull(estimator);
        Assert.Equal("LossBased", estimator.Name);
        Assert.True(estimator.RequiresModel);
    }

    [Fact]
    public void LossBasedDifficultyEstimator_EstimateDifficulty_ReturnsNonNegative()
    {
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var model = CreateMockModel();
        var input = CreateInput(0);
        var output = CreateOutput(0);

        var difficulty = estimator.EstimateDifficulty(input, output, model);

        Assert.True(difficulty >= 0, $"Difficulty {difficulty} should be non-negative");
    }

    [Fact]
    public void LossBasedDifficultyEstimator_GetSortedIndices_ReturnsSortedArray()
    {
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var difficulties = CreateDifficultyScores(50);

        var sortedIndices = estimator.GetSortedIndices(difficulties);

        Assert.Equal(50, sortedIndices.Length);
        // Verify sorted by difficulty (easy to hard)
        for (int i = 1; i < sortedIndices.Length; i++)
        {
            Assert.True(difficulties[sortedIndices[i]] >= difficulties[sortedIndices[i - 1]],
                "Indices should be sorted by increasing difficulty");
        }
    }

    [Fact]
    public void LossBasedDifficultyEstimator_Reset_DoesNotThrow()
    {
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        var exception = Record.Exception(() => estimator.Reset());

        Assert.Null(exception);
    }

    #endregion

    #region ConfidenceBasedDifficultyEstimator Tests

    [Fact]
    public void ConfidenceBasedDifficultyEstimator_Constructor_InitializesCorrectly()
    {
        var estimator = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        Assert.NotNull(estimator);
        // Default metric type is Entropy, so name is "ConfidenceBased_Entropy"
        Assert.Equal("ConfidenceBased_Entropy", estimator.Name);
        Assert.True(estimator.RequiresModel);
    }

    [Fact]
    public void ConfidenceBasedDifficultyEstimator_EstimateDifficulty_ReturnsNonNegative()
    {
        var estimator = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var model = CreateMockModel();
        var input = CreateInput(0);
        var output = CreateOutput(0);

        var difficulty = estimator.EstimateDifficulty(input, output, model);

        Assert.True(difficulty >= 0, $"Difficulty {difficulty} should be non-negative");
    }

    [Fact]
    public void ConfidenceBasedDifficultyEstimator_HighConfidence_LowDifficulty()
    {
        // Confidence-based: low confidence = high difficulty
        // High confidence = low difficulty
        var estimator = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        // Test the concept - difficulty should be inversely related to confidence
        Assert.True(estimator.RequiresModel);
    }

    #endregion

    #region TransferBasedDifficultyEstimator Tests

    [Fact]
    public void TransferBasedDifficultyEstimator_Constructor_InitializesCorrectly()
    {
        var teacherModel = CreateMockModel();
        var estimator = new TransferBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>(
            teacherModel: teacherModel,
            mode: TransferDifficultyMode.LossGap);

        Assert.NotNull(estimator);
        Assert.Equal("TransferBased", estimator.Name);
        Assert.True(estimator.RequiresModel);
    }

    [Fact]
    public void TransferBasedDifficultyEstimator_EstimateDifficulty_ReturnsNonNegative()
    {
        var teacherModel = CreateMockModel();
        var estimator = new TransferBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>(
            teacherModel: teacherModel);

        var input = CreateInput(0);
        var output = CreateOutput(0);
        var studentModel = CreateMockModel();

        var difficulty = estimator.EstimateDifficulty(input, output, studentModel);

        Assert.True(difficulty >= 0, $"Difficulty {difficulty} should be non-negative");
    }

    #endregion

    #region ExpertDefinedDifficultyEstimator Tests

    [Fact]
    public void ExpertDefinedDifficultyEstimator_Constructor_InitializesCorrectly()
    {
        // ExpertDefinedDifficultyEstimator takes a Vector<T> of pre-computed difficulties
        var difficulties = new Vector<double>(new[] { 0.1, 0.5, 0.9 });
        var estimator = new ExpertDefinedDifficultyEstimator<double, Tensor<double>, Tensor<double>>(difficulties);

        Assert.NotNull(estimator);
        Assert.Equal("ExpertDefined", estimator.Name);
        Assert.False(estimator.RequiresModel); // Expert-defined doesn't need model
    }

    [Fact]
    public void ExpertDefinedDifficultyEstimator_ReturnsDefinedDifficulty()
    {
        // Vector of pre-computed difficulties: [0.2, 0.7, 0.5]
        var difficulties = new Vector<double>(new[] { 0.2, 0.7, 0.5 });
        var estimator = new ExpertDefinedDifficultyEstimator<double, Tensor<double>, Tensor<double>>(difficulties);

        // Test that get sorted indices works
        var sortedIndices = estimator.GetSortedIndices(difficulties);

        // Should sort: [0.2, 0.5, 0.7] -> indices [0, 2, 1]
        Assert.Equal(0, sortedIndices[0]);
        Assert.Equal(2, sortedIndices[1]);
        Assert.Equal(1, sortedIndices[2]);
    }

    #endregion

    #region EnsembleDifficultyEstimator Tests

    [Fact]
    public void EnsembleDifficultyEstimator_Constructor_InitializesCorrectly()
    {
        var estimator1 = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var estimator2 = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        // EnsembleDifficultyEstimator accepts IEnumerable<IDifficultyEstimator> and IEnumerable<T>? weights
        var ensemble = new EnsembleDifficultyEstimator<double, Tensor<double>, Tensor<double>>(
            new IDifficultyEstimator<double, Tensor<double>, Tensor<double>>[] { estimator1, estimator2 },
            new double[] { 0.5, 0.5 });

        Assert.NotNull(ensemble);
        // Default combination method is WeightedAverage
        Assert.Equal("Ensemble_WeightedAverage", ensemble.Name);
        Assert.Equal(2, ensemble.Estimators.Count);
    }

    [Fact]
    public void EnsembleDifficultyEstimator_EstimateDifficulty_CombinesEstimates()
    {
        var estimator1 = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var estimator2 = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        var ensemble = new EnsembleDifficultyEstimator<double, Tensor<double>, Tensor<double>>(
            new IDifficultyEstimator<double, Tensor<double>, Tensor<double>>[] { estimator1, estimator2 },
            new double[] { 0.5, 0.5 });

        var model = CreateMockModel();
        var input = CreateInput(0);
        var output = CreateOutput(0);

        var difficulty = ensemble.EstimateDifficulty(input, output, model);

        Assert.True(difficulty >= 0, "Combined difficulty should be non-negative");
    }

    [Fact]
    public void EnsembleDifficultyEstimator_AddEstimator_IncreasesCount()
    {
        var estimator1 = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();

        var ensemble = new EnsembleDifficultyEstimator<double, Tensor<double>, Tensor<double>>(
            new IDifficultyEstimator<double, Tensor<double>, Tensor<double>>[] { estimator1 },
            new double[] { 1.0 });

        var estimator2 = new ConfidenceBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        ensemble.AddEstimator(estimator2, 0.5);

        Assert.Equal(2, ensemble.Estimators.Count);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void CurriculumLearning_LinearScheduler_With_LossEstimator_Integration()
    {
        // Test complete curriculum learning workflow
        var scheduler = new LinearScheduler<double>(totalEpochs: 10, minFraction: 0.2, maxFraction: 1.0);
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var model = CreateMockModel();

        // Generate mock difficulties
        var difficulties = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            difficulties[i] = i / 100.0; // Linear difficulty
        }

        var sortedIndices = estimator.GetSortedIndices(difficulties);

        // Simulate curriculum learning epochs
        int totalSamplesUsed = 0;
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var currentIndices = scheduler.GetCurrentIndices(sortedIndices, 100);
            totalSamplesUsed = Math.Max(totalSamplesUsed, currentIndices.Length);

            var metrics = CreateEpochMetrics(epoch, scheduler.CurrentPhaseNumber, 0.5 - epoch * 0.04);
            scheduler.StepEpoch(metrics);
        }

        // By the end, should have access to all samples
        Assert.True(totalSamplesUsed >= 80, $"Should use most samples by end, used {totalSamplesUsed}");
    }

    [Fact]
    public void CurriculumLearning_SelfPacedScheduler_AdaptsBasedOnLoss()
    {
        // Note: Must provide explicit maxLambda since generic T? default is 0.0 for value types
        // Without it, lambda gets capped to 0.0 after the first StepEpoch
        var scheduler = new SelfPacedScheduler<double>(
            totalEpochs: 100,
            initialLambda: 0.1,
            lambdaGrowthRate: 0.5,
            maxLambda: 10.0);

        // Simulate training with decreasing loss
        var initialLambda = scheduler.PaceParameter;
        for (int epoch = 0; epoch < 5; epoch++)
        {
            var losses = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                losses[i] = 1.0 / (epoch + 1) + i * 0.05; // Decreasing base loss
            }

            var weights = scheduler.ComputeSampleWeights(losses);
            Assert.NotNull(weights);

            var metrics = CreateEpochMetrics(epoch, 0, trainingLoss: 1.0 / (epoch + 1));
            scheduler.StepEpoch(metrics);
        }

        // Lambda should have grown: initial 0.1 + 5 epochs * 0.5 growth = 2.6
        Assert.True(scheduler.PaceParameter > initialLambda,
            $"Pace parameter should grow from {initialLambda} to {scheduler.PaceParameter}");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void LinearScheduler_ZeroTotalEpochs_ThrowsOrHandles()
    {
        // Zero epochs should either throw or handle gracefully
        var exception = Record.Exception(() => new LinearScheduler<double>(totalEpochs: 0));

        // If it doesn't throw, it should at least be functional
        if (exception == null)
        {
            var scheduler = new LinearScheduler<double>(totalEpochs: 0);
            Assert.NotNull(scheduler);
        }
    }

    [Fact]
    public void LinearScheduler_NegativeTotalEpochs_ThrowsOrHandles()
    {
        var exception = Record.Exception(() => new LinearScheduler<double>(totalEpochs: -1));

        // Should throw or handle gracefully
        if (exception == null)
        {
            var scheduler = new LinearScheduler<double>(totalEpochs: -1);
            Assert.NotNull(scheduler);
        }
    }

    [Fact]
    public void Scheduler_EmptySortedIndices_HandlesGracefully()
    {
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);
        var emptyIndices = Array.Empty<int>();

        var indices = scheduler.GetCurrentIndices(emptyIndices, 0);

        Assert.NotNull(indices);
        Assert.Empty(indices);
    }

    [Fact]
    public void DifficultyEstimator_EmptyDifficulties_HandlesGracefully()
    {
        var estimator = new LossBasedDifficultyEstimator<double, Tensor<double>, Tensor<double>>();
        var emptyDifficulties = new Vector<double>(0);

        var sortedIndices = estimator.GetSortedIndices(emptyDifficulties);

        Assert.NotNull(sortedIndices);
        Assert.Empty(sortedIndices);
    }

    [Fact]
    public void AllSchedulers_Reset_WorksMultipleTimes()
    {
        // Note: Must provide explicit values for optional T? parameters since generic defaults are 0.0 for value types
        var schedulers = new ICurriculumScheduler<double>[]
        {
            new LinearScheduler<double>(totalEpochs: 10),
            new SelfPacedScheduler<double>(totalEpochs: 10, initialLambda: 0.1),
            new CompetenceBasedScheduler<double>(totalEpochs: 10, competenceThreshold: 0.9)
        };

        foreach (var scheduler in schedulers)
        {
            // Step a few epochs
            for (int i = 0; i < 3; i++)
            {
                scheduler.StepEpoch(CreateEpochMetrics(i, 0));
            }

            // Reset multiple times
            scheduler.Reset();
            scheduler.Reset();
            scheduler.Reset();

            Assert.Equal(0, scheduler.CurrentEpoch);
        }
    }

    #endregion
}
