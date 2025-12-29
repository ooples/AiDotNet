using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for AdaptiveParametersHelper to verify genetic algorithm parameter adaptation.
/// </summary>
public class AdaptiveParametersHelperIntegrationTests
{
    #region Helper Methods

    private static OptimizationStepData<double, Matrix<double>, Vector<double>> CreateStepData(double fitnessScore)
    {
        return new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            FitnessScore = fitnessScore
        };
    }

    private static GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>> CreateDefaultOptions()
    {
        return new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            CrossoverRateIncrease = 1.05,
            CrossoverRateDecay = 0.95,
            MutationRateIncrease = 1.05,
            MutationRateDecay = 0.95,
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.1
        };
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Improvement Scenario

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FitnessImproves_IncreasesCrossoverRate()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Crossover rate should increase by 5% (multiply by 1.05)
        Assert.Equal(0.5 * 1.05, crossoverRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FitnessImproves_DecreasesMutationRate()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Mutation rate should decrease by 5% (multiply by 0.95)
        Assert.Equal(0.05 * 0.95, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_SmallImprovement_StillUpdatesRates()
    {
        var currentStep = CreateStepData(0.8001);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Even small improvement should increase crossover rate
        Assert.True(crossoverRate > 0.5);
        Assert.True(mutationRate < 0.05);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - No Improvement Scenario

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FitnessDecreases_DecreasesCrossoverRate()
    {
        var currentStep = CreateStepData(0.7);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Crossover rate should decrease by 5% (multiply by 0.95)
        Assert.Equal(0.5 * 0.95, crossoverRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FitnessDecreases_IncreasesMutationRate()
    {
        var currentStep = CreateStepData(0.7);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Mutation rate should increase by 5% (multiply by 1.05)
        Assert.Equal(0.05 * 1.05, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FitnessUnchanged_TreatedAsNoImprovement()
    {
        var currentStep = CreateStepData(0.8);
        var previousStep = CreateStepData(0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Equal fitness is treated as no improvement - crossover decreases, mutation increases
        Assert.Equal(0.5 * 0.95, crossoverRate, 10);
        Assert.Equal(0.05 * 1.05, mutationRate, 10);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Boundary Clamping

    [Fact]
    public void UpdateAdaptiveGeneticParameters_CrossoverRateExceedsMax_ClampedToMax()
    {
        var currentStep = CreateStepData(1.0);
        var previousStep = CreateStepData(0.5);
        var options = CreateDefaultOptions();

        // Start near max
        double crossoverRate = 0.88;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // 0.88 * 1.05 = 0.924, but max is 0.9, so should be clamped
        Assert.Equal(options.MaxCrossoverRate, crossoverRate);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_CrossoverRateBelowMin_ClampedToMin()
    {
        var currentStep = CreateStepData(0.5);
        var previousStep = CreateStepData(1.0);
        var options = CreateDefaultOptions();

        // Start near min
        double crossoverRate = 0.105;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // 0.105 * 0.95 = 0.09975, but min is 0.1, so should be clamped
        Assert.Equal(options.MinCrossoverRate, crossoverRate);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_MutationRateExceedsMax_ClampedToMax()
    {
        var currentStep = CreateStepData(0.5);
        var previousStep = CreateStepData(1.0);
        var options = CreateDefaultOptions();

        // Start near max mutation
        double crossoverRate = 0.5;
        double mutationRate = 0.098;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // 0.098 * 1.05 = 0.1029, but max is 0.1, so should be clamped
        Assert.Equal(options.MaxMutationRate, mutationRate);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_MutationRateBelowMin_ClampedToMin()
    {
        var currentStep = CreateStepData(1.0);
        var previousStep = CreateStepData(0.5);
        var options = CreateDefaultOptions();

        // Start near min mutation
        double crossoverRate = 0.5;
        double mutationRate = 0.00105;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // 0.00105 * 0.95 = 0.0009975, but min is 0.001, so should be clamped
        Assert.Equal(options.MinMutationRate, mutationRate);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_BothRatesExceedBounds_BothClamped()
    {
        var currentStep = CreateStepData(1.0);
        var previousStep = CreateStepData(0.5);
        var options = CreateDefaultOptions();

        // Start at extremes
        double crossoverRate = 0.89;  // Will increase beyond 0.9
        double mutationRate = 0.00105;  // Will decrease below 0.001

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(options.MaxCrossoverRate, crossoverRate);
        Assert.Equal(options.MinMutationRate, mutationRate);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Custom Options

    [Fact]
    public void UpdateAdaptiveGeneticParameters_CustomIncreaseFactor_AppliesCorrectly()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.5);
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            CrossoverRateIncrease = 1.20,  // 20% increase
            MutationRateDecay = 0.80,  // 20% decrease
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.1
        };

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(0.5 * 1.20, crossoverRate, 10);
        Assert.Equal(0.05 * 0.80, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_CustomDecayFactor_AppliesCorrectly()
    {
        var currentStep = CreateStepData(0.5);
        var previousStep = CreateStepData(0.9);
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            CrossoverRateDecay = 0.80,  // 20% decrease
            MutationRateIncrease = 1.20,  // 20% increase
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.5
        };

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(0.5 * 0.80, crossoverRate, 10);
        Assert.Equal(0.05 * 1.20, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_CustomBounds_ClampsCorrectly()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.5);
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            CrossoverRateIncrease = 1.50,  // Aggressive increase
            MutationRateDecay = 0.50,  // Aggressive decrease
            MinCrossoverRate = 0.2,
            MaxCrossoverRate = 0.7,  // Lower max
            MinMutationRate = 0.01,  // Higher min
            MaxMutationRate = 0.2
        };

        double crossoverRate = 0.6;  // Will exceed 0.7 after increase
        double mutationRate = 0.015;  // Will go below 0.01 after decay

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(0.7, crossoverRate);  // Clamped to custom max
        Assert.Equal(0.01, mutationRate);  // Clamped to custom min
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Multiple Iterations

    [Fact]
    public void UpdateAdaptiveGeneticParameters_MultipleImprovements_RatesConvergeTowardBounds()
    {
        var options = CreateDefaultOptions();
        double crossoverRate = 0.5;
        double mutationRate = 0.05;
        double initialCrossover = crossoverRate;
        double initialMutation = mutationRate;

        // Simulate 20 consecutive improvements
        for (int i = 0; i < 20; i++)
        {
            var currentStep = CreateStepData(0.5 + 0.02 * (i + 1));
            var previousStep = CreateStepData(0.5 + 0.02 * i);

            AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
                ref crossoverRate, ref mutationRate, currentStep, previousStep, options);
        }

        // Crossover should have increased toward max
        Assert.True(crossoverRate > initialCrossover, "Crossover should have increased");
        Assert.True(crossoverRate <= options.MaxCrossoverRate, "Crossover should be at or below max");
        // Mutation should have decreased toward min
        Assert.True(mutationRate < initialMutation, "Mutation should have decreased");
        Assert.True(mutationRate >= options.MinMutationRate, "Mutation should be at or above min");
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_MultipleDeclines_RatesConvergeTowardBounds()
    {
        var options = CreateDefaultOptions();
        double crossoverRate = 0.5;
        double mutationRate = 0.05;
        double initialCrossover = crossoverRate;
        double initialMutation = mutationRate;

        // Simulate 20 consecutive declines
        for (int i = 0; i < 20; i++)
        {
            var currentStep = CreateStepData(0.9 - 0.02 * (i + 1));
            var previousStep = CreateStepData(0.9 - 0.02 * i);

            AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
                ref crossoverRate, ref mutationRate, currentStep, previousStep, options);
        }

        // Crossover should have decreased toward min
        Assert.True(crossoverRate < initialCrossover, "Crossover should have decreased");
        Assert.True(crossoverRate >= options.MinCrossoverRate, "Crossover should be at or above min");
        // Mutation should have increased toward max
        Assert.True(mutationRate > initialMutation, "Mutation should have increased");
        Assert.True(mutationRate <= options.MaxMutationRate, "Mutation should be at or below max");
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_AlternatingResults_RatesOscillate()
    {
        var options = CreateDefaultOptions();
        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        var history = new List<(double crossover, double mutation)>();

        // Simulate alternating improvements and declines
        for (int i = 0; i < 10; i++)
        {
            var currentStep = CreateStepData(i % 2 == 0 ? 0.9 : 0.5);
            var previousStep = CreateStepData(i % 2 == 0 ? 0.5 : 0.9);

            AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
                ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

            history.Add((crossoverRate, mutationRate));
        }

        // Rates should oscillate, never hitting bounds
        Assert.True(crossoverRate > options.MinCrossoverRate);
        Assert.True(crossoverRate < options.MaxCrossoverRate);
        Assert.True(mutationRate > options.MinMutationRate);
        Assert.True(mutationRate < options.MaxMutationRate);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Float Type

    [Fact]
    public void UpdateAdaptiveGeneticParameters_Float_WorksCorrectly()
    {
        var currentStep = new OptimizationStepData<float, Matrix<float>, Vector<float>>
        {
            FitnessScore = 0.9f
        };
        var previousStep = new OptimizationStepData<float, Matrix<float>, Vector<float>>
        {
            FitnessScore = 0.8f
        };
        var options = new GeneticAlgorithmOptimizerOptions<float, Matrix<float>, Vector<float>>
        {
            CrossoverRateIncrease = 1.05,
            CrossoverRateDecay = 0.95,
            MutationRateIncrease = 1.05,
            MutationRateDecay = 0.95,
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.1
        };

        float crossoverRate = 0.5f;
        float mutationRate = 0.05f;

        AdaptiveParametersHelper<float, Matrix<float>, Vector<float>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(0.5f * 1.05f, crossoverRate, 4);
        Assert.Equal(0.05f * 0.95f, mutationRate, 4);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Edge Cases

    [Fact]
    public void UpdateAdaptiveGeneticParameters_ZeroFitnessScores_HandlesCorrectly()
    {
        var currentStep = CreateStepData(0.0);
        var previousStep = CreateStepData(0.0);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Equal scores = no improvement, so crossover decreases, mutation increases
        Assert.Equal(0.5 * 0.95, crossoverRate, 10);
        Assert.Equal(0.05 * 1.05, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_NegativeFitnessScores_HandlesCorrectly()
    {
        var currentStep = CreateStepData(-0.5);
        var previousStep = CreateStepData(-0.8);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // -0.5 > -0.8, so this is an improvement
        Assert.Equal(0.5 * 1.05, crossoverRate, 10);
        Assert.Equal(0.05 * 0.95, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_VeryLargeFitnessScores_HandlesCorrectly()
    {
        var currentStep = CreateStepData(1e10);
        var previousStep = CreateStepData(1e9);
        var options = CreateDefaultOptions();

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Large improvement should still work correctly
        Assert.Equal(0.5 * 1.05, crossoverRate, 10);
        Assert.Equal(0.05 * 0.95, mutationRate, 10);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_RatesAtBounds_StayAtBounds()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.5);
        var options = CreateDefaultOptions();

        // Start at max crossover and min mutation
        double crossoverRate = options.MaxCrossoverRate;
        double mutationRate = options.MinMutationRate;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Should remain at bounds
        Assert.Equal(options.MaxCrossoverRate, crossoverRate);
        Assert.Equal(options.MinMutationRate, mutationRate);
    }

    [Fact]
    public void UpdateAdaptiveGeneticParameters_FactorOfOne_RatesUnchanged()
    {
        var currentStep = CreateStepData(0.9);
        var previousStep = CreateStepData(0.5);
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            CrossoverRateIncrease = 1.0,  // No change
            CrossoverRateDecay = 1.0,
            MutationRateIncrease = 1.0,
            MutationRateDecay = 1.0,  // No change
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.1
        };

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Matrix<double>, Vector<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        // Rates should not change when factor is 1.0
        Assert.Equal(0.5, crossoverRate, 10);
        Assert.Equal(0.05, mutationRate, 10);
    }

    #endregion

    #region UpdateAdaptiveGeneticParameters Tests - Tensor Types

    [Fact]
    public void UpdateAdaptiveGeneticParameters_TensorTypes_WorksCorrectly()
    {
        var currentStep = new OptimizationStepData<double, Tensor<double>, Tensor<double>>
        {
            FitnessScore = 0.9
        };
        var previousStep = new OptimizationStepData<double, Tensor<double>, Tensor<double>>
        {
            FitnessScore = 0.8
        };
        var options = new GeneticAlgorithmOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            CrossoverRateIncrease = 1.05,
            CrossoverRateDecay = 0.95,
            MutationRateIncrease = 1.05,
            MutationRateDecay = 0.95,
            MinCrossoverRate = 0.1,
            MaxCrossoverRate = 0.9,
            MinMutationRate = 0.001,
            MaxMutationRate = 0.1
        };

        double crossoverRate = 0.5;
        double mutationRate = 0.05;

        AdaptiveParametersHelper<double, Tensor<double>, Tensor<double>>.UpdateAdaptiveGeneticParameters(
            ref crossoverRate, ref mutationRate, currentStep, previousStep, options);

        Assert.Equal(0.5 * 1.05, crossoverRate, 10);
        Assert.Equal(0.05 * 0.95, mutationRate, 10);
    }

    #endregion
}
