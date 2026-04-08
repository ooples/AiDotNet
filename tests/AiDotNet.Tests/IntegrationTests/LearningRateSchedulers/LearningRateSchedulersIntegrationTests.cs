using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LearningRateSchedulers;

/// <summary>
/// Integration tests for learning rate scheduler classes.
/// Tests learning rate computation, stepping, and reset functionality.
/// </summary>
public class LearningRateSchedulersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region ConstantLRScheduler Tests

    [Fact]
    public void ConstantLRScheduler_CurrentLearningRate_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new ConstantLRScheduler(0.01);

        // Act & Assert
        Assert.Equal(0.01, scheduler.CurrentLearningRate, Tolerance);
        Assert.Equal(0.01, scheduler.BaseLearningRate, Tolerance);
    }

    [Fact]
    public void ConstantLRScheduler_Step_MaintainsConstantRate()
    {
        // Arrange
        var scheduler = new ConstantLRScheduler(0.001);

        // Act - Step multiple times
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should remain constant
        Assert.Equal(0.001, scheduler.CurrentLearningRate, Tolerance);
        Assert.Equal(10, scheduler.CurrentStep);
    }

    [Fact]
    public void ConstantLRScheduler_Reset_ResetsStepCount()
    {
        // Arrange
        var scheduler = new ConstantLRScheduler(0.01);
        scheduler.Step();
        scheduler.Step();

        // Act
        scheduler.Reset();

        // Assert
        Assert.Equal(0, scheduler.CurrentStep);
        Assert.Equal(0.01, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region StepLRScheduler Tests

    [Fact]
    public void StepLRScheduler_Initial_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 10, gamma: 0.1);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
        Assert.Equal(0.1, scheduler.BaseLearningRate, Tolerance);
    }

    [Fact]
    public void StepLRScheduler_StepBeforeDecay_MaintainsRate()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 5, gamma: 0.5);

        // Act - Step 4 times (before decay)
        for (int i = 0; i < 4; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should still be base rate
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void StepLRScheduler_StepAtDecay_ReducesRate()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 5, gamma: 0.5);

        // Act - Step 5 times (at decay point)
        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should be reduced by gamma
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void StepLRScheduler_MultipleDecays_ReducesRateCorrectly()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 3, gamma: 0.5);

        // Act - Step 9 times (3 decay periods)
        for (int i = 0; i < 9; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should be 0.1 * 0.5^3 = 0.0125
        Assert.Equal(0.0125, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void StepLRScheduler_GetLearningRateAtStep_ReturnsCorrectRate()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 5, gamma: 0.5);

        // Act & Assert
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(4), Tolerance);
        Assert.Equal(0.05, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.025, scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    #endregion

    #region ExponentialLRScheduler Tests

    [Fact]
    public void ExponentialLRScheduler_Initial_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(0.1, gamma: 0.95);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void ExponentialLRScheduler_Step_DecaysExponentially()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(0.1, gamma: 0.9);

        // Act
        scheduler.Step();

        // Assert - Rate should be 0.1 * 0.9 = 0.09
        Assert.Equal(0.09, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void ExponentialLRScheduler_MultipleSteps_DecaysCorrectly()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(0.1, gamma: 0.9);

        // Act - Step 3 times
        scheduler.Step();
        scheduler.Step();
        scheduler.Step();

        // Assert - Rate should be 0.1 * 0.9^3 = 0.0729
        Assert.Equal(0.0729, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region CosineAnnealingLRScheduler Tests

    [Fact]
    public void CosineAnnealingLRScheduler_Initial_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 100, etaMin: 0.001);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void CosineAnnealingLRScheduler_AtTMax_ReturnsMinimumRate()
    {
        // Arrange
        var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 10, etaMin: 0.001);

        // Act - Step to T_max
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        // Assert - Should be at or near minimum
        Assert.True(scheduler.CurrentLearningRate <= 0.002);
    }

    [Fact]
    public void CosineAnnealingLRScheduler_RateDecreasesMonotonically()
    {
        // Arrange
        var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 10, etaMin: 0.001);

        // Act & Assert
        double previousRate = scheduler.CurrentLearningRate;
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
            Assert.True(scheduler.CurrentLearningRate <= previousRate);
            previousRate = scheduler.CurrentLearningRate;
        }
    }

    #endregion

    #region PolynomialLRScheduler Tests

    [Fact]
    public void PolynomialLRScheduler_Initial_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new PolynomialLRScheduler(0.1, totalSteps: 100, power: 2.0, endLearningRate: 0.0);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void PolynomialLRScheduler_AtEnd_ReturnsEndRate()
    {
        // Arrange
        var scheduler = new PolynomialLRScheduler(0.1, totalSteps: 10, power: 1.0, endLearningRate: 0.01);

        // Act - Step to end
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        // Assert
        Assert.Equal(0.01, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void PolynomialLRScheduler_LinearDecay_DecaysLinearly()
    {
        // Arrange - power=1.0 gives linear decay
        var scheduler = new PolynomialLRScheduler(0.1, totalSteps: 10, power: 1.0, endLearningRate: 0.0);

        // Act - Step halfway
        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        // Assert - Should be at 50% of the way
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region LinearWarmupScheduler Tests

    [Fact]
    public void LinearWarmupScheduler_Initial_StartsLow()
    {
        // Arrange
        var scheduler = new LinearWarmupScheduler(0.1, warmupSteps: 10);

        // Assert - Should start very low
        Assert.True(scheduler.CurrentLearningRate < 0.1);
    }

    [Fact]
    public void LinearWarmupScheduler_AfterWarmup_ReachesBaseLearningRate()
    {
        // Arrange
        var scheduler = new LinearWarmupScheduler(0.1, warmupSteps: 5);

        // Act - Complete warmup
        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        // Assert - Should reach base learning rate
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void LinearWarmupScheduler_DuringWarmup_IncreasesLinearly()
    {
        // Arrange
        var scheduler = new LinearWarmupScheduler(0.1, warmupSteps: 10);
        double initialRate = scheduler.CurrentLearningRate;

        // Act
        scheduler.Step();

        // Assert - Rate should increase
        Assert.True(scheduler.CurrentLearningRate > initialRate);
    }

    #endregion

    #region CyclicLRScheduler Tests

    [Fact]
    public void CyclicLRScheduler_Initial_StartsAtBaseLearningRate()
    {
        // Arrange
        var scheduler = new CyclicLRScheduler(baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 10);

        // Assert
        Assert.Equal(0.001, scheduler.BaseLearningRate, Tolerance);
    }

    [Fact]
    public void CyclicLRScheduler_Step_IncreasesTowardsMax()
    {
        // Arrange
        var scheduler = new CyclicLRScheduler(baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 10);
        double initialRate = scheduler.CurrentLearningRate;

        // Act
        scheduler.Step();

        // Assert - Rate should increase
        Assert.True(scheduler.CurrentLearningRate > initialRate);
    }

    [Fact]
    public void CyclicLRScheduler_Cycle_RateCycles()
    {
        // Arrange
        var scheduler = new CyclicLRScheduler(baseLearningRate: 0.001, maxLearningRate: 0.1, stepSizeUp: 5, stepSizeDown: 5);

        // Act - Complete one cycle
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        // Assert - Should return close to base
        Assert.True(scheduler.CurrentLearningRate < 0.05);
    }

    #endregion

    #region OneCycleLRScheduler Tests

    [Fact]
    public void OneCycleLRScheduler_Initial_StartsLow()
    {
        // Arrange
        var scheduler = new OneCycleLRScheduler(maxLearningRate: 0.1, totalSteps: 100);

        // Assert - Should start at a fraction of max
        Assert.True(scheduler.CurrentLearningRate < 0.1);
    }

    [Fact]
    public void OneCycleLRScheduler_AtPeak_ReachesMaxLearningRate()
    {
        // Arrange
        var scheduler = new OneCycleLRScheduler(maxLearningRate: 0.1, totalSteps: 100, pctStart: 0.3);

        // Act - Step to peak (30% of total steps)
        for (int i = 0; i < 30; i++)
        {
            scheduler.Step();
        }

        // Assert - Should be at or near max
        Assert.True(scheduler.CurrentLearningRate > 0.08);
    }

    #endregion

    #region MultiStepLRScheduler Tests

    [Fact]
    public void MultiStepLRScheduler_BeforeMilestone_MaintainsRate()
    {
        // Arrange
        var scheduler = new MultiStepLRScheduler(0.1, milestones: new[] { 10, 20, 30 }, gamma: 0.1);

        // Act - Step before first milestone
        for (int i = 0; i < 9; i++)
        {
            scheduler.Step();
        }

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void MultiStepLRScheduler_AtMilestone_ReducesRate()
    {
        // Arrange
        var scheduler = new MultiStepLRScheduler(0.1, milestones: new[] { 5, 10, 15 }, gamma: 0.5);

        // Act - Step to first milestone
        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should be reduced
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void MultiStepLRScheduler_MultipleMilestones_ReducesCorrectly()
    {
        // Arrange
        var scheduler = new MultiStepLRScheduler(0.1, milestones: new[] { 2, 4, 6 }, gamma: 0.5);

        // Act - Step past all milestones
        for (int i = 0; i < 7; i++)
        {
            scheduler.Step();
        }

        // Assert - Rate should be 0.1 * 0.5^3 = 0.0125
        Assert.Equal(0.0125, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region ReduceOnPlateauScheduler Tests

    [Fact]
    public void ReduceOnPlateauScheduler_Initial_ReturnsBaseLearningRate()
    {
        // Arrange
        var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.1, patience: 10);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void ReduceOnPlateauScheduler_StepWithImprovement_MaintainsRate()
    {
        // Arrange
        var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.5, patience: 5);

        // Act - Report improving metrics
        scheduler.Step(0.5);
        scheduler.Step(0.4);
        scheduler.Step(0.3);

        // Assert
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region State Management Tests

    [Fact]
    public void LearningRateScheduler_GetState_ReturnsNonNullState()
    {
        // Arrange
        var scheduler = new StepLRScheduler(0.1, stepSize: 10, gamma: 0.1);
        scheduler.Step();
        scheduler.Step();

        // Act
        var state = scheduler.GetState();

        // Assert
        Assert.NotNull(state);
        Assert.True(state.Count > 0);
    }

    [Fact]
    public void LearningRateScheduler_LoadState_RestoresScheduler()
    {
        // Arrange
        var scheduler1 = new StepLRScheduler(0.1, stepSize: 5, gamma: 0.5);
        scheduler1.Step();
        scheduler1.Step();
        scheduler1.Step();
        var state = scheduler1.GetState();

        var scheduler2 = new StepLRScheduler(0.1, stepSize: 5, gamma: 0.5);

        // Act
        scheduler2.LoadState(state);

        // Assert
        Assert.Equal(scheduler1.CurrentStep, scheduler2.CurrentStep);
        Assert.Equal(scheduler1.CurrentLearningRate, scheduler2.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void LearningRateScheduler_Reset_RestoresInitialState()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(0.1, gamma: 0.9);
        scheduler.Step();
        scheduler.Step();
        scheduler.Step();

        // Act
        scheduler.Reset();

        // Assert
        Assert.Equal(0, scheduler.CurrentStep);
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllSchedulers_HandleLargeNumberOfSteps()
    {
        // Arrange
        var schedulers = new ILearningRateScheduler[]
        {
            new ConstantLRScheduler(0.01),
            new StepLRScheduler(0.01, stepSize: 100, gamma: 0.5),
            new ExponentialLRScheduler(0.01, gamma: 0.999),
            new CosineAnnealingLRScheduler(0.01, tMax: 1000, etaMin: 0.0001),
            new PolynomialLRScheduler(0.01, totalSteps: 1000, power: 2.0, endLearningRate: 0.0001)
        };

        // Act & Assert
        foreach (var scheduler in schedulers)
        {
            for (int i = 0; i < 1000; i++)
            {
                scheduler.Step();
            }
            Assert.False(double.IsNaN(scheduler.CurrentLearningRate));
            Assert.False(double.IsInfinity(scheduler.CurrentLearningRate));
            Assert.True(scheduler.CurrentLearningRate >= 0);
        }
    }

    [Fact]
    public void AllSchedulers_ReturnPositiveLearningRates()
    {
        // Arrange
        var schedulers = new ILearningRateScheduler[]
        {
            new ConstantLRScheduler(0.01),
            new StepLRScheduler(0.01, stepSize: 5, gamma: 0.5),
            new ExponentialLRScheduler(0.01, gamma: 0.9)
        };

        // Act & Assert
        foreach (var scheduler in schedulers)
        {
            Assert.True(scheduler.CurrentLearningRate > 0);
            for (int i = 0; i < 50; i++)
            {
                scheduler.Step();
                Assert.True(scheduler.CurrentLearningRate >= 0);
            }
        }
    }

    #endregion
}
