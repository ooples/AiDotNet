using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for optimizer and learning rate scheduler combinations.
/// Tests that optimizers correctly integrate with LR schedulers and that
/// the learning rate changes are properly applied during optimization.
/// </summary>
public class OptimizerSchedulerIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region AdamW + Scheduler Tests

    [Fact]
    public void AdamWOptimizer_WithStepLRScheduler_LearningRateDecays()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.01, stepSize: 2, gamma: 0.5);
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        // Act - Step the optimizer multiple times to trigger scheduler decay
        double initialLR = scheduler.CurrentLearningRate;
        optimizer.UpdateParameters(parameters, gradient);
        optimizer.StepScheduler(); // Epoch 1
        optimizer.UpdateParameters(parameters, gradient);
        optimizer.StepScheduler(); // Epoch 2 - should trigger decay

        // Assert
        Assert.Equal(0.01, initialLR, Tolerance);
        Assert.Equal(0.005, scheduler.CurrentLearningRate, Tolerance); // 0.01 * 0.5
    }

    [Fact]
    public void AdamWOptimizer_WithCosineScheduler_LearningRateAnneals()
    {
        // Arrange
        var scheduler = new CosineAnnealingLRScheduler(
            baseLearningRate: 0.1, tMax: 10, etaMin: 0.001);
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        // Act - Step through several epochs
        var learningRates = new List<double>();
        learningRates.Add(scheduler.CurrentLearningRate);

        for (int i = 0; i < 10; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
            learningRates.Add(scheduler.CurrentLearningRate);
        }

        // Assert - Learning rate should decrease monotonically (cosine annealing)
        for (int i = 1; i < learningRates.Count; i++)
        {
            Assert.True(learningRates[i] <= learningRates[i - 1] + Tolerance,
                $"LR at step {i} ({learningRates[i]}) should be <= LR at step {i - 1} ({learningRates[i - 1]})");
        }

        // Final LR should be at or near minimum
        Assert.True(learningRates[^1] <= 0.01);
    }

    [Fact]
    public void AdamWOptimizer_WithLinearWarmupScheduler_LearningRateWarmsUp()
    {
        // Arrange
        var scheduler = new LinearWarmupScheduler(baseLearningRate: 0.1, warmupSteps: 5);
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        // Act - Step through warmup period
        var learningRates = new List<double>();
        learningRates.Add(scheduler.CurrentLearningRate);

        for (int i = 0; i < 5; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
            learningRates.Add(scheduler.CurrentLearningRate);
        }

        // Assert - Learning rate should increase during warmup
        for (int i = 1; i < learningRates.Count; i++)
        {
            Assert.True(learningRates[i] >= learningRates[i - 1] - Tolerance,
                $"LR at step {i} ({learningRates[i]}) should be >= LR at step {i - 1} ({learningRates[i - 1]})");
        }

        // Should reach base LR after warmup
        Assert.Equal(0.1, learningRates[^1], Tolerance);
    }

    #endregion

    #region Adam + Scheduler Tests

    [Fact]
    public void AdamOptimizer_WithExponentialScheduler_LearningRateDecays()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.01, gamma: 0.9);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.5, 0.5 });

        // Act
        double lrBeforeStep = scheduler.CurrentLearningRate;
        optimizer.UpdateParameters(parameters, gradient);
        optimizer.StepScheduler();
        double lrAfterStep = scheduler.CurrentLearningRate;

        // Assert
        Assert.Equal(0.01, lrBeforeStep, Tolerance);
        Assert.Equal(0.009, lrAfterStep, Tolerance); // 0.01 * 0.9
    }

    [Fact]
    public void AdamOptimizer_WithPolynomialScheduler_LearningRateDecays()
    {
        // Arrange
        var scheduler = new PolynomialLRScheduler(
            baseLearningRate: 0.1, totalSteps: 10, power: 2.0, endLearningRate: 0.01);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.5, 0.5 });

        // Act - Step to completion
        for (int i = 0; i < 10; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        // Assert - Should reach end learning rate
        Assert.Equal(0.01, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region Lion + Scheduler Tests

    [Fact]
    public void LionOptimizer_WithCosineScheduler_LearningRateAnneals()
    {
        // Arrange
        var scheduler = new CosineAnnealingLRScheduler(
            baseLearningRate: 0.0001, tMax: 5, etaMin: 0.00001);
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.0001,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        // Act
        double initialLR = scheduler.CurrentLearningRate;
        for (int i = 0; i < 5; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }
        double finalLR = scheduler.CurrentLearningRate;

        // Assert
        Assert.Equal(0.0001, initialLR, Tolerance);
        Assert.True(finalLR < initialLR);
    }

    #endregion

    #region Scheduler Step Mode Tests

    [Fact]
    public void Optimizer_StepPerBatch_StepsOnEveryBatch()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 3, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Act - 3 batches should trigger decay
        for (int i = 0; i < 3; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        // Assert
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance); // 0.1 * 0.5
    }

    [Fact]
    public void Optimizer_StepPerEpoch_OnlyStepsOnEpochEnd()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 2, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Act - Simulate 5 batches per epoch for 2 epochs
        for (int epoch = 0; epoch < 2; epoch++)
        {
            for (int batch = 0; batch < 5; batch++)
            {
                optimizer.UpdateParameters(parameters, gradient);
            }
            optimizer.StepScheduler(); // Only step at epoch end
        }

        // Assert - After 2 epochs, should have decayed once (stepSize=2)
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Optimizer_Reset_ResetsLearningRate()
    {
        // Arrange
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Step several times to decay LR
        for (int i = 0; i < 5; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        double lrBeforeReset = optimizer.GetCurrentLearningRate();

        // Act
        optimizer.Reset();
        double lrAfterReset = optimizer.GetCurrentLearningRate();

        // Assert
        Assert.True(lrBeforeReset < 0.1, $"LR before reset should have decayed below 0.1, was {lrBeforeReset}");
        Assert.Equal(0.1, lrAfterReset, Tolerance);
    }

    #endregion

    #region Null Scheduler Tests

    [Fact]
    public void Optimizer_WithNullScheduler_UsesConstantLearningRate()
    {
        // Arrange - No scheduler provided
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        // Act - Multiple updates
        var initialParams = parameters;
        for (int i = 0; i < 10; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler(); // Should be no-op without scheduler
        }

        // Assert - Optimizer should still work and update parameters
        Assert.True(parameters[0] < initialParams[0]);
        Assert.True(parameters[1] < initialParams[1]);
        Assert.True(parameters[2] < initialParams[2]);
    }

    #endregion

    #region Multiple Optimizer Types Tests

    [Fact]
    public void AdamOptimizer_WorksWithStepScheduler()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.01, stepSize: 2, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1 });

        // Act
        double lrBefore = scheduler.CurrentLearningRate;

        for (int i = 0; i < 2; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        double lrAfter = scheduler.CurrentLearningRate;

        // Assert
        Assert.Equal(0.01, lrBefore, Tolerance);
        Assert.Equal(0.005, lrAfter, Tolerance); // After 2 steps with stepSize=2
    }

    [Fact]
    public void AdamWOptimizer_WorksWithStepScheduler()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.01, stepSize: 2, gamma: 0.5);
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1 });

        // Act
        double lrBefore = scheduler.CurrentLearningRate;

        for (int i = 0; i < 2; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        double lrAfter = scheduler.CurrentLearningRate;

        // Assert
        Assert.Equal(0.01, lrBefore, Tolerance);
        Assert.Equal(0.005, lrAfter, Tolerance);
    }

    [Fact]
    public void LionOptimizer_WorksWithStepScheduler()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.01, stepSize: 2, gamma: 0.5);
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1 });

        // Act
        double lrBefore = scheduler.CurrentLearningRate;

        for (int i = 0; i < 2; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        double lrAfter = scheduler.CurrentLearningRate;

        // Assert
        Assert.Equal(0.01, lrBefore, Tolerance);
        Assert.Equal(0.005, lrAfter, Tolerance);
    }

    #endregion

    #region OneCycle Scheduler Tests

    [Fact]
    public void AdamWOptimizer_WithOneCycleScheduler_CompletesFullCycle()
    {
        // Arrange
        var scheduler = new OneCycleLRScheduler(
            maxLearningRate: 0.1, totalSteps: 20, pctStart: 0.3);
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1 });

        // Act - Complete the full cycle
        var learningRates = new List<double>();
        learningRates.Add(scheduler.CurrentLearningRate);

        for (int i = 0; i < 20; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
            learningRates.Add(scheduler.CurrentLearningRate);
        }

        // Assert
        // Should start low, peak around 30%, then decay
        double startLR = learningRates[0];
        double peakLR = learningRates.Max();
        double endLR = learningRates[^1];

        Assert.True(startLR < peakLR, "Start LR should be less than peak");
        Assert.True(endLR < peakLR, "End LR should be less than peak");
        Assert.True(peakLR > 0.08, "Peak should be near max LR");
    }

    #endregion

    #region Cyclic LR Scheduler Tests

    [Fact]
    public void AdamOptimizer_WithCyclicScheduler_LearningRateCycles()
    {
        // Arrange
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.01, stepSizeUp: 5, stepSizeDown: 5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Act - Complete one full cycle (10 steps)
        var learningRates = new List<double>();

        for (int i = 0; i < 10; i++)
        {
            learningRates.Add(scheduler.CurrentLearningRate);
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.StepScheduler();
        }

        // Assert - LR should cycle: increase for 5 steps, decrease for 5 steps
        // Check that there's variation in learning rates
        double minLR = learningRates.Min();
        double maxLR = learningRates.Max();
        Assert.True(maxLR > minLR * 2, "LR should vary significantly in cyclic mode");
    }

    #endregion

    #region GetCurrentLearningRate Tests

    [Fact]
    public void Optimizer_GetCurrentLearningRate_ReturnsSchedulerControlledRate()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 1, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Act
        double initialLR = optimizer.GetCurrentLearningRate();
        optimizer.StepScheduler();
        double afterStepLR = optimizer.GetCurrentLearningRate();

        // Assert
        Assert.Equal(0.1, initialLR, Tolerance);
        Assert.Equal(0.05, afterStepLR, Tolerance);
    }

    [Fact]
    public void Optimizer_OnEpochEnd_StepsSchedulerInEpochMode()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 1, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Act
        double lrBeforeEpochEnd = optimizer.GetCurrentLearningRate();
        optimizer.OnEpochEnd();
        double lrAfterEpochEnd = optimizer.GetCurrentLearningRate();

        // Assert
        Assert.Equal(0.1, lrBeforeEpochEnd, Tolerance);
        Assert.Equal(0.05, lrAfterEpochEnd, Tolerance);
    }

    [Fact]
    public void Optimizer_OnBatchEnd_StepsSchedulerInBatchMode()
    {
        // Arrange
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 1, gamma: 0.5);
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Act
        double lrBeforeBatchEnd = optimizer.GetCurrentLearningRate();
        optimizer.OnBatchEnd();
        double lrAfterBatchEnd = optimizer.GetCurrentLearningRate();

        // Assert
        Assert.Equal(0.1, lrBeforeBatchEnd, Tolerance);
        Assert.Equal(0.05, lrAfterBatchEnd, Tolerance);
    }

    #endregion
}
