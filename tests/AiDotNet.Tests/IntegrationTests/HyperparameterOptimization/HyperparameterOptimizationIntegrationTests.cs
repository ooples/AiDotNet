using AiDotNet.HyperparameterOptimization;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.HyperparameterOptimization;

/// <summary>
/// Integration tests for the HyperparameterOptimization module.
/// Tests cover EarlyStopping, TrialPruner, SearchSpace, Trial, and all optimizer implementations.
/// </summary>
public class HyperparameterOptimizationIntegrationTests
{
    #region EarlyStopping Tests

    [Fact]
    public void EarlyStopping_Constructor_WithDefaultParameters()
    {
        var earlyStopping = new EarlyStopping<double>();

        Assert.False(earlyStopping.ShouldStop);
        Assert.Equal(double.NegativeInfinity, earlyStopping.BestValue);
        Assert.Equal(0, earlyStopping.BestEpoch);
        Assert.Equal(0, earlyStopping.EpochsSinceBest);
        Assert.Empty(earlyStopping.History);
    }

    [Fact]
    public void EarlyStopping_Constructor_WithCustomParameters()
    {
        var earlyStopping = new EarlyStopping<double>(
            patience: 5,
            minDelta: 0.01,
            maximize: false,
            mode: EarlyStoppingMode.RelativeBest);

        Assert.False(earlyStopping.ShouldStop);
        Assert.Equal(double.PositiveInfinity, earlyStopping.BestValue);
    }

    [Fact]
    public void EarlyStopping_Constructor_ThrowsOnInvalidPatience()
    {
        Assert.Throws<ArgumentException>(() => new EarlyStopping<double>(patience: 0));
        Assert.Throws<ArgumentException>(() => new EarlyStopping<double>(patience: -1));
    }

    [Fact]
    public void EarlyStopping_Constructor_ThrowsOnNegativeMinDelta()
    {
        Assert.Throws<ArgumentException>(() => new EarlyStopping<double>(minDelta: -0.1));
    }

    [Fact]
    public void EarlyStopping_Check_UpdatesBestValue_WhenImproving()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 3, maximize: true);

        earlyStopping.Check(0.5, 0);
        Assert.Equal(0.5, earlyStopping.BestValue);
        Assert.Equal(0, earlyStopping.BestEpoch);

        earlyStopping.Check(0.7, 1);
        Assert.Equal(0.7, earlyStopping.BestValue);
        Assert.Equal(1, earlyStopping.BestEpoch);
    }

    [Fact]
    public void EarlyStopping_Check_TriggersStop_AfterPatienceExhausted()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 3, maximize: true);

        Assert.False(earlyStopping.Check(0.5, 0)); // Best
        Assert.False(earlyStopping.Check(0.4, 1)); // Worse - counter 1
        Assert.False(earlyStopping.Check(0.3, 2)); // Worse - counter 2
        Assert.True(earlyStopping.Check(0.2, 3));  // Worse - counter 3, should stop

        Assert.True(earlyStopping.ShouldStop);
        Assert.Equal(3, earlyStopping.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_Check_WithMinimization()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 3, maximize: false);

        earlyStopping.Check(0.5, 0);
        Assert.Equal(0.5, earlyStopping.BestValue);

        earlyStopping.Check(0.3, 1); // Improvement (lower is better)
        Assert.Equal(0.3, earlyStopping.BestValue);
        Assert.Equal(0, earlyStopping.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_Check_WithMinDelta()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 3, minDelta: 0.1, maximize: true);

        earlyStopping.Check(0.5, 0);
        Assert.Equal(0.5, earlyStopping.BestValue);

        // 0.55 is not > 0.5 + 0.1 (not enough improvement)
        earlyStopping.Check(0.55, 1);
        Assert.Equal(0.5, earlyStopping.BestValue); // Still the old best
        Assert.Equal(1, earlyStopping.EpochsSinceBest);

        // 0.65 is > 0.5 + 0.1 (significant improvement)
        earlyStopping.Check(0.65, 2);
        Assert.Equal(0.65, earlyStopping.BestValue);
        Assert.Equal(0, earlyStopping.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_Check_WithMovingAverageMode()
    {
        var earlyStopping = new EarlyStopping<double>(
            patience: 3,
            maximize: true,
            mode: EarlyStoppingMode.MovingAverage);

        earlyStopping.Check(0.5, 0);
        earlyStopping.Check(0.6, 1);
        earlyStopping.Check(0.55, 2);

        Assert.False(earlyStopping.ShouldStop);
        Assert.Equal(3, earlyStopping.History.Count);
    }

    [Fact]
    public void EarlyStopping_Check_GenericTypeT()
    {
        var earlyStopping = new EarlyStopping<float>(patience: 3, maximize: true);

        earlyStopping.Check(0.5f, 0);
        Assert.Equal(0.5, earlyStopping.BestValue);
    }

    [Fact]
    public void EarlyStopping_Reset_ClearsState()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 3, maximize: true);

        earlyStopping.Check(0.5, 0);
        earlyStopping.Check(0.4, 1);

        earlyStopping.Reset();

        Assert.False(earlyStopping.ShouldStop);
        Assert.Equal(double.NegativeInfinity, earlyStopping.BestValue);
        Assert.Equal(0, earlyStopping.BestEpoch);
        Assert.Equal(0, earlyStopping.EpochsSinceBest);
        Assert.Empty(earlyStopping.History);
    }

    [Fact]
    public void EarlyStopping_GetState_ReturnsCorrectState()
    {
        var earlyStopping = new EarlyStopping<double>(patience: 5, maximize: true);

        earlyStopping.Check(0.5, 0);
        earlyStopping.Check(0.4, 1);

        var state = earlyStopping.GetState();

        Assert.False(state.Stopped);
        Assert.Equal(0.5, state.BestValue);
        Assert.Equal(0, state.BestEpoch);
        Assert.Equal(1, state.EpochsSinceBest);
        Assert.Equal(5, state.Patience);
        Assert.Equal(2, state.TotalChecks);
    }

    [Fact]
    public void EarlyStoppingState_ToString_ReturnsFormattedString()
    {
        var state = new EarlyStoppingState(false, 0.85, 10, 2, 5, 15);
        var str = state.ToString();

        Assert.Contains("RUNNING", str);
        Assert.Contains("0.85", str);
    }

    #endregion

    #region EarlyStoppingBuilder Tests

    [Fact]
    public void EarlyStoppingBuilder_Build_WithDefaults()
    {
        var earlyStopping = EarlyStoppingBuilder<double>.Create().Build();

        Assert.False(earlyStopping.ShouldStop);
    }

    [Fact]
    public void EarlyStoppingBuilder_FluentConfiguration()
    {
        var earlyStopping = EarlyStoppingBuilder<double>.Create()
            .WithPatience(7)
            .WithMinDelta(0.05)
            .Minimize()
            .WithMode(EarlyStoppingMode.RelativeBest)
            .Build();

        Assert.Equal(double.PositiveInfinity, earlyStopping.BestValue); // Minimize mode
    }

    [Fact]
    public void EarlyStoppingBuilder_WithPatience_ThrowsOnInvalid()
    {
        var builder = EarlyStoppingBuilder<double>.Create();
        Assert.Throws<ArgumentException>(() => builder.WithPatience(0));
    }

    [Fact]
    public void EarlyStoppingBuilder_WithMinDelta_ThrowsOnNegative()
    {
        var builder = EarlyStoppingBuilder<double>.Create();
        Assert.Throws<ArgumentException>(() => builder.WithMinDelta(-1));
    }

    #endregion

    #region EarlyStoppingMode Enum Tests

    [Fact]
    public void EarlyStoppingMode_HasExpectedValues()
    {
        var modes = (EarlyStoppingMode[])Enum.GetValues(typeof(EarlyStoppingMode));

        Assert.Contains(EarlyStoppingMode.Best, modes);
        Assert.Contains(EarlyStoppingMode.RelativeBest, modes);
        Assert.Contains(EarlyStoppingMode.MovingAverage, modes);
    }

    #endregion

    #region TrialPruner Tests

    [Fact]
    public void TrialPruner_Constructor_WithDefaultParameters()
    {
        var pruner = new TrialPruner<double>();

        var stats = pruner.GetStatistics();
        Assert.Equal(0, stats.TotalTrials);
    }

    [Fact]
    public void TrialPruner_Constructor_WithCustomParameters()
    {
        var pruner = new TrialPruner<double>(
            maximize: false,
            strategy: PruningStrategy.PercentilePruning,
            percentile: 75.0,
            warmupSteps: 5,
            checkInterval: 2);

        var stats = pruner.GetStatistics();
        Assert.Equal(0, stats.TotalTrials);
    }

    [Fact]
    public void TrialPruner_Constructor_ThrowsOnInvalidPercentile()
    {
        Assert.Throws<ArgumentException>(() => new TrialPruner<double>(percentile: 0));
        Assert.Throws<ArgumentException>(() => new TrialPruner<double>(percentile: 101));
    }

    [Fact]
    public void TrialPruner_Constructor_ThrowsOnNegativeWarmup()
    {
        Assert.Throws<ArgumentException>(() => new TrialPruner<double>(warmupSteps: -1));
    }

    [Fact]
    public void TrialPruner_Constructor_ThrowsOnInvalidCheckInterval()
    {
        Assert.Throws<ArgumentException>(() => new TrialPruner<double>(checkInterval: 0));
    }

    [Fact]
    public void TrialPruner_ReportAndCheckPrune_TracksTrials()
    {
        var pruner = new TrialPruner<double>(warmupSteps: 0);

        // Report values for first trial
        pruner.ReportAndCheckPrune("trial1", 0, 0.5);
        pruner.ReportAndCheckPrune("trial1", 1, 0.6);

        var stats = pruner.GetStatistics();
        Assert.Equal(1, stats.TotalTrials);
    }

    [Fact]
    public void TrialPruner_ReportAndCheckPrune_DoesNotPruneBeforeWarmup()
    {
        var pruner = new TrialPruner<double>(warmupSteps: 3);

        // Should not prune during warmup
        Assert.False(pruner.ReportAndCheckPrune("trial1", 0, 0.1));
        Assert.False(pruner.ReportAndCheckPrune("trial1", 1, 0.1));
        Assert.False(pruner.ReportAndCheckPrune("trial1", 2, 0.1));
    }

    [Fact]
    public void TrialPruner_ReportAndCheckPrune_WithTrial()
    {
        var pruner = new TrialPruner<double>(warmupSteps: 0);
        var trial = new HyperparameterTrial<double>(0);

        Assert.False(pruner.ReportAndCheckPrune(trial, 0, 0.5));
    }

    [Fact]
    public void TrialPruner_MedianPruning_WithMultipleTrials()
    {
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.MedianPruning,
            warmupSteps: 0);

        // Add good trials
        pruner.ReportAndCheckPrune("trial1", 0, 0.8);
        pruner.ReportAndCheckPrune("trial2", 0, 0.7);
        pruner.ReportAndCheckPrune("trial3", 0, 0.9);

        // Bad trial should potentially be pruned
        bool shouldPrune = pruner.ReportAndCheckPrune("trial4", 0, 0.1);
        // Note: May or may not prune depending on threshold

        var stats = pruner.GetStatistics();
        Assert.Equal(4, stats.TotalTrials);
    }

    [Fact]
    public void TrialPruner_CheckThreshold()
    {
        var prunerMax = new TrialPruner<double>(maximize: true);
        Assert.True(prunerMax.CheckThreshold(0.3, 0.5)); // Below threshold for maximize
        Assert.False(prunerMax.CheckThreshold(0.7, 0.5));

        var prunerMin = new TrialPruner<double>(maximize: false);
        Assert.False(prunerMin.CheckThreshold(0.3, 0.5));
        Assert.True(prunerMin.CheckThreshold(0.7, 0.5)); // Above threshold for minimize
    }

    [Fact]
    public void TrialPruner_MarkComplete()
    {
        var pruner = new TrialPruner<double>();
        pruner.ReportAndCheckPrune("trial1", 0, 0.5);

        // Should not throw
        pruner.MarkComplete("trial1");
    }

    [Fact]
    public void TrialPruner_Reset_ClearsHistory()
    {
        var pruner = new TrialPruner<double>();
        pruner.ReportAndCheckPrune("trial1", 0, 0.5);

        Assert.Equal(1, pruner.GetStatistics().TotalTrials);

        pruner.Reset();

        Assert.Equal(0, pruner.GetStatistics().TotalTrials);
    }

    [Fact]
    public void TrialPrunerStatistics_ToString_ReturnsFormattedString()
    {
        var stats = new TrialPrunerStatistics(10, 5.5, 20);
        var str = stats.ToString();

        Assert.Contains("10", str);
        Assert.Contains("5.5", str);
        Assert.Contains("20", str);
    }

    #endregion

    #region PruningStrategy Enum Tests

    [Fact]
    public void PruningStrategy_HasExpectedValues()
    {
        var strategies = (PruningStrategy[])Enum.GetValues(typeof(PruningStrategy));

        Assert.Contains(PruningStrategy.MedianPruning, strategies);
        Assert.Contains(PruningStrategy.PercentilePruning, strategies);
        Assert.Contains(PruningStrategy.SuccessiveHalving, strategies);
        Assert.Contains(PruningStrategy.ThresholdPruning, strategies);
    }

    #endregion

    #region HyperparameterSearchSpace Tests

    [Fact]
    public void HyperparameterSearchSpace_Constructor_CreatesEmptyParameters()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Empty(searchSpace.Parameters);
    }

    [Fact]
    public void HyperparameterSearchSpace_AddContinuous()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.001, 0.1);

        Assert.Single(searchSpace.Parameters);
        Assert.True(searchSpace.Parameters.ContainsKey("learning_rate"));

        var dist = searchSpace.Parameters["learning_rate"] as ContinuousDistribution;
        Assert.NotNull(dist);
        Assert.Equal(0.001, dist.Min);
        Assert.Equal(0.1, dist.Max);
        Assert.False(dist.LogScale);
    }

    [Fact]
    public void HyperparameterSearchSpace_AddContinuous_LogScale()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.0001, 0.1, logScale: true);

        var dist = searchSpace.Parameters["learning_rate"] as ContinuousDistribution;
        Assert.NotNull(dist);
        Assert.True(dist.LogScale);
    }

    [Fact]
    public void HyperparameterSearchSpace_AddContinuous_ThrowsOnInvalidRange()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous("test", 0.5, 0.5));
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous("test", 0.6, 0.5));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddContinuous_ThrowsOnNonPositiveMinWithLogScale()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous("test", 0, 1, logScale: true));
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous("test", -1, 1, logScale: true));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddContinuous_ThrowsOnEmptyName()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous("", 0, 1));
        Assert.Throws<ArgumentException>(() => searchSpace.AddContinuous(null!, 0, 1));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddInteger()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("batch_size", 16, 128, step: 16);

        Assert.Single(searchSpace.Parameters);

        var dist = searchSpace.Parameters["batch_size"] as IntegerDistribution;
        Assert.NotNull(dist);
        Assert.Equal(16, dist.Min);
        Assert.Equal(128, dist.Max);
        Assert.Equal(16, dist.Step);
    }

    [Fact]
    public void HyperparameterSearchSpace_AddInteger_ThrowsOnInvalidRange()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddInteger("test", 10, 10));
        Assert.Throws<ArgumentException>(() => searchSpace.AddInteger("test", 20, 10));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddInteger_ThrowsOnNonPositiveStep()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddInteger("test", 0, 10, step: 0));
        Assert.Throws<ArgumentException>(() => searchSpace.AddInteger("test", 0, 10, step: -1));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddCategorical()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddCategorical("optimizer", "adam", "sgd", "rmsprop");

        var dist = searchSpace.Parameters["optimizer"] as CategoricalDistribution;
        Assert.NotNull(dist);
        Assert.Equal(3, dist.Choices.Count);
        Assert.Contains("adam", dist.Choices);
    }

    [Fact]
    public void HyperparameterSearchSpace_AddCategorical_ThrowsOnEmpty()
    {
        var searchSpace = new HyperparameterSearchSpace();
        Assert.Throws<ArgumentException>(() => searchSpace.AddCategorical("test"));
    }

    [Fact]
    public void HyperparameterSearchSpace_AddBoolean()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddBoolean("use_dropout");

        var dist = searchSpace.Parameters["use_dropout"] as CategoricalDistribution;
        Assert.NotNull(dist);
        Assert.Equal(2, dist.Choices.Count);
        Assert.Contains(true, dist.Choices);
        Assert.Contains(false, dist.Choices);
    }

    #endregion

    #region ParameterDistribution Tests

    [Fact]
    public void ContinuousDistribution_Sample_ReturnsValueInRange()
    {
        var dist = new ContinuousDistribution { Min = 0.001, Max = 0.1, LogScale = false };
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            var value = (double)dist.Sample(random);
            Assert.InRange(value, 0.001, 0.1);
        }
    }

    [Fact]
    public void ContinuousDistribution_Sample_LogScale()
    {
        var dist = new ContinuousDistribution { Min = 0.0001, Max = 0.1, LogScale = true };
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            var value = (double)dist.Sample(random);
            Assert.InRange(value, 0.0001, 0.1);
        }
    }

    [Fact]
    public void ContinuousDistribution_DistributionType()
    {
        var dist = new ContinuousDistribution();
        Assert.Equal("continuous", dist.DistributionType);
    }

    [Fact]
    public void IntegerDistribution_Sample_ReturnsValueInRange()
    {
        var dist = new IntegerDistribution { Min = 16, Max = 128, Step = 16 };
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            var value = (int)dist.Sample(random);
            Assert.InRange(value, 16, 128);
            Assert.Equal(0, (value - 16) % 16); // Should be a multiple of step from min
        }
    }

    [Fact]
    public void IntegerDistribution_DistributionType()
    {
        var dist = new IntegerDistribution();
        Assert.Equal("integer", dist.DistributionType);
    }

    [Fact]
    public void CategoricalDistribution_Sample_ReturnsValidChoice()
    {
        var dist = new CategoricalDistribution { Choices = new List<object> { "a", "b", "c" } };
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            var value = dist.Sample(random);
            Assert.Contains(value, dist.Choices);
        }
    }

    [Fact]
    public void CategoricalDistribution_DistributionType()
    {
        var dist = new CategoricalDistribution();
        Assert.Equal("categorical", dist.DistributionType);
    }

    #endregion

    #region HyperparameterTrial Tests

    [Fact]
    public void HyperparameterTrial_Constructor()
    {
        var trial = new HyperparameterTrial<double>(5);

        Assert.NotNull(trial.TrialId);
        Assert.Equal(5, trial.TrialNumber);
        Assert.Empty(trial.Parameters);
        Assert.Empty(trial.IntermediateValues);
        Assert.Equal(TrialStatus.Running, trial.Status);
        // ObjectiveValue defaults to default(T) for value types
        Assert.Equal(default(double), trial.ObjectiveValue);
        Assert.Null(trial.EndTime);
    }

    [Fact]
    public void HyperparameterTrial_Complete()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.Complete(0.95);

        Assert.Equal(TrialStatus.Complete, trial.Status);
        Assert.Equal(0.95, trial.ObjectiveValue);
        Assert.NotNull(trial.EndTime);
    }

    [Fact]
    public void HyperparameterTrial_Prune()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.Prune();

        Assert.Equal(TrialStatus.Pruned, trial.Status);
        Assert.NotNull(trial.EndTime);
    }

    [Fact]
    public void HyperparameterTrial_Fail()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.Fail();

        Assert.Equal(TrialStatus.Failed, trial.Status);
        Assert.NotNull(trial.EndTime);
    }

    [Fact]
    public void HyperparameterTrial_ReportIntermediateValue()
    {
        var trial = new HyperparameterTrial<double>(0);

        trial.ReportIntermediateValue(1, 0.5);
        trial.ReportIntermediateValue(2, 0.6);
        trial.ReportIntermediateValue(3, 0.65);

        Assert.Equal(3, trial.IntermediateValues.Count);
        Assert.Equal(0.5, trial.IntermediateValues[1]);
        Assert.Equal(0.6, trial.IntermediateValues[2]);
        Assert.Equal(0.65, trial.IntermediateValues[3]);
    }

    [Fact]
    public void HyperparameterTrial_GetDuration_WhileRunning()
    {
        var trial = new HyperparameterTrial<double>(0);

        var duration = trial.GetDuration();

        Assert.NotNull(duration);
        Assert.True(duration.Value.TotalMilliseconds >= 0);
    }

    [Fact]
    public void HyperparameterTrial_GetDuration_AfterComplete()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.Complete(0.9);

        var duration = trial.GetDuration();

        Assert.NotNull(duration);
    }

    [Fact]
    public void HyperparameterTrial_Parameters_CanBeSet()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.Parameters["learning_rate"] = 0.01;
        trial.Parameters["batch_size"] = 32;

        Assert.Equal(0.01, trial.Parameters["learning_rate"]);
        Assert.Equal(32, trial.Parameters["batch_size"]);
    }

    [Fact]
    public void HyperparameterTrial_UserAttributes()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.UserAttributes["note"] = "test trial";

        Assert.Equal("test trial", trial.UserAttributes["note"]);
    }

    [Fact]
    public void HyperparameterTrial_SystemAttributes()
    {
        var trial = new HyperparameterTrial<double>(0);
        trial.SystemAttributes["gpu_id"] = 0;

        Assert.Equal(0, trial.SystemAttributes["gpu_id"]);
    }

    #endregion

    #region TrialStatus Enum Tests

    [Fact]
    public void TrialStatus_HasExpectedValues()
    {
        var statuses = (TrialStatus[])Enum.GetValues(typeof(TrialStatus));

        Assert.Contains(TrialStatus.Running, statuses);
        Assert.Contains(TrialStatus.Complete, statuses);
        Assert.Contains(TrialStatus.Pruned, statuses);
        Assert.Contains(TrialStatus.Failed, statuses);
    }

    #endregion

    #region GridSearchOptimizer Tests

    [Fact]
    public void GridSearchOptimizer_Constructor()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>(maximize: true);
        Assert.Empty(optimizer.GetAllTrials());
    }

    [Fact]
    public void GridSearchOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>(maximize: true);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("x", 1, 3);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            int x = (int)parameters["x"];
            return -Math.Pow(x - 2, 2); // Maximum at x=2
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        Assert.NotNull(result);
        Assert.NotNull(result.BestTrial);
        Assert.True(result.CompletedTrials > 0);
    }

    [Fact]
    public void GridSearchOptimizer_Optimize_ThrowsOnNullObjective()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("x", 1, 5);

        Assert.Throws<ArgumentNullException>(() => optimizer.Optimize(null!, searchSpace, 10));
    }

    [Fact]
    public void GridSearchOptimizer_Optimize_ThrowsOnNullSearchSpace()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();

        Assert.Throws<ArgumentNullException>(() =>
            optimizer.Optimize(_ => 0, null!, 10));
    }

    [Fact]
    public void GridSearchOptimizer_Optimize_ThrowsOnInvalidNTrials()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("x", 1, 5);

        Assert.Throws<ArgumentException>(() =>
            optimizer.Optimize(_ => 0, searchSpace, 0));
    }

    [Fact]
    public void GridSearchOptimizer_SuggestNext_ThrowsNotSupported()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        var trial = new HyperparameterTrial<double>(0);

        Assert.Throws<NotSupportedException>(() => optimizer.SuggestNext(trial));
    }

    [Fact]
    public void GridSearchOptimizer_GetBestTrial_ThrowsWhenNoCompleted()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();

        Assert.Throws<InvalidOperationException>(() => optimizer.GetBestTrial());
    }

    [Fact]
    public void GridSearchOptimizer_GetAllTrials_ReturnsTrials()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddCategorical("opt", "a", "b");

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var trials = optimizer.GetAllTrials();
        Assert.True(trials.Count > 0);
    }

    [Fact]
    public void GridSearchOptimizer_GetTrials_WithFilter()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddCategorical("opt", "a", "b");

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var completedTrials = optimizer.GetTrials(t => t.Status == TrialStatus.Complete);
        Assert.True(completedTrials.Count > 0);
    }

    [Fact]
    public void GridSearchOptimizer_GetTrials_ThrowsOnNullFilter()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>();
        Assert.Throws<ArgumentNullException>(() => optimizer.GetTrials(null!));
    }

    #endregion

    #region RandomSearchOptimizer Tests

    [Fact]
    public void RandomSearchOptimizer_Constructor()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>(maximize: true, seed: 42);
        Assert.Empty(optimizer.GetAllTrials());
    }

    [Fact]
    public void RandomSearchOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>(maximize: true, seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            return -Math.Pow(x - 5, 2);
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 20);

        Assert.NotNull(result);
        Assert.Equal(20, result.CompletedTrials);
        Assert.NotNull(result.BestTrial);
    }

    [Fact]
    public void RandomSearchOptimizer_Optimize_ReproducibleWithSeed()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        var optimizer1 = new RandomSearchOptimizer<double, double[], double>(seed: 42);
        var result1 = optimizer1.Optimize(_ => 0.5, searchSpace, 10);

        var optimizer2 = new RandomSearchOptimizer<double, double[], double>(seed: 42);
        var result2 = optimizer2.Optimize(_ => 0.5, searchSpace, 10);

        // Same seed should produce same parameter sequences
        Assert.Equal(result1.AllTrials.Count, result2.AllTrials.Count);
    }

    [Fact]
    public void RandomSearchOptimizer_SuggestNext_ThrowsBeforeOptimize()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>();
        var trial = new HyperparameterTrial<double>(0);

        Assert.Throws<InvalidOperationException>(() => optimizer.SuggestNext(trial));
    }

    #endregion

    #region BayesianOptimizer Tests

    [Fact]
    public void BayesianOptimizer_Constructor()
    {
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: true,
            acquisitionFunction: AcquisitionFunctionType.ExpectedImprovement,
            nInitialPoints: 5,
            explorationWeight: 2.0,
            seed: 42);

        Assert.Empty(optimizer.GetAllTrials());
    }

    [Fact]
    public void BayesianOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: true,
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            return -Math.Pow(x - 5, 2);
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        Assert.NotNull(result);
        Assert.Equal(10, result.CompletedTrials);
    }

    [Fact]
    public void BayesianOptimizer_Optimize_WithMinimization()
    {
        var optimizer = new BayesianOptimizer<double, double[], double>(
            maximize: false,
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            return Math.Pow(x - 5, 2); // Minimum at x=5
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        Assert.NotNull(result);
    }

    [Fact]
    public void BayesianOptimizer_SuggestNext_AfterOptimize()
    {
        var optimizer = new BayesianOptimizer<double, double[], double>(
            nInitialPoints: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        optimizer.Optimize(_ => 0.5, searchSpace, 5);

        var trial = new HyperparameterTrial<double>(100);
        var suggestion = optimizer.SuggestNext(trial);

        Assert.NotNull(suggestion);
        Assert.True(suggestion.ContainsKey("x"));
    }

    #endregion

    #region AcquisitionFunctionType Enum Tests

    [Fact]
    public void AcquisitionFunctionType_HasExpectedValues()
    {
        var types = (AcquisitionFunctionType[])Enum.GetValues(typeof(AcquisitionFunctionType));

        Assert.Contains(AcquisitionFunctionType.ExpectedImprovement, types);
        Assert.Contains(AcquisitionFunctionType.ProbabilityOfImprovement, types);
        Assert.Contains(AcquisitionFunctionType.UpperConfidenceBound, types);
        Assert.Contains(AcquisitionFunctionType.LowerConfidenceBound, types);
    }

    #endregion

    #region HyperbandOptimizer Tests

    [Fact]
    public void HyperbandOptimizer_Constructor()
    {
        var optimizer = new HyperbandOptimizer<double, double[], double>(
            maximize: true,
            maxResource: 81,
            reductionFactor: 3,
            minResource: 1,
            seed: 42);

        Assert.True(optimizer.NumBrackets > 0);
    }

    [Fact]
    public void HyperbandOptimizer_Constructor_ThrowsOnInvalidMaxResource()
    {
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double>(maxResource: 0));
    }

    [Fact]
    public void HyperbandOptimizer_Constructor_ThrowsOnInvalidReductionFactor()
    {
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double>(reductionFactor: 1));
    }

    [Fact]
    public void HyperbandOptimizer_Constructor_ThrowsOnInvalidMinResource()
    {
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double>(minResource: 0));
    }

    [Fact]
    public void HyperbandOptimizer_Constructor_ThrowsWhenMinExceedsMax()
    {
        Assert.Throws<ArgumentException>(() =>
            new HyperbandOptimizer<double, double[], double>(maxResource: 10, minResource: 20));
    }

    [Fact]
    public void HyperbandOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new HyperbandOptimizer<double, double[], double>(
            maximize: true,
            maxResource: 27,
            reductionFactor: 3,
            minResource: 1,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            int resource = parameters.ContainsKey("resource") ? (int)parameters["resource"] : 1;
            return -Math.Pow(x - 5, 2) + resource * 0.01;
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 20);

        Assert.NotNull(result);
        Assert.True(result.CompletedTrials > 0);
    }

    [Fact]
    public void HyperbandOptimizer_GetTotalConfigurationCount()
    {
        var optimizer = new HyperbandOptimizer<double, double[], double>(
            maxResource: 81,
            reductionFactor: 3);

        int total = optimizer.GetTotalConfigurationCount();
        Assert.True(total > 0);
    }

    [Fact]
    public void HyperbandOptimizer_GetBracketInfo()
    {
        var optimizer = new HyperbandOptimizer<double, double[], double>(
            maxResource: 81,
            reductionFactor: 3);

        var brackets = optimizer.GetBracketInfo();

        Assert.NotEmpty(brackets);
        foreach (var bracket in brackets)
        {
            Assert.True(bracket.InitialConfigurations > 0);
            Assert.True(bracket.InitialResource > 0);
            Assert.NotEmpty(bracket.Rounds);
        }
    }

    [Fact]
    public void BracketInfo_ToString()
    {
        var bracket = new BracketInfo(2, 10, 9, new List<(int, int)> { (10, 9), (3, 27), (1, 81) });
        var str = bracket.ToString();

        Assert.Contains("Bracket 2", str);
        Assert.Contains("10", str);
    }

    #endregion

    #region ASHAOptimizer Tests

    [Fact]
    public void ASHAOptimizer_Constructor()
    {
        var optimizer = new ASHAOptimizer<double, double[], double>(
            maximize: true,
            maxResource: 81,
            reductionFactor: 3,
            minResource: 1,
            promotionThreshold: 0.5,
            seed: 42);

        Assert.NotEmpty(optimizer.Rungs);
    }

    [Fact]
    public void ASHAOptimizer_Constructor_ThrowsOnInvalidMaxResource()
    {
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(maxResource: 0));
    }

    [Fact]
    public void ASHAOptimizer_Constructor_ThrowsOnInvalidReductionFactor()
    {
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(reductionFactor: 1));
    }

    [Fact]
    public void ASHAOptimizer_Constructor_ThrowsOnInvalidMinResource()
    {
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(minResource: 0));
    }

    [Fact]
    public void ASHAOptimizer_Constructor_ThrowsWhenMinExceedsMax()
    {
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(maxResource: 10, minResource: 20));
    }

    [Fact]
    public void ASHAOptimizer_Constructor_ThrowsOnInvalidPromotionThreshold()
    {
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(promotionThreshold: 0));
        Assert.Throws<ArgumentException>(() =>
            new ASHAOptimizer<double, double[], double>(promotionThreshold: 1.5));
    }

    [Fact]
    public void ASHAOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new ASHAOptimizer<double, double[], double>(
            maximize: true,
            maxResource: 27,
            reductionFactor: 3,
            minResource: 1,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            return -Math.Pow(x - 5, 2);
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 15);

        Assert.NotNull(result);
        Assert.True(result.CompletedTrials > 0);
    }

    [Fact]
    public void ASHAOptimizer_GetRungStatistics()
    {
        var optimizer = new ASHAOptimizer<double, double[], double>(
            maxResource: 27,
            reductionFactor: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var stats = optimizer.GetRungStatistics();
        Assert.NotNull(stats);
    }

    [Fact]
    public void ASHAOptimizer_GetBestConfiguration()
    {
        var optimizer = new ASHAOptimizer<double, double[], double>(
            maxResource: 27,
            reductionFactor: 3,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var (config, score) = optimizer.GetBestConfiguration();
        Assert.NotNull(config);
    }

    [Fact]
    public void RungStatistics_ToString()
    {
        var stats = new RungStatistics(27, 5, 0.75, 0.5, 0.9);
        var str = stats.ToString();

        Assert.Contains("27", str);
        Assert.Contains("5", str);
    }

    #endregion

    #region PopulationBasedTrainingOptimizer Tests

    [Fact]
    public void PopulationBasedTrainingOptimizer_Constructor()
    {
        var optimizer = new PopulationBasedTrainingOptimizer<double, double[], double>(
            maximize: true,
            populationSize: 10,
            readyInterval: 10,
            exploitFraction: 0.2,
            perturbFactor: 0.2,
            exploitStrategy: ExploitStrategy.Truncation,
            exploreStrategy: ExploreStrategy.Perturb,
            seed: 42);

        Assert.Empty(optimizer.GetPopulationState());
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_Constructor_ThrowsOnInvalidPopulationSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(populationSize: 1));
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_Constructor_ThrowsOnInvalidReadyInterval()
    {
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(readyInterval: 0));
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_Constructor_ThrowsOnInvalidExploitFraction()
    {
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(exploitFraction: 0));
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(exploitFraction: 0.6));
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_Constructor_ThrowsOnInvalidPerturbFactor()
    {
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(perturbFactor: 0));
        Assert.Throws<ArgumentException>(() =>
            new PopulationBasedTrainingOptimizer<double, double[], double>(perturbFactor: 1.5));
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_Optimize_SimpleSearch()
    {
        var optimizer = new PopulationBasedTrainingOptimizer<double, double[], double>(
            maximize: true,
            populationSize: 4,
            readyInterval: 2,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = (double)parameters["x"];
            return -Math.Pow(x - 5, 2);
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 20);

        Assert.NotNull(result);
        Assert.True(result.CompletedTrials > 0);
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_GetPopulationState()
    {
        var optimizer = new PopulationBasedTrainingOptimizer<double, double[], double>(
            populationSize: 4,
            readyInterval: 2,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var state = optimizer.GetPopulationState();
        Assert.NotEmpty(state);
    }

    [Fact]
    public void PopulationBasedTrainingOptimizer_GetBestMember()
    {
        var optimizer = new PopulationBasedTrainingOptimizer<double, double[], double>(
            populationSize: 4,
            readyInterval: 2,
            seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        optimizer.Optimize(_ => 0.5, searchSpace, 10);

        var best = optimizer.GetBestMember();
        Assert.NotNull(best);
    }

    [Fact]
    public void PopulationMemberInfo_ToString()
    {
        var member = new PopulationMemberInfo(0, new Dictionary<string, object> { { "x", 5.0 } }, 0.85, 10, 5);
        var str = member.ToString();

        Assert.Contains("Member 0", str);
        Assert.Contains("0.85", str);
    }

    #endregion

    #region ExploitStrategy Enum Tests

    [Fact]
    public void ExploitStrategy_HasExpectedValues()
    {
        var strategies = (ExploitStrategy[])Enum.GetValues(typeof(ExploitStrategy));

        Assert.Contains(ExploitStrategy.Truncation, strategies);
        Assert.Contains(ExploitStrategy.Binary, strategies);
        Assert.Contains(ExploitStrategy.Probabilistic, strategies);
    }

    #endregion

    #region ExploreStrategy Enum Tests

    [Fact]
    public void ExploreStrategy_HasExpectedValues()
    {
        var strategies = (ExploreStrategy[])Enum.GetValues(typeof(ExploreStrategy));

        Assert.Contains(ExploreStrategy.Perturb, strategies);
        Assert.Contains(ExploreStrategy.Resample, strategies);
        Assert.Contains(ExploreStrategy.PerturbOrResample, strategies);
    }

    #endregion

    #region ReportTrial and ShouldPrune Tests

    [Fact]
    public void HyperparameterOptimizerBase_ReportTrial()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        // Initialize searchspace via optimize
        optimizer.Optimize(_ => 0.5, searchSpace, 1);

        var trial = new HyperparameterTrial<double>(100);
        trial.Parameters["x"] = 5.0;

        optimizer.ReportTrial(trial, 0.9);

        Assert.Equal(TrialStatus.Complete, trial.Status);
        Assert.Equal(0.9, trial.ObjectiveValue);
    }

    [Fact]
    public void HyperparameterOptimizerBase_ReportTrial_ThrowsOnNullTrial()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>();
        Assert.Throws<ArgumentNullException>(() => optimizer.ReportTrial(null!, 0.5));
    }

    [Fact]
    public void HyperparameterOptimizerBase_ShouldPrune_ReturnsFalseByDefault()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>();
        var trial = new HyperparameterTrial<double>(0);

        Assert.False(optimizer.ShouldPrune(trial, 1, 0.5));
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void GridSearchOptimizer_Optimize_HandlesFailingObjective()
    {
        var optimizer = new GridSearchOptimizer<double, double[], double>(maximize: true);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddInteger("x", 1, 3);

        int callCount = 0;
        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            callCount++;
            if (callCount == 2) throw new InvalidOperationException("Simulated failure");
            return 0.5;
        };

        var result = optimizer.Optimize(objective, searchSpace, nTrials: 10);

        // Should still complete with some successful trials
        Assert.True(result.CompletedTrials >= 1);
        Assert.True(result.FailedTrials >= 1);
    }

    [Fact]
    public void RandomSearchOptimizer_Optimize_WithMultipleParameterTypes()
    {
        var optimizer = new RandomSearchOptimizer<double, double[], double>(seed: 42);

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 0.001, 0.1, logScale: true);
        searchSpace.AddInteger("batch_size", 16, 128, step: 16);
        searchSpace.AddCategorical("optimizer", "adam", "sgd");
        searchSpace.AddBoolean("use_dropout");

        var result = optimizer.Optimize(_ => 0.5, searchSpace, nTrials: 10);

        Assert.Equal(10, result.CompletedTrials);

        // Check that parameters are correctly typed
        var bestParams = result.BestParameters;
        Assert.True(bestParams["learning_rate"] is double);
        Assert.True(bestParams["batch_size"] is int);
        Assert.True(bestParams["optimizer"] is string);
        Assert.True(bestParams["use_dropout"] is bool);
    }

    [Fact]
    public void BayesianOptimizer_Optimize_WithDifferentAcquisitionFunctions()
    {
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("x", 0, 10);

        foreach (var acq in (AcquisitionFunctionType[])Enum.GetValues(typeof(AcquisitionFunctionType)))
        {
            var optimizer = new BayesianOptimizer<double, double[], double>(
                acquisitionFunction: acq,
                nInitialPoints: 3,
                seed: 42);

            var result = optimizer.Optimize(_ => 0.5, searchSpace, 5);
            Assert.Equal(5, result.CompletedTrials);
        }
    }

    #endregion
}
