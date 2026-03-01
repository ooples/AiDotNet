using AiDotNet.HyperparameterOptimization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.HyperparameterOptimization;

/// <summary>
/// Deep math-correctness integration tests for EarlyStopping and TrialPruner.
/// Each test hand-computes the expected state/decision and verifies the code matches.
/// </summary>
public class HyperparameterOptimizationDeepMathIntegrationTests
{
    private const double Eps = 1e-10;

    // ========================================================================
    // EarlyStopping - Best Mode (maximize)
    // ========================================================================

    [Fact]
    public void EarlyStopping_BestMode_Maximize_FirstValueAlwaysImproves()
    {
        // First value always improves since bestValue starts at -Infinity
        var es = new EarlyStopping<double>(patience: 3, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        bool stopped = es.Check(-1000.0, 0);

        Assert.False(stopped);
        Assert.Equal(-1000.0, es.BestValue);
        Assert.Equal(0, es.BestEpoch);
        Assert.Equal(0, es.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_BestMode_Maximize_PatienceExhaustion()
    {
        // patience=3, maximize=true, minDelta=0
        // values: 10, 20, 15, 15, 15 → should stop at index 4 (3 non-improvements after best at index 1)
        var es = new EarlyStopping<double>(patience: 3, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        Assert.False(es.Check(10.0, 0)); // improves (10 > -inf)
        Assert.Equal(10.0, es.BestValue);

        Assert.False(es.Check(20.0, 1)); // improves (20 > 10)
        Assert.Equal(20.0, es.BestValue);
        Assert.Equal(1, es.BestEpoch);

        Assert.False(es.Check(15.0, 2)); // no improvement (15 < 20), counter=1
        Assert.Equal(1, es.EpochsSinceBest);

        Assert.False(es.Check(15.0, 3)); // no improvement, counter=2
        Assert.Equal(2, es.EpochsSinceBest);

        Assert.True(es.Check(15.0, 4)); // no improvement, counter=3 >= patience → stopped
        Assert.True(es.ShouldStop);
        Assert.Equal(20.0, es.BestValue);
        Assert.Equal(1, es.BestEpoch);
        Assert.Equal(3, es.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_BestMode_Maximize_MinDelta_ExactBoundary()
    {
        // patience=2, maximize=true, minDelta=0.5
        // bestValue=10: need value > 10 + 0.5 = 10.5
        // value=10.5 should NOT be improvement (not strictly greater)
        // value=10.5 + epsilon should be improvement
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.5, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0, 0); // bestValue = 10.0

        es.Check(10.5, 1); // 10.5 > 10.0 + 0.5 = 10.5? No (not strictly greater)
        Assert.Equal(1, es.EpochsSinceBest); // counter incremented
        Assert.Equal(10.0, es.BestValue); // best unchanged

        es.Check(10.5 + 1e-12, 2); // slightly above threshold → should improve
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.True(es.BestValue > 10.0); // best updated
    }

    [Fact]
    public void EarlyStopping_BestMode_Maximize_ImprovementResetsCounter()
    {
        // After 2 non-improvements, an improvement should reset counter to 0
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0, 0); // best=10
        es.Check(5.0, 1);  // counter=1
        es.Check(5.0, 2);  // counter=2

        Assert.Equal(2, es.EpochsSinceBest);
        Assert.False(es.ShouldStop);

        es.Check(15.0, 3); // improvement! counter resets to 0, best=15
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(15.0, es.BestValue);
        Assert.Equal(3, es.BestEpoch);
    }

    [Fact]
    public void EarlyStopping_BestMode_Maximize_StaysStoppedAfterTrigger()
    {
        // Once stopped, subsequent checks should still return true
        var es = new EarlyStopping<double>(patience: 1, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0, 0); // best=10
        Assert.True(es.Check(5.0, 1)); // counter=1 >= patience=1 → stopped

        // Even with a better value, still stopped
        Assert.True(es.Check(100.0, 2));
        Assert.True(es.ShouldStop);
    }

    // ========================================================================
    // EarlyStopping - Best Mode (minimize)
    // ========================================================================

    [Fact]
    public void EarlyStopping_BestMode_Minimize_FirstValueAlwaysImproves()
    {
        var es = new EarlyStopping<double>(patience: 3, minDelta: 0.0, maximize: false, mode: EarlyStoppingMode.Best);

        bool stopped = es.Check(1000.0, 0); // 1000 < +Infinity → improved

        Assert.False(stopped);
        Assert.Equal(1000.0, es.BestValue);
    }

    [Fact]
    public void EarlyStopping_BestMode_Minimize_PatienceExhaustion()
    {
        // minimize: lower is better
        // values: 10, 5, 8, 8 → patience=2, should stop at index 3
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.0, maximize: false, mode: EarlyStoppingMode.Best);

        Assert.False(es.Check(10.0, 0)); // best=10
        Assert.False(es.Check(5.0, 1));  // best=5 (5 < 10)
        Assert.False(es.Check(8.0, 2));  // 8 < 5? No → counter=1
        Assert.True(es.Check(8.0, 3));   // 8 < 5? No → counter=2 → stopped

        Assert.Equal(5.0, es.BestValue);
        Assert.Equal(1, es.BestEpoch);
    }

    [Fact]
    public void EarlyStopping_BestMode_Minimize_MinDelta()
    {
        // minimize=true, minDelta=0.1, bestValue=5.0
        // need value < 5.0 - 0.1 = 4.9 to count as improvement
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.1, maximize: false, mode: EarlyStoppingMode.Best);

        es.Check(5.0, 0); // best=5.0

        es.Check(4.95, 1); // 4.95 < 4.9? No → not improved
        Assert.Equal(1, es.EpochsSinceBest);

        es.Check(4.89, 2); // 4.89 < 4.9? Yes → improved!
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(4.89, es.BestValue, 10);
    }

    // ========================================================================
    // EarlyStopping - RelativeBest Mode
    // ========================================================================

    [Fact]
    public void EarlyStopping_RelativeBest_Maximize_PositiveValues()
    {
        // maximize=true, minDelta=0.1 (10%)
        // bestValue=100: threshold = 100 * (1 + 0.1) = 110
        // value=109 → not improved (109 < 110)
        // value=111 → improved (111 > 110)
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.1, maximize: true, mode: EarlyStoppingMode.RelativeBest);

        es.Check(100.0, 0); // best=100 (first value always improves since bestValue was -Inf)

        es.Check(109.0, 1); // 109 > 100 * 1.1 = 110? No → counter=1
        Assert.Equal(1, es.EpochsSinceBest);

        es.Check(111.0, 2); // 111 > 110? Yes → improved, best=111
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(111.0, es.BestValue);
    }

    [Fact]
    public void EarlyStopping_RelativeBest_Maximize_NegativeValues()
    {
        // With negative bestValue and maximize=true:
        // bestValue=-10, minDelta=0.1
        // Correct threshold = -10 + |-10| * 0.1 = -10 + 1 = -9
        // value=-10.5 → -10.5 > -9? No → NOT improved (correct: -10.5 is worse than -10)
        // value=-8.5 → -8.5 > -9? Yes → improved (correct: -8.5 is better than -10)
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.1, maximize: true, mode: EarlyStoppingMode.RelativeBest);

        es.Check(-10.0, 0); // best=-10

        es.Check(-10.5, 1); // -10.5 is WORSE than -10 when maximizing → should NOT improve
        Assert.Equal(1, es.EpochsSinceBest);
        Assert.Equal(-10.0, es.BestValue);

        es.Check(-8.5, 2); // -8.5 > -9 → improved
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(-8.5, es.BestValue);
    }

    [Fact]
    public void EarlyStopping_RelativeBest_Minimize_NegativeValues()
    {
        // With negative bestValue and minimize=true:
        // bestValue=-10, minDelta=0.1
        // Correct threshold = -10 - |-10| * 0.1 = -10 - 1 = -11
        // value=-9.5 → -9.5 < -11? No → NOT improved (correct: -9.5 is worse than -10 for minimize)
        // value=-11.5 → -11.5 < -11? Yes → improved (correct: -11.5 is more negative = better for minimize)
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.1, maximize: false, mode: EarlyStoppingMode.RelativeBest);

        es.Check(-10.0, 0); // best=-10

        es.Check(-9.5, 1); // -9.5 is WORSE than -10 when minimizing → should NOT improve
        Assert.Equal(1, es.EpochsSinceBest);
        Assert.Equal(-10.0, es.BestValue);

        es.Check(-11.5, 2); // -11.5 < -11 → improved
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(-11.5, es.BestValue);
    }

    [Fact]
    public void EarlyStopping_RelativeBest_Minimize_PositiveValues()
    {
        // minimize=true, minDelta=0.2 (20%)
        // bestValue=100: threshold = 100 * (1 - 0.2) = 80
        // value=85 → not improved (85 > 80, need to go below)
        // value=79 → improved (79 < 80)
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.2, maximize: false, mode: EarlyStoppingMode.RelativeBest);

        es.Check(100.0, 0); // best=100

        es.Check(85.0, 1); // 85 < 100 * 0.8 = 80? No → counter=1
        Assert.Equal(1, es.EpochsSinceBest);

        es.Check(79.0, 2); // 79 < 80? Yes → improved
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(79.0, es.BestValue);
    }

    // ========================================================================
    // EarlyStopping - MovingAverage Mode
    // ========================================================================

    [Fact]
    public void EarlyStopping_MovingAverage_FirstTwoCallsAlwaysImprove()
    {
        // First call: history.Count=1 < 2 → always improves
        // Second call: history.Count=2 → computes average, but still might improve
        var es = new EarlyStopping<double>(patience: 3, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.MovingAverage);

        es.Check(10.0, 0);
        Assert.Equal(0, es.EpochsSinceBest); // first always improves

        // Second call: history=[10, 5], windowSize=min(3,1)=1, avg of skip(0).take(1)=[10], avg=10
        // 5 > 10? No → counter=1
        es.Check(5.0, 1);
        Assert.Equal(1, es.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_MovingAverage_HandComputedWindow()
    {
        // patience=2, maximize=true, minDelta=0
        // Step-by-step trace of moving average window:
        //
        // v0=10: history=[10], count<2 → improved, best=10
        // v1=8:  history=[10,8], windowSize=min(2,1)=1
        //        skip(2-1-1)=skip(0), take(1)=[10], avg=10
        //        8 > 10? No → counter=1
        // v2=12: history=[10,8,12], windowSize=min(2,2)=2
        //        skip(3-2-1)=skip(0), take(2)=[10,8], avg=9
        //        12 > 9? Yes → improved, best=12, counter=0
        // v3=9:  history=[10,8,12,9], windowSize=min(2,3)=2
        //        skip(4-2-1)=skip(1), take(2)=[8,12], avg=10
        //        9 > 10? No → counter=1
        // v4=10: history=[10,8,12,9,10], windowSize=min(2,4)=2
        //        skip(5-2-1)=skip(2), take(2)=[12,9], avg=10.5
        //        10 > 10.5? No → counter=2 → stopped
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.MovingAverage);

        Assert.False(es.Check(10.0, 0));
        Assert.Equal(0, es.EpochsSinceBest);

        Assert.False(es.Check(8.0, 1));
        Assert.Equal(1, es.EpochsSinceBest);

        Assert.False(es.Check(12.0, 2));
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(12.0, es.BestValue);

        Assert.False(es.Check(9.0, 3));
        Assert.Equal(1, es.EpochsSinceBest);

        Assert.True(es.Check(10.0, 4));
        Assert.Equal(2, es.EpochsSinceBest);
        Assert.True(es.ShouldStop);
    }

    [Fact]
    public void EarlyStopping_MovingAverage_WindowSizeCappedByPatience()
    {
        // patience=2, after many values the window should be at most patience-sized
        // values: 1, 2, 3, 4, 5, 6, 7, 8
        // At v7=8: history has 8 elements, windowSize=min(2, 7)=2
        // skip(8-2-1)=skip(5), take(2)=[6,7], avg=6.5
        // 8 > 6.5? Yes → improved
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.MovingAverage);

        for (int i = 1; i <= 8; i++)
            es.Check((double)i, i - 1);

        // Should not have stopped with monotonically increasing values
        Assert.False(es.ShouldStop);
    }

    [Fact]
    public void EarlyStopping_MovingAverage_MinDelta()
    {
        // patience=3, maximize=true, minDelta=1.0
        // values: 10, 10.5
        // v0=10: improved (first)
        // v1=10.5: history=[10,10.5], window=[10], avg=10
        //          10.5 > 10 + 1.0 = 11? No → counter=1
        var es = new EarlyStopping<double>(patience: 3, minDelta: 1.0, maximize: true, mode: EarlyStoppingMode.MovingAverage);

        es.Check(10.0, 0);
        es.Check(10.5, 1);
        Assert.Equal(1, es.EpochsSinceBest); // small improvement not enough with minDelta=1.0
    }

    // ========================================================================
    // EarlyStopping - Reset and State
    // ========================================================================

    [Fact]
    public void EarlyStopping_Reset_ClearsAllState()
    {
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0, 0);
        es.Check(5.0, 1);  // counter=1
        es.Check(5.0, 2);  // counter=2 → stopped

        Assert.True(es.ShouldStop);
        Assert.Equal(10.0, es.BestValue);
        Assert.Equal(3, es.History.Count);

        es.Reset();

        Assert.False(es.ShouldStop);
        Assert.Equal(double.NegativeInfinity, es.BestValue);
        Assert.Equal(0, es.BestEpoch);
        Assert.Equal(0, es.EpochsSinceBest);
        Assert.Equal(0, es.History.Count);
    }

    [Fact]
    public void EarlyStopping_GetState_ReturnsCorrectSnapshot()
    {
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0, 0);
        es.Check(20.0, 1);
        es.Check(15.0, 2);

        var state = es.GetState();

        Assert.False(state.Stopped);
        Assert.Equal(20.0, state.BestValue);
        Assert.Equal(1, state.BestEpoch);
        Assert.Equal(1, state.EpochsSinceBest);
        Assert.Equal(5, state.Patience);
        Assert.Equal(3, state.TotalChecks);
    }

    [Fact]
    public void EarlyStopping_History_RecordsAllValues()
    {
        var es = new EarlyStopping<double>(patience: 10, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        double[] values = { 1.0, 3.0, 2.0, 5.0, 4.0 };
        for (int i = 0; i < values.Length; i++)
            es.Check(values[i], i);

        Assert.Equal(values.Length, es.History.Count);
        for (int i = 0; i < values.Length; i++)
            Assert.Equal(values[i], es.History[i]);
    }

    [Fact]
    public void EarlyStopping_AutoEpoch_WhenEpochNotProvided()
    {
        // When epoch=-1 (default), it should use _history.Count as epoch
        var es = new EarlyStopping<double>(patience: 5, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        es.Check(10.0); // epoch auto-assigned to 0 (history count before add was 0)
        Assert.Equal(0, es.BestEpoch);

        es.Check(20.0); // epoch auto-assigned to 1
        Assert.Equal(1, es.BestEpoch);

        es.Check(5.0);  // no improvement
        es.Check(30.0); // epoch auto-assigned to 3
        Assert.Equal(3, es.BestEpoch);
    }

    // ========================================================================
    // EarlyStopping - Builder
    // ========================================================================

    [Fact]
    public void EarlyStopping_Builder_ProducesCorrectConfig()
    {
        var es = EarlyStoppingBuilder<double>.Create()
            .WithPatience(7)
            .WithMinDelta(0.05)
            .Minimize()
            .WithMode(EarlyStoppingMode.RelativeBest)
            .Build();

        // Test that builder settings are applied:
        // minimize + RelativeBest with minDelta=0.05
        // bestValue starts at +Inf for minimize
        es.Check(100.0, 0); // best=100

        // threshold = 100 * (1 - 0.05) = 95
        // 96 < 95? No → counter=1
        es.Check(96.0, 1);
        Assert.Equal(1, es.EpochsSinceBest);

        // 94 < 95? Yes → improved
        es.Check(94.0, 2);
        Assert.Equal(0, es.EpochsSinceBest);
    }

    [Fact]
    public void EarlyStopping_Builder_InvalidPatience_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            EarlyStoppingBuilder<double>.Create().WithPatience(0));
    }

    [Fact]
    public void EarlyStopping_Builder_NegativeMinDelta_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            EarlyStoppingBuilder<double>.Create().WithMinDelta(-0.1));
    }

    // ========================================================================
    // EarlyStopping - Patience=1 edge case
    // ========================================================================

    [Fact]
    public void EarlyStopping_Patience1_StopsOnFirstNonImprovement()
    {
        var es = new EarlyStopping<double>(patience: 1, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        Assert.False(es.Check(10.0, 0)); // best=10
        Assert.True(es.Check(9.0, 1));   // counter=1 >= 1 → stopped immediately
    }

    // ========================================================================
    // EarlyStopping - Generic T parameter (float)
    // ========================================================================

    [Fact]
    public void EarlyStopping_Float_TypeParameter()
    {
        var es = new EarlyStopping<float>(patience: 2, minDelta: 0.0, maximize: true, mode: EarlyStoppingMode.Best);

        Assert.False(es.Check(10.0f, 0));
        Assert.False(es.Check(20.0f, 1));
        Assert.False(es.Check(15.0f, 2)); // counter=1
        Assert.True(es.Check(15.0f, 3));  // counter=2 → stopped

        Assert.Equal(20.0, es.BestValue, 5);
    }

    // ========================================================================
    // TrialPruner - GetPercentile correctness
    // ========================================================================

    [Fact]
    public void TrialPruner_MedianPruning_HandComputedMedian_OddCount()
    {
        // 3 other trials at step 5 with values: 10, 20, 30
        // Median (50th percentile): index = 0.5 * (3-1) = 1.0 → sorted[1] = 20
        // Current trial value=15 < 20 → should prune (maximize=true)
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        // Populate 3 completed trials
        pruner.ReportAndCheckPrune("trial_a", 5, 10.0);
        pruner.ReportAndCheckPrune("trial_b", 5, 30.0);
        pruner.ReportAndCheckPrune("trial_c", 5, 20.0);

        // Check current trial
        bool shouldPrune = pruner.ReportAndCheckPrune("trial_current", 5, 15.0);
        Assert.True(shouldPrune); // 15 < median(10,20,30)=20 → prune
    }

    [Fact]
    public void TrialPruner_MedianPruning_HandComputedMedian_EvenCount()
    {
        // 4 other trials with values: 10, 20, 30, 40
        // Median: index = 0.5 * (4-1) = 1.5 → interpolate sorted[1] and sorted[2]
        // sorted = [10, 20, 30, 40], median = 20 * 0.5 + 30 * 0.5 = 25
        // Current value=26 > 25 → should NOT prune
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 3, 10.0);
        pruner.ReportAndCheckPrune("b", 3, 20.0);
        pruner.ReportAndCheckPrune("c", 3, 30.0);
        pruner.ReportAndCheckPrune("d", 3, 40.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("current", 3, 26.0);
        Assert.False(shouldPrune); // 26 > 25 → don't prune
    }

    [Fact]
    public void TrialPruner_MedianPruning_ExactlyAtMedian_ShouldNotPrune()
    {
        // values: [10, 20, 30], median=20
        // Current value=20: 20 < 20 → false → should NOT prune
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("current", 1, 20.0);
        Assert.False(shouldPrune); // exactly at median → not pruned
    }

    [Fact]
    public void TrialPruner_MedianPruning_Minimize()
    {
        // minimize: prune if value > median
        // values: [10, 20, 30], median=20
        // Current=25 → 25 > 20 → prune
        var pruner = new TrialPruner<double>(maximize: false, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("current", 1, 25.0);
        Assert.True(shouldPrune); // 25 > 20 → prune (minimize mode)
    }

    [Fact]
    public void TrialPruner_MedianPruning_SingleOtherTrial_NeverPrunes()
    {
        // Only 1 other trial → valuesAtStep.Count < 2 → never prune
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("other", 1, 100.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("current", 1, 0.0);
        Assert.False(shouldPrune); // not enough data
    }

    // ========================================================================
    // TrialPruner - Warmup Steps
    // ========================================================================

    [Fact]
    public void TrialPruner_WarmupSteps_NoPruningDuringWarmup()
    {
        // warmupSteps=5: steps 0-4 should never prune regardless of value
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 5);

        // Add other trials with high values
        for (int s = 0; s <= 10; s++)
        {
            pruner.ReportAndCheckPrune("good1", s, 100.0);
            pruner.ReportAndCheckPrune("good2", s, 200.0);
            pruner.ReportAndCheckPrune("good3", s, 150.0);
        }

        // Current trial is terrible but during warmup
        for (int s = 0; s < 5; s++)
        {
            bool shouldPrune = pruner.ReportAndCheckPrune("bad", s, -1000.0);
            Assert.False(shouldPrune); // warmup period
        }

        // At step 5, warmup is over - should now prune the terrible trial
        bool prunedAfterWarmup = pruner.ReportAndCheckPrune("bad", 5, -1000.0);
        Assert.True(prunedAfterWarmup); // -1000 < median → prune
    }

    // ========================================================================
    // TrialPruner - Check Interval
    // ========================================================================

    [Fact]
    public void TrialPruner_CheckInterval_OnlyChecksOnMultiples()
    {
        // checkInterval=3, warmupSteps=0
        // Should only prune at steps 0, 3, 6, 9, ...
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.MedianPruning,
            warmupSteps: 0,
            checkInterval: 3);

        // Add good other trials
        for (int s = 0; s <= 10; s++)
        {
            pruner.ReportAndCheckPrune("good1", s, 100.0);
            pruner.ReportAndCheckPrune("good2", s, 200.0);
        }

        // Bad trial: value=0 should be pruned, but only at interval multiples
        Assert.True(pruner.ReportAndCheckPrune("bad", 0, 0.0));   // step 0 % 3 == 0 → check → prune

        // Reset and try non-multiple steps
        pruner.Reset();
        for (int s = 0; s <= 10; s++)
        {
            pruner.ReportAndCheckPrune("good1", s, 100.0);
            pruner.ReportAndCheckPrune("good2", s, 200.0);
        }

        Assert.False(pruner.ReportAndCheckPrune("bad2", 1, 0.0)); // step 1 % 3 != 0 → skip
        Assert.False(pruner.ReportAndCheckPrune("bad2", 2, 0.0)); // step 2 % 3 != 0 → skip
        Assert.True(pruner.ReportAndCheckPrune("bad2", 3, 0.0));  // step 3 % 3 == 0 → check → prune
    }

    // ========================================================================
    // TrialPruner - PercentilePruning
    // ========================================================================

    [Fact]
    public void TrialPruner_PercentilePruning_Maximize_KeepTop25()
    {
        // percentile=25 with maximize=true
        // threshold = GetPercentile(values, 100 - 25) = GetPercentile(values, 75)
        // other trial values: [10, 20, 30, 40]
        // 75th percentile: index = 0.75 * 3 = 2.25
        // interpolate: sorted[2] * 0.75 + sorted[3] * 0.25 = 30 * 0.75 + 40 * 0.25 = 22.5 + 10 = 32.5
        // value < 32.5 → prune
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.PercentilePruning,
            percentile: 25,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);
        pruner.ReportAndCheckPrune("d", 1, 40.0);

        Assert.True(pruner.ReportAndCheckPrune("cur", 1, 30.0));  // 30 < 32.5 → prune
        Assert.False(pruner.ReportAndCheckPrune("cur2", 1, 35.0)); // 35 > 32.5 → keep
    }

    [Fact]
    public void TrialPruner_PercentilePruning_Minimize_KeepBottom25()
    {
        // percentile=25 with minimize=false
        // threshold = GetPercentile(values, 25)
        // other trial values: [10, 20, 30, 40]
        // 25th percentile: index = 0.25 * 3 = 0.75
        // interpolate: sorted[0] * 0.25 + sorted[1] * 0.75 = 10 * 0.25 + 20 * 0.75 = 2.5 + 15 = 17.5
        // value > 17.5 → prune (for minimize, higher is worse)
        var pruner = new TrialPruner<double>(
            maximize: false,
            strategy: PruningStrategy.PercentilePruning,
            percentile: 25,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);
        pruner.ReportAndCheckPrune("d", 1, 40.0);

        Assert.True(pruner.ReportAndCheckPrune("cur", 1, 20.0));   // 20 > 17.5 → prune
        Assert.False(pruner.ReportAndCheckPrune("cur2", 1, 15.0)); // 15 < 17.5 → keep
    }

    // ========================================================================
    // TrialPruner - SuccessiveHalving
    // ========================================================================

    [Fact]
    public void TrialPruner_SuccessiveHalving_HandComputed()
    {
        // Need at least 4 other trials for successive halving
        // maximize=true, other values: [10, 20, 30, 40]
        // sorted desc: [40, 30, 20, 10]
        // topHalfCount = 4/2 = 2, threshold = sorted[2] = 20
        // value < 20 → prune
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.SuccessiveHalving,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);
        pruner.ReportAndCheckPrune("d", 1, 40.0);

        Assert.True(pruner.ReportAndCheckPrune("cur", 1, 15.0));  // 15 < 20 → prune
        Assert.False(pruner.ReportAndCheckPrune("cur2", 1, 25.0)); // 25 > 20 → keep
    }

    [Fact]
    public void TrialPruner_SuccessiveHalving_ExactlyAtThreshold()
    {
        // threshold = sorted[topHalfCount]
        // values: [10, 20, 30, 40], sorted desc: [40, 30, 20, 10], threshold=20
        // value=20: 20 < 20 → false → NOT pruned
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.SuccessiveHalving,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);
        pruner.ReportAndCheckPrune("d", 1, 40.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("cur", 1, 20.0);
        Assert.False(shouldPrune); // exactly at threshold → not pruned
    }

    [Fact]
    public void TrialPruner_SuccessiveHalving_TooFewTrials_NeverPrunes()
    {
        // Need at least 4 other trials. With 3 → never prune
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.SuccessiveHalving,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 100.0);
        pruner.ReportAndCheckPrune("b", 1, 200.0);
        pruner.ReportAndCheckPrune("c", 1, 300.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("cur", 1, 0.0);
        Assert.False(shouldPrune); // only 3 other trials < 4
    }

    [Fact]
    public void TrialPruner_SuccessiveHalving_Minimize()
    {
        // minimize=true, other values: [10, 20, 30, 40]
        // sorted asc: [10, 20, 30, 40]
        // topHalfCount = 4/2 = 2, threshold = sorted[2] = 30
        // value > 30 → prune
        var pruner = new TrialPruner<double>(
            maximize: false,
            strategy: PruningStrategy.SuccessiveHalving,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 20.0);
        pruner.ReportAndCheckPrune("c", 1, 30.0);
        pruner.ReportAndCheckPrune("d", 1, 40.0);

        Assert.True(pruner.ReportAndCheckPrune("cur", 1, 35.0));  // 35 > 30 → prune
        Assert.False(pruner.ReportAndCheckPrune("cur2", 1, 25.0)); // 25 < 30 → keep
    }

    // ========================================================================
    // TrialPruner - ThresholdPruning
    // ========================================================================

    [Fact]
    public void TrialPruner_ThresholdPruning_NeverAutoprunes()
    {
        // ThresholdPruning always returns false from ShouldPrune
        // Must use CheckThreshold explicitly
        var pruner = new TrialPruner<double>(
            maximize: true,
            strategy: PruningStrategy.ThresholdPruning,
            warmupSteps: 0);

        pruner.ReportAndCheckPrune("other", 1, 100.0);
        pruner.ReportAndCheckPrune("other2", 1, 200.0);

        bool shouldPrune = pruner.ReportAndCheckPrune("cur", 1, 0.0);
        Assert.False(shouldPrune); // ThresholdPruning doesn't auto-prune
    }

    [Fact]
    public void TrialPruner_CheckThreshold_Maximize()
    {
        var pruner = new TrialPruner<double>(maximize: true);

        Assert.True(pruner.CheckThreshold(5.0, 10.0));  // 5 < 10 → prune
        Assert.False(pruner.CheckThreshold(15.0, 10.0)); // 15 > 10 → keep
        Assert.False(pruner.CheckThreshold(10.0, 10.0)); // 10 < 10? No → keep
    }

    [Fact]
    public void TrialPruner_CheckThreshold_Minimize()
    {
        var pruner = new TrialPruner<double>(maximize: false);

        Assert.True(pruner.CheckThreshold(15.0, 10.0));  // 15 > 10 → prune
        Assert.False(pruner.CheckThreshold(5.0, 10.0));   // 5 < 10 → keep
        Assert.False(pruner.CheckThreshold(10.0, 10.0));  // 10 > 10? No → keep
    }

    // ========================================================================
    // TrialPruner - GetValuesAtStep uses latest value at or before step
    // ========================================================================

    [Fact]
    public void TrialPruner_GetValuesAtStep_UsesLatestValueAtOrBeforeStep()
    {
        // Trial "other" reports at steps 1, 3, 5
        // When checking at step 4, should use value from step 3 (latest <= 4)
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("other1", 1, 100.0);
        pruner.ReportAndCheckPrune("other1", 3, 50.0);  // value drops at step 3
        pruner.ReportAndCheckPrune("other1", 5, 80.0);

        pruner.ReportAndCheckPrune("other2", 1, 60.0);
        pruner.ReportAndCheckPrune("other2", 3, 70.0);
        pruner.ReportAndCheckPrune("other2", 5, 90.0);

        // At step 4: other1's latest is step 3 → value 50, other2's latest is step 3 → value 70
        // median of [50, 70] = 60
        // current=55 < 60 → prune
        bool shouldPrune = pruner.ReportAndCheckPrune("current", 4, 55.0);
        Assert.True(shouldPrune);

        // current2=65 > 60 → keep
        bool shouldPrune2 = pruner.ReportAndCheckPrune("current2", 4, 65.0);
        Assert.False(shouldPrune2);
    }

    // ========================================================================
    // TrialPruner - Statistics
    // ========================================================================

    [Fact]
    public void TrialPruner_Statistics_TracksTrials()
    {
        var pruner = new TrialPruner<double>(maximize: true, warmupSteps: 0);

        pruner.ReportAndCheckPrune("t1", 0, 10.0);
        pruner.ReportAndCheckPrune("t1", 1, 20.0);
        pruner.ReportAndCheckPrune("t1", 2, 30.0);

        pruner.ReportAndCheckPrune("t2", 0, 5.0);
        pruner.ReportAndCheckPrune("t2", 1, 15.0);

        pruner.ReportAndCheckPrune("t3", 0, 50.0);

        var stats = pruner.GetStatistics();

        Assert.Equal(3, stats.TotalTrials);
        Assert.Equal(3, stats.MaxSteps); // t1 has 3 steps
        Assert.Equal(2.0, stats.AverageSteps, 5); // (3+2+1)/3 = 2.0
    }

    // ========================================================================
    // TrialPruner - Reset
    // ========================================================================

    [Fact]
    public void TrialPruner_Reset_ClearsHistory()
    {
        var pruner = new TrialPruner<double>(maximize: true, warmupSteps: 0);

        pruner.ReportAndCheckPrune("t1", 0, 100.0);
        pruner.ReportAndCheckPrune("t2", 0, 200.0);

        pruner.Reset();

        var stats = pruner.GetStatistics();
        Assert.Equal(0, stats.TotalTrials);

        // After reset, a trial should not be pruned (no history to compare)
        bool shouldPrune = pruner.ReportAndCheckPrune("new_trial", 0, 0.0);
        Assert.False(shouldPrune); // no other trials
    }

    // ========================================================================
    // TrialPruner - Constructor validation
    // ========================================================================

    [Fact]
    public void TrialPruner_InvalidPercentile_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new TrialPruner<double>(percentile: 0));
        Assert.Throws<ArgumentException>(() =>
            new TrialPruner<double>(percentile: 101));
        Assert.Throws<ArgumentException>(() =>
            new TrialPruner<double>(percentile: -5));
    }

    [Fact]
    public void TrialPruner_InvalidWarmupSteps_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new TrialPruner<double>(warmupSteps: -1));
    }

    [Fact]
    public void TrialPruner_InvalidCheckInterval_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new TrialPruner<double>(checkInterval: 0));
    }

    // ========================================================================
    // TrialPruner - Percentile computation edge cases
    // ========================================================================

    [Fact]
    public void TrialPruner_Percentile_TwoValues()
    {
        // 2 other trials: values [10, 30]
        // Median: index = 0.5 * (2-1) = 0.5
        // interpolate: sorted[0]*0.5 + sorted[1]*0.5 = 10*0.5 + 30*0.5 = 20
        // current=19 < 20 → prune
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        pruner.ReportAndCheckPrune("a", 1, 10.0);
        pruner.ReportAndCheckPrune("b", 1, 30.0);

        Assert.True(pruner.ReportAndCheckPrune("cur", 1, 19.0));  // 19 < 20 → prune
        Assert.False(pruner.ReportAndCheckPrune("cur2", 1, 21.0)); // 21 > 20 → keep
    }

    // ========================================================================
    // Integration: EarlyStopping + TrialPruner working together
    // ========================================================================

    [Fact]
    public void Integration_EarlyStoppingAndPruning_SimulatedTraining()
    {
        // Simulate a hyperparameter search with 3 trials
        // Trial 1: steady improvement (should complete)
        // Trial 2: stagnates early (should be stopped by early stopping)
        // Trial 3: worse than others (should be pruned)
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 2);

        var es1 = new EarlyStopping<double>(patience: 3, maximize: true, mode: EarlyStoppingMode.Best);
        var es2 = new EarlyStopping<double>(patience: 3, maximize: true, mode: EarlyStoppingMode.Best);
        var es3 = new EarlyStopping<double>(patience: 3, maximize: true, mode: EarlyStoppingMode.Best);

        // Simulated accuracy values over 10 steps
        double[] trial1 = { 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92 };
        double[] trial2 = { 0.50, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52 };
        double[] trial3 = { 0.30, 0.32, 0.33, 0.33, 0.34, 0.34, 0.35, 0.35, 0.35, 0.35 };

        int trial1Steps = 0, trial2Steps = 0, trial3Steps = 0;
        bool trial2EarlyStopped = false, trial3Pruned = false;

        for (int step = 0; step < 10; step++)
        {
            // Trial 1
            pruner.ReportAndCheckPrune("trial1", step, trial1[step]);
            es1.Check(trial1[step], step);
            trial1Steps++;

            // Trial 2 (early stopping check)
            if (!trial2EarlyStopped)
            {
                pruner.ReportAndCheckPrune("trial2", step, trial2[step]);
                trial2EarlyStopped = es2.Check(trial2[step], step);
                trial2Steps++;
            }

            // Trial 3 (pruning check)
            if (!trial3Pruned)
            {
                trial3Pruned = pruner.ReportAndCheckPrune("trial3", step, trial3[step]);
                trial3Steps++;
            }
        }

        // Trial 1 should complete all 10 steps
        Assert.Equal(10, trial1Steps);
        Assert.False(es1.ShouldStop);

        // Trial 2 should be early-stopped after stagnating
        // best at step 1 (0.52), then steps 2,3,4 are non-improving → stopped at step 4
        Assert.True(trial2EarlyStopped);
        Assert.True(trial2Steps < 10);

        // Trial 3 should be pruned (worse than median of trial1 and trial2)
        Assert.True(trial3Pruned);
        Assert.True(trial3Steps < 10);
    }

    // ========================================================================
    // EarlyStopping - Multiple resets
    // ========================================================================

    [Fact]
    public void EarlyStopping_MultipleResets_IndependentBehavior()
    {
        var es = new EarlyStopping<double>(patience: 2, maximize: true, mode: EarlyStoppingMode.Best);

        // First run
        es.Check(10.0, 0);
        es.Check(5.0, 1);
        es.Check(5.0, 2); // stopped
        Assert.True(es.ShouldStop);

        // Reset and run differently
        es.Reset();
        Assert.False(es.ShouldStop);

        es.Check(1.0, 0);
        es.Check(2.0, 1); // improves
        es.Check(3.0, 2); // improves
        Assert.False(es.ShouldStop);
        Assert.Equal(3.0, es.BestValue);
    }

    // ========================================================================
    // EarlyStopping - MovingAverage with minimize mode
    // ========================================================================

    [Fact]
    public void EarlyStopping_MovingAverage_Minimize_HandComputed()
    {
        // patience=2, minimize=true, minDelta=0
        // v0=10: improved (first)
        // v1=12: history=[10,12], window=[10], avg=10, 12 < 10? No → counter=1
        // v2=8: history=[10,12,8], windowSize=min(2,2)=2, skip(0).take(2)=[10,12], avg=11
        //       8 < 11? Yes → improved, best=8, counter=0
        // v3=9: history=[10,12,8,9], windowSize=min(2,3)=2, skip(1).take(2)=[12,8], avg=10
        //       9 < 10? Yes → improved, best=9, counter=0
        // v4=10: history=[10,12,8,9,10], windowSize=min(2,4)=2, skip(2).take(2)=[8,9], avg=8.5
        //        10 < 8.5? No → counter=1
        // v5=11: history=[10,12,8,9,10,11], windowSize=min(2,5)=2, skip(3).take(2)=[9,10], avg=9.5
        //        11 < 9.5? No → counter=2 → stopped
        var es = new EarlyStopping<double>(patience: 2, minDelta: 0.0, maximize: false, mode: EarlyStoppingMode.MovingAverage);

        Assert.False(es.Check(10.0, 0));
        Assert.Equal(0, es.EpochsSinceBest);

        Assert.False(es.Check(12.0, 1));
        Assert.Equal(1, es.EpochsSinceBest);

        Assert.False(es.Check(8.0, 2));
        Assert.Equal(0, es.EpochsSinceBest);

        Assert.False(es.Check(9.0, 3));
        Assert.Equal(0, es.EpochsSinceBest);

        Assert.False(es.Check(10.0, 4));
        Assert.Equal(1, es.EpochsSinceBest);

        Assert.True(es.Check(11.0, 5));
        Assert.Equal(2, es.EpochsSinceBest);
        Assert.True(es.ShouldStop);
    }

    // ========================================================================
    // TrialPruner - Multiple steps with evolving history
    // ========================================================================

    [Fact]
    public void TrialPruner_MultiStep_TrialCanSurviveThenGetPruned()
    {
        // A trial that starts OK but falls behind should eventually get pruned
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        // Two good trials that improve steadily
        double[] good1 = { 10, 20, 30, 40, 50 };
        double[] good2 = { 15, 25, 35, 45, 55 };
        double[] bad   = { 12, 18, 22, 23, 23 }; // starts OK, stagnates

        bool pruned = false;
        int prunedAt = -1;
        for (int s = 0; s < 5; s++)
        {
            pruner.ReportAndCheckPrune("good1", s, good1[s]);
            pruner.ReportAndCheckPrune("good2", s, good2[s]);

            if (!pruned)
            {
                pruned = pruner.ReportAndCheckPrune("bad", s, bad[s]);
                if (pruned) prunedAt = s;
            }
        }

        // At step 0: median of [10,15]=12.5, bad=12 < 12.5 → prune immediately
        // Actually let me recalculate: at step 0, good1 reports 10, good2 reports 15
        // Then bad reports 12. Other trials at step 0: good1=10, good2=15
        // median of [10, 15] = 12.5, bad=12 < 12.5 → prune
        Assert.True(pruned);
        Assert.Equal(0, prunedAt);
    }

    // ========================================================================
    // EarlyStopping constructor validation
    // ========================================================================

    [Fact]
    public void EarlyStopping_InvalidPatience_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new EarlyStopping<double>(patience: 0));
    }

    [Fact]
    public void EarlyStopping_NegativeMinDelta_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new EarlyStopping<double>(minDelta: -0.1));
    }

    // ========================================================================
    // TrialPruner - ExcludesCurrentTrial from comparison
    // ========================================================================

    [Fact]
    public void TrialPruner_ExcludesCurrentTrialFromComparison()
    {
        // If the ONLY other trial is the current one, it shouldn't compare with itself
        var pruner = new TrialPruner<double>(maximize: true, strategy: PruningStrategy.MedianPruning, warmupSteps: 0);

        // First report from "trial1"
        bool p1 = pruner.ReportAndCheckPrune("trial1", 0, 100.0);
        Assert.False(p1); // no other trials to compare with

        // Second report from same trial
        bool p2 = pruner.ReportAndCheckPrune("trial1", 1, 50.0);
        Assert.False(p2); // still only 1 trial total
    }
}
