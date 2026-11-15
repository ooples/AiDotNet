using System;
using System.Collections.Generic;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Every learning-rate scheduler runs on the fused compiled path. A per-batch scheduler maps to a plan schedule whose
/// per-step rates equal the eager scheduler's bit for bit (batch 1 at the starting rate, batch N at what the (N - 1)th
/// Step() returns); anything a formula cannot follow - a metric, state across steps, a scheduler already part-way
/// through, a non-per-batch cadence - gets the external rate the batch-end step keeps current.
/// </summary>
public class FusedLrScheduleEverySchedulerTests
{
    public static IEnumerable<object[]> PerBatchSchedulers()
    {
        yield return new object[] { "Cosine", new Func<ILearningRateScheduler>(() => new CosineAnnealingLRScheduler(0.1, 40, 1e-4)) };
        yield return new object[] { "Exponential", new Func<ILearningRateScheduler>(() => new ExponentialLRScheduler(0.1, 0.97)) };
        yield return new object[] { "Noam", new Func<ILearningRateScheduler>(() => new NoamSchedule(256, 20)) };
        yield return new object[] { "LinearWarmup", new Func<ILearningRateScheduler>(() => new LinearWarmupScheduler(0.1, 10, 80, 1e-3, LinearWarmupScheduler.DecayMode.Cosine, 1e-4)) };
        yield return new object[] { "Step", new Func<ILearningRateScheduler>(() => new StepLRScheduler(0.1, 7, 0.5)) };
        // Floors the Tensors schedules lack: 0.1 * 0.97^n falls below 0.01 at n = 76, 0.1 * 0.5^(n/7) at n = 28.
        yield return new object[] { "ExponentialFloor", new Func<ILearningRateScheduler>(() => new ExponentialLRScheduler(0.1, 0.97, 0.01)) };
        yield return new object[] { "StepFloor", new Func<ILearningRateScheduler>(() => new StepLRScheduler(0.1, 7, 0.5, 0.01)) };
        yield return new object[] { "Cyclic", new Func<ILearningRateScheduler>(() => new CyclicLRScheduler(0.001, 0.1, 10, 10)) };
        yield return new object[] { "CyclicTriangular2Asym", new Func<ILearningRateScheduler>(() => new CyclicLRScheduler(0.001, 0.1, 5, 13, CyclicLRScheduler.CyclicMode.Triangular2)) };
        yield return new object[] { "CyclicExpRange", new Func<ILearningRateScheduler>(() => new CyclicLRScheduler(0.001, 0.1, 6, 9, CyclicLRScheduler.CyclicMode.ExponentialRange, 0.99)) };
        yield return new object[] { "WsdLinear", new Func<ILearningRateScheduler>(() => new WarmupStableDecayScheduler(3e-4, 10, 50, 30)) };
        yield return new object[] { "WsdCosine", new Func<ILearningRateScheduler>(() => new WarmupStableDecayScheduler(3e-4, 0, 20, 40, WarmupStableDecayScheduler.DecayShape.Cosine, 1e-5)) };
        yield return new object[] { "WsdSqrt", new Func<ILearningRateScheduler>(() => new WarmupStableDecayScheduler(3e-4, 5, 5, 0, WarmupStableDecayScheduler.DecayShape.OneMinusSqrt, 2e-5)) };
        yield return new object[] { "Polynomial", new Func<ILearningRateScheduler>(() => new PolynomialLRScheduler(0.1, 60, 2.0, 1e-3)) };
        yield return new object[] { "MultiStep", new Func<ILearningRateScheduler>(() => new MultiStepLRScheduler(0.1, new[] { 0, 5, 20, 40 }, 0.5, 1e-3)) };
        yield return new object[] { "TriStage", new Func<ILearningRateScheduler>(() => new TriStageScheduler(0.1, 8, 10, 60, 1e-4, 2.0)) };
        yield return new object[] { "TriStageNoWarmup", new Func<ILearningRateScheduler>(() => new TriStageScheduler(0.1, 0, 10, 60)) };
        yield return new object[] { "NoamHoldAnnealing", new Func<ILearningRateScheduler>(() => new NoamHoldAnnealingScheduler(1e-3, 10, 5, 0.5)) };
        yield return new object[] { "WarmRestarts", new Func<ILearningRateScheduler>(() => new CosineAnnealingWarmRestartsScheduler(0.1, 7, 2, 1e-4)) };
        yield return new object[] { "OneCycleCosine", new Func<ILearningRateScheduler>(() => new OneCycleLRScheduler(0.1, 80)) };
        yield return new object[] { "OneCycleLinear", new Func<ILearningRateScheduler>(() => new OneCycleLRScheduler(0.1, 50, 0.25, 10, 100, OneCycleLRScheduler.AnnealingStrategy.Linear)) };
        yield return new object[] { "Sequential", new Func<ILearningRateScheduler>(() => new SequentialLRScheduler(
            new List<ILearningRateScheduler>
            {
                new LinearWarmupScheduler(0.1, 5, 5),
                new SequentialLRScheduler(new List<ILearningRateScheduler> { new ConstantLRScheduler(0.05), new PolynomialLRScheduler(0.05, 30) }, new[] { 10 }),
                new StepLRScheduler(0.01, 3, 0.5),
            },
            new[] { 12, 40 })) };
    }

    private static LrSchedule? Map(ILearningRateScheduler scheduler, SchedulerStepMode mode = SchedulerStepMode.StepPerBatch)
    {
        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { LearningRateScheduler = scheduler, SchedulerStepMode = mode });
        Assert.True(((IFusedOptimizerSpec)adam).TryGetFusedOptimizerConfig(out var cfg), $"{scheduler.GetType().Name} declined the fused path.");
        return cfg.Schedule;
    }

    [Theory]
    [MemberData(nameof(PerBatchSchedulers))]
    public void PerBatchScheduler_MapsToAPlanSchedule_BitIdenticalToEager(string name, Func<ILearningRateScheduler> make)
    {
        var fused = Map(make());
        Assert.NotNull(fused);
        Assert.IsNotType<ExternalLrSchedule>(fused);
        var eager = make();
        double eagerLr = eager.CurrentLearningRate;
        for (int n = 1; n <= 300; n++)
        {
            double fusedLr = fused!.GetLr(n);
            Assert.True(BitConverter.DoubleToInt64Bits(eagerLr) == BitConverter.DoubleToInt64Bits(fusedLr),
                $"{name}: batch {n} eager {eagerLr:R} vs fused {fusedLr:R}");
            eagerLr = eager.Step(double.NaN);   // the optimizer always steps with its last metric
        }
    }

    [Theory]
    [MemberData(nameof(PerBatchSchedulers))]
    public void BuiltInSchedulers_Checkpoint_AndCustomOnesSayTheyCannot(string name, Func<ILearningRateScheduler> make)
    {
        var fused = Map(make())!;
        // A floored schedule reads the eager formula live, so it has no checkpoint form.
        bool formula = name is "ExponentialFloor" or "StepFloor";
        if (formula) { Assert.Null(fused.TryCaptureCheckpoint()); return; }
        var restored = fused.TryCaptureCheckpoint()!.ToSchedule();
        for (int n = 1; n <= 300; n++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(fused.GetLr(n)), BitConverter.DoubleToInt64Bits(restored.GetLr(n)));
    }

    public static IEnumerable<object[]> ExternalCases()
    {
        yield return new object[] { "ReduceOnPlateau", new Func<ILearningRateScheduler>(() => new ReduceOnPlateauScheduler(0.1, patience: 2)), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "AdaptiveFitness", new Func<ILearningRateScheduler>(() => new AdaptiveFitnessScheduler(0.1)), SchedulerStepMode.StepPerBatch };
        // Custom formulas run on the external rate: the batch-end Step() already calls them once per step, and a
        // delegate or override may keep state (this one counts its calls), so the plan must not call them again.
        yield return new object[] { "Lambda", new Func<ILearningRateScheduler>(() => new LambdaLRScheduler(0.1, s => 1.0 / (1 + 0.1 * s))), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "StatefulLambda", new Func<ILearningRateScheduler>(() => { int calls = 0; return new LambdaLRScheduler(0.1, _ => 1.0 / ++calls); }), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "PureSubclass", new Func<ILearningRateScheduler>(() => new HalvingScheduler(0.1)), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "StatefulSubclass", new Func<ILearningRateScheduler>(() => new CountingScheduler(0.1)), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "AlreadyStepped", new Func<ILearningRateScheduler>(() => { var s = new StepLRScheduler(0.1, 2); s.Step(); return s; }), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "SequentialWithAStatefulChild", new Func<ILearningRateScheduler>(() => new SequentialLRScheduler(
            new List<ILearningRateScheduler> { new ConstantLRScheduler(0.1), new CountingScheduler(0.05) }, new[] { 3 })), SchedulerStepMode.StepPerBatch };
        yield return new object[] { "WarmupThenEpoch", new Func<ILearningRateScheduler>(() => new LinearWarmupScheduler(0.1, 4, 20)), SchedulerStepMode.WarmupThenEpoch };
        yield return new object[] { "PerEpoch", new Func<ILearningRateScheduler>(() => new WarmupStableDecayScheduler(0.1, 2, 4, 4)), SchedulerStepMode.StepPerEpoch };
    }

    [Theory]
    [MemberData(nameof(ExternalCases))]
    public void SchedulesNoFormulaCanFollow_RunFusedOnTheExternalRate(string name, Func<ILearningRateScheduler> make, SchedulerStepMode mode)
    {
        var scheduler = make();
        var fused = Map(scheduler, mode);
        var external = Assert.IsType<ExternalLrSchedule>(fused);
        Assert.True(Math.Abs(external.LearningRate - scheduler.CurrentLearningRate) <= 1e-7 * scheduler.CurrentLearningRate, name);
    }

    /// <summary>Only the formula is overridden, so the scheduler is a pure function of its step count.</summary>
    private sealed class HalvingScheduler : LearningRateSchedulerBase
    {
        public HalvingScheduler(double lr) : base(lr, 1e-4) { }
        protected override double ComputeLearningRate(int step) => _baseLearningRate * Math.Pow(0.5, step / 4);
    }

    /// <summary>Steps keep state the formula does not see, so no formula can stand in for it.</summary>
    private sealed class CountingScheduler : LearningRateSchedulerBase
    {
        private int _calls;
        public CountingScheduler(double lr) : base(lr) { }
        public override double Step()
        {
            _calls++;
            _currentStep++;
            _currentLearningRate = _baseLearningRate / (1 + _calls * _calls);
            return _currentLearningRate;
        }
        protected override double ComputeLearningRate(int step) => _baseLearningRate;
    }
}
