using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Maps a per-batch learning-rate scheduler onto the schedule a fused compiled training plan evaluates on its own,
/// one optimizer step at a time. Every mapping reproduces the eager per-batch sequence bit for bit: optimizer step 1
/// runs at the rate the scheduler starts at, step N at the rate <c>Step()</c> returns the (N - 1)th time.
/// </summary>
/// <remarks>
/// Built-in schedulers map to the Tensors schedule of the same formula, which a plan checkpoint can save and restore.
/// An exponential or step schedule with a nonzero floor maps to an adapter that reads its floored formula, since the
/// Tensors schedule has no floor. Anything else - a custom formula or delegate (a <see cref="LambdaLRScheduler"/>
/// delegate may keep state, so re-reading it per query need not match one call per step), a schedule that reacts to
/// a metric, keeps state across steps, or was already stepped - is declined here; the optimizer then gives the plan
/// an external rate that its batch-end step keeps current.
/// </remarks>
internal static class FusedLrScheduleMapper
{
    public static bool TryMap(ILearningRateScheduler scheduler, out LrSchedule? schedule)
    {
        schedule = null;
        // A plan counts its steps from 1; a scheduler already part-way through would be replayed from its start.
        if (scheduler is not LearningRateSchedulerBase stepped || stepped.CurrentStep != 0)
            return false;
        schedule = MapBuiltIn(scheduler);
        return schedule is not null;
    }

    // Exact types only: a subclass may override the formula the mapping copies.
    private static LrSchedule? MapBuiltIn(ILearningRateScheduler scheduler)
    {
        switch (scheduler)
        {
            case ConstantLRScheduler c when Is<ConstantLRScheduler>(c):
                return LrSchedule.Constant(c.CurrentLearningRate);
            case CosineAnnealingLRScheduler cosine when Is<CosineAnnealingLRScheduler>(cosine):
                // The eager formula operation for operation: Cosine(lr, tMax + 1, etaMin) matches it only to the last bit.
                return LrSchedule.CosineAnnealing(cosine.BaseLearningRate, cosine.TMax, cosine.EtaMin);
            case ExponentialLRScheduler expo when Is<ExponentialLRScheduler>(expo):
                // LrSchedule.Exponential has no floor; a floored schedule reads the eager formula instead.
                return expo.MinLearningRate == 0
                    ? LrSchedule.Exponential(expo.BaseLearningRate, expo.Gamma)
                    : new FormulaLrSchedule(expo, expo.CurrentLearningRate);
            case NoamSchedule noam when Is<NoamSchedule>(noam):
                return LrSchedule.Noam(noam.ModelDimension, noam.WarmupSteps, noam.Factor);
            case LinearWarmupScheduler warmup when Is<LinearWarmupScheduler>(warmup):
                return LrSchedule.LinearWarmup(
                    warmup.BaseLearningRate, warmup.WarmupSteps, warmup.TotalSteps, warmup.WarmupInitLr,
                    warmup.CurrentDecayMode switch
                    {
                        LinearWarmupScheduler.DecayMode.Linear => WarmupDecayMode.Linear,
                        LinearWarmupScheduler.DecayMode.Cosine => WarmupDecayMode.Cosine,
                        _ => WarmupDecayMode.Constant,
                    },
                    warmup.EndLr);
            case StepLRScheduler stepLr when Is<StepLRScheduler>(stepLr):
                return stepLr.MinLearningRate == 0
                    ? LrSchedule.Step(stepLr.BaseLearningRate, stepLr.StepSize, stepLr.Gamma)
                    : new FormulaLrSchedule(stepLr, stepLr.CurrentLearningRate);
            case CyclicLRScheduler cyclic when Is<CyclicLRScheduler>(cyclic):
                return LrSchedule.Cyclic(cyclic.BaseLearningRate, cyclic.MaxLearningRate, cyclic.StepSizeUp, cyclic.StepSizeDown,
                    cyclic.Mode switch
                    {
                        CyclicLRScheduler.CyclicMode.Triangular2 => CyclicLrMode.Triangular2,
                        CyclicLRScheduler.CyclicMode.ExponentialRange => CyclicLrMode.ExponentialRange,
                        _ => CyclicLrMode.Triangular,
                    },
                    cyclic.Gamma);
            case WarmupStableDecayScheduler wsd when Is<WarmupStableDecayScheduler>(wsd):
                return LrSchedule.WarmupStableDecay(wsd.BaseLearningRate, wsd.WarmupSteps, wsd.DecayStartStep, wsd.DecaySteps,
                    wsd.Shape switch
                    {
                        WarmupStableDecayScheduler.DecayShape.Cosine => StableDecayShape.Cosine,
                        WarmupStableDecayScheduler.DecayShape.OneMinusSqrt => StableDecayShape.OneMinusSqrt,
                        _ => StableDecayShape.Linear,
                    },
                    wsd.EndLr);
            case PolynomialLRScheduler poly when Is<PolynomialLRScheduler>(poly):
                return LrSchedule.Polynomial(poly.BaseLearningRate, poly.TotalSteps, poly.Power, poly.EndLearningRate);
            case MultiStepLRScheduler multi when Is<MultiStepLRScheduler>(multi):
                return LrSchedule.MultiStep(multi.BaseLearningRate, multi.Milestones.ToArray(), multi.Gamma, multi.MinLearningRate);
            case TriStageScheduler tri when Is<TriStageScheduler>(tri):
                return LrSchedule.TriStage(tri.BaseLearningRate, tri.WarmupSteps, tri.HoldSteps, tri.TotalSteps, tri.MinLearningRate, tri.DecayPower);
            case NoamHoldAnnealingScheduler hold:   // sealed
                return LrSchedule.NoamHoldAnnealing(hold.PeakLearningRate, hold.WarmupSteps, hold.HoldSteps, hold.DecayRate);
            case CosineAnnealingWarmRestartsScheduler sgdr when Is<CosineAnnealingWarmRestartsScheduler>(sgdr):
                return LrSchedule.CosineWarmRestarts(sgdr.BaseLearningRate, sgdr.T0, sgdr.TMult, sgdr.EtaMin);
            case OneCycleLRScheduler one when Is<OneCycleLRScheduler>(one):
                return LrSchedule.OneCycleWithAnneal(one.MaxLearningRate, one.TotalSteps, one.PctStart, one.DivFactor, one.FinalDivFactor,
                    one.Strategy == OneCycleLRScheduler.AnnealingStrategy.Linear ? OneCycleAnneal.Linear : OneCycleAnneal.Cosine);
            case SequentialLRScheduler sequential when Is<SequentialLRScheduler>(sequential):
            {
                // Each child steps only while active, from its own start, so each must be unstepped and mappable.
                var children = new LrSchedule[sequential.Schedulers.Count];
                for (int k = 0; k < children.Length; k++)
                {
                    if (!TryMap(sequential.Schedulers[k], out var child) || child is null)
                        return null;
                    children[k] = child;
                }
                return LrSchedule.Sequential(children, sequential.Milestones.ToArray());
            }
            default:
                return null;
        }
    }

    private static bool Is<TScheduler>(ILearningRateScheduler scheduler) => scheduler.GetType() == typeof(TScheduler);

    /// <summary>
    /// Reads a pure scheduler's formula: step 1 at its starting rate, step N at <c>GetLearningRateAtStep(N - 1)</c>,
    /// which is what its <c>Step()</c> returns the (N - 1)th time. It has no checkpoint form (the formula is read live),
    /// so a plan using it cannot be serialized; the optimizer's own scheduler state still is.
    /// </summary>
    private sealed class FormulaLrSchedule : LrSchedule
    {
        private readonly LearningRateSchedulerBase _scheduler;
        private readonly double _initial;

        public FormulaLrSchedule(LearningRateSchedulerBase scheduler, double initial)
        {
            _scheduler = scheduler;
            _initial = initial;
        }

        public override double GetLr(int step) => step <= 1 ? _initial : _scheduler.GetLearningRateAtStep(step - 1);
    }
}
