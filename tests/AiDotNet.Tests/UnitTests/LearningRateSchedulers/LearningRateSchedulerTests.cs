using AiDotNet.LearningRateSchedulers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.LearningRateSchedulers
{
    public class LearningRateSchedulerTests
    {
        #region StepLR Tests

        [Fact(Timeout = 60000)]
        public async Task StepLR_InitializesWithCorrectLearningRate()
        {
            var scheduler = new StepLRScheduler(0.1, stepSize: 10, gamma: 0.5);
            Assert.Equal(0.1, scheduler.CurrentLearningRate);
            Assert.Equal(0.1, scheduler.BaseLearningRate);
        }

        [Fact(Timeout = 60000)]
        public async Task StepLR_DecaysAtStepSize()
        {
            var scheduler = new StepLRScheduler(0.1, stepSize: 3, gamma: 0.5);

            // Steps 1-2: LR should remain at 0.1 (before stepSize)
            for (int i = 0; i < 2; i++)
            {
                scheduler.Step();
            }
            Assert.Equal(0.1, scheduler.CurrentLearningRate, 6);

            // Step 3: Should decay to 0.05 (at stepSize)
            // Per PyTorch StepLR: lr = initial_lr * gamma ** (step // step_size)
            scheduler.Step();
            Assert.Equal(0.05, scheduler.CurrentLearningRate, 6);
        }

        [Fact(Timeout = 60000)]
        public async Task StepLR_Reset_RestoresInitialState()
        {
            var scheduler = new StepLRScheduler(0.1, stepSize: 2, gamma: 0.5);

            for (int i = 0; i < 5; i++) scheduler.Step();
            Assert.NotEqual(0.1, scheduler.CurrentLearningRate);

            scheduler.Reset();
            Assert.Equal(0.1, scheduler.CurrentLearningRate);
            Assert.Equal(0, scheduler.CurrentStep);
        }

        #endregion

        #region CosineAnnealing Tests

        [Fact(Timeout = 60000)]
        public async Task CosineAnnealing_InitializesCorrectly()
        {
            var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 100, etaMin: 0.001);
            Assert.Equal(0.1, scheduler.CurrentLearningRate);
        }

        [Fact(Timeout = 60000)]
        public async Task CosineAnnealing_DecreasesToMinimum()
        {
            var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 10, etaMin: 0.01);

            // Run for full cycle
            for (int i = 0; i < 10; i++)
            {
                scheduler.Step();
            }

            // At tMax, should be at etaMin
            Assert.True(scheduler.CurrentLearningRate <= 0.1);
            Assert.True(scheduler.CurrentLearningRate >= 0.01);
        }

        [Fact(Timeout = 60000)]
        public async Task CosineAnnealing_FollowsCosineShape()
        {
            var scheduler = new CosineAnnealingLRScheduler(1.0, tMax: 4, etaMin: 0.0);

            // At step 0, LR = 1.0
            Assert.Equal(1.0, scheduler.CurrentLearningRate, 6);

            // At step 2 (halfway), LR should be around 0.5
            scheduler.Step();
            scheduler.Step();
            double midpointLr = scheduler.CurrentLearningRate;
            Assert.True(midpointLr > 0.3 && midpointLr < 0.7);
        }

        #endregion

        #region OneCycle Tests

        [Fact(Timeout = 60000)]
        public async Task OneCycle_InitializesCorrectly()
        {
            var scheduler = new OneCycleLRScheduler(0.1, totalSteps: 100);
            Assert.True(scheduler.CurrentLearningRate < 0.1); // Starts low
        }

        [Fact(Timeout = 60000)]
        public async Task OneCycle_ReachesPeakAtPctStart()
        {
            var scheduler = new OneCycleLRScheduler(0.1, totalSteps: 100, pctStart: 0.3);

            // After warmup phase (30 steps), should be near max
            for (int i = 0; i < 30; i++)
            {
                scheduler.Step();
            }

            // Should be close to max LR
            Assert.True(scheduler.CurrentLearningRate >= 0.05);
        }

        [Fact(Timeout = 60000)]
        public async Task OneCycle_DecaysAfterPeak()
        {
            var scheduler = new OneCycleLRScheduler(0.1, totalSteps: 100, pctStart: 0.3);

            // Warmup
            for (int i = 0; i < 30; i++) scheduler.Step();
            double peakLr = scheduler.CurrentLearningRate;

            // Continue past peak
            for (int i = 0; i < 50; i++) scheduler.Step();

            Assert.True(scheduler.CurrentLearningRate < peakLr);
        }

        #endregion

        #region LinearWarmup Tests

        [Fact(Timeout = 60000)]
        public async Task LinearWarmup_StartsAtInitialLr()
        {
            var scheduler = new LinearWarmupScheduler(
                baseLearningRate: 0.1,
                warmupSteps: 10,
                totalSteps: 100,
                warmupInitLr: 0.001);

            Assert.Equal(0.001, scheduler.CurrentLearningRate, 6);
        }

        [Fact(Timeout = 60000)]
        public async Task LinearWarmup_ReachesPeakAfterWarmup()
        {
            var scheduler = new LinearWarmupScheduler(
                baseLearningRate: 0.1,
                warmupSteps: 10,
                totalSteps: 100,
                warmupInitLr: 0.0);

            // During warmup
            for (int i = 0; i < 10; i++)
            {
                scheduler.Step();
            }

            // Should be at or near peak
            Assert.True(scheduler.CurrentLearningRate >= 0.09);
        }

        [Fact(Timeout = 60000)]
        public async Task LinearWarmup_DecaysAfterPeak()
        {
            var scheduler = new LinearWarmupScheduler(
                baseLearningRate: 0.1,
                warmupSteps: 10,
                totalSteps: 100,
                warmupInitLr: 0.0,
                decayMode: LinearWarmupScheduler.DecayMode.Linear,
                endLr: 0.001);

            // Warmup
            for (int i = 0; i < 10; i++) scheduler.Step();
            double peakLr = scheduler.CurrentLearningRate;

            // Decay phase
            for (int i = 0; i < 50; i++) scheduler.Step();

            Assert.True(scheduler.CurrentLearningRate < peakLr);
        }

        #endregion

        #region ExponentialLR Tests

        [Fact(Timeout = 60000)]
        public async Task ExponentialLR_DecaysExponentially()
        {
            var scheduler = new ExponentialLRScheduler(1.0, gamma: 0.9);

            scheduler.Step();
            Assert.Equal(0.9, scheduler.CurrentLearningRate, 6);

            scheduler.Step();
            Assert.Equal(0.81, scheduler.CurrentLearningRate, 6);

            scheduler.Step();
            Assert.Equal(0.729, scheduler.CurrentLearningRate, 6);
        }

        #endregion

        #region ReduceOnPlateau Tests

        [Fact(Timeout = 60000)]
        public async Task ReduceOnPlateau_DoesNotReduceWhenImproving()
        {
            var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.5, patience: 3);

            // Improving metrics (decreasing)
            scheduler.Step(1.0);
            scheduler.Step(0.9);
            scheduler.Step(0.8);
            scheduler.Step(0.7);

            Assert.Equal(0.1, scheduler.CurrentLearningRate, 6);
        }

        [Fact(Timeout = 60000)]
        public async Task ReduceOnPlateau_ReducesAfterPatience()
        {
            var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.5, patience: 2);

            // Plateau (not improving)
            scheduler.Step(1.0);
            scheduler.Step(1.0);
            scheduler.Step(1.0);
            scheduler.Step(1.0);

            // Should have reduced after patience exhausted
            Assert.True(scheduler.CurrentLearningRate < 0.1);
        }

        [Fact(Timeout = 60000)]
        public async Task ReduceOnPlateau_RespectsMinLearningRate()
        {
            var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.1, patience: 1, minLearningRate: 0.001);

            // Force multiple reductions
            for (int i = 0; i < 20; i++)
            {
                scheduler.Step(1.0);
            }

            Assert.True(scheduler.CurrentLearningRate >= 0.001);
        }

        #endregion

        #region CyclicLR Tests

        [Fact(Timeout = 60000)]
        public async Task CyclicLR_OscillatesBetweenBounds()
        {
            var scheduler = new CyclicLRScheduler(baseLearningRate: 0.001, maxLearningRate: 0.01, stepSizeUp: 5);

            double minObserved = double.MaxValue;
            double maxObserved = double.MinValue;

            for (int i = 0; i < 20; i++)
            {
                scheduler.Step();
                minObserved = Math.Min(minObserved, scheduler.CurrentLearningRate);
                maxObserved = Math.Max(maxObserved, scheduler.CurrentLearningRate);
            }

            Assert.True(minObserved >= 0.001);
            Assert.True(maxObserved <= 0.01);
        }

        #endregion

        #region Factory Tests

        [Fact(Timeout = 60000)]
        public async Task Factory_CreateForCNN_ReturnsStepLR()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForCNN();
            Assert.IsType<StepLRScheduler>(scheduler);
        }

        [Fact(Timeout = 60000)]
        public async Task Factory_CreateForTransformer_ReturnsLinearWarmup()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForTransformer();
            Assert.IsType<LinearWarmupScheduler>(scheduler);
        }

        [Fact(Timeout = 60000)]
        public async Task Factory_CreateForSuperConvergence_ReturnsOneCycle()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForSuperConvergence();
            Assert.IsType<OneCycleLRScheduler>(scheduler);
        }

        [Fact(Timeout = 60000)]
        public async Task Factory_CreateAdaptive_ReturnsReduceOnPlateau()
        {
            var scheduler = LearningRateSchedulerFactory.CreateAdaptive();
            Assert.IsType<ReduceOnPlateauScheduler>(scheduler);
        }

        [Fact(Timeout = 60000)]
        public async Task Factory_Create_ReturnsCorrectType()
        {
            Assert.IsType<StepLRScheduler>(
                LearningRateSchedulerFactory.Create(LearningRateSchedulerType.Step, 0.1));
            Assert.IsType<CosineAnnealingLRScheduler>(
                LearningRateSchedulerFactory.Create(LearningRateSchedulerType.CosineAnnealing, 0.1, 100));
            Assert.IsType<OneCycleLRScheduler>(
                LearningRateSchedulerFactory.Create(LearningRateSchedulerType.OneCycle, 0.1, 100));
        }

        #endregion

        #region State Serialization Tests

        [Fact(Timeout = 60000)]
        public async Task StepLR_GetState_ContainsRequiredKeys()
        {
            var scheduler = new StepLRScheduler(0.1, 10, 0.5);
            scheduler.Step();
            scheduler.Step();

            var state = scheduler.GetState();

            Assert.True(state.ContainsKey("current_step"));
            Assert.True(state.ContainsKey("current_lr"));
            Assert.True(state.ContainsKey("base_lr"));
        }

        [Fact(Timeout = 60000)]
        public async Task StepLR_LoadState_RestoresState()
        {
            var scheduler1 = new StepLRScheduler(0.1, 5, 0.5);

            // Build some state
            for (int i = 0; i < 10; i++) scheduler1.Step();

            var state = scheduler1.GetState();

            // Create new scheduler and load state
            var scheduler2 = new StepLRScheduler(0.1, 5, 0.5);
            scheduler2.LoadState(state);

            Assert.Equal(scheduler1.CurrentStep, scheduler2.CurrentStep);
            Assert.Equal(scheduler1.CurrentLearningRate, scheduler2.CurrentLearningRate);
        }

        #endregion

        #region SequentialLR Tests

        [Fact(Timeout = 60000)]
        public async Task SequentialLR_SwitchesSchedulersAtMilestones()
        {
            var schedulers = new List<ILearningRateScheduler>
            {
                new LinearWarmupScheduler(0.1, warmupSteps: 5, totalSteps: 5, warmupInitLr: 0.01),
                new CosineAnnealingLRScheduler(0.1, tMax: 10, etaMin: 0.01)
            };

            var sequential = new SequentialLRScheduler(schedulers, new[] { 5 });

            // During first scheduler
            Assert.Equal(0, sequential.CurrentSchedulerIndex);

            // Warmup phase
            for (int i = 0; i < 5; i++) sequential.Step();
            Assert.Equal(0, sequential.CurrentSchedulerIndex);

            // After milestone, should switch
            sequential.Step();
            Assert.Equal(1, sequential.CurrentSchedulerIndex);
        }

        #endregion

        #region NoamSchedule Tests

        // Locks down the t = step + 1 mapping the schedule uses to align the
        // library's "Step at end of batch" convention (currentStep is
        // 0-based "batches completed") with the Vaswani 2017 paper's
        // 1-based t. PR #1270 review-comments o_xq + o_yU + o_yf flagged
        // that the previous implementation gave the same LR for the first
        // two batches, lagged the formula by one step throughout, and
        // jumped to the peak LR on Reset() instead of restoring the
        // warmup-start.

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_InitialLR_IsWarmupStart_NotPeak()
        {
            // d_model=512, warmup=4000, factor=1 — paper-canonical Vaswani recipe.
            var scheduler = new NoamSchedule(modelDimension: 512, warmupSteps: 4000);

            // Peak LR (at t=warmup) — what `_baseLearningRate` is set to.
            double peak = scheduler.BaseLearningRate;
            // Warmup-start (at t=1): formula gives factor * d_model^-0.5 * 1 * warmup^-1.5.
            double expectedStart = Math.Pow(512, -0.5) * 1.0 * Math.Pow(4000, -1.5);

            // Initial LR (before any Step()) MUST be the tiny warmup-start,
            // not the peak. This was the regression where ctor pre-set the
            // peak LR via the base ctor and the override only happened
            // after step=1 — meaning the very first Train() call before
            // OnBatchEnd ticks would use peak LR.
            Assert.Equal(expectedStart, scheduler.CurrentLearningRate, 12);
            Assert.True(scheduler.CurrentLearningRate < peak / 100,
                "Initial LR should be much smaller than peak (warmup-start, not peak).");
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_FirstTwoSteps_GiveDistinctLRs()
        {
            // Original bug (review-comment o_xq): the ctor pre-set
            // ComputeLearningRate(1) AND the first Step() also computed
            // ComputeLearningRate(1) — so batches 1 and 2 both saw lr(t=1).
            var scheduler = new NoamSchedule(modelDimension: 512, warmupSteps: 4000);

            double initial = scheduler.CurrentLearningRate;  // batch 1 reads this (lr(t=1))
            double afterStep1 = scheduler.Step();             // becomes lr(t=2), batch 2 reads
            double afterStep2 = scheduler.Step();             // becomes lr(t=3), batch 3 reads

            Assert.NotEqual(initial, afterStep1);
            Assert.NotEqual(afterStep1, afterStep2);
            // During warmup, lr is monotonically increasing.
            Assert.True(afterStep1 > initial);
            Assert.True(afterStep2 > afterStep1);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_StepsToWarmupBoundary_HitsPeak()
        {
            int warmup = 100;
            var scheduler = new NoamSchedule(modelDimension: 512, warmupSteps: warmup);

            // After (warmup - 1) Step() calls, currentStep = warmup - 1, t = warmup → peak.
            for (int i = 0; i < warmup - 1; i++)
            {
                scheduler.Step();
            }

            double peak = scheduler.BaseLearningRate;
            Assert.Equal(peak, scheduler.CurrentLearningRate, 10);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_PostWarmup_DecaysAsInverseSqrt()
        {
            int warmup = 100;
            var scheduler = new NoamSchedule(modelDimension: 512, warmupSteps: warmup);

            // Step well past the warmup so we're in the decay phase.
            for (int i = 0; i < 400; i++)
            {
                scheduler.Step();
            }
            double lrAt400 = scheduler.CurrentLearningRate;

            for (int i = 0; i < 400; i++)
            {
                scheduler.Step();
            }
            double lrAt800 = scheduler.CurrentLearningRate;

            // Decay phase: lr(t) ∝ t^-0.5 ⇒ lr(t=801)/lr(t=401) = sqrt(401/801) ≈ 0.7077.
            // (currentStep = 400 → t = 401; currentStep = 800 → t = 801.)
            double ratio = lrAt800 / lrAt400;
            double expectedRatio = Math.Sqrt(401.0 / 801.0);
            Assert.Equal(expectedRatio, ratio, 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_Reset_RestoresWarmupStart_NotPeak()
        {
            var scheduler = new NoamSchedule(modelDimension: 512, warmupSteps: 4000);
            double initialLr = scheduler.CurrentLearningRate;

            // Advance past warmup so currentLR is now well past the warmup-start.
            for (int i = 0; i < 5000; i++)
            {
                scheduler.Step();
            }
            Assert.NotEqual(initialLr, scheduler.CurrentLearningRate);

            scheduler.Reset();

            // Reset MUST restore warmup-start LR, not the peak (which is
            // what `_baseLearningRate` is set to). Without the Reset
            // override (review-comment o_yf), the base.Reset() would snap
            // currentLR to peak and the next training run would skip warmup.
            Assert.Equal(0, scheduler.CurrentStep);
            Assert.Equal(initialLr, scheduler.CurrentLearningRate, 12);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task NoamSchedule_ZeroOrNegativeWarmup_Throws()
        {
            Assert.Throws<ArgumentException>(() => new NoamSchedule(modelDimension: 512, warmupSteps: 0));
            Assert.Throws<ArgumentException>(() => new NoamSchedule(modelDimension: 512, warmupSteps: -1));
            Assert.Throws<ArgumentException>(() => new NoamSchedule(modelDimension: 0, warmupSteps: 4000));
            Assert.Throws<ArgumentException>(() => new NoamSchedule(modelDimension: 512, warmupSteps: 4000, factor: 0));
            await Task.CompletedTask;
        }

        #endregion
    }
}
