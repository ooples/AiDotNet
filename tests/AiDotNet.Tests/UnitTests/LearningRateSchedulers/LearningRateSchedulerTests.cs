using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNetTests.UnitTests.LearningRateSchedulers
{
    public class LearningRateSchedulerTests
    {
        #region StepLR Tests

        [Fact]
        public void StepLR_InitializesWithCorrectLearningRate()
        {
            var scheduler = new StepLRScheduler(0.1, stepSize: 10, gamma: 0.5);
            Assert.Equal(0.1, scheduler.CurrentLearningRate);
            Assert.Equal(0.1, scheduler.BaseLearningRate);
        }

        [Fact]
        public void StepLR_DecaysAtStepSize()
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

        [Fact]
        public void StepLR_Reset_RestoresInitialState()
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

        [Fact]
        public void CosineAnnealing_InitializesCorrectly()
        {
            var scheduler = new CosineAnnealingLRScheduler(0.1, tMax: 100, etaMin: 0.001);
            Assert.Equal(0.1, scheduler.CurrentLearningRate);
        }

        [Fact]
        public void CosineAnnealing_DecreasesToMinimum()
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

        [Fact]
        public void CosineAnnealing_FollowsCosineShape()
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

        [Fact]
        public void OneCycle_InitializesCorrectly()
        {
            var scheduler = new OneCycleLRScheduler(0.1, totalSteps: 100);
            Assert.True(scheduler.CurrentLearningRate < 0.1); // Starts low
        }

        [Fact]
        public void OneCycle_ReachesPeakAtPctStart()
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

        [Fact]
        public void OneCycle_DecaysAfterPeak()
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

        [Fact]
        public void LinearWarmup_StartsAtInitialLr()
        {
            var scheduler = new LinearWarmupScheduler(
                baseLearningRate: 0.1,
                warmupSteps: 10,
                totalSteps: 100,
                warmupInitLr: 0.001);

            Assert.Equal(0.001, scheduler.CurrentLearningRate, 6);
        }

        [Fact]
        public void LinearWarmup_ReachesPeakAfterWarmup()
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

        [Fact]
        public void LinearWarmup_DecaysAfterPeak()
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

        [Fact]
        public void ExponentialLR_DecaysExponentially()
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

        [Fact]
        public void ReduceOnPlateau_DoesNotReduceWhenImproving()
        {
            var scheduler = new ReduceOnPlateauScheduler(0.1, factor: 0.5, patience: 3);

            // Improving metrics (decreasing)
            scheduler.Step(1.0);
            scheduler.Step(0.9);
            scheduler.Step(0.8);
            scheduler.Step(0.7);

            Assert.Equal(0.1, scheduler.CurrentLearningRate, 6);
        }

        [Fact]
        public void ReduceOnPlateau_ReducesAfterPatience()
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

        [Fact]
        public void ReduceOnPlateau_RespectsMinLearningRate()
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

        [Fact]
        public void CyclicLR_OscillatesBetweenBounds()
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

        [Fact]
        public void Factory_CreateForCNN_ReturnsStepLR()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForCNN();
            Assert.IsType<StepLRScheduler>(scheduler);
        }

        [Fact]
        public void Factory_CreateForTransformer_ReturnsLinearWarmup()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForTransformer();
            Assert.IsType<LinearWarmupScheduler>(scheduler);
        }

        [Fact]
        public void Factory_CreateForSuperConvergence_ReturnsOneCycle()
        {
            var scheduler = LearningRateSchedulerFactory.CreateForSuperConvergence();
            Assert.IsType<OneCycleLRScheduler>(scheduler);
        }

        [Fact]
        public void Factory_CreateAdaptive_ReturnsReduceOnPlateau()
        {
            var scheduler = LearningRateSchedulerFactory.CreateAdaptive();
            Assert.IsType<ReduceOnPlateauScheduler>(scheduler);
        }

        [Fact]
        public void Factory_Create_ReturnsCorrectType()
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

        [Fact]
        public void StepLR_GetState_ContainsRequiredKeys()
        {
            var scheduler = new StepLRScheduler(0.1, 10, 0.5);
            scheduler.Step();
            scheduler.Step();

            var state = scheduler.GetState();

            Assert.True(state.ContainsKey("current_step"));
            Assert.True(state.ContainsKey("current_lr"));
            Assert.True(state.ContainsKey("base_lr"));
        }

        [Fact]
        public void StepLR_LoadState_RestoresState()
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

        [Fact]
        public void SequentialLR_SwitchesSchedulersAtMilestones()
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
    }
}
