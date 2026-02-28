using System;
using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LearningRateSchedulers;

/// <summary>
/// Deep mathematical correctness tests for learning rate schedulers.
/// Each test verifies exact hand-calculated expected values.
/// </summary>
public class LRSchedulerDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region CosineAnnealing

    [Fact]
    public void CosineAnnealing_AtStep0_ReturnsBaseLR()
    {
        // lr = etaMin + 0.5*(baseLR - etaMin)*(1 + cos(0)) = etaMin + (baseLR - etaMin) = baseLR
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.001);
        double lr = scheduler.GetLearningRateAtStep(0);
        Assert.Equal(0.1, lr, Tolerance);
    }

    [Fact]
    public void CosineAnnealing_AtTMax_ReturnsEtaMin()
    {
        // lr = etaMin + 0.5*(baseLR - etaMin)*(1 + cos(pi)) = etaMin + 0 = etaMin
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.001);
        double lr = scheduler.GetLearningRateAtStep(100);
        Assert.Equal(0.001, lr, Tolerance);
    }

    [Fact]
    public void CosineAnnealing_AtHalfway_HandCalculated()
    {
        // step=50, tMax=100
        // lr = 0.001 + 0.5*(0.1-0.001)*(1 + cos(pi*50/100))
        //    = 0.001 + 0.5*0.099*(1 + cos(pi/2))
        //    = 0.001 + 0.5*0.099*(1 + 0)
        //    = 0.001 + 0.0495 = 0.0505
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.001);
        double lr = scheduler.GetLearningRateAtStep(50);
        double expected = 0.001 + 0.5 * 0.099 * (1.0 + Math.Cos(Math.PI * 50.0 / 100.0));
        Assert.Equal(expected, lr, Tolerance);
    }

    [Fact]
    public void CosineAnnealing_AtQuarter_HandCalculated()
    {
        // step=25, tMax=100
        // cos(pi*25/100) = cos(pi/4) = sqrt(2)/2 ≈ 0.7071
        // lr = 0.001 + 0.5*0.099*(1 + 0.7071) = 0.001 + 0.0495*1.7071 ≈ 0.08550
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.001);
        double lr = scheduler.GetLearningRateAtStep(25);
        double expected = 0.001 + 0.5 * 0.099 * (1.0 + Math.Cos(Math.PI * 25.0 / 100.0));
        Assert.Equal(expected, lr, Tolerance);
    }

    [Fact]
    public void CosineAnnealing_BeyondTMax_ClampsToEtaMin()
    {
        // After tMax, step is clamped → cos(pi) = -1 → lr = etaMin
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.001);
        double lr = scheduler.GetLearningRateAtStep(200);
        Assert.Equal(0.001, lr, Tolerance);
    }

    [Fact]
    public void CosineAnnealing_Monotonic_Decreasing()
    {
        // Cosine annealing should be monotonically decreasing from 0 to tMax
        var scheduler = new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 100, etaMin: 0.0);
        double prev = scheduler.GetLearningRateAtStep(0);
        for (int step = 1; step <= 100; step++)
        {
            double current = scheduler.GetLearningRateAtStep(step);
            Assert.True(current <= prev + 1e-12, $"LR increased at step {step}: {prev} → {current}");
            prev = current;
        }
    }

    [Fact]
    public void CosineAnnealing_InvalidTMax_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: 0));
        Assert.Throws<ArgumentException>(() =>
            new CosineAnnealingLRScheduler(baseLearningRate: 0.1, tMax: -1));
    }

    #endregion

    #region ExponentialLR

    [Fact]
    public void Exponential_HandCalculated()
    {
        // lr = base * gamma^step
        // base=0.1, gamma=0.9
        // step=0: 0.1 * 0.9^0 = 0.1
        // step=1: 0.1 * 0.9^1 = 0.09
        // step=5: 0.1 * 0.9^5 = 0.1 * 0.59049 = 0.059049
        // step=10: 0.1 * 0.9^10 = 0.1 * 0.34868 = 0.034868
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);

        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.09, scheduler.GetLearningRateAtStep(1), Tolerance);
        Assert.Equal(0.1 * Math.Pow(0.9, 5), scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.1 * Math.Pow(0.9, 10), scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    [Fact]
    public void Exponential_StepMethod_AdvancesCorrectly()
    {
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.5);

        // Initial state
        Assert.Equal(0, scheduler.CurrentStep);
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);

        // Step 1: 0.1 * 0.5^1 = 0.05
        double lr1 = scheduler.Step();
        Assert.Equal(0.05, lr1, Tolerance);
        Assert.Equal(1, scheduler.CurrentStep);

        // Step 2: 0.1 * 0.5^2 = 0.025
        double lr2 = scheduler.Step();
        Assert.Equal(0.025, lr2, Tolerance);
    }

    [Fact]
    public void Exponential_WithMinLR_Floors()
    {
        // gamma=0.1, base=0.1, min=0.01
        // step=0: 0.1
        // step=1: 0.01
        // step=2: 0.001 → clamped to 0.01
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.1, minLearningRate: 0.01);
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(1), Tolerance);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(2), Tolerance); // floored
    }

    [Fact]
    public void Exponential_InvalidGamma_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ExponentialLRScheduler(0.1, gamma: 0.0));
        Assert.Throws<ArgumentException>(() => new ExponentialLRScheduler(0.1, gamma: -0.5));
        Assert.Throws<ArgumentException>(() => new ExponentialLRScheduler(0.1, gamma: 1.5));
    }

    #endregion

    #region StepLR

    [Fact]
    public void StepLR_HandCalculated()
    {
        // base=0.1, stepSize=3, gamma=0.5
        // step 0: floor(0/3)=0, 0.1*0.5^0 = 0.1
        // step 1: floor(1/3)=0, 0.1*0.5^0 = 0.1
        // step 2: floor(2/3)=0, 0.1*0.5^0 = 0.1
        // step 3: floor(3/3)=1, 0.1*0.5^1 = 0.05
        // step 4: floor(4/3)=1, 0.1*0.5^1 = 0.05
        // step 5: floor(5/3)=1, 0.1*0.5^1 = 0.05
        // step 6: floor(6/3)=2, 0.1*0.5^2 = 0.025
        var scheduler = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 3, gamma: 0.5);

        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(2), Tolerance);
        Assert.Equal(0.05, scheduler.GetLearningRateAtStep(3), Tolerance);
        Assert.Equal(0.05, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.025, scheduler.GetLearningRateAtStep(6), Tolerance);
    }

    [Fact]
    public void StepLR_StepMethod_DecaysAtCorrectInterval()
    {
        var scheduler = new StepLRScheduler(baseLearningRate: 1.0, stepSize: 2, gamma: 0.1);

        // Steps 0: base=1.0
        Assert.Equal(1.0, scheduler.CurrentLearningRate, Tolerance);

        // Step to 1: floor(1/2)=0 → 1.0
        scheduler.Step();
        Assert.Equal(1.0, scheduler.CurrentLearningRate, Tolerance);

        // Step to 2: floor(2/2)=1 → 0.1
        scheduler.Step();
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);

        // Step to 3: floor(3/2)=1 → 0.1
        scheduler.Step();
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);

        // Step to 4: floor(4/2)=2 → 0.01
        scheduler.Step();
        Assert.Equal(0.01, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void StepLR_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new StepLRScheduler(0.1, stepSize: 0));
        Assert.Throws<ArgumentException>(() => new StepLRScheduler(0.1, stepSize: -1));
        Assert.Throws<ArgumentException>(() => new StepLRScheduler(0.1, stepSize: 10, gamma: 0.0));
    }

    #endregion

    #region PolynomialLR

    [Fact]
    public void Polynomial_LinearDecay_HandCalculated()
    {
        // power=1 (linear), base=0.1, end=0.01, totalSteps=10
        // lr = (0.1-0.01) * (1 - step/10)^1 + 0.01
        // step 0: 0.09 * 1.0 + 0.01 = 0.1
        // step 5: 0.09 * 0.5 + 0.01 = 0.055
        // step 10: end_lr = 0.01
        var scheduler = new PolynomialLRScheduler(baseLearningRate: 0.1, totalSteps: 10, power: 1.0, endLearningRate: 0.01);

        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.055, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    [Fact]
    public void Polynomial_QuadraticDecay_HandCalculated()
    {
        // power=2, base=0.1, end=0.0, totalSteps=10
        // lr = 0.1 * (1 - step/10)^2
        // step 0: 0.1 * 1.0 = 0.1
        // step 5: 0.1 * (0.5)^2 = 0.1 * 0.25 = 0.025
        // step 9: 0.1 * (0.1)^2 = 0.1 * 0.01 = 0.001
        var scheduler = new PolynomialLRScheduler(baseLearningRate: 0.1, totalSteps: 10, power: 2.0, endLearningRate: 0.0);

        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.025, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.001, scheduler.GetLearningRateAtStep(9), Tolerance);
        Assert.Equal(0.0, scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    [Fact]
    public void Polynomial_BeyondTotalSteps_ReturnsEndLR()
    {
        var scheduler = new PolynomialLRScheduler(baseLearningRate: 0.1, totalSteps: 10, power: 1.0, endLearningRate: 0.01);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(20), Tolerance);
    }

    [Fact]
    public void Polynomial_LinearDecay_Monotonic()
    {
        var scheduler = new PolynomialLRScheduler(baseLearningRate: 0.1, totalSteps: 100, power: 1.0, endLearningRate: 0.0);
        double prev = scheduler.GetLearningRateAtStep(0);
        for (int step = 1; step <= 100; step++)
        {
            double current = scheduler.GetLearningRateAtStep(step);
            Assert.True(current <= prev + 1e-12, $"LR increased at step {step}: {prev} → {current}");
            prev = current;
        }
    }

    [Fact]
    public void Polynomial_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new PolynomialLRScheduler(0.1, totalSteps: 0));
        Assert.Throws<ArgumentException>(() =>
            new PolynomialLRScheduler(0.1, totalSteps: 10, power: 0.0));
        Assert.Throws<ArgumentException>(() =>
            new PolynomialLRScheduler(0.1, totalSteps: 10, power: -1.0));
    }

    #endregion

    #region LinearWarmup

    [Fact]
    public void LinearWarmup_WarmupPhase_HandCalculated()
    {
        // warmupSteps=10, base=0.1, warmupInitLr=0.0
        // At step k (during warmup): lr = 0 + (0.1 - 0) * k/10 = 0.01 * k
        var scheduler = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 10, totalSteps: 100,
            warmupInitLr: 0.0, decayMode: LinearWarmupScheduler.DecayMode.Constant);

        Assert.Equal(0.0, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(1), Tolerance);
        Assert.Equal(0.05, scheduler.GetLearningRateAtStep(5), Tolerance);
        // At step 10 (warmup boundary), we are IN the constant phase now
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    [Fact]
    public void LinearWarmup_ConstantAfterWarmup()
    {
        var scheduler = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 5, totalSteps: 100,
            decayMode: LinearWarmupScheduler.DecayMode.Constant);

        // After warmup, should stay at baseLR
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(50), Tolerance);
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(100), Tolerance);
    }

    [Fact]
    public void LinearWarmup_LinearDecay_HandCalculated()
    {
        // warmup: 10 steps (0→0.1), then linear decay over 90 steps (0.1→0.01)
        var scheduler = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 10, totalSteps: 100,
            warmupInitLr: 0.0, decayMode: LinearWarmupScheduler.DecayMode.Linear, endLr: 0.01);

        // At step 10: base LR = 0.1
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(10), Tolerance);

        // At step 55: halfway through decay (decayStep=45, decaySteps=90)
        // progress = 45/90 = 0.5
        // lr = 0.1 - (0.1-0.01)*0.5 = 0.1 - 0.045 = 0.055
        Assert.Equal(0.055, scheduler.GetLearningRateAtStep(55), Tolerance);

        // At step 100: end LR
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(100), Tolerance);
    }

    [Fact]
    public void LinearWarmup_CosineDecay_BoundaryValues()
    {
        var scheduler = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 10, totalSteps: 110,
            warmupInitLr: 0.0, decayMode: LinearWarmupScheduler.DecayMode.Cosine, endLr: 0.0);

        // At warmup end (step 10): base LR
        Assert.Equal(0.1, scheduler.GetLearningRateAtStep(10), Tolerance);

        // At total steps: end LR
        Assert.Equal(0.0, scheduler.GetLearningRateAtStep(110), Tolerance);

        // Halfway through decay (step 60, decayStep=50, decaySteps=100):
        // cosine = (1 + cos(pi*0.5)) / 2 = (1+0)/2 = 0.5
        // lr = 0 + (0.1 - 0) * 0.5 = 0.05
        Assert.Equal(0.05, scheduler.GetLearningRateAtStep(60), Tolerance);
    }

    [Fact]
    public void LinearWarmup_CustomInitLR_HandCalculated()
    {
        // Start from 0.01, warm up to 0.1 in 10 steps
        // lr = 0.01 + (0.1 - 0.01) * step/10 = 0.01 + 0.009 * step
        var scheduler = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 10, totalSteps: 100,
            warmupInitLr: 0.01, decayMode: LinearWarmupScheduler.DecayMode.Constant);

        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.055, scheduler.GetLearningRateAtStep(5), Tolerance);
    }

    [Fact]
    public void LinearWarmup_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearWarmupScheduler(0.1, warmupSteps: -1));
    }

    #endregion

    #region CyclicLR

    [Fact]
    public void CyclicLR_Triangular_HandCalculated()
    {
        // base=0.001, max=0.01, stepSizeUp=5, stepSizeDown=5
        // Cycle length = 10
        // Amplitude = 0.009
        // step 0: scale=0/5=0 → lr = 0.001 + 0.009*0 = 0.001
        // step 3: scale=3/5=0.6 → lr = 0.001 + 0.009*0.6 = 0.0064
        // step 5: scale=1-0/5=1.0 → lr = 0.001 + 0.009*1.0 = 0.01
        // step 8: scale=1-3/5=0.4 → lr = 0.001 + 0.009*0.4 = 0.0046
        // step 10: new cycle, step 0 again
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.01,
            stepSizeUp: 5, stepSizeDown: 5,
            mode: CyclicLRScheduler.CyclicMode.Triangular);

        Assert.Equal(0.001, scheduler.GetLearningRateAtStep(0), Tolerance);
        Assert.Equal(0.0064, scheduler.GetLearningRateAtStep(3), Tolerance);
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(5), Tolerance);
        Assert.Equal(0.0046, scheduler.GetLearningRateAtStep(8), Tolerance);
        // Next cycle starts at step 10
        Assert.Equal(0.001, scheduler.GetLearningRateAtStep(10), Tolerance);
    }

    [Fact]
    public void CyclicLR_Triangular2_AmplitudeHalves()
    {
        // base=0.001, max=0.01, stepSizeUp=5
        // Cycle 0: amplitude=0.009 → peak = 0.01
        // Cycle 1: amplitude=0.009/2=0.0045 → peak = 0.0055
        // Cycle 2: amplitude=0.009/4=0.00225 → peak = 0.00325
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.01,
            stepSizeUp: 5, stepSizeDown: 5,
            mode: CyclicLRScheduler.CyclicMode.Triangular2);

        // Peak of cycle 0 (step 5)
        double peak0 = scheduler.GetLearningRateAtStep(5);
        Assert.Equal(0.01, peak0, Tolerance);

        // Peak of cycle 1 (step 15)
        double peak1 = scheduler.GetLearningRateAtStep(15);
        Assert.Equal(0.001 + 0.009 / 2.0, peak1, Tolerance);

        // Peak of cycle 2 (step 25)
        double peak2 = scheduler.GetLearningRateAtStep(25);
        Assert.Equal(0.001 + 0.009 / 4.0, peak2, Tolerance);
    }

    [Fact]
    public void CyclicLR_Periodic_RepeatsExactly()
    {
        // Triangular mode should repeat exactly
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.01,
            stepSizeUp: 5, stepSizeDown: 5,
            mode: CyclicLRScheduler.CyclicMode.Triangular);

        for (int step = 0; step < 10; step++)
        {
            double lr_cycle0 = scheduler.GetLearningRateAtStep(step);
            double lr_cycle1 = scheduler.GetLearningRateAtStep(step + 10);
            Assert.Equal(lr_cycle0, lr_cycle1, Tolerance);
        }
    }

    [Fact]
    public void CyclicLR_AlwaysBounded()
    {
        // LR should always be between base and max
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 0.01,
            stepSizeUp: 7, stepSizeDown: 3,
            mode: CyclicLRScheduler.CyclicMode.Triangular);

        for (int step = 0; step < 100; step++)
        {
            double lr = scheduler.GetLearningRateAtStep(step);
            Assert.True(lr >= 0.001 - 1e-10, $"LR below base at step {step}: {lr}");
            Assert.True(lr <= 0.01 + 1e-10, $"LR above max at step {step}: {lr}");
        }
    }

    [Fact]
    public void CyclicLR_AsymmetricSteps_HandCalculated()
    {
        // stepSizeUp=3, stepSizeDown=7, cycle length=10
        // base=0.0, max=1.0
        // Ascending (steps 0-2): scale = step/3
        //   step 0: 0.0, step 1: 1/3, step 2: 2/3
        // At peak (step 3): entering descending, scale = 1 - 0/7 = 1.0
        // Descending (steps 3-9): scale = 1 - (step-3)/7
        //   step 3: 1.0, step 5: 1-2/7=5/7, step 9: 1-6/7=1/7
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001, maxLearningRate: 1.0,
            stepSizeUp: 3, stepSizeDown: 7,
            mode: CyclicLRScheduler.CyclicMode.Triangular);

        // Ascending
        Assert.Equal(0.001, scheduler.GetLearningRateAtStep(0), 1e-6);
        // step 1: scale = 1/3, lr = 0.001 + 0.999 * (1/3) = 0.334
        double lr1 = scheduler.GetLearningRateAtStep(1);
        Assert.Equal(0.001 + 0.999 / 3.0, lr1, 1e-6);

        // Peak at step 3: scale=1-0/7=1.0, lr = 0.001 + 0.999 = 1.0
        Assert.Equal(1.0, scheduler.GetLearningRateAtStep(3), 1e-6);
    }

    [Fact]
    public void CyclicLR_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new CyclicLRScheduler(0.01, 0.001)); // max < base
        Assert.Throws<ArgumentException>(() =>
            new CyclicLRScheduler(0.001, 0.01, stepSizeUp: 0));
    }

    #endregion

    #region Constant LR

    [Fact]
    public void ConstantLR_NeverChanges()
    {
        var scheduler = new ConstantLRScheduler(baseLearningRate: 0.05);

        for (int step = 0; step < 100; step++)
        {
            Assert.Equal(0.05, scheduler.GetLearningRateAtStep(step), Tolerance);
        }

        // Step 10 times, should still be 0.05
        for (int i = 0; i < 10; i++) scheduler.Step();
        Assert.Equal(0.05, scheduler.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region State Management

    [Fact]
    public void Reset_RestoresInitialState()
    {
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.5);
        scheduler.Step(); // lr = 0.05
        scheduler.Step(); // lr = 0.025

        Assert.Equal(2, scheduler.CurrentStep);
        Assert.Equal(0.025, scheduler.CurrentLearningRate, Tolerance);

        scheduler.Reset();
        Assert.Equal(0, scheduler.CurrentStep);
        Assert.Equal(0.1, scheduler.CurrentLearningRate, Tolerance);
    }

    [Fact]
    public void GetState_ContainsExpectedKeys()
    {
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        scheduler.Step();

        var state = scheduler.GetState();
        Assert.True(state.ContainsKey("base_lr"));
        Assert.True(state.ContainsKey("current_lr"));
        Assert.True(state.ContainsKey("current_step"));
        Assert.True(state.ContainsKey("gamma"));

        Assert.Equal(0.1, Convert.ToDouble(state["base_lr"]), Tolerance);
        Assert.Equal(1, Convert.ToInt32(state["current_step"]));
    }

    [Fact]
    public void LoadState_RestoresCorrectState()
    {
        var scheduler1 = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        scheduler1.Step();
        scheduler1.Step();
        scheduler1.Step();
        var state = scheduler1.GetState();

        var scheduler2 = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        scheduler2.LoadState(state);

        Assert.Equal(scheduler1.CurrentStep, scheduler2.CurrentStep);
        Assert.Equal(scheduler1.CurrentLearningRate, scheduler2.CurrentLearningRate, Tolerance);
    }

    #endregion

    #region Base Class Validation

    [Fact]
    public void BaseLR_ZeroOrNegative_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ExponentialLRScheduler(baseLearningRate: 0.0));
        Assert.Throws<ArgumentException>(() => new ExponentialLRScheduler(baseLearningRate: -0.1));
    }

    [Fact]
    public void MinLR_Negative_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9, minLearningRate: -0.01));
    }

    [Fact]
    public void GetLearningRateAtStep_NegativeStep_Throws()
    {
        var scheduler = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        Assert.Throws<ArgumentException>(() => scheduler.GetLearningRateAtStep(-1));
    }

    #endregion

    #region Cross-Scheduler Consistency

    [Fact]
    public void Exponential_Vs_StepLR_AtDecayPoints()
    {
        // StepLR with stepSize=1 and gamma=0.9 should match Exponential with gamma=0.9
        // because floor(step/1) = step, so lr = base * 0.9^step
        var exponential = new ExponentialLRScheduler(baseLearningRate: 0.1, gamma: 0.9);
        var stepLR = new StepLRScheduler(baseLearningRate: 0.1, stepSize: 1, gamma: 0.9);

        for (int step = 0; step < 20; step++)
        {
            Assert.Equal(
                exponential.GetLearningRateAtStep(step),
                stepLR.GetLearningRateAtStep(step),
                1e-10);
        }
    }

    [Fact]
    public void Polynomial_Power1_Vs_LinearWarmup_LinearDecay()
    {
        // Polynomial with power=1 starting from step 0 should match
        // LinearWarmup with 0 warmup steps and linear decay
        var poly = new PolynomialLRScheduler(
            baseLearningRate: 0.1, totalSteps: 100, power: 1.0, endLearningRate: 0.01);
        var warmup = new LinearWarmupScheduler(
            baseLearningRate: 0.1, warmupSteps: 0, totalSteps: 100,
            warmupInitLr: 0.1, decayMode: LinearWarmupScheduler.DecayMode.Linear, endLr: 0.01);

        for (int step = 0; step <= 100; step++)
        {
            Assert.Equal(
                poly.GetLearningRateAtStep(step),
                warmup.GetLearningRateAtStep(step),
                1e-8);
        }
    }

    #endregion
}
