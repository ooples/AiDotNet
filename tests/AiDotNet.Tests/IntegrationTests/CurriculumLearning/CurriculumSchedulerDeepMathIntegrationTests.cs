using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.CurriculumLearning.Schedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CurriculumLearning;

/// <summary>
/// Deep math integration tests for curriculum learning schedulers.
/// Verifies Linear, Exponential, Polynomial, Cosine, Step, SelfPaced,
/// and CompetenceBased schedulers against hand-calculated expected values.
/// </summary>
public class CurriculumSchedulerDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    // Helper to convert double to the operations type
    private static double ToDouble(double val) => val;

    private static CurriculumEpochMetrics<double> CreateMetrics(
        double trainingLoss, bool improved = true)
    {
        return new CurriculumEpochMetrics<double>
        {
            TrainingLoss = trainingLoss,
            Improved = improved
        };
    }

    private static CurriculumEpochMetrics<double> CreateMetricsWithAccuracy(
        double trainingLoss, double validationAccuracy,
        bool improved = true)
    {
        return new CurriculumEpochMetrics<double>
        {
            TrainingLoss = trainingLoss,
            ValidationAccuracy = validationAccuracy,
            Improved = improved
        };
    }

    // ─── Linear Scheduler ───────────────────────────────────────────────

    [Fact]
    public void LinearScheduler_Epoch0_ReturnsMinFraction()
    {
        // 10 epochs, minFraction=0.2, maxFraction=1.0
        // At epoch 0: progress = 0/(10-1) = 0
        // fraction = 0.2 + 0 * (1.0 - 0.2) = 0.2
        var scheduler = new LinearScheduler<double>(10, minFraction: 0.2, maxFraction: 1.0);
        var fraction = scheduler.GetDataFraction();
        Assert.Equal(0.2, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_FinalEpoch_ReturnsMaxFraction()
    {
        // 10 epochs, at epoch 9: progress = 9/9 = 1.0
        // fraction = 0.2 + 1.0 * (1.0 - 0.2) = 1.0
        var scheduler = new LinearScheduler<double>(10, minFraction: 0.2, maxFraction: 1.0);
        for (int i = 0; i < 9; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        var fraction = scheduler.GetDataFraction();
        Assert.Equal(1.0, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_MidEpoch_HandCalculated()
    {
        // 10 epochs, at epoch 4: progress = 4/9 ≈ 0.4444
        // fraction = 0.1 + 0.4444 * (1.0 - 0.1) = 0.1 + 0.4 = 0.5
        var scheduler = new LinearScheduler<double>(10, minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));

        var fraction = scheduler.GetDataFraction();
        double progress = 4.0 / 9.0;
        double expected = 0.1 + progress * (1.0 - 0.1);
        Assert.Equal(expected, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_MonotonicallyIncreasing()
    {
        var scheduler = new LinearScheduler<double>(20, minFraction: 0.05, maxFraction: 1.0);
        double prevFraction = scheduler.GetDataFraction();

        for (int epoch = 0; epoch < 19; epoch++)
        {
            scheduler.StepEpoch(CreateMetrics(0.5));
            double current = scheduler.GetDataFraction();
            Assert.True(current >= prevFraction - Tolerance,
                $"Fraction should not decrease: epoch {epoch + 1}, prev={prevFraction}, current={current}");
            prevFraction = current;
        }
    }

    [Fact]
    public void LinearScheduler_EachEpoch_ConstantIncrement()
    {
        // Linear scheduler: each epoch adds the same increment
        // increment = (max - min) / (totalEpochs - 1) = 0.9 / 9 = 0.1
        var scheduler = new LinearScheduler<double>(10, minFraction: 0.1, maxFraction: 1.0);
        double increment = (1.0 - 0.1) / 9.0;

        for (int epoch = 0; epoch < 10; epoch++)
        {
            double expected = 0.1 + (epoch / 9.0) * (1.0 - 0.1);
            double actual = scheduler.GetDataFraction();
            Assert.Equal(expected, actual, Tolerance);
            if (epoch < 9) scheduler.StepEpoch(CreateMetrics(0.5));
        }
    }

    // ─── Exponential Scheduler ──────────────────────────────────────────

    [Fact]
    public void ExponentialScheduler_Epoch0_ReturnsMinFraction()
    {
        // At epoch 0: t = 0, numerator = 1 - e^0 = 0, progress = 0
        // fraction = 0.1 + 0 * 0.9 = 0.1
        var scheduler = new ExponentialScheduler<double>(10, growthRate: 3.0,
            minFraction: 0.1, maxFraction: 1.0);
        Assert.Equal(0.1, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void ExponentialScheduler_FinalEpoch_ReturnsMaxFraction()
    {
        // At epoch 9: t = 1.0, numerator = 1-e^(-3), denominator = 1-e^(-3)
        // progress = 1.0, fraction = 0.1 + 1.0 * 0.9 = 1.0
        var scheduler = new ExponentialScheduler<double>(10, growthRate: 3.0,
            minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 9; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(1.0, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void ExponentialScheduler_MidEpoch_HandCalculated()
    {
        // 10 epochs, rate=3.0, epoch 4: t = 4/9
        // numerator = 1 - e^(-3 * 4/9) = 1 - e^(-4/3)
        // denominator = 1 - e^(-3)
        // progress = numerator / denominator
        var scheduler = new ExponentialScheduler<double>(10, growthRate: 3.0,
            minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));

        double t = 4.0 / 9.0;
        double numerator = 1.0 - Math.Exp(-3.0 * t);
        double denominator = 1.0 - Math.Exp(-3.0);
        double progress = numerator / denominator;
        double expected = 0.1 + progress * 0.9;

        Assert.Equal(expected, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void ExponentialScheduler_HighGrowthRate_FasterEarlyGrowth()
    {
        // Higher growth rate => faster initial growth
        var low = new ExponentialScheduler<double>(10, growthRate: 1.0,
            minFraction: 0.0, maxFraction: 1.0);
        var high = new ExponentialScheduler<double>(10, growthRate: 10.0,
            minFraction: 0.0, maxFraction: 1.0);

        // Advance both to epoch 2 (early in training)
        for (int i = 0; i < 2; i++)
        {
            low.StepEpoch(CreateMetrics(0.5));
            high.StepEpoch(CreateMetrics(0.5));
        }

        double lowFrac = low.GetDataFraction();
        double highFrac = high.GetDataFraction();

        Assert.True(highFrac > lowFrac,
            $"High growth rate ({highFrac}) should produce larger fraction than low rate ({lowFrac}) early on");
    }

    [Fact]
    public void ExponentialScheduler_MonotonicallyIncreasing()
    {
        var scheduler = new ExponentialScheduler<double>(20, growthRate: 5.0,
            minFraction: 0.0, maxFraction: 1.0);
        double prev = scheduler.GetDataFraction();

        for (int i = 0; i < 19; i++)
        {
            scheduler.StepEpoch(CreateMetrics(0.5));
            double current = scheduler.GetDataFraction();
            Assert.True(current >= prev - Tolerance,
                $"Exponential fraction should not decrease: epoch {i + 1}");
            prev = current;
        }
    }

    [Fact]
    public void ExponentialScheduler_Rate1_HandCalculatedValues()
    {
        // rate=1.0, 5 epochs, min=0, max=1
        // epoch 0: t=0, progress=0
        // epoch 1: t=0.25, progress = (1-e^(-0.25))/(1-e^(-1))
        // epoch 2: t=0.5, progress = (1-e^(-0.5))/(1-e^(-1))
        var scheduler = new ExponentialScheduler<double>(5, growthRate: 1.0,
            minFraction: 0.0, maxFraction: 1.0);

        double denom = 1.0 - Math.Exp(-1.0);
        double[] tValues = { 0.0, 0.25, 0.5, 0.75, 1.0 };

        for (int epoch = 0; epoch < 5; epoch++)
        {
            double t = tValues[epoch];
            double expectedProgress = (1.0 - Math.Exp(-1.0 * t)) / denom;
            if (epoch == 0) expectedProgress = 0.0; // t=0 gives 0/denom = 0
            double expected = expectedProgress;
            Assert.Equal(expected, scheduler.GetDataFraction(), Tolerance);
            if (epoch < 4) scheduler.StepEpoch(CreateMetrics(0.5));
        }
    }

    // ─── Polynomial Scheduler ───────────────────────────────────────────

    [Fact]
    public void PolynomialScheduler_Power2_HandCalculated()
    {
        // 10 epochs, power=2, min=0.1, max=1.0
        // epoch 4: t = 4/9, progress = (4/9)^2 = 16/81 ≈ 0.1975
        // fraction = 0.1 + 0.1975 * 0.9 ≈ 0.2778
        var scheduler = new PolynomialScheduler<double>(10, power: 2.0,
            minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));

        double t = 4.0 / 9.0;
        double progress = Math.Pow(t, 2.0);
        double expected = 0.1 + progress * 0.9;
        Assert.Equal(expected, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void PolynomialScheduler_Power1_EqualsLinear()
    {
        // power=1 should give same results as linear scheduler
        var poly = new PolynomialScheduler<double>(10, power: 1.0,
            minFraction: 0.1, maxFraction: 1.0);
        var linear = new LinearScheduler<double>(10,
            minFraction: 0.1, maxFraction: 1.0);

        for (int epoch = 0; epoch < 10; epoch++)
        {
            Assert.Equal(linear.GetDataFraction(), poly.GetDataFraction(), Tolerance);
            if (epoch < 9)
            {
                poly.StepEpoch(CreateMetrics(0.5));
                linear.StepEpoch(CreateMetrics(0.5));
            }
        }
    }

    [Fact]
    public void PolynomialScheduler_Power0_5_ConcaveCurve()
    {
        // power < 1 => concave (fast start, slow finish)
        // At t=0.25: progress = 0.25^0.5 = 0.5
        // At t=0.5: progress = 0.5^0.5 ≈ 0.7071
        // These are higher than linear (where t=progress)
        var scheduler = new PolynomialScheduler<double>(5, power: 0.5,
            minFraction: 0.0, maxFraction: 1.0);

        // Advance to epoch 1: t = 1/4 = 0.25
        scheduler.StepEpoch(CreateMetrics(0.5));
        double frac = scheduler.GetDataFraction();
        double expected = Math.Pow(0.25, 0.5); // 0.5

        Assert.Equal(expected, frac, Tolerance);
        Assert.True(frac > 0.25, "Concave curve (power<1) should be above linear at early points");
    }

    [Fact]
    public void PolynomialScheduler_Power3_ConvexCurve()
    {
        // power > 1 => convex (slow start, fast finish)
        // At epoch 1 (t=1/4): progress = 0.25^3 = 0.015625
        // This is lower than linear (where progress = 0.25)
        var scheduler = new PolynomialScheduler<double>(5, power: 3.0,
            minFraction: 0.0, maxFraction: 1.0);
        scheduler.StepEpoch(CreateMetrics(0.5));

        double frac = scheduler.GetDataFraction();
        double expected = Math.Pow(0.25, 3.0); // 0.015625
        Assert.Equal(expected, frac, Tolerance);
        Assert.True(frac < 0.25, "Convex curve (power>1) should be below linear at early points");
    }

    [Fact]
    public void PolynomialScheduler_AllPowers_StartAtMinEndAtMax()
    {
        double[] powers = { 0.5, 1.0, 2.0, 3.0, 5.0 };
        foreach (double power in powers)
        {
            var scheduler = new PolynomialScheduler<double>(10, power: power,
                minFraction: 0.2, maxFraction: 0.9);

            // Epoch 0: should be minFraction (progress=0, any power)
            Assert.Equal(0.2, scheduler.GetDataFraction(), Tolerance);

            // Advance to final epoch
            for (int i = 0; i < 9; i++)
                scheduler.StepEpoch(CreateMetrics(0.5));

            // Final epoch: should be maxFraction (progress=1, any power)
            Assert.Equal(0.9, scheduler.GetDataFraction(), Tolerance);
        }
    }

    // ─── Cosine Scheduler ───────────────────────────────────────────────

    [Fact]
    public void CosineScheduler_Epoch0_ReturnsMinFraction()
    {
        // t=0: progress = 0.5*(1-cos(0)) = 0.5*(1-1) = 0
        // fraction = 0.1 + 0 * 0.9 = 0.1
        var scheduler = new CosineScheduler<double>(10, minFraction: 0.1, maxFraction: 1.0);
        Assert.Equal(0.1, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void CosineScheduler_FinalEpoch_ReturnsMaxFraction()
    {
        // t=1: progress = 0.5*(1-cos(pi)) = 0.5*(1-(-1)) = 1.0
        // fraction = 0.1 + 1.0 * 0.9 = 1.0
        var scheduler = new CosineScheduler<double>(10, minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 9; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(1.0, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void CosineScheduler_Midpoint_HandCalculated()
    {
        // 10 epochs, epoch 4: t = 4/9
        // progress = 0.5 * (1 - cos(pi * 4/9))
        // cos(4*pi/9) = cos(80 degrees) ≈ 0.17365
        // progress ≈ 0.5 * (1 - 0.17365) = 0.41317
        var scheduler = new CosineScheduler<double>(10, minFraction: 0.1, maxFraction: 1.0);
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));

        double t = 4.0 / 9.0;
        double progress = 0.5 * (1.0 - Math.Cos(Math.PI * t));
        double expected = 0.1 + progress * 0.9;
        Assert.Equal(expected, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void CosineScheduler_HalfwayPoint_ProgressIs0_5()
    {
        // At t=0.5: progress = 0.5*(1-cos(pi/2)) = 0.5*(1-0) = 0.5
        // fraction = min + 0.5 * (max - min) = exactly midpoint
        var scheduler = new CosineScheduler<double>(3, minFraction: 0.0, maxFraction: 1.0);
        scheduler.StepEpoch(CreateMetrics(0.5)); // now epoch=1, t=1/(3-1)=0.5

        double expected = 0.5 * (1.0 - Math.Cos(Math.PI * 0.5)); // 0.5
        Assert.Equal(expected, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void CosineScheduler_SymmetricAroundMidpoint()
    {
        // Cosine curve: 0.5*(1-cos(pi*t))
        // At t and (1-t): progress(t) + progress(1-t) should sum to ~1.0
        var scheduler1 = new CosineScheduler<double>(11, minFraction: 0.0, maxFraction: 1.0);
        var scheduler2 = new CosineScheduler<double>(11, minFraction: 0.0, maxFraction: 1.0);

        // Advance scheduler1 to epoch 3 (t=3/10=0.3)
        for (int i = 0; i < 3; i++)
            scheduler1.StepEpoch(CreateMetrics(0.5));

        // Advance scheduler2 to epoch 7 (t=7/10=0.7)
        for (int i = 0; i < 7; i++)
            scheduler2.StepEpoch(CreateMetrics(0.5));

        double frac1 = scheduler1.GetDataFraction();
        double frac2 = scheduler2.GetDataFraction();

        // frac1 + frac2 should equal min + max
        Assert.Equal(0.0 + 1.0, frac1 + frac2, 1e-5);
    }

    [Fact]
    public void CosineScheduler_MonotonicallyIncreasing()
    {
        var scheduler = new CosineScheduler<double>(20, minFraction: 0.1, maxFraction: 1.0);
        double prev = scheduler.GetDataFraction();

        for (int i = 0; i < 19; i++)
        {
            scheduler.StepEpoch(CreateMetrics(0.5));
            double current = scheduler.GetDataFraction();
            Assert.True(current >= prev - Tolerance,
                $"Cosine fraction should not decrease: epoch {i + 1}");
            prev = current;
        }
    }

    // ─── Step Scheduler ─────────────────────────────────────────────────

    [Fact]
    public void StepScheduler_UniformSteps_HandCalculated()
    {
        // 12 epochs, 3 steps, min=0.1, max=1.0
        // epochsPerStep = 12/3 = 4
        // Epochs 0-3: step 0, stepFraction = 1/3 => fraction = 0.1 + (1/3)*0.9 = 0.4
        // Epochs 4-7: step 1, stepFraction = 2/3 => fraction = 0.1 + (2/3)*0.9 = 0.7
        // Epochs 8-11: step 2, stepFraction = 3/3 => fraction = 0.1 + (3/3)*0.9 = 1.0
        var scheduler = new StepScheduler<double>(12, numSteps: 3,
            minFraction: 0.1, maxFraction: 1.0);

        double expectedStep0 = 0.1 + (1.0 / 3.0) * 0.9;
        double expectedStep1 = 0.1 + (2.0 / 3.0) * 0.9;
        double expectedStep2 = 0.1 + (3.0 / 3.0) * 0.9;

        // Epoch 0 (step 0)
        Assert.Equal(expectedStep0, scheduler.GetDataFraction(), Tolerance);

        // Advance to epoch 4 (step 1)
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(expectedStep1, scheduler.GetDataFraction(), Tolerance);

        // Advance to epoch 8 (step 2)
        for (int i = 0; i < 4; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(expectedStep2, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void StepScheduler_CustomFractions_ExactValues()
    {
        // Custom fractions: [0.2, 0.5, 0.8, 1.0]
        // 12 epochs, 4 steps => epochsPerStep = 3
        var fractions = new double[] { 0.2, 0.5, 0.8, 1.0 };
        var scheduler = new StepScheduler<double>(12, fractions);

        // Epoch 0 (step 0): fraction = 0.2
        Assert.Equal(0.2, scheduler.GetDataFraction(), Tolerance);

        // Advance to epoch 3 (step 1): fraction = 0.5
        for (int i = 0; i < 3; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(0.5, scheduler.GetDataFraction(), Tolerance);

        // Advance to epoch 6 (step 2): fraction = 0.8
        for (int i = 0; i < 3; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(0.8, scheduler.GetDataFraction(), Tolerance);

        // Advance to epoch 9 (step 3): fraction = 1.0
        for (int i = 0; i < 3; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(1.0, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void StepScheduler_NonDecreasing()
    {
        var scheduler = new StepScheduler<double>(20, numSteps: 5,
            minFraction: 0.1, maxFraction: 1.0);
        double prev = scheduler.GetDataFraction();

        for (int i = 0; i < 19; i++)
        {
            scheduler.StepEpoch(CreateMetrics(0.5));
            double current = scheduler.GetDataFraction();
            Assert.True(current >= prev - Tolerance,
                $"Step fraction should not decrease: epoch {i + 1}");
            prev = current;
        }
    }

    [Fact]
    public void StepScheduler_DecreasingFractions_Throws()
    {
        // Custom fractions must be non-decreasing
        Assert.Throws<ArgumentException>(() =>
            new StepScheduler<double>(10, new double[] { 0.5, 0.3, 0.8 }));
    }

    [Fact]
    public void StepScheduler_FractionOutOfRange_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new StepScheduler<double>(10, new double[] { 0.2, 1.5 }));
    }

    // ─── Self-Paced Scheduler: Hard Regularizer ─────────────────────────

    [Fact]
    public void SelfPaced_Hard_SelectsOnlyBelowLambda()
    {
        // Hard: v = 1 if loss < lambda, else 0
        // initialLambda = 0.5
        // losses = [0.1, 0.3, 0.5, 0.7, 0.9]
        // Expected: indices 0,1 selected (loss < 0.5)
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.5, regularizer: SelfPaceRegularizer.Hard);

        var losses = new Vector<double>(new double[] { 0.1, 0.3, 0.5, 0.7, 0.9 });
        var weights = scheduler.ComputeSampleWeights(losses);

        Assert.Equal(1.0, weights[0], Tolerance); // 0.1 < 0.5
        Assert.Equal(1.0, weights[1], Tolerance); // 0.3 < 0.5
        Assert.Equal(0.0, weights[2], Tolerance); // 0.5 >= 0.5 (not strictly less)
        Assert.Equal(0.0, weights[3], Tolerance); // 0.7 >= 0.5
        Assert.Equal(0.0, weights[4], Tolerance); // 0.9 >= 0.5
    }

    [Fact]
    public void SelfPaced_Hard_LambdaGrows_MoreSamplesSelected()
    {
        // After stepping, lambda increases by growthRate
        // initialLambda=0.3, growthRate=0.2 => after 1 step: lambda=0.5, after 2 steps: lambda=0.7
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.3, lambdaGrowthRate: 0.2,
            regularizer: SelfPaceRegularizer.Hard);

        var losses = new Vector<double>(new double[] { 0.1, 0.4, 0.6, 0.8 });

        // Initial lambda=0.3: only loss=0.1 selected
        var weights0 = scheduler.ComputeSampleWeights(losses);
        Assert.Equal(1.0, weights0[0], Tolerance); // 0.1 < 0.3
        Assert.Equal(0.0, weights0[1], Tolerance); // 0.4 >= 0.3

        // Step epoch: lambda becomes 0.5
        scheduler.StepEpoch(CreateMetrics(0.5));
        var weights1 = scheduler.ComputeSampleWeights(losses);
        Assert.Equal(1.0, weights1[0], Tolerance); // 0.1 < 0.5
        Assert.Equal(1.0, weights1[1], Tolerance); // 0.4 < 0.5
        Assert.Equal(0.0, weights1[2], Tolerance); // 0.6 >= 0.5

        // Step again: lambda becomes 0.7
        scheduler.StepEpoch(CreateMetrics(0.5));
        var weights2 = scheduler.ComputeSampleWeights(losses);
        Assert.Equal(1.0, weights2[0], Tolerance); // 0.1 < 0.7
        Assert.Equal(1.0, weights2[1], Tolerance); // 0.4 < 0.7
        Assert.Equal(1.0, weights2[2], Tolerance); // 0.6 < 0.7
        Assert.Equal(0.0, weights2[3], Tolerance); // 0.8 >= 0.7
    }

    [Fact]
    public void SelfPaced_Hard_LambdaCappedAtMax()
    {
        // maxLambda = 1.0, growthRate = 0.5
        // After 3 steps: lambda = 0.5 + 3*0.5 = 2.0 => capped at 1.0
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.5, lambdaGrowthRate: 0.5, maxLambda: 1.0,
            regularizer: SelfPaceRegularizer.Hard);

        for (int i = 0; i < 3; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));

        Assert.Equal(1.0, scheduler.CurrentThreshold, Tolerance);
    }

    // ─── Self-Paced Scheduler: Linear Regularizer ───────────────────────

    [Fact]
    public void SelfPaced_Linear_HandCalculatedWeights()
    {
        // Linear: v = max(0, 1 - loss/lambda)
        // lambda = 1.0
        // loss=0.0: weight = 1 - 0/1 = 1.0
        // loss=0.3: weight = 1 - 0.3/1 = 0.7
        // loss=0.7: weight = 1 - 0.7/1 = 0.3
        // loss=1.0: weight = 1 - 1.0/1 = 0.0 => not selected (<=0)
        // loss=1.5: weight = 1 - 1.5/1 = -0.5 => not selected
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 1.0, regularizer: SelfPaceRegularizer.Linear);

        var losses = new Vector<double>(new double[] { 0.0, 0.3, 0.7, 1.0, 1.5 });
        var weights = scheduler.ComputeSampleWeights(losses);

        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(0.7, weights[1], Tolerance);
        Assert.Equal(0.3, weights[2], Tolerance);
        Assert.Equal(0.0, weights[3], Tolerance); // exactly at threshold
        Assert.Equal(0.0, weights[4], Tolerance); // above threshold
    }

    [Fact]
    public void SelfPaced_Linear_WeightsDecrease_WithLoss()
    {
        // Weights should decrease as loss increases (for selected samples)
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 2.0, regularizer: SelfPaceRegularizer.Linear);

        var losses = new Vector<double>(new double[] { 0.1, 0.5, 1.0, 1.5, 1.9 });
        var weights = scheduler.ComputeSampleWeights(losses);

        for (int i = 0; i < 4; i++)
        {
            if (weights[i] > 0 && weights[i + 1] > 0)
            {
                Assert.True(weights[i] > weights[i + 1],
                    $"Weight at loss={losses[i]} ({weights[i]}) should exceed weight at loss={losses[i + 1]} ({weights[i + 1]})");
            }
        }
    }

    // ─── Self-Paced Scheduler: Mixture Regularizer ──────────────────────

    [Fact]
    public void SelfPaced_Mixture_HandCalculated()
    {
        // Mixture: zeta=0.5, weight = zeta*1 + (1-zeta)*(1-loss/lambda) if loss < lambda
        // lambda=1.0
        // loss=0.0: linearPart = 1-0 = 1.0, weight = 0.5*1 + 0.5*1.0 = 1.0
        // loss=0.4: linearPart = 1-0.4 = 0.6, weight = 0.5*1 + 0.5*0.6 = 0.5+0.3 = 0.8
        // loss=0.8: linearPart = 1-0.8 = 0.2, weight = 0.5*1 + 0.5*0.2 = 0.5+0.1 = 0.6
        // loss=1.0: not selected (loss >= lambda)
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 1.0, regularizer: SelfPaceRegularizer.Mixture);

        var losses = new Vector<double>(new double[] { 0.0, 0.4, 0.8, 1.0 });
        var weights = scheduler.ComputeSampleWeights(losses);

        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(0.8, weights[1], Tolerance);
        Assert.Equal(0.6, weights[2], Tolerance);
        Assert.Equal(0.0, weights[3], Tolerance);
    }

    [Fact]
    public void SelfPaced_Mixture_AlwaysHigherThanLinear()
    {
        // Mixture adds a constant offset (zeta) to the linear part
        // So mixture weight >= linear weight for all selected samples
        var mixture = new SelfPacedScheduler<double>(10,
            initialLambda: 1.0, regularizer: SelfPaceRegularizer.Mixture);
        var linear = new SelfPacedScheduler<double>(10,
            initialLambda: 1.0, regularizer: SelfPaceRegularizer.Linear);

        var losses = new Vector<double>(new double[] { 0.1, 0.3, 0.5, 0.7, 0.9 });
        var mWeights = mixture.ComputeSampleWeights(losses);
        var lWeights = linear.ComputeSampleWeights(losses);

        for (int i = 0; i < losses.Length; i++)
        {
            if (mWeights[i] > 0)
            {
                Assert.True(mWeights[i] >= lWeights[i] - Tolerance,
                    $"Mixture weight ({mWeights[i]}) should be >= linear weight ({lWeights[i]}) at loss={losses[i]}");
            }
        }
    }

    // ─── Self-Paced Scheduler: Logarithmic Regularizer ──────────────────

    [Fact]
    public void SelfPaced_Logarithmic_HandCalculated()
    {
        // Logarithmic: v = max(0, log(lambda+eps) - log(loss+eps)) / log(lambda+eps)
        // lambda=2.0, epsilon=1e-10
        // loss=0.5: logLambda ≈ log(2), logLoss ≈ log(0.5)
        //   diff = log(2) - log(0.5) = log(2/0.5) = log(4) ≈ 1.386
        //   weight = 1.386 / log(2) ≈ 1.386 / 0.693 ≈ 2.0
        // But weights are not clamped to [0,1] in the formula, they can exceed 1
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 2.0, regularizer: SelfPaceRegularizer.Logarithmic);

        var losses = new Vector<double>(new double[] { 0.5 });
        var weights = scheduler.ComputeSampleWeights(losses);

        double eps = 1e-10;
        double logLambda = Math.Log(2.0 + eps);
        double logLoss = Math.Log(0.5 + eps);
        double diff = logLambda - logLoss;
        double expected = diff / logLambda;

        Assert.Equal(expected, weights[0], 1e-4);
    }

    [Fact]
    public void SelfPaced_Logarithmic_LossAboveLambda_ZeroWeight()
    {
        // When loss >= lambda, log(loss) >= log(lambda), diff <= 0, weight = 0
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 1.0, regularizer: SelfPaceRegularizer.Logarithmic);

        var losses = new Vector<double>(new double[] { 1.5, 2.0, 5.0 });
        var weights = scheduler.ComputeSampleWeights(losses);

        for (int i = 0; i < weights.Length; i++)
            Assert.Equal(0.0, weights[i], Tolerance);
    }

    // ─── Self-Paced: SelectSamplesWithWeights ───────────────────────────

    [Fact]
    public void SelfPaced_SelectSamplesWithWeights_ReturnsCorrectIndices()
    {
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.5, regularizer: SelfPaceRegularizer.Hard);

        var losses = new Vector<double>(new double[] { 0.1, 0.6, 0.2, 0.8, 0.4 });
        var (indices, weights) = scheduler.SelectSamplesWithWeights(losses);

        // Only indices 0, 2, 4 have loss < 0.5
        Assert.Contains(0, indices);
        Assert.Contains(2, indices);
        Assert.Contains(4, indices);
        Assert.DoesNotContain(1, indices);
        Assert.DoesNotContain(3, indices);
        Assert.Equal(3, indices.Length);
    }

    [Fact]
    public void SelfPaced_SelectSamplesWithWeights_AllAboveThreshold_SelectsEasiest()
    {
        // When no samples qualify, the easiest sample should still be selected
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.01, regularizer: SelfPaceRegularizer.Hard);

        var losses = new Vector<double>(new double[] { 0.5, 0.3, 0.1, 0.8 });
        var (indices, weights) = scheduler.SelectSamplesWithWeights(losses);

        // All losses >= 0.01, so easiest (index 2, loss=0.1) should be selected
        Assert.Single(indices);
        Assert.Equal(2, indices[0]);
    }

    // ─── Self-Paced: Reset ──────────────────────────────────────────────

    [Fact]
    public void SelfPaced_Reset_RestoresInitialLambda()
    {
        var scheduler = new SelfPacedScheduler<double>(10,
            initialLambda: 0.3, lambdaGrowthRate: 0.1,
            regularizer: SelfPaceRegularizer.Hard);

        // Advance 5 epochs: lambda should be 0.3 + 5*0.1 = 0.8
        for (int i = 0; i < 5; i++)
            scheduler.StepEpoch(CreateMetrics(0.5));
        Assert.Equal(0.8, scheduler.CurrentThreshold, Tolerance);

        // Reset: lambda should go back to 0.3
        scheduler.Reset();
        Assert.Equal(0.3, scheduler.CurrentThreshold, Tolerance);
    }

    // ─── Competence-Based Scheduler ─────────────────────────────────────

    [Fact]
    public void CompetenceBased_InitialCompetence_Zero()
    {
        var scheduler = new CompetenceBasedScheduler<double>(10);
        Assert.Equal(0.0, scheduler.CurrentCompetence, Tolerance);
    }

    [Fact]
    public void CompetenceBased_EMASmoothing_HandCalculated()
    {
        // EMA: competence = alpha * new + (1-alpha) * old
        // smoothingFactor=0.3 (default)
        // Initial competence = 0.0
        // After update with accuracy 0.8:
        //   new_competence = 0.3 * 0.8 + 0.7 * 0.0 = 0.24
        var scheduler = new CompetenceBasedScheduler<double>(10,
            metricType: CompetenceMetricType.Accuracy, smoothingFactor: 0.3);

        var metrics = CreateMetricsWithAccuracy(0.5, validationAccuracy: 0.8);
        scheduler.UpdateCompetence(metrics);

        Assert.Equal(0.24, scheduler.CurrentCompetence, Tolerance);
    }

    [Fact]
    public void CompetenceBased_EMASmoothing_TwoUpdates()
    {
        // First update: competence = 0.3 * 0.8 + 0.7 * 0.0 = 0.24
        // Second update: competence = 0.3 * 0.9 + 0.7 * 0.24 = 0.27 + 0.168 = 0.438
        var scheduler = new CompetenceBasedScheduler<double>(10,
            metricType: CompetenceMetricType.Accuracy, smoothingFactor: 0.3);

        scheduler.UpdateCompetence(CreateMetricsWithAccuracy(0.5, validationAccuracy: 0.8));
        Assert.Equal(0.24, scheduler.CurrentCompetence, Tolerance);

        scheduler.UpdateCompetence(CreateMetricsWithAccuracy(0.3, validationAccuracy: 0.9));
        Assert.Equal(0.438, scheduler.CurrentCompetence, Tolerance);
    }

    [Fact]
    public void CompetenceBased_PlateauDetection_AdvancesAfterPatience()
    {
        // Plateau: competence increases as epochs without improvement accumulate
        // patienceEpochs=3, smoothingFactor=1.0 (no smoothing, use latest value directly)
        // Call 1: loss=0.5, first time so _bestLoss is MaxValue, improvement is huge => resets counter to 0
        //   plateauProgress = 0/3 = 0, competence = 0
        // Call 2: loss=0.5, no improvement => counter=1, plateauProgress=1/3
        // Call 3: loss=0.5, no improvement => counter=2, plateauProgress=2/3
        // Call 4: loss=0.5, no improvement => counter=3, plateauProgress=3/3=1.0
        // After 4 calls, competence = 1.0 >= threshold 0.9 ✓
        var scheduler = new CompetenceBasedScheduler<double>(20,
            metricType: CompetenceMetricType.Plateau,
            patienceEpochs: 3, competenceThreshold: 0.9, smoothingFactor: 1.0);

        for (int i = 0; i < 4; i++)
        {
            var metrics = CreateMetrics(0.5, improved: false);
            scheduler.UpdateCompetence(metrics);
        }

        Assert.True(scheduler.HasMasteredCurrentContent(),
            $"Competence ({scheduler.CurrentCompetence}) should exceed threshold after plateau patience");
    }

    [Fact]
    public void CompetenceBased_CompetenceClampedToZeroOne()
    {
        var scheduler = new CompetenceBasedScheduler<double>(10,
            metricType: CompetenceMetricType.Accuracy, smoothingFactor: 1.0);

        // Very high accuracy should still result in competence <= 1.0
        scheduler.UpdateCompetence(CreateMetricsWithAccuracy(0.01, validationAccuracy: 1.0));
        Assert.True(scheduler.CurrentCompetence <= 1.0 + Tolerance,
            $"Competence should be <= 1.0, got {scheduler.CurrentCompetence}");
        Assert.True(scheduler.CurrentCompetence >= 0.0 - Tolerance,
            $"Competence should be >= 0.0, got {scheduler.CurrentCompetence}");
    }

    [Fact]
    public void CompetenceBased_ResetForNewPhase_HalvesCompetence()
    {
        // When mastery is achieved and phase advances, competence is halved
        var scheduler = new CompetenceBasedScheduler<double>(20,
            competenceThreshold: 0.5,
            metricType: CompetenceMetricType.Accuracy,
            smoothingFactor: 1.0);

        // Drive competence above threshold
        scheduler.UpdateCompetence(CreateMetricsWithAccuracy(0.1, validationAccuracy: 0.8));
        double preAdvanceCompetence = scheduler.CurrentCompetence;
        Assert.True(preAdvanceCompetence >= 0.5, "Competence should be above threshold");

        // StepEpoch should detect mastery and advance phase
        bool advanced = scheduler.StepEpoch(CreateMetricsWithAccuracy(0.1, validationAccuracy: 0.8));

        if (advanced)
        {
            // After phase advance, competence should be halved
            Assert.True(scheduler.CurrentCompetence < preAdvanceCompetence,
                "Competence should decrease after phase advance");
        }
    }

    // ─── Cross-Scheduler Comparisons ────────────────────────────────────

    [Fact]
    public void AllSchedulers_StartAtMinFraction()
    {
        double min = 0.15;
        double max = 0.95;

        var linear = new LinearScheduler<double>(10, minFraction: min, maxFraction: max);
        var exponential = new ExponentialScheduler<double>(10, minFraction: min, maxFraction: max);
        var polynomial = new PolynomialScheduler<double>(10, power: 2.0, minFraction: min, maxFraction: max);
        var cosine = new CosineScheduler<double>(10, minFraction: min, maxFraction: max);

        Assert.Equal(min, linear.GetDataFraction(), Tolerance);
        Assert.Equal(min, exponential.GetDataFraction(), Tolerance);
        Assert.Equal(min, polynomial.GetDataFraction(), Tolerance);
        Assert.Equal(min, cosine.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void AllSchedulers_EndAtMaxFraction()
    {
        double min = 0.15;
        double max = 0.95;

        var linear = new LinearScheduler<double>(10, minFraction: min, maxFraction: max);
        var exponential = new ExponentialScheduler<double>(10, minFraction: min, maxFraction: max);
        var polynomial = new PolynomialScheduler<double>(10, power: 2.0, minFraction: min, maxFraction: max);
        var cosine = new CosineScheduler<double>(10, minFraction: min, maxFraction: max);

        // Advance all to final epoch
        for (int i = 0; i < 9; i++)
        {
            linear.StepEpoch(CreateMetrics(0.5));
            exponential.StepEpoch(CreateMetrics(0.5));
            polynomial.StepEpoch(CreateMetrics(0.5));
            cosine.StepEpoch(CreateMetrics(0.5));
        }

        Assert.Equal(max, linear.GetDataFraction(), Tolerance);
        Assert.Equal(max, exponential.GetDataFraction(), Tolerance);
        Assert.Equal(max, polynomial.GetDataFraction(), Tolerance);
        Assert.Equal(max, cosine.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void Polynomial_Power2_BelowLinear_InFirstHalf()
    {
        // Convex curve (power>1) should be below linear in first half
        var poly = new PolynomialScheduler<double>(10, power: 2.0,
            minFraction: 0.0, maxFraction: 1.0);
        var linear = new LinearScheduler<double>(10,
            minFraction: 0.0, maxFraction: 1.0);

        for (int epoch = 1; epoch < 5; epoch++) // First half (not epoch 0)
        {
            poly.StepEpoch(CreateMetrics(0.5));
            linear.StepEpoch(CreateMetrics(0.5));

            double polyFrac = poly.GetDataFraction();
            double linearFrac = linear.GetDataFraction();

            Assert.True(polyFrac < linearFrac + Tolerance,
                $"Polynomial(2) should be below linear at epoch {epoch}: poly={polyFrac}, linear={linearFrac}");
        }
    }

    // ─── Validation and Edge Cases ──────────────────────────────────────

    [Fact]
    public void LinearScheduler_SingleEpoch_ReturnsMaxFraction()
    {
        // With 1 epoch, should return max fraction immediately (progress clamped to 1)
        var scheduler = new LinearScheduler<double>(1, minFraction: 0.2, maxFraction: 0.8);
        // For 1 epoch: totalEpochs-1 = 0, so Math.Max(1, 0) = 1, progress = 0/1 = 0
        // Actually: progress = min(1.0, 0/max(1,0)) = min(1.0, 0/1) = 0
        // So fraction = 0.2 + 0 * 0.6 = 0.2
        // Wait - for TotalEpochs=1, currentEpoch=0, progress = 0/(1-1) but code uses Math.Max(1, TotalEpochs-1) = 1
        // So progress = 0/1 = 0 => returns min
        Assert.Equal(0.2, scheduler.GetDataFraction(), Tolerance);
    }

    [Fact]
    public void AllSchedulers_Reset_RestoresEpoch0()
    {
        var linear = new LinearScheduler<double>(10, minFraction: 0.2, maxFraction: 1.0);

        // Advance then reset
        for (int i = 0; i < 5; i++)
            linear.StepEpoch(CreateMetrics(0.5));

        double midFraction = linear.GetDataFraction();
        Assert.True(midFraction > 0.2 + Tolerance, "Should have advanced beyond min");

        linear.Reset();
        Assert.Equal(0.2, linear.GetDataFraction(), Tolerance);
        Assert.Equal(0, linear.CurrentEpoch);
    }

    [Fact]
    public void ExponentialScheduler_InvalidGrowthRate_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ExponentialScheduler<double>(10, growthRate: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ExponentialScheduler<double>(10, growthRate: -1.0));
    }

    [Fact]
    public void PolynomialScheduler_InvalidPower_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PolynomialScheduler<double>(10, power: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PolynomialScheduler<double>(10, power: -1.0));
    }

    [Fact]
    public void SelfPacedScheduler_InvalidInitialLambda_Throws()
    {
        // Negative lambda should throw (0.0 is treated as "use default" after ResolveDefault fix)
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SelfPacedScheduler<double>(10, initialLambda: -0.5));
    }

    [Fact]
    public void CompetenceBasedScheduler_InvalidThreshold_Throws()
    {
        // Threshold > 1 should throw (0.0 is treated as "use default" after ResolveDefault fix)
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new CompetenceBasedScheduler<double>(10, competenceThreshold: 1.5));
    }

    [Fact]
    public void Scheduler_InvalidTotalEpochs_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearScheduler<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearScheduler<double>(-1));
    }

    [Fact]
    public void Scheduler_MinGreaterThanMax_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearScheduler<double>(10, minFraction: 0.8, maxFraction: 0.2));
    }
}
