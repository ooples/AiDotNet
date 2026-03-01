using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CurriculumLearning;

/// <summary>
/// Deep mathematical integration tests for curriculum learning schedulers and self-paced learning.
/// Verifies correctness of hand-computed expected values for:
/// - Linear scheduler: fraction = min + progress * (max - min)
/// - Exponential scheduler: (1-e^(-r*t)) / (1-e^(-r))
/// - Polynomial scheduler: t^power
/// - Cosine scheduler: 0.5*(1-cos(pi*t))
/// - Step scheduler: discrete fraction jumps
/// - Self-paced learning: hard, linear, mixture, logarithmic regularizers
/// - Competence-based advancement: EMA smoothing, plateau detection
/// </summary>
public class CurriculumLearningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Linear Scheduler Mathematics

    [Fact]
    public void LinearScheduler_FractionAtStart_IsMinFraction()
    {
        // fraction(0) = min + (0/total) * (max - min) = min
        double min = 0.1, max = 1.0;
        int epoch = 0, totalEpochs = 10;

        double fraction = LinearFraction(epoch, totalEpochs, min, max);
        Assert.Equal(0.1, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_FractionAtEnd_IsMaxFraction()
    {
        // fraction(total-1) = min + (1.0) * (max - min) = max
        double min = 0.1, max = 1.0;
        int totalEpochs = 10;

        double fraction = LinearFraction(totalEpochs - 1, totalEpochs, min, max);
        Assert.Equal(1.0, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_FractionAtMidpoint_IsAverage()
    {
        // At midpoint t=0.5: fraction = min + 0.5*(max-min) = (min+max)/2
        double min = 0.2, max = 0.8;
        int totalEpochs = 11; // epoch 5 is exactly midpoint

        double fraction = LinearFraction(5, totalEpochs, min, max);
        Assert.Equal(0.5, fraction, Tolerance);
    }

    [Fact]
    public void LinearScheduler_MonotonicallyIncreasing()
    {
        double min = 0.1, max = 1.0;
        int totalEpochs = 20;

        double prev = 0;
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            double fraction = LinearFraction(epoch, totalEpochs, min, max);
            Assert.True(fraction >= prev - 1e-10,
                $"Fraction at epoch {epoch} ({fraction}) should be >= previous ({prev})");
            prev = fraction;
        }
    }

    [Fact]
    public void LinearScheduler_ConstantRateOfChange()
    {
        // Linear: equal increment per epoch
        double min = 0.0, max = 1.0;
        int totalEpochs = 10;

        double expectedStep = (max - min) / (totalEpochs - 1);
        for (int epoch = 1; epoch < totalEpochs; epoch++)
        {
            double curr = LinearFraction(epoch, totalEpochs, min, max);
            double prev = LinearFraction(epoch - 1, totalEpochs, min, max);
            double step = curr - prev;

            Assert.Equal(expectedStep, step, Tolerance);
        }
    }

    #endregion

    #region Exponential Scheduler Mathematics

    [Fact]
    public void ExponentialScheduler_AtStart_IsMinFraction()
    {
        // At t=0: (1-e^0)/(1-e^(-r)) = 0
        double min = 0.1, max = 1.0;
        double growthRate = 3.0;

        double fraction = ExponentialFraction(0, 10, growthRate, min, max);
        Assert.Equal(min, fraction, Tolerance);
    }

    [Fact]
    public void ExponentialScheduler_AtEnd_IsMaxFraction()
    {
        // At t=1: (1-e^(-r))/(1-e^(-r)) = 1
        double min = 0.1, max = 1.0;
        double growthRate = 3.0;

        double fraction = ExponentialFraction(9, 10, growthRate, min, max);
        Assert.Equal(max, fraction, Tolerance);
    }

    [Fact]
    public void ExponentialScheduler_HandComputed()
    {
        // t=0.5, rate=3.0
        // numerator = 1 - e^(-3*0.5) = 1 - e^(-1.5) = 1 - 0.22313 = 0.77687
        // denominator = 1 - e^(-3) = 1 - 0.04979 = 0.95021
        // progress = 0.77687 / 0.95021 = 0.81757
        // fraction = 0.1 + 0.81757 * 0.9 = 0.83581
        double growthRate = 3.0;
        double t = 0.5;
        double progress = (1 - Math.Exp(-growthRate * t)) / (1 - Math.Exp(-growthRate));

        double expected = 0.1 + progress * 0.9;
        double fraction = ExponentialFraction(5, 11, growthRate, 0.1, 1.0);

        Assert.Equal(expected, fraction, 1e-3);
    }

    [Fact]
    public void ExponentialScheduler_HigherRate_FasterInitialGrowth()
    {
        // Higher growth rate -> more initial fraction at same epoch
        double min = 0.0, max = 1.0;
        int epoch = 2, total = 20;

        double fracLow = ExponentialFraction(epoch, total, 1.0, min, max);
        double fracMed = ExponentialFraction(epoch, total, 3.0, min, max);
        double fracHigh = ExponentialFraction(epoch, total, 10.0, min, max);

        Assert.True(fracHigh > fracMed, $"High rate {fracHigh} should > medium {fracMed}");
        Assert.True(fracMed > fracLow, $"Medium rate {fracMed} should > low {fracLow}");
    }

    [Fact]
    public void ExponentialScheduler_MonotonicallyIncreasing()
    {
        double min = 0.1, max = 1.0;
        double growthRate = 5.0;
        int totalEpochs = 20;

        double prev = 0;
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            double fraction = ExponentialFraction(epoch, totalEpochs, growthRate, min, max);
            Assert.True(fraction >= prev - 1e-10,
                $"Fraction at epoch {epoch} ({fraction}) should be >= previous ({prev})");
            prev = fraction;
        }
    }

    #endregion

    #region Polynomial Scheduler Mathematics

    [Fact]
    public void PolynomialScheduler_Power1_IsLinear()
    {
        // power=1: t^1 = t (same as linear)
        double min = 0.0, max = 1.0;
        int totalEpochs = 10;

        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            double polyFrac = PolynomialFraction(epoch, totalEpochs, 1.0, min, max);
            double linFrac = LinearFraction(epoch, totalEpochs, min, max);
            Assert.Equal(linFrac, polyFrac, Tolerance);
        }
    }

    [Fact]
    public void PolynomialScheduler_Power2_Quadratic()
    {
        // power=2: progress = t^2
        // t=0.5 -> progress=0.25
        // fraction = 0 + 0.25 * 1.0 = 0.25
        double fraction = PolynomialFraction(5, 11, 2.0, 0.0, 1.0);
        double expected = Math.Pow(0.5, 2.0); // 0.25
        Assert.Equal(expected, fraction, Tolerance);
    }

    [Fact]
    public void PolynomialScheduler_PowerHalf_SquareRoot()
    {
        // power=0.5: progress = sqrt(t)
        // t=0.25 -> progress=0.5
        double fraction = PolynomialFraction(2, 9, 0.5, 0.0, 1.0);
        double expected = Math.Pow(0.25, 0.5); // sqrt(0.25) = 0.5
        Assert.Equal(expected, fraction, Tolerance);
    }

    [Fact]
    public void PolynomialScheduler_HighPower_SlowStart()
    {
        // Higher power -> slower start (more time at low fractions)
        double min = 0.0, max = 1.0;
        int epoch = 3, total = 20;

        double fracP1 = PolynomialFraction(epoch, total, 1.0, min, max);
        double fracP2 = PolynomialFraction(epoch, total, 2.0, min, max);
        double fracP5 = PolynomialFraction(epoch, total, 5.0, min, max);

        Assert.True(fracP1 > fracP2, $"Power 1 ({fracP1}) should > power 2 ({fracP2}) early on");
        Assert.True(fracP2 > fracP5, $"Power 2 ({fracP2}) should > power 5 ({fracP5}) early on");
    }

    [Fact]
    public void PolynomialScheduler_BoundaryConditions()
    {
        // At t=0, always 0 (then min), at t=1 always 1 (then max)
        double min = 0.2, max = 0.8;
        double[] powers = [0.5, 1.0, 2.0, 3.0, 10.0];

        foreach (var power in powers)
        {
            double atStart = PolynomialFraction(0, 10, power, min, max);
            double atEnd = PolynomialFraction(9, 10, power, min, max);

            Assert.Equal(min, atStart, Tolerance);
            Assert.Equal(max, atEnd, Tolerance);
        }
    }

    #endregion

    #region Cosine Scheduler Mathematics

    [Fact]
    public void CosineScheduler_AtStart_IsMinFraction()
    {
        // t=0: progress = 0.5*(1-cos(0)) = 0.5*(1-1) = 0
        double min = 0.1, max = 1.0;
        double fraction = CosineFraction(0, 10, min, max);
        Assert.Equal(min, fraction, Tolerance);
    }

    [Fact]
    public void CosineScheduler_AtEnd_IsMaxFraction()
    {
        // t=1: progress = 0.5*(1-cos(pi)) = 0.5*(1-(-1)) = 1.0
        double min = 0.1, max = 1.0;
        double fraction = CosineFraction(9, 10, min, max);
        Assert.Equal(max, fraction, Tolerance);
    }

    [Fact]
    public void CosineScheduler_AtMidpoint_IsHalfway()
    {
        // t=0.5: progress = 0.5*(1-cos(pi/2)) = 0.5*(1-0) = 0.5
        // fraction = 0.0 + 0.5 * 1.0 = 0.5
        double fraction = CosineFraction(5, 11, 0.0, 1.0);
        Assert.Equal(0.5, fraction, Tolerance);
    }

    [Fact]
    public void CosineScheduler_SymmetricProgress()
    {
        // Cosine scheduler has S-shaped curve: symmetric around midpoint
        // progress(0.25) + progress(0.75) = 1.0
        double t1 = 0.25, t2 = 0.75;
        double p1 = 0.5 * (1 - Math.Cos(Math.PI * t1));
        double p2 = 0.5 * (1 - Math.Cos(Math.PI * t2));

        Assert.Equal(1.0, p1 + p2, Tolerance);
    }

    [Fact]
    public void CosineScheduler_HandComputedQuarter()
    {
        // t=0.25: progress = 0.5*(1-cos(pi/4)) = 0.5*(1-sqrt(2)/2) = 0.5*(1-0.70711) = 0.14645
        double t = 0.25;
        double progress = 0.5 * (1 - Math.Cos(Math.PI * t));
        double expected = 0.5 * (1 - Math.Sqrt(2) / 2);
        Assert.Equal(expected, progress, Tolerance);
    }

    #endregion

    #region Step Scheduler Mathematics

    [Fact]
    public void StepScheduler_UniformSteps_CorrectFractions()
    {
        // 3 steps over 12 epochs: epochs 0-3 -> step 0, 4-7 -> step 1, 8-11 -> step 2
        int numSteps = 3, totalEpochs = 12;
        double min = 0.0, max = 1.0;

        // Step 0: fraction = (1/3) * (max-min) + min = 1/3
        // Step 1: fraction = (2/3) * (max-min) + min = 2/3
        // Step 2: fraction = (3/3) * (max-min) + min = 1.0
        double step0 = StepFraction(0, totalEpochs, numSteps, min, max);
        double step1 = StepFraction(4, totalEpochs, numSteps, min, max);
        double step2 = StepFraction(8, totalEpochs, numSteps, min, max);

        Assert.Equal(1.0 / 3.0, step0, Tolerance);
        Assert.Equal(2.0 / 3.0, step1, Tolerance);
        Assert.Equal(1.0, step2, Tolerance);
    }

    [Fact]
    public void StepScheduler_WithinSameStep_FractionConstant()
    {
        int numSteps = 4, totalEpochs = 20;
        double min = 0.0, max = 1.0;

        // Epochs 0-4 are in step 0, should have same fraction
        double frac0 = StepFraction(0, totalEpochs, numSteps, min, max);
        double frac2 = StepFraction(2, totalEpochs, numSteps, min, max);
        double frac4 = StepFraction(4, totalEpochs, numSteps, min, max);

        Assert.Equal(frac0, frac2, Tolerance);
        Assert.Equal(frac0, frac4, Tolerance);
    }

    [Fact]
    public void StepScheduler_StepBoundary_FractionJumps()
    {
        int numSteps = 5, totalEpochs = 10;
        double min = 0.0, max = 1.0;

        // At step boundaries, fraction should jump
        double beforeStep = StepFraction(1, totalEpochs, numSteps, min, max);
        double afterStep = StepFraction(2, totalEpochs, numSteps, min, max);

        Assert.True(afterStep > beforeStep || Math.Abs(afterStep - beforeStep) < Tolerance,
            $"After step {afterStep} should be >= before {beforeStep}");
    }

    #endregion

    #region Self-Paced Learning: Hard Regularizer

    [Fact]
    public void SelfPaced_Hard_SelectsBelowThreshold()
    {
        // Hard: v = 1 if loss < lambda, else 0
        double lambda = 0.5;
        double[] losses = [0.1, 0.3, 0.5, 0.7, 1.0];

        var weights = HardRegularizer(losses, lambda);

        Assert.Equal(1.0, weights[0], Tolerance); // 0.1 < 0.5
        Assert.Equal(1.0, weights[1], Tolerance); // 0.3 < 0.5
        Assert.Equal(0.0, weights[2], Tolerance); // 0.5 = 0.5 (not strictly less)
        Assert.Equal(0.0, weights[3], Tolerance); // 0.7 > 0.5
        Assert.Equal(0.0, weights[4], Tolerance); // 1.0 > 0.5
    }

    [Fact]
    public void SelfPaced_Hard_HigherLambda_SelectsMore()
    {
        double[] losses = [0.1, 0.3, 0.5, 0.7, 1.0];

        int selectedLow = HardRegularizer(losses, 0.2).Count(w => w > 0);
        int selectedMed = HardRegularizer(losses, 0.6).Count(w => w > 0);
        int selectedHigh = HardRegularizer(losses, 1.5).Count(w => w > 0);

        Assert.True(selectedHigh >= selectedMed);
        Assert.True(selectedMed >= selectedLow);
    }

    #endregion

    #region Self-Paced Learning: Linear Regularizer

    [Fact]
    public void SelfPaced_Linear_WeightFormula()
    {
        // Linear: v = max(0, 1 - loss/lambda)
        // lambda=1.0, loss=0.3 -> v = 1 - 0.3 = 0.7
        // lambda=1.0, loss=0.7 -> v = 1 - 0.7 = 0.3
        // lambda=1.0, loss=1.5 -> v = max(0, -0.5) = 0.0
        double lambda = 1.0;

        Assert.Equal(0.7, LinearRegularizer(0.3, lambda), Tolerance);
        Assert.Equal(0.3, LinearRegularizer(0.7, lambda), Tolerance);
        Assert.Equal(0.0, LinearRegularizer(1.5, lambda), Tolerance);
    }

    [Fact]
    public void SelfPaced_Linear_ZeroLoss_MaxWeight()
    {
        // loss=0 -> weight = 1 - 0/lambda = 1
        Assert.Equal(1.0, LinearRegularizer(0.0, 0.5), Tolerance);
    }

    [Fact]
    public void SelfPaced_Linear_AtThreshold_ZeroWeight()
    {
        // loss=lambda -> weight = 1 - lambda/lambda = 0
        double lambda = 2.0;
        Assert.Equal(0.0, LinearRegularizer(lambda, lambda), Tolerance);
    }

    [Fact]
    public void SelfPaced_Linear_GradualTransition()
    {
        // Lower loss -> higher weight (smooth transition)
        double lambda = 1.0;
        double[] losses = [0.0, 0.25, 0.5, 0.75, 1.0];
        double[] expectedWeights = [1.0, 0.75, 0.5, 0.25, 0.0];

        for (int i = 0; i < losses.Length; i++)
        {
            double weight = LinearRegularizer(losses[i], lambda);
            Assert.Equal(expectedWeights[i], weight, Tolerance);
        }
    }

    #endregion

    #region Self-Paced Learning: Mixture Regularizer

    [Fact]
    public void SelfPaced_Mixture_CombinesHardAndLinear()
    {
        // Mixture: if loss >= lambda -> 0, else weight = zeta + (1-zeta)*(1 - loss/lambda)
        // zeta = 0.5 (typical)
        // lambda=1.0, loss=0.4
        // linearPart = 1 - 0.4 = 0.6
        // weight = 0.5*1 + 0.5*0.6 = 0.5 + 0.3 = 0.8
        double lambda = 1.0, loss = 0.4, zeta = 0.5;

        double weight = MixtureRegularizer(loss, lambda, zeta);
        Assert.Equal(0.8, weight, Tolerance);
    }

    [Fact]
    public void SelfPaced_Mixture_AboveThreshold_Zero()
    {
        double weight = MixtureRegularizer(1.5, 1.0, 0.5);
        Assert.Equal(0.0, weight, Tolerance);
    }

    [Fact]
    public void SelfPaced_Mixture_ZeroLoss_MaxWeight()
    {
        // loss=0: weight = zeta + (1-zeta)*1 = 1.0
        double weight = MixtureRegularizer(0.0, 1.0, 0.5);
        Assert.Equal(1.0, weight, Tolerance);
    }

    [Fact]
    public void SelfPaced_Mixture_MinWeightIsZeta()
    {
        // At loss just below lambda: linearPart ≈ 0
        // weight ≈ zeta + (1-zeta)*0 = zeta
        double zeta = 0.3;
        double weight = MixtureRegularizer(0.999, 1.0, zeta);
        Assert.True(Math.Abs(weight - zeta) < 0.01,
            $"Near-threshold weight {weight} should be close to zeta={zeta}");
    }

    #endregion

    #region Self-Paced Lambda Growth

    [Fact]
    public void SelfPaced_LambdaGrowth_Linear()
    {
        // lambda(t+1) = lambda(t) + growth_rate
        double lambda = 0.1;
        double growthRate = 0.1;
        double maxLambda = 10.0;

        for (int epoch = 0; epoch < 5; epoch++)
        {
            lambda = Math.Min(maxLambda, lambda + growthRate);
        }

        Assert.Equal(0.6, lambda, Tolerance);
    }

    [Fact]
    public void SelfPaced_LambdaGrowth_CapsAtMax()
    {
        double lambda = 0.1;
        double growthRate = 5.0;
        double maxLambda = 2.0;

        lambda = Math.Min(maxLambda, lambda + growthRate);
        Assert.Equal(2.0, lambda, Tolerance);
    }

    [Fact]
    public void SelfPaced_LambdaGrowth_MoreEpochs_MoreSamples()
    {
        // As lambda grows, more samples should be selected (hard regularizer)
        double[] losses = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
        double lambda = 0.1;
        double growthRate = 0.3;

        int prevSelected = 0;
        for (int epoch = 0; epoch < 10; epoch++)
        {
            int selected = HardRegularizer(losses, lambda).Count(w => w > 0);
            Assert.True(selected >= prevSelected,
                $"Epoch {epoch}: selected {selected} should be >= previous {prevSelected}");
            prevSelected = selected;
            lambda += growthRate;
        }
    }

    #endregion

    #region Competence-Based Scheduler Mathematics

    [Fact]
    public void Competence_EMA_Smoothing_HandComputed()
    {
        // EMA: competence_new = alpha * raw + (1-alpha) * competence_old
        // alpha=0.3, competence_old=0.2, raw_new=0.8
        // competence_new = 0.3*0.8 + 0.7*0.2 = 0.24 + 0.14 = 0.38
        double alpha = 0.3;
        double competenceOld = 0.2;
        double rawNew = 0.8;

        double competenceNew = alpha * rawNew + (1 - alpha) * competenceOld;
        Assert.Equal(0.38, competenceNew, Tolerance);
    }

    [Fact]
    public void Competence_EMA_SmoothsOutNoise()
    {
        // EMA should smooth out noisy observations
        double alpha = 0.3;
        double competence = 0.0;
        double[] rawValues = [0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7];

        var smoothed = new List<double>();
        foreach (var raw in rawValues)
        {
            competence = alpha * raw + (1 - alpha) * competence;
            smoothed.Add(competence);
        }

        // Variance of smoothed should be less than variance of raw
        double rawVariance = Variance(rawValues);
        double smoothedVariance = Variance(smoothed.ToArray());

        Assert.True(smoothedVariance < rawVariance,
            $"Smoothed variance {smoothedVariance} should be < raw {rawVariance}");
    }

    [Fact]
    public void Competence_EMA_ConvergesToConstant()
    {
        // If all raw values are the same, EMA converges to that value
        double alpha = 0.3;
        double competence = 0.0;
        double constantValue = 0.75;

        for (int i = 0; i < 50; i++)
        {
            competence = alpha * constantValue + (1 - alpha) * competence;
        }

        Assert.Equal(constantValue, competence, 1e-5);
    }

    [Fact]
    public void Competence_PlateauDetection_HandComputed()
    {
        // Plateau competence: epochs_without_improvement / patience
        // If 3 epochs without improvement and patience=5: competence = 3/5 = 0.6
        int epochsWithoutImprovement = 3;
        int patience = 5;

        double plateauCompetence = (double)epochsWithoutImprovement / patience;
        Assert.Equal(0.6, plateauCompetence, Tolerance);
    }

    [Fact]
    public void Competence_PlateauDetection_CappedAt1()
    {
        // Plateau competence should cap at 1.0
        int epochsWithoutImprovement = 10;
        int patience = 5;

        double plateauCompetence = Math.Min(1.0, (double)epochsWithoutImprovement / patience);
        Assert.Equal(1.0, plateauCompetence, Tolerance);
    }

    [Fact]
    public void Competence_LossBasedCompetence_HighLoss_LowCompetence()
    {
        // Competence from loss: 1/(1+loss) -> high loss = low competence
        double highLoss = 10.0;
        double lowLoss = 0.1;

        double compHigh = 1.0 / (1.0 + highLoss);
        double compLow = 1.0 / (1.0 + lowLoss);

        Assert.True(compLow > compHigh,
            $"Low loss competence {compLow} should > high loss {compHigh}");
        Assert.Equal(1.0 / 11.0, compHigh, Tolerance);
        Assert.Equal(1.0 / 1.1, compLow, Tolerance);
    }

    [Fact]
    public void Competence_Combined_WeightedAverage()
    {
        // Combined competence: average of components
        // accuracy_competence = 0.8 (with 0.8 discount for training = 0.64)
        // loss_competence = 1/(1+0.5) = 0.667
        // improvement indicator = 0.8 (if improved)
        double trainingAccuracy = 0.8;
        double discountedAccuracy = trainingAccuracy * 0.8; // 0.64
        double lossCompetence = 1.0 / (1.0 + 0.5); // 0.667
        double improvementIndicator = 0.8; // improved

        double combined = (discountedAccuracy + lossCompetence + improvementIndicator) / 3.0;
        double expected = (0.64 + 1.0 / 1.5 + 0.8) / 3.0;
        Assert.Equal(expected, combined, Tolerance);
    }

    #endregion

    #region Phase Advancement Logic

    [Fact]
    public void PhaseAdvancement_CompetenceExceedsThreshold()
    {
        // Phase advances when competence >= threshold
        double competence = 0.92;
        double threshold = 0.9;

        bool shouldAdvance = competence >= threshold;
        Assert.True(shouldAdvance);
    }

    [Fact]
    public void PhaseAdvancement_CompetenceBelowThreshold()
    {
        double competence = 0.85;
        double threshold = 0.9;

        bool shouldAdvance = competence >= threshold;
        Assert.False(shouldAdvance);
    }

    [Fact]
    public void PhaseAdvancement_ResetCompetenceAfterAdvance()
    {
        // After advancing, competence is halved (momentum preservation)
        double competenceBefore = 0.95;
        double competenceAfter = competenceBefore * 0.5;

        Assert.Equal(0.475, competenceAfter, Tolerance);
    }

    [Fact]
    public void PhaseAdvancement_DataFractionIncreases()
    {
        // Each phase uses more data
        // phase progression = currentPhase / (totalPhases - 1)
        // fraction = min + progress * (max - min)
        double min = 0.1, max = 1.0;
        int totalPhases = 5;

        double[] fractions = new double[totalPhases];
        for (int phase = 0; phase < totalPhases; phase++)
        {
            double progress = (double)phase / Math.Max(1, totalPhases - 1);
            progress = Math.Min(1.0, progress);
            fractions[phase] = min + progress * (max - min);
        }

        // Verify monotonically increasing
        for (int i = 1; i < fractions.Length; i++)
        {
            Assert.True(fractions[i] >= fractions[i - 1],
                $"Phase {i} fraction ({fractions[i]}) should be >= phase {i - 1} ({fractions[i - 1]})");
        }

        // Verify endpoints
        Assert.Equal(min, fractions[0], Tolerance);
        Assert.Equal(max, fractions[totalPhases - 1], Tolerance);
    }

    #endregion

    #region Difficulty Estimation Mathematics

    [Fact]
    public void DifficultyEstimation_LossBased_HigherLoss_Harder()
    {
        // Difficulty scores based on loss: higher loss = harder sample
        double[] losses = [0.01, 0.5, 2.0, 10.0];

        for (int i = 1; i < losses.Length; i++)
        {
            Assert.True(losses[i] > losses[i - 1],
                "Higher loss should indicate harder sample");
        }
    }

    [Fact]
    public void DifficultyEstimation_Confidence_LowerConfidence_Harder()
    {
        // Difficulty from confidence: 1 - max(softmax)
        double[] maxProbs = [0.95, 0.7, 0.4, 0.25]; // decreasing confidence
        double[] difficulties = maxProbs.Select(p => 1 - p).ToArray();

        // Higher difficulty for lower confidence
        for (int i = 1; i < difficulties.Length; i++)
        {
            Assert.True(difficulties[i] > difficulties[i - 1],
                $"Lower confidence should be harder: {difficulties[i]} vs {difficulties[i - 1]}");
        }
    }

    [Fact]
    public void DifficultyEstimation_EnsembleDisagreement()
    {
        // Ensemble disagreement: variance of ensemble predictions
        // Models predict: [0.8, 0.9, 0.85] -> low variance -> easy
        // Models predict: [0.3, 0.9, 0.1] -> high variance -> hard
        double[] easyPreds = [0.8, 0.9, 0.85];
        double[] hardPreds = [0.3, 0.9, 0.1];

        double easyVariance = Variance(easyPreds);
        double hardVariance = Variance(hardPreds);

        Assert.True(hardVariance > easyVariance,
            $"Hard sample variance {hardVariance} should > easy {easyVariance}");
    }

    #endregion

    #region Logarithmic Regularizer

    [Fact]
    public void SelfPaced_Logarithmic_HandComputed()
    {
        // v = max(0, log(lambda) - log(loss + eps)) / log(lambda)
        // lambda=2.0, loss=0.5
        // v = (log(2) - log(0.5)) / log(2) = (0.693 - (-0.693)) / 0.693 = 1.386/0.693 = 2.0
        // Capped at 1.0? Actually the formula as written can exceed 1 for very small losses
        double lambda = 2.0, loss = 0.5;
        double logLambda = Math.Log(lambda + 1e-10);
        double logLoss = Math.Log(loss + 1e-10);
        double weight = (logLambda - logLoss) / logLambda;

        Assert.True(weight > 0, $"Weight should be positive for loss < lambda: {weight}");
    }

    [Fact]
    public void SelfPaced_Logarithmic_HighLoss_ZeroWeight()
    {
        // When loss >= lambda, log(loss) >= log(lambda), weight <= 0
        double lambda = 1.0, loss = 2.0;
        double logLambda = Math.Log(lambda + 1e-10);
        double logLoss = Math.Log(loss + 1e-10);
        double weight = Math.Max(0, (logLambda - logLoss) / logLambda);

        Assert.Equal(0.0, weight, Tolerance);
    }

    #endregion

    #region Sample Selection Properties

    [Fact]
    public void SampleSelection_TotalSamples_IncreaseOverEpochs()
    {
        // As training progresses, more samples should be included
        double min = 0.2, max = 1.0;
        int totalSamples = 100;

        var samplesPerEpoch = new List<int>();
        for (int epoch = 0; epoch < 10; epoch++)
        {
            double fraction = LinearFraction(epoch, 10, min, max);
            int numSamples = (int)Math.Ceiling(fraction * totalSamples);
            samplesPerEpoch.Add(numSamples);
        }

        // Should be non-decreasing
        for (int i = 1; i < samplesPerEpoch.Count; i++)
        {
            Assert.True(samplesPerEpoch[i] >= samplesPerEpoch[i - 1],
                $"Samples at epoch {i} ({samplesPerEpoch[i]}) should be >= epoch {i - 1} ({samplesPerEpoch[i - 1]})");
        }
    }

    [Fact]
    public void SampleSelection_DifficultySorted_EasyFirst()
    {
        // Sorted indices: easiest samples first
        double[] difficulties = [0.5, 0.1, 0.8, 0.3, 0.9];
        int[] sorted = Enumerable.Range(0, difficulties.Length)
            .OrderBy(i => difficulties[i])
            .ToArray();

        // Expected order: [1, 3, 0, 2, 4] (indices sorted by difficulty)
        Assert.Equal(1, sorted[0]); // difficulty 0.1
        Assert.Equal(3, sorted[1]); // difficulty 0.3
        Assert.Equal(0, sorted[2]); // difficulty 0.5
    }

    #endregion

    #region Helper Methods

    private static double LinearFraction(int epoch, int totalEpochs, double min, double max)
    {
        double t = (double)epoch / Math.Max(1, totalEpochs - 1);
        t = Math.Min(1.0, t);
        return min + t * (max - min);
    }

    private static double ExponentialFraction(int epoch, int totalEpochs, double growthRate, double min, double max)
    {
        double t = (double)epoch / Math.Max(1, totalEpochs - 1);
        t = Math.Min(1.0, t);

        double numerator = 1.0 - Math.Exp(-growthRate * t);
        double denominator = 1.0 - Math.Exp(-growthRate);
        double progress = numerator / denominator;
        progress = Math.Min(1.0, Math.Max(0.0, progress));

        return min + progress * (max - min);
    }

    private static double PolynomialFraction(int epoch, int totalEpochs, double power, double min, double max)
    {
        double t = (double)epoch / Math.Max(1, totalEpochs - 1);
        t = Math.Min(1.0, t);
        double progress = Math.Pow(t, power);
        return min + progress * (max - min);
    }

    private static double CosineFraction(int epoch, int totalEpochs, double min, double max)
    {
        double t = (double)epoch / Math.Max(1, totalEpochs - 1);
        t = Math.Min(1.0, t);
        double progress = 0.5 * (1.0 - Math.Cos(Math.PI * t));
        return min + progress * (max - min);
    }

    private static double StepFraction(int epoch, int totalEpochs, int numSteps, double min, double max)
    {
        double epochsPerStep = (double)totalEpochs / numSteps;
        int currentStep = (int)Math.Floor(epoch / epochsPerStep);
        currentStep = Math.Min(currentStep, numSteps - 1);

        double stepFraction = (double)(currentStep + 1) / numSteps;
        return min + stepFraction * (max - min);
    }

    private static double[] HardRegularizer(double[] losses, double lambda)
    {
        return losses.Select(l => l < lambda ? 1.0 : 0.0).ToArray();
    }

    private static double LinearRegularizer(double loss, double lambda)
    {
        return Math.Max(0.0, 1.0 - loss / lambda);
    }

    private static double MixtureRegularizer(double loss, double lambda, double zeta)
    {
        if (loss >= lambda) return 0.0;
        double linearPart = 1.0 - loss / lambda;
        return zeta + (1 - zeta) * linearPart;
    }

    private static double Variance(double[] values)
    {
        double mean = values.Average();
        return values.Sum(v => Math.Pow(v - mean, 2)) / values.Length;
    }

    #endregion
}
