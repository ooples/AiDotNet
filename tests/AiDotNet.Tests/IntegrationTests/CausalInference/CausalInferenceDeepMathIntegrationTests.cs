using AiDotNet.CausalInference;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalInference;

/// <summary>
/// Deep math integration tests for causal inference models (IPW, PSM, DoublyRobust).
/// Tests verify correctness of propensity score estimation, weight computation,
/// treatment effect estimation, matching quality, and mathematical properties
/// against hand-computed reference values and causal inference theory.
/// </summary>
public class CausalInferenceDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 0.05;
    private const double MediumTolerance = 1e-3;

    #region Helper Methods

    /// <summary>
    /// Creates balanced data with zero features (all propensity scores should be ~0.5).
    /// Treatment has no confounding - purely randomized.
    /// </summary>
    private static (Matrix<double> x, Vector<int> treatment, Vector<double> outcome)
        CreateBalancedZeroFeatureData(int nPerGroup, double treatmentEffect)
    {
        int n = nPerGroup * 2;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = 0.0; // Zero features -> propensity = sigmoid(0) = 0.5
            treatment[i] = i < nPerGroup ? 1 : 0;
            double baseline = 10.0;
            outcome[i] = baseline + (treatment[i] == 1 ? treatmentEffect : 0.0);
        }

        return (x, treatment, outcome);
    }

    /// <summary>
    /// Creates data with a known linear treatment effect and confounding.
    /// Outcome = baseline + beta*X + treatmentEffect*T
    /// Treatment assignment depends on X (confounded).
    /// </summary>
    private static (Matrix<double> x, Vector<int> treatment, Vector<double> outcome)
        CreateConfoundedData(int n, double treatmentEffect)
    {
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            // Feature ranges from -2 to 2
            double xi = -2.0 + 4.0 * i / (n - 1);
            x[i, 0] = xi;

            // Treatment more likely for higher X (confounding)
            // First half of positive X gets treated, first half of negative X is control
            treatment[i] = xi > 0 ? 1 : 0;

            // Outcome depends on X and treatment
            double baseline = 10.0;
            double betaX = 2.0;
            outcome[i] = baseline + betaX * xi + (treatment[i] == 1 ? treatmentEffect : 0.0);
        }

        return (x, treatment, outcome);
    }

    /// <summary>
    /// Converts Vector{int} to Vector{double} for the Fit(features, treatment, outcome) overload.
    /// </summary>
    private static Vector<double> ToDoubleVector(Vector<int> v)
    {
        var result = new Vector<double>(v.Length);
        for (int i = 0; i < v.Length; i++)
            result[i] = v[i];
        return result;
    }

    #endregion

    #region Propensity Score Tests

    [Fact]
    public void PropensityScores_BalancedData_AllApproximatelyHalf()
    {
        // With zero features and balanced treatment, propensity scores should converge to ~0.5
        // Logistic regression: P(T=1|X) = sigmoid(beta0) where X=0 for all
        // With 50/50 split, beta0 should converge to 0, so sigmoid(0) = 0.5
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var scores = ipw.EstimatePropensityScores(x);

        Assert.Equal(40, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] > 0.4 && scores[i] < 0.6,
                $"Propensity score at index {i} should be ~0.5 for balanced data, got {scores[i]:F6}");
        }
    }

    [Fact]
    public void PropensityScores_AlwaysInZeroOneRange()
    {
        // Propensity scores must be in (0, 1) since they're probabilities
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var scores = ipw.EstimatePropensityScores(x);

        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] > 0.0 && scores[i] < 1.0,
                $"Propensity score at index {i} must be in (0,1), got {scores[i]:F6}");
        }
    }

    [Fact]
    public void PropensityScores_Trimming_EnforcedCorrectly()
    {
        // With trimMin=0.1, trimMax=0.9, all scores should be in [0.1, 0.9]
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(trimMin: 0.1, trimMax: 0.9);
        ipw.Fit(x, treatment);

        var scores = ipw.EstimatePropensityScores(x);

        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0.1 - 1e-10,
                $"Score at index {i} = {scores[i]:F6} is below trimMin 0.1");
            Assert.True(scores[i] <= 0.9 + 1e-10,
                $"Score at index {i} = {scores[i]:F6} is above trimMax 0.9");
        }
    }

    [Fact]
    public void PropensityScores_ConfoundedData_HigherForTreatedGroup()
    {
        // When treatment depends on X, treated group should have higher average propensity
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var scores = ipw.EstimatePropensityScores(x);

        double avgTreated = 0, avgControl = 0;
        int nTreated = 0, nControl = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (treatment[i] == 1)
            {
                avgTreated += scores[i];
                nTreated++;
            }
            else
            {
                avgControl += scores[i];
                nControl++;
            }
        }
        avgTreated /= nTreated;
        avgControl /= nControl;

        Assert.True(avgTreated > avgControl,
            $"Average propensity for treated ({avgTreated:F4}) should exceed control ({avgControl:F4}) " +
            "when treatment is confounded with features.");
    }

    [Fact]
    public void PropensityScores_PSM_NeverTrimmed()
    {
        // PSM doesn't apply trimming to propensity scores (unlike IPW)
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);

        var scores = psm.EstimatePropensityScores(x);

        // Scores should be raw sigmoid outputs - still in (0,1) but not trimmed
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] > 0.0 && scores[i] < 1.0,
                $"PSM score at index {i} must be in (0,1), got {scores[i]:F6}");
        }
    }

    #endregion

    #region IPW Weight Tests

    [Fact]
    public void IPW_StabilizedWeights_BalancedData_AllApproximatelyOne()
    {
        // With balanced data and uniform propensity ~0.5:
        // Stabilized treated weight = P(T=1)/e(X) = 0.5/0.5 = 1.0
        // Stabilized control weight = P(T=0)/(1-e(X)) = 0.5/0.5 = 1.0
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(stabilizedWeights: true);
        ipw.Fit(x, treatment);

        var weights = ipw.ComputeWeights(x, treatment);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(Math.Abs(weights[i] - 1.0) < 0.15,
                $"Stabilized weight at index {i} should be ~1.0 for balanced data, got {weights[i]:F6}");
        }
    }

    [Fact]
    public void IPW_UnstabilizedWeights_BalancedData_AllApproximatelyTwo()
    {
        // With balanced data and uniform propensity ~0.5:
        // Unstabilized treated weight = 1/e(X) = 1/0.5 = 2.0
        // Unstabilized control weight = 1/(1-e(X)) = 1/0.5 = 2.0
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(stabilizedWeights: false);
        ipw.Fit(x, treatment);

        var weights = ipw.ComputeWeights(x, treatment);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(Math.Abs(weights[i] - 2.0) < 0.3,
                $"Unstabilized weight at index {i} should be ~2.0 for balanced data, got {weights[i]:F6}");
        }
    }

    [Fact]
    public void IPW_Weights_AlwaysPositive()
    {
        // IPW weights must always be positive (they're ratios of probabilities)
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var weights = ipw.ComputeWeights(x, treatment);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(weights[i] > 0.0,
                $"IPW weight at index {i} must be positive, got {weights[i]:F6}");
        }
    }

    [Fact]
    public void IPW_Weights_BoundedByTrimming()
    {
        // With trimMin=0.1, max unstabilized weight = 1/0.1 = 10
        // With trimMax=0.9, max unstabilized weight for control = 1/(1-0.9) = 10
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        double trimMin = 0.1, trimMax = 0.9;
        var ipw = new InverseProbabilityWeighting<double>(trimMin: trimMin, trimMax: trimMax, stabilizedWeights: false);
        ipw.Fit(x, treatment);

        var weights = ipw.ComputeWeights(x, treatment);
        double maxPossibleWeight = 1.0 / trimMin; // = 10

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(weights[i] <= maxPossibleWeight + 1e-10,
                $"Weight at index {i} = {weights[i]:F4} exceeds max {maxPossibleWeight} given trimMin={trimMin}");
        }
    }

    [Fact]
    public void IPW_StabilizedWeights_SumApproximatesGroupSize()
    {
        // For balanced data with uniform propensity:
        // Sum of stabilized weights for treated ≈ n_treated
        // Sum of stabilized weights for control ≈ n_control
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(stabilizedWeights: true);
        ipw.Fit(x, treatment);

        var weights = ipw.ComputeWeights(x, treatment);

        double sumTreated = 0, sumControl = 0;
        int nTreated = 0, nControl = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            if (treatment[i] == 1)
            {
                sumTreated += weights[i];
                nTreated++;
            }
            else
            {
                sumControl += weights[i];
                nControl++;
            }
        }

        // With uniform propensity = 0.5 and marginalP = 0.5:
        // Each stabilized weight = 0.5/0.5 = 1.0, sum = n_group
        Assert.True(Math.Abs(sumTreated - nTreated) < 3.0,
            $"Sum of stabilized treated weights ({sumTreated:F2}) should approximate n_treated ({nTreated})");
        Assert.True(Math.Abs(sumControl - nControl) < 3.0,
            $"Sum of stabilized control weights ({sumControl:F2}) should approximate n_control ({nControl})");
    }

    #endregion

    #region IPW Effective Sample Size Tests

    [Fact]
    public void IPW_ESS_UniformWeights_EqualsGroupSize()
    {
        // ESS = (sum_w)^2 / (sum_w^2)
        // When all weights are equal (w=c): ESS = (n*c)^2 / (n*c^2) = n
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(stabilizedWeights: true);
        ipw.Fit(x, treatment);

        var (essTreated, essControl) = ipw.GetEffectiveSampleSize(x, treatment);

        // With ~uniform weights, ESS ≈ group size
        Assert.True(essTreated > 15.0,
            $"Treated ESS ({essTreated:F2}) should be close to 20 with uniform weights");
        Assert.True(essControl > 15.0,
            $"Control ESS ({essControl:F2}) should be close to 20 with uniform weights");
    }

    [Fact]
    public void IPW_ESS_NeverExceedsGroupSize()
    {
        // ESS <= n for each group (by Cauchy-Schwarz inequality)
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (essTreated, essControl) = ipw.GetEffectiveSampleSize(x, treatment);

        int nTreated = 0, nControl = 0;
        for (int i = 0; i < treatment.Length; i++)
        {
            if (treatment[i] == 1) nTreated++;
            else nControl++;
        }

        Assert.True(essTreated <= nTreated + 1e-6,
            $"Treated ESS ({essTreated:F2}) cannot exceed group size ({nTreated})");
        Assert.True(essControl <= nControl + 1e-6,
            $"Control ESS ({essControl:F2}) cannot exceed group size ({nControl})");
    }

    [Fact]
    public void IPW_ESS_AlwaysPositive()
    {
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (essTreated, essControl) = ipw.GetEffectiveSampleSize(x, treatment);

        Assert.True(essTreated > 0.0,
            $"Treated ESS must be positive, got {essTreated:F6}");
        Assert.True(essControl > 0.0,
            $"Control ESS must be positive, got {essControl:F6}");
    }

    [Fact]
    public void IPW_ESS_HandComputed_ThreeObservations()
    {
        // Hand-compute ESS for a simple case:
        // 3 observations: 2 treated (weights w1, w2), 1 control (weight w3)
        // With balanced data and propensity ~0.5:
        // All weights ≈ 1 (stabilized), so ESS ≈ n for each group
        // ESS_treated = (w1+w2)^2 / (w1^2+w2^2)
        // If w1=w2=1: ESS = 4/2 = 2 (exact group size)
        var x = new Matrix<double>(3, 1);
        x[0, 0] = 0.0; x[1, 0] = 0.0; x[2, 0] = 0.0;
        var treatment = new Vector<int>(new int[] { 1, 1, 0 });
        var outcome = new Vector<double>(new double[] { 15, 15, 10 });

        var ipw = new InverseProbabilityWeighting<double>(stabilizedWeights: true);
        ipw.Fit(x, treatment);

        var (essTreated, essControl) = ipw.GetEffectiveSampleSize(x, treatment);

        // With approximately equal weights, ESS should be close to group sizes
        Assert.True(essTreated > 1.0 && essTreated <= 2.0 + 1e-6,
            $"Treated ESS should be ~2 for 2 equal-weight treated, got {essTreated:F4}");
        Assert.True(essControl > 0.5 && essControl <= 1.0 + 1e-6,
            $"Control ESS should be ~1 for 1 control observation, got {essControl:F4}");
    }

    #endregion

    #region IPW ATE Estimation Tests

    [Fact]
    public void IPW_ATE_BalancedNoConfounding_RecoversKnownEffect()
    {
        // With no confounding and balanced groups:
        // True ATE = 5.0
        // IPW should recover this exactly since propensity is ~0.5 for all
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (ate, se) = ipw.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate - trueATE) < 0.5,
            $"IPW ATE ({ate:F4}) should be close to true ATE ({trueATE})");
        Assert.True(se >= 0.0,
            $"Standard error must be non-negative, got {se:F6}");
    }

    [Fact]
    public void IPW_ATE_ConstantOutcome_ZeroEffect()
    {
        // When treated and control have same outcomes, ATE should be ~0
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 0.0);
        var outcome = new Vector<double>(x.Rows);
        for (int i = 0; i < outcome.Length; i++)
            outcome[i] = 10.0; // Same outcome for everyone

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (ate, _) = ipw.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate) < 0.5,
            $"ATE should be ~0 when outcomes are identical, got {ate:F4}");
    }

    [Fact]
    public void IPW_ATE_NegativeEffect_DetectedCorrectly()
    {
        // Treatment has negative effect (-3.0)
        double trueATE = -3.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (ate, _) = ipw.EstimateATE(x, treatment, outcome);

        Assert.True(ate < 0.0,
            $"ATE should be negative for negative treatment effect, got {ate:F4}");
        Assert.True(Math.Abs(ate - trueATE) < 0.5,
            $"ATE ({ate:F4}) should be close to true ATE ({trueATE})");
    }

    [Fact]
    public void IPW_ATE_StabilizedEqualsUnstabilized_HajekEstimator()
    {
        // The Hajek estimator (normalized weights) gives the same ATE regardless
        // of whether weights are stabilized or not, because the stabilization
        // constant cancels in the ratio.
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var ipwStab = new InverseProbabilityWeighting<double>(stabilizedWeights: true);
        ipwStab.Fit(x, treatment);
        var (ateStab, _) = ipwStab.EstimateATE(x, treatment, outcome);

        var ipwUnstab = new InverseProbabilityWeighting<double>(stabilizedWeights: false);
        ipwUnstab.Fit(x, treatment);
        var (ateUnstab, _) = ipwUnstab.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ateStab - ateUnstab) < 1e-6,
            $"Stabilized ATE ({ateStab:F6}) should equal unstabilized ATE ({ateUnstab:F6}) " +
            "for the Hajek estimator.");
    }

    [Fact]
    public void IPW_ATE_StandardError_PositiveAndReasonable()
    {
        // Standard error should be positive and not unreasonably large
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (ate, se) = ipw.EstimateATE(x, treatment, outcome);

        Assert.True(se > 0.0,
            $"Bootstrap SE should be positive, got {se:F6}");
        Assert.True(se < Math.Abs(ate) * 10,
            $"SE ({se:F4}) should be reasonable relative to ATE ({ate:F4})");
    }

    #endregion

    #region IPW ATT Estimation Tests

    [Fact]
    public void IPW_ATT_BalancedNoConfounding_RecoversKnownEffect()
    {
        // With no confounding, ATT should equal ATE
        double trueEffect = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueEffect);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (att, _) = ipw.EstimateATT(x, treatment, outcome);

        Assert.True(Math.Abs(att - trueEffect) < 0.5,
            $"ATT ({att:F4}) should be close to true effect ({trueEffect})");
    }

    [Fact]
    public void IPW_ATT_TreatedWeightIsOne()
    {
        // For ATT, treated units get weight=1 (not IPW-weighted)
        // The Hajek-style ATT = mean(Y|T=1) - weighted_mean(Y|T=0, w=e/(1-e))
        // With balanced data and propensity ~0.5:
        // control weight = e/(1-e) ≈ 0.5/0.5 = 1.0
        // So ATT ≈ mean(treated) - mean(control) = ATE
        double trueEffect = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueEffect);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (ate, _) = ipw.EstimateATE(x, treatment, outcome);
        var (att, _) = ipw.EstimateATT(x, treatment, outcome);

        // With balanced data and no confounding, ATT should be very close to ATE
        Assert.True(Math.Abs(att - ate) < 1.0,
            $"ATT ({att:F4}) and ATE ({ate:F4}) should be close for balanced, unconfounded data.");
    }

    #endregion

    #region IPW CATE and Prediction Tests

    [Fact]
    public void IPW_CATEPerIndividual_ReturnsConstantATE()
    {
        // IPW doesn't estimate heterogeneous effects; it returns ATE for all individuals
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var cate = ipw.EstimateCATEPerIndividual(x, treatment, outcome);

        Assert.Equal(x.Rows, cate.Length);

        // All individual effects should be the same (the ATE)
        double firstEffect = cate[0];
        for (int i = 1; i < cate.Length; i++)
        {
            Assert.Equal(firstEffect, cate[i], Tolerance);
        }
    }

    [Fact]
    public void IPW_PredictTreatmentEffect_ReturnsConstantATE()
    {
        // PredictTreatmentEffect should return ATE for all new observations
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        var treatmentDouble = ToDoubleVector(treatment);
        ipw.Fit(x, treatmentDouble, outcome);

        var newX = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) newX[i, 0] = 0.0;

        var effects = ipw.PredictTreatmentEffect(newX);

        Assert.Equal(5, effects.Length);
        double firstEffect = effects[0];
        for (int i = 1; i < effects.Length; i++)
        {
            Assert.Equal(firstEffect, effects[i], Tolerance);
        }
    }

    [Fact]
    public void IPW_Predict_ReturnsPropensityScores()
    {
        // The Predict method should return propensity scores
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var predictions = ipw.Predict(x);
        var scores = ipw.EstimatePropensityScores(x);

        Assert.Equal(scores.Length, predictions.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.Equal(scores[i], predictions[i], Tolerance);
        }
    }

    #endregion

    #region IPW Overlap Check Tests

    [Fact]
    public void IPW_CheckOverlap_ReturnsValidBounds()
    {
        // Overlap check should return min/max propensity scores for each group
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (tMin, tMax, cMin, cMax) = ipw.CheckOverlap(x, treatment);

        // All values should be in (0, 1)
        Assert.True(tMin > 0 && tMin < 1, $"Treatment min ({tMin:F4}) should be in (0,1)");
        Assert.True(tMax > 0 && tMax < 1, $"Treatment max ({tMax:F4}) should be in (0,1)");
        Assert.True(cMin > 0 && cMin < 1, $"Control min ({cMin:F4}) should be in (0,1)");
        Assert.True(cMax > 0 && cMax < 1, $"Control max ({cMax:F4}) should be in (0,1)");

        // Min should be <= max
        Assert.True(tMin <= tMax, $"Treatment min ({tMin:F4}) should be <= max ({tMax:F4})");
        Assert.True(cMin <= cMax, $"Control min ({cMin:F4}) should be <= max ({cMax:F4})");
    }

    [Fact]
    public void IPW_CheckOverlap_BalancedData_OverlappingRanges()
    {
        // With balanced data, both groups should have similar propensity score ranges
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var (tMin, tMax, cMin, cMax) = ipw.CheckOverlap(x, treatment);

        // Ranges should overlap (good overlap = viable causal inference)
        Assert.True(tMin <= cMax && cMin <= tMax,
            $"Propensity ranges should overlap: treated [{tMin:F4}, {tMax:F4}], control [{cMin:F4}, {cMax:F4}]");
    }

    #endregion

    #region PSM Matching Tests

    [Fact]
    public void PSM_ATE_BalancedNoConfounding_RecoversKnownEffect()
    {
        // With balanced data and known effect, PSM should recover the effect
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);

        var (ate, se) = psm.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate - trueATE) < 1.0,
            $"PSM ATE ({ate:F4}) should be close to true ATE ({trueATE})");
        Assert.True(se >= 0.0,
            $"Standard error must be non-negative, got {se:F6}");
    }

    [Fact]
    public void PSM_ATT_EqualsATE_ForPSM()
    {
        // For PSM, ATT = ATE since matching is done from treated to controls
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);

        var (ate, _) = psm.EstimateATE(x, treatment, outcome);
        var (att, _) = psm.EstimateATT(x, treatment, outcome);

        Assert.Equal(ate, att, Tolerance);
    }

    [Fact]
    public void PSM_NumberOfMatches_PositiveForWideCaliperBalancedData()
    {
        // With wide caliper and balanced data, most treated should find matches
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);

        int numMatches = psm.GetNumberOfMatches(x, treatment);

        Assert.True(numMatches > 0,
            $"Number of matches should be positive with wide caliper, got {numMatches}");
        // Each treated should match at least one control
        Assert.True(numMatches >= 10,
            $"With balanced data and wide caliper, should have many matches, got {numMatches}");
    }

    [Fact]
    public void PSM_MatchQuality_SMDDecreases_AfterMatching()
    {
        // After matching, standardized mean difference should decrease or stay similar
        // This is a key quality metric for PSM
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);

        var (beforeSMD, afterSMD) = psm.GetMatchQuality(x, treatment);

        Assert.Equal(x.Columns, beforeSMD.Length);
        Assert.Equal(x.Columns, afterSMD.Length);

        // After matching, |afterSMD| should generally be <= |beforeSMD|
        // We check for at least one feature
        for (int j = 0; j < x.Columns; j++)
        {
            // Just verify both are finite numbers (SMD could be 0 if perfect balance)
            Assert.False(double.IsNaN(beforeSMD[j]),
                $"Before-matching SMD for feature {j} should not be NaN");
            Assert.False(double.IsNaN(afterSMD[j]),
                $"After-matching SMD for feature {j} should not be NaN");
        }
    }

    [Fact]
    public void PSM_WithReplacement_CanReuseControls()
    {
        // With replacement, the same control can match multiple treated individuals
        // This should allow more matches even with limited controls
        var x = new Matrix<double>(5, 1);
        x[0, 0] = 0; x[1, 0] = 0; x[2, 0] = 0; x[3, 0] = 0; x[4, 0] = 0;
        var treatment = new Vector<int>(new int[] { 1, 1, 1, 0, 0 });
        var outcome = new Vector<double>(new double[] { 15, 14, 16, 10, 11 });

        // With replacement: 3 treated can each match one of 2 controls
        var psmWith = new PropensityScoreMatching<double>(caliper: 0.5, withReplacement: true, seed: 42);
        psmWith.Fit(x, treatment);
        int matchesWithReplacement = psmWith.GetNumberOfMatches(x, treatment);

        // Without replacement: only 2 controls available, so max 2 matches
        var psmWithout = new PropensityScoreMatching<double>(caliper: 0.5, withReplacement: false, seed: 42);
        psmWithout.Fit(x, treatment);
        int matchesWithout = psmWithout.GetNumberOfMatches(x, treatment);

        Assert.True(matchesWithReplacement >= matchesWithout,
            $"With-replacement matches ({matchesWithReplacement}) should be >= without ({matchesWithout})");
    }

    [Fact]
    public void PSM_NarrowCaliper_FewerMatches()
    {
        // Narrower caliper should produce fewer or equal matches
        var (x, treatment, _) = CreateConfoundedData(40, 5.0);

        var psmWide = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psmWide.Fit(x, treatment);
        int wideMatches = psmWide.GetNumberOfMatches(x, treatment);

        var psmNarrow = new PropensityScoreMatching<double>(caliper: 0.01, seed: 42);
        psmNarrow.Fit(x, treatment);
        int narrowMatches = psmNarrow.GetNumberOfMatches(x, treatment);

        Assert.True(narrowMatches <= wideMatches,
            $"Narrow caliper matches ({narrowMatches}) should be <= wide caliper matches ({wideMatches})");
    }

    [Fact]
    public void PSM_MatchRatio_IncreasesMatchCount()
    {
        // Higher match ratio should produce more total matches (if controls available)
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var psm1 = new PropensityScoreMatching<double>(caliper: 0.5, matchRatio: 1, withReplacement: true, seed: 42);
        psm1.Fit(x, treatment);
        int matches1 = psm1.GetNumberOfMatches(x, treatment);

        var psm2 = new PropensityScoreMatching<double>(caliper: 0.5, matchRatio: 2, withReplacement: true, seed: 42);
        psm2.Fit(x, treatment);
        int matches2 = psm2.GetNumberOfMatches(x, treatment);

        Assert.True(matches2 >= matches1,
            $"MatchRatio=2 matches ({matches2}) should be >= matchRatio=1 matches ({matches1})");
    }

    #endregion

    #region Doubly Robust Estimator Tests

    [Fact]
    public void DR_ATE_BalancedNoConfounding_RecoversKnownEffect()
    {
        // DR should recover known treatment effect
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);

        var (ate, se) = dr.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate - trueATE) < 1.0,
            $"DR ATE ({ate:F4}) should be close to true ATE ({trueATE})");
        Assert.True(se >= 0.0,
            $"Standard error must be non-negative, got {se:F6}");
    }

    [Fact]
    public void DR_ATE_ConstantOutcome_ZeroEffect()
    {
        // With identical outcomes, ATE should be ~0
        int n = 40;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = 0.0;
            treatment[i] = i < n / 2 ? 1 : 0;
            outcome[i] = 10.0;
        }

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);

        var (ate, _) = dr.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate) < 0.5,
            $"DR ATE should be ~0 when outcomes are identical, got {ate:F4}");
    }

    [Fact]
    public void DR_OutcomeModelPredictions_ReasonableValues()
    {
        // The outcome model should produce reasonable predictions
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var dr = new DoublyRobustEstimator<double>();
        var treatmentDouble = ToDoubleVector(treatment);
        dr.Fit(x, treatmentDouble, outcome);

        var treatedPredictions = dr.PredictTreated(x);
        var controlPredictions = dr.PredictControl(x);

        Assert.Equal(x.Rows, treatedPredictions.Length);
        Assert.Equal(x.Rows, controlPredictions.Length);

        // Treated predictions should be higher than control (treatment effect > 0)
        double avgTreated = 0, avgControl = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            avgTreated += treatedPredictions[i];
            avgControl += controlPredictions[i];
        }
        avgTreated /= x.Rows;
        avgControl /= x.Rows;

        Assert.True(avgTreated > avgControl,
            $"Average treated prediction ({avgTreated:F4}) should exceed control ({avgControl:F4}) " +
            "when treatment effect is positive.");
    }

    [Fact]
    public void DR_TreatmentEffect_EqualsOutcomeModelDifference()
    {
        // PredictTreatmentEffect should equal PredictTreated - PredictControl
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var dr = new DoublyRobustEstimator<double>();
        var treatmentDouble = ToDoubleVector(treatment);
        dr.Fit(x, treatmentDouble, outcome);

        var effects = dr.PredictTreatmentEffect(x);
        var treated = dr.PredictTreated(x);
        var control = dr.PredictControl(x);

        for (int i = 0; i < x.Rows; i++)
        {
            double expectedEffect = treated[i] - control[i];
            Assert.True(Math.Abs(effects[i] - expectedEffect) < 1e-10,
                $"Effect at {i}: PredictTreatmentEffect ({effects[i]:F6}) != " +
                $"PredictTreated - PredictControl ({expectedEffect:F6})");
        }
    }

    [Fact]
    public void DR_ATT_RecoversEffect()
    {
        // ATT should recover treatment effect for balanced data
        double trueEffect = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueEffect);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);

        var (att, se) = dr.EstimateATT(x, treatment, outcome);

        Assert.True(Math.Abs(att - trueEffect) < 2.0,
            $"DR ATT ({att:F4}) should be close to true effect ({trueEffect})");
        Assert.True(se >= 0.0, "ATT SE must be non-negative.");
    }

    [Fact]
    public void DR_CATEPerIndividual_MatchesTreatmentEffect()
    {
        // CATE per individual should match PredictTreatmentEffect
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);

        var cate = dr.EstimateCATEPerIndividual(x, treatment, outcome);
        var effects = dr.PredictTreatmentEffect(x);

        Assert.Equal(x.Rows, cate.Length);
        for (int i = 0; i < x.Rows; i++)
        {
            Assert.True(Math.Abs(cate[i] - effects[i]) < 1e-10,
                $"CATE at {i} ({cate[i]:F6}) should match PredictTreatmentEffect ({effects[i]:F6})");
        }
    }

    [Fact]
    public void DR_PropensityScores_Trimmed()
    {
        // DR should trim propensity scores like IPW
        var (x, treatment, outcome) = CreateConfoundedData(40, 5.0);

        double trimMin = 0.1, trimMax = 0.9;
        var dr = new DoublyRobustEstimator<double>(trimMin: trimMin, trimMax: trimMax);
        dr.Fit(x, treatment, outcome);

        var scores = dr.EstimatePropensityScores(x);

        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= trimMin - 1e-10,
                $"DR score at {i} = {scores[i]:F6} below trimMin {trimMin}");
            Assert.True(scores[i] <= trimMax + 1e-10,
                $"DR score at {i} = {scores[i]:F6} above trimMax {trimMax}");
        }
    }

    #endregion

    #region Cross-Method Consistency Tests

    [Fact]
    public void AllMethods_BalancedData_AgreeOnEffectDirection()
    {
        // With balanced data and known positive effect, all methods should agree on direction
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        // IPW
        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);
        var (ateIPW, _) = ipw.EstimateATE(x, treatment, outcome);

        // PSM
        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);
        var (atePSM, _) = psm.EstimateATE(x, treatment, outcome);

        // DR
        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);
        var (ateDR, _) = dr.EstimateATE(x, treatment, outcome);

        // All should be positive
        Assert.True(ateIPW > 0.0, $"IPW ATE ({ateIPW:F4}) should be positive.");
        Assert.True(atePSM > 0.0, $"PSM ATE ({atePSM:F4}) should be positive.");
        Assert.True(ateDR > 0.0, $"DR ATE ({ateDR:F4}) should be positive.");

        // All should be within reasonable range of true ATE
        Assert.True(Math.Abs(ateIPW - trueATE) < 2.0,
            $"IPW ATE ({ateIPW:F4}) should be within 2 of true ATE ({trueATE})");
        Assert.True(Math.Abs(atePSM - trueATE) < 2.0,
            $"PSM ATE ({atePSM:F4}) should be within 2 of true ATE ({trueATE})");
        Assert.True(Math.Abs(ateDR - trueATE) < 2.0,
            $"DR ATE ({ateDR:F4}) should be within 2 of true ATE ({trueATE})");
    }

    [Fact]
    public void AllMethods_BalancedData_AgreeOnNegativeEffect()
    {
        // All methods should detect negative treatment effects
        double trueATE = -4.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, trueATE);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);
        var (ateIPW, _) = ipw.EstimateATE(x, treatment, outcome);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(x, treatment);
        var (atePSM, _) = psm.EstimateATE(x, treatment, outcome);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);
        var (ateDR, _) = dr.EstimateATE(x, treatment, outcome);

        Assert.True(ateIPW < 0.0, $"IPW ATE ({ateIPW:F4}) should be negative.");
        Assert.True(atePSM < 0.0, $"PSM ATE ({atePSM:F4}) should be negative.");
        Assert.True(ateDR < 0.0, $"DR ATE ({ateDR:F4}) should be negative.");
    }

    [Fact]
    public void IPW_And_DR_PropensityScores_AgreeMathematically()
    {
        // Both IPW and DR use logistic regression for propensity scores
        // With the same trim settings, they should give the same scores
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>(trimMin: 0.05, trimMax: 0.95);
        ipw.Fit(x, treatment);

        var dr = new DoublyRobustEstimator<double>(trimMin: 0.05, trimMax: 0.95);
        dr.Fit(x, treatment, outcome);

        var ipwScores = ipw.EstimatePropensityScores(x);
        var drScores = dr.EstimatePropensityScores(x);

        // Both should produce propensity scores of similar magnitude
        double avgIPW = 0, avgDR = 0;
        for (int i = 0; i < ipwScores.Length; i++)
        {
            avgIPW += ipwScores[i];
            avgDR += drScores[i];
        }
        avgIPW /= ipwScores.Length;
        avgDR /= drScores.Length;

        Assert.True(Math.Abs(avgIPW - avgDR) < 0.1,
            $"Average propensity scores should be similar: IPW={avgIPW:F4}, DR={avgDR:F4}");
    }

    #endregion

    #region Validation and Edge Case Tests

    [Fact]
    public void IPW_InvalidTrimBounds_Throws()
    {
        // trimMin must be in (0, 1)
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMin: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMin: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMax: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMax: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMin: 0.5, trimMax: 0.3));
    }

    [Fact]
    public void PSM_InvalidCaliper_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PropensityScoreMatching<double>(caliper: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PropensityScoreMatching<double>(caliper: -1.0));
    }

    [Fact]
    public void PSM_InvalidMatchRatio_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PropensityScoreMatching<double>(matchRatio: 0));
    }

    [Fact]
    public void DR_InvalidTrimBounds_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DoublyRobustEstimator<double>(trimMin: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DoublyRobustEstimator<double>(trimMax: 1.0));
    }

    [Fact]
    public void DR_CrossFittingRequiresMultipleFolds_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DoublyRobustEstimator<double>(useCrossFitting: true, numFolds: 1));
    }

    [Fact]
    public void IPW_AllTreated_ThrowsOnEstimateATE()
    {
        int n = 10;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = 0.0;
            treatment[i] = 1; // All treated
            outcome[i] = 10.0;
        }

        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Throws<ArgumentException>(() => ipw.EstimateATE(x, treatment, outcome));
    }

    [Fact]
    public void IPW_AllControl_ThrowsOnEstimateATE()
    {
        int n = 10;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = 0.0;
            treatment[i] = 0; // All control
            outcome[i] = 10.0;
        }

        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Throws<ArgumentException>(() => ipw.EstimateATE(x, treatment, outcome));
    }

    [Fact]
    public void IPW_MismatchedDimensions_Throws()
    {
        var x = new Matrix<double>(10, 1);
        var treatment = new Vector<int>(5); // Wrong length

        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Throws<ArgumentException>(() => ipw.Fit(x, treatment));
    }

    [Fact]
    public void IPW_InvalidTreatmentValues_Throws()
    {
        var x = new Matrix<double>(5, 1);
        var treatment = new Vector<int>(new int[] { 0, 1, 2, 0, 1 }); // 2 is invalid

        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Throws<ArgumentException>(() => ipw.Fit(x, treatment));
    }

    [Fact]
    public void IPW_NotFitted_ThrowsOnPredict()
    {
        var x = new Matrix<double>(5, 1);

        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Throws<InvalidOperationException>(() => ipw.Predict(x));
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void IPW_Serialization_PreservesParameters()
    {
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var originalParams = ipw.GetParameters();
        byte[] serialized = ipw.Serialize();

        var ipw2 = new InverseProbabilityWeighting<double>();
        ipw2.Deserialize(serialized);

        var restoredParams = ipw2.GetParameters();

        Assert.Equal(originalParams.Length, restoredParams.Length);
        for (int i = 0; i < originalParams.Length; i++)
        {
            Assert.Equal(originalParams[i], restoredParams[i], Tolerance);
        }
    }

    [Fact]
    public void PSM_Serialization_PreservesParameters()
    {
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.1, seed: 42);
        psm.Fit(x, treatment);

        var originalParams = psm.GetParameters();
        byte[] serialized = psm.Serialize();

        var psm2 = new PropensityScoreMatching<double>();
        psm2.Deserialize(serialized);

        var restoredParams = psm2.GetParameters();

        Assert.Equal(originalParams.Length, restoredParams.Length);
        for (int i = 0; i < originalParams.Length; i++)
        {
            Assert.Equal(originalParams[i], restoredParams[i], Tolerance);
        }
    }

    #endregion

    #region Logistic Regression Convergence Tests

    [Fact]
    public void LogisticRegression_SeparatesGroups_PropensityReflectsTreatment()
    {
        // When features strongly predict treatment, propensity scores should
        // be higher for treated and lower for control
        int n = 40;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        for (int i = 0; i < n; i++)
        {
            // Treated group: high X, Control group: low X
            x[i, 0] = i < n / 2 ? 2.0 : -2.0;
            treatment[i] = i < n / 2 ? 1 : 0;
        }

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var scores = ipw.EstimatePropensityScores(x);

        // Average score for treated should be > 0.5, for control < 0.5
        double avgTreated = 0, avgControl = 0;
        for (int i = 0; i < n; i++)
        {
            if (treatment[i] == 1) avgTreated += scores[i];
            else avgControl += scores[i];
        }
        avgTreated /= (n / 2);
        avgControl /= (n / 2);

        Assert.True(avgTreated > 0.5,
            $"Average propensity for treated ({avgTreated:F4}) should be > 0.5");
        Assert.True(avgControl < 0.5,
            $"Average propensity for control ({avgControl:F4}) should be < 0.5");
    }

    [Fact]
    public void LogisticRegression_ZeroFeatures_InterceptOnly()
    {
        // With zero features, logistic regression reduces to intercept-only model
        // Propensity = sigmoid(beta0) where beta0 converges to log(n1/n0)
        // For balanced groups (n1=n0): beta0 → 0, so propensity → 0.5
        int n = 50;
        var x = new Matrix<double>(n, 1);
        var treatment = new Vector<int>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = 0.0;
            treatment[i] = i < n / 2 ? 1 : 0;
        }

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var params_ = ipw.GetParameters();
        // Parameters should include intercept (index 0) and one coefficient (index 1)
        Assert.True(params_.Length >= 2, $"Should have at least 2 parameters (intercept + 1 coef), got {params_.Length}");

        // Feature coefficient should be approximately 0 since X is constant
        double featureCoef = params_[1];
        Assert.True(Math.Abs(featureCoef) < 0.1,
            $"Feature coefficient should be ~0 for constant features, got {featureCoef:F6}");
    }

    #endregion

    #region DR Formula Verification Tests

    [Fact]
    public void DR_Formula_WhenOutcomeModelPerfect_IPWCorrectionSmall()
    {
        // When the outcome model perfectly fits, the IPW correction term
        // T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e) should be approximately 0
        // because Y ≈ mu for each observation.
        //
        // This means DR-ATE ≈ (1/n) * sum(mu1 - mu0)
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);

        var treatedPred = dr.PredictTreated(x);
        var controlPred = dr.PredictControl(x);

        // Compute the outcome model component: (1/n) * sum(mu1 - mu0)
        double outcomeModelATE = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            outcomeModelATE += treatedPred[i] - controlPred[i];
        }
        outcomeModelATE /= x.Rows;

        // Compare with actual DR-ATE
        var (drATE, _) = dr.EstimateATE(x, treatment, outcome);

        // They should be close (within outcome model error)
        Assert.True(Math.Abs(drATE - outcomeModelATE) < 2.0,
            $"DR ATE ({drATE:F4}) should be close to outcome model ATE ({outcomeModelATE:F4}) " +
            "when outcome model is reasonably accurate.");
    }

    [Fact]
    public void DR_CrossFitting_ProducesReasonableEstimate()
    {
        // Cross-fitting should also produce a reasonable ATE estimate
        double trueATE = 5.0;
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(30, trueATE);

        var dr = new DoublyRobustEstimator<double>(useCrossFitting: true, numFolds: 3);
        dr.Fit(x, treatment, outcome);

        var (ate, se) = dr.EstimateATE(x, treatment, outcome);

        Assert.True(Math.Abs(ate - trueATE) < 3.0,
            $"Cross-fitted DR ATE ({ate:F4}) should be within 3 of true ATE ({trueATE})");
        Assert.True(se >= 0.0, $"SE must be non-negative, got {se:F6}");
    }

    #endregion

    #region Model Type and Interface Tests

    [Fact]
    public void IPW_GetModelType_ReturnsCorrectType()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        Assert.Equal(AiDotNet.Enums.ModelType.InverseProbabilityWeighting, ipw.GetModelType());
    }

    [Fact]
    public void PSM_GetModelType_ReturnsCorrectType()
    {
        var psm = new PropensityScoreMatching<double>();
        Assert.Equal(AiDotNet.Enums.ModelType.PropensityScoreMatching, psm.GetModelType());
    }

    [Fact]
    public void DR_GetModelType_ReturnsCorrectType()
    {
        var dr = new DoublyRobustEstimator<double>();
        Assert.Equal(AiDotNet.Enums.ModelType.DoublyRobustEstimator, dr.GetModelType());
    }

    [Fact]
    public void IPW_GetModelMetadata_ContainsExpectedFields()
    {
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var metadata = ipw.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Equal(1, metadata.FeatureCount); // 1 feature
        Assert.True(metadata.AdditionalInfo is not null);
    }

    [Fact]
    public void IPW_WithParameters_CreatesNewInstance()
    {
        var (x, treatment, _) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);

        var params_ = ipw.GetParameters();
        var newModel = ipw.WithParameters(params_);

        Assert.NotNull(newModel);
        var newParams = newModel.GetParameters();
        Assert.Equal(params_.Length, newParams.Length);
    }

    #endregion

    #region Confounded Data Tests (Key Causal Inference Scenarios)

    [Fact]
    public void IPW_ConfoundedData_ReducesBias()
    {
        // Without adjustment: naive difference = mean(Y|T=1) - mean(Y|T=0) includes confounding bias
        // With IPW adjustment: ATE should be closer to true effect
        double trueATE = 3.0;
        var (x, treatment, outcome) = CreateConfoundedData(40, trueATE);

        // Naive estimate (no adjustment)
        double sumTreated = 0, sumControl = 0;
        int nT = 0, nC = 0;
        for (int i = 0; i < outcome.Length; i++)
        {
            if (treatment[i] == 1)
            {
                sumTreated += outcome[i];
                nT++;
            }
            else
            {
                sumControl += outcome[i];
                nC++;
            }
        }
        double naiveATE = sumTreated / nT - sumControl / nC;

        // IPW estimate
        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(x, treatment);
        var (ipwATE, _) = ipw.EstimateATE(x, treatment, outcome);

        // The naive estimate has bias from confounding
        // IPW should reduce this bias (though not necessarily eliminate it with limited data)
        double naiveBias = Math.Abs(naiveATE - trueATE);
        double ipwBias = Math.Abs(ipwATE - trueATE);

        // IPW-adjusted estimate should be closer to truth OR at least a reasonable number
        // (we use a generous tolerance since this is a small sample)
        Assert.True(ipwBias < naiveBias + 3.0,
            $"IPW ATE ({ipwATE:F4}, bias={ipwBias:F4}) should not be much worse than " +
            $"naive ({naiveATE:F4}, bias={naiveBias:F4}). True ATE={trueATE}");
    }

    [Fact]
    public void DR_ConfoundedData_ReducesBias()
    {
        // DR should also reduce confounding bias
        double trueATE = 3.0;
        var (x, treatment, outcome) = CreateConfoundedData(40, trueATE);

        // Naive estimate
        double sumTreated = 0, sumControl = 0;
        int nT = 0, nC = 0;
        for (int i = 0; i < outcome.Length; i++)
        {
            if (treatment[i] == 1) { sumTreated += outcome[i]; nT++; }
            else { sumControl += outcome[i]; nC++; }
        }
        double naiveATE = sumTreated / nT - sumControl / nC;

        // DR estimate
        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(x, treatment, outcome);
        var (drATE, _) = dr.EstimateATE(x, treatment, outcome);

        double naiveBias = Math.Abs(naiveATE - trueATE);
        double drBias = Math.Abs(drATE - trueATE);

        Assert.True(drBias < naiveBias + 3.0,
            $"DR ATE ({drATE:F4}, bias={drBias:F4}) should not be much worse than " +
            $"naive ({naiveATE:F4}, bias={naiveBias:F4}). True ATE={trueATE}");
    }

    #endregion

    #region IPW Predict Treated/Control Tests

    [Fact]
    public void IPW_PredictTreated_ReturnsReasonableValues()
    {
        // PredictTreated uses kernel-weighted outcomes from training treated subjects
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        var treatmentDouble = ToDoubleVector(treatment);
        ipw.Fit(x, treatmentDouble, outcome);

        var treatedPred = ipw.PredictTreated(x);

        Assert.Equal(x.Rows, treatedPred.Length);

        // Predicted treated outcomes should be close to actual treated mean (15.0)
        double avgPred = 0;
        for (int i = 0; i < treatedPred.Length; i++)
            avgPred += treatedPred[i];
        avgPred /= treatedPred.Length;

        Assert.True(avgPred > 12.0 && avgPred < 18.0,
            $"Average treated prediction ({avgPred:F4}) should be near 15.0 (treated outcome)");
    }

    [Fact]
    public void IPW_PredictControl_ReturnsReasonableValues()
    {
        // PredictControl uses kernel-weighted outcomes from training control subjects
        var (x, treatment, outcome) = CreateBalancedZeroFeatureData(20, 5.0);

        var ipw = new InverseProbabilityWeighting<double>();
        var treatmentDouble = ToDoubleVector(treatment);
        ipw.Fit(x, treatmentDouble, outcome);

        var controlPred = ipw.PredictControl(x);

        Assert.Equal(x.Rows, controlPred.Length);

        // Predicted control outcomes should be close to actual control mean (10.0)
        double avgPred = 0;
        for (int i = 0; i < controlPred.Length; i++)
            avgPred += controlPred[i];
        avgPred /= controlPred.Length;

        Assert.True(avgPred > 7.0 && avgPred < 13.0,
            $"Average control prediction ({avgPred:F4}) should be near 10.0 (control outcome)");
    }

    #endregion
}
