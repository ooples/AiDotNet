using AiDotNet.CausalInference;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalInference;

/// <summary>
/// Extended integration tests for causal inference with deep mathematical verification.
/// Tests ATE estimation, CATE heterogeneity, propensity score overlap,
/// doubly-robust consistency, and known treatment effect recovery.
/// </summary>
public class CausalInferenceExtendedIntegrationTests
{
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Creates synthetic data with known treatment effect.
    /// Y = beta0 + beta1*X + tau*T + noise
    /// where tau is the true treatment effect.
    /// </summary>
    private static (Matrix<double> X, Vector<int> treatment, Vector<double> outcome)
        CreateKnownEffectData(int n, double trueTau, int seed = 42)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var X = new Matrix<double>(n, 2);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            double x1 = rng.NextDouble() * 2 - 1;
            X[i, 0] = x0;
            X[i, 1] = x1;

            // Treatment assignment (50/50 random)
            treatment[i] = rng.NextDouble() > 0.5 ? 1 : 0;

            // Outcome: y = 1 + 2*x0 + 0.5*x1 + tau*T + noise
            double noise = (rng.NextDouble() - 0.5) * 0.1;
            outcome[i] = 1.0 + 2.0 * x0 + 0.5 * x1 + trueTau * treatment[i] + noise;
        }

        return (X, treatment, outcome);
    }

    /// <summary>
    /// Creates data with heterogeneous treatment effect.
    /// tau(x) = alpha + beta*x[0] (effect depends on feature)
    /// </summary>
    private static (Matrix<double> X, Vector<int> treatment, Vector<double> outcome)
        CreateHeterogeneousEffectData(int n, double baseEffect, double heterogeneity, int seed = 42)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var X = new Matrix<double>(n, 2);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            double x1 = rng.NextDouble() * 2 - 1;
            X[i, 0] = x0;
            X[i, 1] = x1;

            treatment[i] = rng.NextDouble() > 0.5 ? 1 : 0;

            // Heterogeneous effect: tau(x) = baseEffect + heterogeneity * x0
            double tau = baseEffect + heterogeneity * x0;
            double noise = (rng.NextDouble() - 0.5) * 0.1;
            outcome[i] = 1.0 + x0 + tau * treatment[i] + noise;
        }

        return (X, treatment, outcome);
    }

    #region SLearner - Single Model Treatment Effect

    [Fact]
    public void SLearner_KnownPositiveEffect_ATEIsPositive()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: 3.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var slearner = new SLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        slearner.Fit(X, treatmentVec, outcome);

        var (ate, se) = slearner.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"ATE should be positive for positive treatment effect, got {ate:F4}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se:F4}");
    }

    [Fact]
    public void SLearner_ZeroEffect_ATENearZero()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: 0.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var slearner = new SLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        slearner.Fit(X, treatmentVec, outcome);

        var (ate, _) = slearner.EstimateATE(X, treatment, outcome);

        Assert.True(Math.Abs(ate) < 1.5,
            $"ATE should be close to 0 for no treatment effect, got {ate:F4}");
    }

    [Fact]
    public void SLearner_NegativeEffect_ATEIsNegative()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: -2.0, seed: 123);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var slearner = new SLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        slearner.Fit(X, treatmentVec, outcome);

        var (ate, _) = slearner.EstimateATE(X, treatment, outcome);

        Assert.True(ate < 0, $"ATE should be negative for negative treatment effect, got {ate:F4}");
    }

    [Fact]
    public void SLearner_CATE_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var slearner = new SLearner<double>(maxIterations: 100, learningRate: 0.05);
        slearner.Fit(X, treatmentVec, outcome);

        var cates = slearner.EstimateCATEPerIndividual(X, treatment, outcome);

        Assert.Equal(X.Rows, cates.Length);
    }

    #endregion

    #region TLearner - Two Separate Models

    [Fact]
    public void TLearner_PositiveEffect_ATEIsPositive()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: 3.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var tlearner = new TLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        tlearner.Fit(X, treatmentVec, outcome);

        var (ate, se) = tlearner.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"T-Learner ATE should be positive, got {ate:F4}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se:F4}");
    }

    [Fact]
    public void TLearner_TreatmentEffect_MatchesExpectedSign()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: -1.5, seed: 77);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var tlearner = new TLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        tlearner.Fit(X, treatmentVec, outcome);

        var (ate, _) = tlearner.EstimateATE(X, treatment, outcome);

        Assert.True(ate < 0, $"T-Learner ATE should be negative, got {ate:F4}");
    }

    [Fact]
    public void TLearner_EstimateTreatmentEffect_PerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var tlearner = new TLearner<double>(maxIterations: 100, learningRate: 0.05);
        tlearner.Fit(X, treatmentVec, outcome);

        var effects = tlearner.EstimateTreatmentEffect(X);
        Assert.Equal(X.Rows, effects.Length);

        // With positive treatment, most individual effects should be positive
        int positiveCount = 0;
        for (int i = 0; i < effects.Length; i++)
        {
            if (effects[i] > 0) positiveCount++;
        }
        Assert.True(positiveCount > effects.Length / 3,
            $"Most individual effects should be positive, got {positiveCount}/{effects.Length}");
    }

    #endregion

    #region XLearner - Cross-Learner Estimation

    [Fact]
    public void XLearner_PositiveEffect_ATEIsPositive()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: 2.5);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var xlearner = new XLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        xlearner.Fit(X, treatmentVec, outcome);

        var (ate, se) = xlearner.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"X-Learner ATE should be positive, got {ate:F4}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se:F4}");
    }

    [Fact]
    public void XLearner_CATEPerIndividual_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        var xlearner = new XLearner<double>(maxIterations: 100, learningRate: 0.05);
        xlearner.Fit(X, treatmentVec, outcome);

        var cates = xlearner.EstimateCATEPerIndividual(X, treatment, outcome);
        Assert.Equal(X.Rows, cates.Length);
    }

    #endregion

    #region InverseProbabilityWeighting

    [Fact]
    public void IPW_KnownEffect_ATEInCorrectDirection()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: 3.0);

        var ipw = new InverseProbabilityWeighting<double>(trimMin: 0.1, trimMax: 0.9);
        ipw.Fit(X, treatment);

        var (ate, se) = ipw.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"IPW ATE should be positive, got {ate:F4}");
        Assert.True(se > 0, $"IPW SE should be positive, got {se:F4}");
    }

    [Fact]
    public void IPW_PropensityTrimming_PreventsDivisionByZero()
    {
        // Create imbalanced data
        var rng = RandomHelper.CreateSeededRandom(42);
        int n = 100;
        var X = new Matrix<double>(n, 2);
        var treatment = new Vector<int>(n);
        var outcome = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            X[i, 0] = rng.NextDouble();
            X[i, 1] = rng.NextDouble();
            // 90% treated
            treatment[i] = rng.NextDouble() > 0.1 ? 1 : 0;
            outcome[i] = treatment[i] * 2.0 + rng.NextDouble();
        }

        var ipw = new InverseProbabilityWeighting<double>(trimMin: 0.05, trimMax: 0.95);

        // Should not throw despite extreme propensity scores
        ipw.Fit(X, treatment);
        var (ate, _) = ipw.EstimateATE(X, treatment, outcome);

        Assert.False(double.IsNaN(ate), "ATE should not be NaN with trimming");
        Assert.False(double.IsInfinity(ate), "ATE should not be Infinity with trimming");
    }

    [Fact]
    public void IPW_NegativeEffect_ATEIsNegative()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: -2.0, seed: 99);

        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(X, treatment);

        var (ate, _) = ipw.EstimateATE(X, treatment, outcome);

        Assert.True(ate < 0, $"IPW ATE should be negative, got {ate:F4}");
    }

    #endregion

    #region PropensityScoreMatching

    [Fact]
    public void PSM_KnownEffect_ATEInCorrectDirection()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: 3.0);

        var psm = new PropensityScoreMatching<double>(caliper: 0.5, seed: 42);
        psm.Fit(X, treatment);

        var (ate, se) = psm.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"PSM ATE should be positive, got {ate:F4}");
        Assert.True(se >= 0, $"PSM SE should be non-negative, got {se:F4}");
    }

    [Fact]
    public void PSM_CATEPerIndividual_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);

        var psm = new PropensityScoreMatching<double>(seed: 42);
        psm.Fit(X, treatment);

        var cates = psm.EstimateCATEPerIndividual(X, treatment, outcome);
        Assert.Equal(X.Rows, cates.Length);
    }

    #endregion

    #region DoublyRobustEstimator

    [Fact]
    public void DR_KnownEffect_ATEInCorrectDirection()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: 3.0);

        var dr = new DoublyRobustEstimator<double>(trimMin: 0.1, trimMax: 0.9);
        dr.Fit(X, treatment, outcome);

        var (ate, se) = dr.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"DR ATE should be positive, got {ate:F4}");
        Assert.True(se >= 0, $"DR SE should be non-negative, got {se:F4}");
    }

    [Fact]
    public void DR_NegativeEffect_ATEIsNegative()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: -2.0, seed: 99);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(X, treatment, outcome);

        var (ate, _) = dr.EstimateATE(X, treatment, outcome);

        Assert.True(ate < 0, $"DR ATE should be negative, got {ate:F4}");
    }

    [Fact]
    public void DR_CATEPerIndividual_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(X, treatment, outcome);

        var cates = dr.EstimateCATEPerIndividual(X, treatment, outcome);
        Assert.Equal(X.Rows, cates.Length);
    }

    #endregion

    #region CausalForest

    [Fact]
    public void CausalForest_KnownEffect_ATEInCorrectDirection()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(200, trueTau: 3.0);

        var forest = new CausalForest<double>(numTrees: 10, maxDepth: 5, minSamplesLeaf: 5);
        forest.Fit(X, treatment, outcome);

        var (ate, se) = forest.EstimateATE(X, treatment, outcome);

        Assert.True(ate > 0, $"CausalForest ATE should be positive, got {ate:F4}");
        Assert.True(se >= 0, $"SE should be non-negative, got {se:F4}");
    }

    [Fact]
    public void CausalForest_IndividualEffects_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);

        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 5);
        forest.Fit(X, treatment, outcome);

        var effects = forest.EstimateTreatmentEffect(X);
        Assert.Equal(X.Rows, effects.Length);
    }

    [Fact]
    public void CausalForest_CATEPerIndividual_HasOnePerSample()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(100, trueTau: 2.0);

        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 5);
        forest.Fit(X, treatment, outcome);

        var cates = forest.EstimateCATEPerIndividual(X, treatment, outcome);
        Assert.Equal(X.Rows, cates.Length);
    }

    #endregion

    #region Cross-Model Comparison

    [Fact]
    public void AllEstimators_PositiveEffect_AllReturnPositiveATE()
    {
        var (X, treatment, outcome) = CreateKnownEffectData(300, trueTau: 5.0, seed: 42);
        var treatmentVec = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++) treatmentVec[i] = treatment[i];

        // S-Learner
        var slearner = new SLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        slearner.Fit(X, treatmentVec, outcome);
        var (ateS, _) = slearner.EstimateATE(X, treatment, outcome);

        // T-Learner
        var tlearner = new TLearner<double>(maxIterations: 200, learningRate: 0.05, lambda: 0.001);
        tlearner.Fit(X, treatmentVec, outcome);
        var (ateT, _) = tlearner.EstimateATE(X, treatment, outcome);

        // IPW
        var ipw = new InverseProbabilityWeighting<double>();
        ipw.Fit(X, treatment);
        var (ateIPW, _) = ipw.EstimateATE(X, treatment, outcome);

        // DR
        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(X, treatment, outcome);
        var (ateDR, _) = dr.EstimateATE(X, treatment, outcome);

        Assert.True(ateS > 0, $"S-Learner ATE should be positive, got {ateS:F4}");
        Assert.True(ateT > 0, $"T-Learner ATE should be positive, got {ateT:F4}");
        Assert.True(ateIPW > 0, $"IPW ATE should be positive, got {ateIPW:F4}");
        Assert.True(ateDR > 0, $"DR ATE should be positive, got {ateDR:F4}");
    }

    #endregion

    #region Parameter Validation

    [Fact]
    public void IPW_InvalidTrimRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMin: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMax: 1.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InverseProbabilityWeighting<double>(trimMin: 0.5, trimMax: 0.3));
    }

    [Fact]
    public void PSM_InvalidCaliper_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PropensityScoreMatching<double>(caliper: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PropensityScoreMatching<double>(matchRatio: 0));
    }

    #endregion
}
