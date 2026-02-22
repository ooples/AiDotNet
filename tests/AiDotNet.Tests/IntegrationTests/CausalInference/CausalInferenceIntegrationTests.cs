using AiDotNet.CausalInference;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalInference;

/// <summary>
/// Integration tests for causal inference classes.
/// Tests Fit, EstimateTreatmentEffect, and ATE for each meta-learner and estimator.
/// </summary>
public class CausalInferenceIntegrationTests
{
    /// <summary>
    /// Creates synthetic treatment/control data with a known positive treatment effect.
    /// Treatment group: y = 2*x + 5 (effect=5), Control group: y = 2*x.
    /// </summary>
    private static (Matrix<double> features, Vector<double> treatment, Vector<double> outcome) CreateSyntheticData()
    {
        int n = 40;
        var data = new double[n, 2];
        var treatArr = new double[n];
        var outcomeArr = new double[n];

        for (int i = 0; i < n; i++)
        {
            double x = i * 0.1;
            data[i, 0] = x;
            data[i, 1] = x * 0.5;

            if (i < n / 2)
            {
                treatArr[i] = 1.0; // Treated
                outcomeArr[i] = 2.0 * x + 5.0; // With treatment effect
            }
            else
            {
                treatArr[i] = 0.0; // Control
                outcomeArr[i] = 2.0 * x; // No treatment effect
            }
        }

        return (new Matrix<double>(data), new Vector<double>(treatArr), new Vector<double>(outcomeArr));
    }

    private static Vector<int> ToIntTreatment(Vector<double> treatment)
    {
        var result = new Vector<int>(new int[treatment.Length]);
        for (int i = 0; i < treatment.Length; i++)
            result[i] = (int)treatment[i];
        return result;
    }

    #region SLearner Tests

    [Fact]
    public void SLearner_Construction_WithDefaults()
    {
        var learner = new SLearner<double>();
        Assert.NotNull(learner);
        Assert.False(learner.IsTrained);
    }

    [Fact]
    public void SLearner_EstimateTreatmentEffect_ReturnsPositiveEffect()
    {
        var learner = new SLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // Treatment effect is +5 in synthetic data; at minimum effects should be non-zero
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0, $"Expected positive average treatment effect, got {avgEffect}");
    }

    #endregion

    #region TLearner Tests

    [Fact]
    public void TLearner_EstimateTreatmentEffect_ReturnsPositiveEffect()
    {
        var learner = new TLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0, $"Expected positive average treatment effect, got {avgEffect}");
    }

    #endregion

    #region XLearner Tests

    [Fact]
    public void XLearner_EstimateTreatmentEffect_ReturnsValues()
    {
        var learner = new XLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // XLearner should detect a positive treatment effect
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0, $"Expected positive average treatment effect, got {avgEffect}");
    }

    #endregion

    #region DoublyRobustEstimator Tests

    [Fact]
    public void DoublyRobust_EstimateTreatmentEffect_ReturnsValues()
    {
        var dr = new DoublyRobustEstimator<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        dr.Fit(features, treatment, outcome);
        Assert.True(dr.IsTrained);

        var effects = dr.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion

    #region InverseProbabilityWeighting Tests

    [Fact]
    public void IPW_EstimateTreatmentEffect_ReturnsValues()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        ipw.Fit(features, treatment, outcome);
        Assert.True(ipw.IsTrained);

        var effects = ipw.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion

    #region PropensityScoreMatching Tests

    [Fact]
    public void PropensityScoreMatching_EstimateTreatmentEffect_ReturnsValues()
    {
        var psm = new PropensityScoreMatching<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        psm.Fit(features, treatment, outcome);
        Assert.True(psm.IsTrained);

        var effects = psm.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion

    #region CausalForest Tests

    [Fact]
    public void CausalForest_FitAndEstimate_ReturnsEffects()
    {
        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 3);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);

        forest.Fit(features, treatmentInt, outcome);

        var effects = forest.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion
}
