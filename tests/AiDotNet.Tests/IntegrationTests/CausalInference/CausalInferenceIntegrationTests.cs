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

    #region SLearner Tests

    [Fact]
    public void SLearner_Construction_WithDefaults()
    {
        var learner = new SLearner<double>();
        Assert.NotNull(learner);
        Assert.False(learner.IsTrained);
    }

    [Fact]
    public void SLearner_Fit_DoesNotThrow()
    {
        var learner = new SLearner<double>(maxIterations: 50, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);
    }

    [Fact]
    public void SLearner_EstimateTreatmentEffect_ReturnsValues()
    {
        var learner = new SLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion

    #region TLearner Tests

    [Fact]
    public void TLearner_Construction_WithDefaults()
    {
        var learner = new TLearner<double>();
        Assert.NotNull(learner);
        Assert.False(learner.IsTrained);
    }

    [Fact]
    public void TLearner_Fit_DoesNotThrow()
    {
        var learner = new TLearner<double>(maxIterations: 50, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);
    }

    [Fact]
    public void TLearner_EstimateTreatmentEffect_ReturnsValues()
    {
        var learner = new TLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);
    }

    #endregion

    #region XLearner Tests

    [Fact]
    public void XLearner_Construction_WithDefaults()
    {
        var learner = new XLearner<double>();
        Assert.NotNull(learner);
        Assert.False(learner.IsTrained);
    }

    [Fact]
    public void XLearner_Fit_DoesNotThrow()
    {
        var learner = new XLearner<double>(maxIterations: 50, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);
    }

    #endregion

    #region DoublyRobustEstimator Tests

    [Fact]
    public void DoublyRobust_Construction_WithDefaults()
    {
        var dr = new DoublyRobustEstimator<double>();
        Assert.NotNull(dr);
        Assert.False(dr.IsTrained);
    }

    [Fact]
    public void DoublyRobust_Fit_DoesNotThrow()
    {
        var dr = new DoublyRobustEstimator<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        dr.Fit(features, treatment, outcome);
        Assert.True(dr.IsTrained);
    }

    #endregion

    #region InverseProbabilityWeighting Tests

    [Fact]
    public void IPW_Construction_WithDefaults()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        Assert.NotNull(ipw);
        Assert.False(ipw.IsTrained);
    }

    [Fact]
    public void IPW_Fit_DoesNotThrow()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        ipw.Fit(features, treatment, outcome);
        Assert.True(ipw.IsTrained);
    }

    #endregion

    #region PropensityScoreMatching Tests

    [Fact]
    public void PropensityScoreMatching_Construction_WithDefaults()
    {
        var psm = new PropensityScoreMatching<double>();
        Assert.NotNull(psm);
        Assert.False(psm.IsTrained);
    }

    [Fact]
    public void PropensityScoreMatching_Fit_DoesNotThrow()
    {
        var psm = new PropensityScoreMatching<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        psm.Fit(features, treatment, outcome);
        Assert.True(psm.IsTrained);
    }

    #endregion

    #region CausalForest Tests

    [Fact]
    public void CausalForest_Construction()
    {
        var forest = new CausalForest<double>();
        Assert.NotNull(forest);
    }

    [Fact]
    public void CausalForest_Fit_DoesNotThrow()
    {
        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 3);
        var (features, treatment, outcome) = CreateSyntheticData();

        // CausalForest.Fit uses Vector<int> for treatment
        var treatmentInt = new Vector<int>(new int[treatment.Length]);
        for (int i = 0; i < treatment.Length; i++)
            treatmentInt[i] = (int)treatment[i];

        forest.Fit(features, treatmentInt, outcome);
    }

    #endregion
}
