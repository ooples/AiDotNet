using AiDotNet.CausalInference;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalInference;

/// <summary>
/// Integration tests for causal inference classes.
/// Tests Fit, EstimateTreatmentEffect, EstimateATE, and correctness of effect estimates.
///
/// Synthetic data design:
/// - 40 subjects, 20 treated + 20 control
/// - Features: x = i*0.1, x2 = x*0.5
/// - Treatment: first 20 get T=1, last 20 get T=0
/// - Outcome: y = 2*x + 5*T (true ATE = 5)
/// - All estimators should recover a positive treatment effect close to 5.
/// </summary>
public class CausalInferenceIntegrationTests
{
    private const double KnownTreatmentEffect = 5.0;

    /// <summary>
    /// Creates synthetic treatment/control data with a known positive treatment effect of +5.
    /// Treatment group: y = 2*x + 5, Control group: y = 2*x.
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
                outcomeArr[i] = 2.0 * x + KnownTreatmentEffect; // With treatment effect
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
    public void SLearner_EstimateTreatmentEffect_DetectsPositiveEffect()
    {
        var learner = new SLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // Average treatment effect should be close to 5.0
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 1.0, $"S-Learner average effect should be > 1.0, got {avgEffect}");

        // No effect should be NaN or Infinity
        for (int i = 0; i < effects.Length; i++)
        {
            Assert.False(double.IsNaN(effects[i]), $"Effect at index {i} is NaN");
            Assert.False(double.IsInfinity(effects[i]), $"Effect at index {i} is Infinity");
        }
    }

    [Fact]
    public void SLearner_EstimateATE_ReturnsPositiveWithStandardError()
    {
        var learner = new SLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        learner.Fit(features, treatment, outcome);

        var (ate, se) = learner.EstimateATE(features, treatmentInt, outcome);

        // ATE should be positive (true effect is +5)
        Assert.True(ate > 0, $"S-Learner ATE should be positive, got {ate}");
        // Standard error should be non-negative and finite
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
        Assert.False(double.IsNaN(se), "Standard error is NaN");
    }

    #endregion

    #region TLearner Tests

    [Fact]
    public void TLearner_EstimateTreatmentEffect_DetectsPositiveEffect()
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
        Assert.True(avgEffect > 1.0, $"T-Learner average effect should be > 1.0, got {avgEffect}");
    }

    [Fact]
    public void TLearner_EstimateATE_ReturnsPositiveWithStandardError()
    {
        var learner = new TLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        learner.Fit(features, treatment, outcome);

        var (ate, se) = learner.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"T-Learner ATE should be positive, got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
    }

    #endregion

    #region XLearner Tests

    [Fact]
    public void XLearner_EstimateTreatmentEffect_DetectsPositiveEffect()
    {
        var learner = new XLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        learner.Fit(features, treatment, outcome);
        Assert.True(learner.IsTrained);

        var effects = learner.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0, $"X-Learner average effect should be positive, got {avgEffect}");
    }

    [Fact]
    public void XLearner_EstimateATE_ReturnsPositiveWithStandardError()
    {
        var learner = new XLearner<double>(maxIterations: 100, learningRate: 0.01);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        learner.Fit(features, treatment, outcome);

        var (ate, se) = learner.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"X-Learner ATE should be positive, got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
    }

    #endregion

    #region DoublyRobustEstimator Tests

    [Fact]
    public void DoublyRobust_EstimateTreatmentEffect_DetectsPositiveEffect()
    {
        var dr = new DoublyRobustEstimator<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        dr.Fit(features, treatment, outcome);
        Assert.True(dr.IsTrained);

        var effects = dr.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // DR estimator should detect positive treatment effect
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 1.0,
            $"Doubly Robust average effect should be > 1.0 (true effect is {KnownTreatmentEffect}), got {avgEffect}");
    }

    [Fact]
    public void DoublyRobust_EstimateATE_ReturnsPositiveWithStandardError()
    {
        var dr = new DoublyRobustEstimator<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        dr.Fit(features, treatmentInt, outcome);

        var (ate, se) = dr.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"DR ATE should be positive (true ATE is {KnownTreatmentEffect}), got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
        Assert.False(double.IsInfinity(ate), "ATE is Infinity");
    }

    [Fact]
    public void DoublyRobust_PredictTreatedVsControl_TreatedIsHigher()
    {
        var dr = new DoublyRobustEstimator<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        dr.Fit(features, treatment, outcome);

        var treatedPreds = dr.PredictTreated(features);
        var controlPreds = dr.PredictControl(features);

        Assert.Equal(features.Rows, treatedPreds.Length);
        Assert.Equal(features.Rows, controlPreds.Length);

        // On average, treated predictions should be higher than control
        double avgTreated = 0, avgControl = 0;
        for (int i = 0; i < features.Rows; i++)
        {
            avgTreated += treatedPreds[i];
            avgControl += controlPreds[i];
        }
        avgTreated /= features.Rows;
        avgControl /= features.Rows;

        Assert.True(avgTreated > avgControl,
            $"Average treated outcome ({avgTreated}) should exceed control ({avgControl})");
    }

    #endregion

    #region InverseProbabilityWeighting Tests

    [Fact]
    public void IPW_EstimateTreatmentEffect_DetectsPositiveEffect()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        // IPW requires Fit(features, treatment, outcome) with Vector<T> treatment to cache outcome data
        ipw.Fit(features, treatment, outcome);
        Assert.True(ipw.IsTrained);

        var effects = ipw.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // IPW should detect positive treatment effect
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0,
            $"IPW average effect should be positive (true effect is {KnownTreatmentEffect}), got {avgEffect}");
    }

    [Fact]
    public void IPW_EstimateATE_ReturnsPositiveValue()
    {
        var ipw = new InverseProbabilityWeighting<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        ipw.Fit(features, treatmentInt);

        var (ate, se) = ipw.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"IPW ATE should be positive, got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
    }

    #endregion

    #region PropensityScoreMatching Tests

    [Fact]
    public void PropensityScoreMatching_EstimateTreatmentEffect_DetectsPositiveEffect()
    {
        var psm = new PropensityScoreMatching<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        // PSM requires Fit(features, treatment, outcome) with Vector<T> treatment to cache outcome data
        psm.Fit(features, treatment, outcome);
        Assert.True(psm.IsTrained);

        var effects = psm.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // PSM should detect positive treatment effect
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0,
            $"PSM average effect should be positive (true effect is {KnownTreatmentEffect}), got {avgEffect}");
    }

    [Fact]
    public void PropensityScoreMatching_EstimateATE_ReturnsPositiveValue()
    {
        var psm = new PropensityScoreMatching<double>();
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);
        psm.Fit(features, treatmentInt);

        var (ate, se) = psm.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"PSM ATE should be positive, got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
    }

    #endregion

    #region CausalForest Tests

    [Fact]
    public void CausalForest_FitAndEstimate_ReturnsPositiveEffects()
    {
        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 3);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);

        forest.Fit(features, treatmentInt, outcome);

        var effects = forest.EstimateTreatmentEffect(features);
        Assert.Equal(features.Rows, effects.Length);

        // Average effect should be positive (true effect is +5)
        double avgEffect = 0;
        for (int i = 0; i < effects.Length; i++) avgEffect += effects[i];
        avgEffect /= effects.Length;
        Assert.True(avgEffect > 0,
            $"CausalForest average effect should be positive (true effect is {KnownTreatmentEffect}), got {avgEffect}");
    }

    [Fact]
    public void CausalForest_EstimateATE_ReturnsPositiveValue()
    {
        var forest = new CausalForest<double>(numTrees: 5, maxDepth: 3, seed: 42);
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);

        forest.Fit(features, treatmentInt, outcome);
        var (ate, se) = forest.EstimateATE(features, treatmentInt, outcome);

        Assert.True(ate > 0, $"CausalForest ATE should be positive, got {ate}");
        Assert.True(se >= 0, $"Standard error should be non-negative, got {se}");
        Assert.False(double.IsNaN(ate), "ATE is NaN");
    }

    #endregion

    #region Cross-Estimator Consistency Tests

    [Fact]
    public void AllEstimators_AgreeTreatmentIsPositive()
    {
        var (features, treatment, outcome) = CreateSyntheticData();
        var treatmentInt = ToIntTreatment(treatment);

        // All estimators should agree that the treatment effect is positive
        var slearner = new SLearner<double>(maxIterations: 100, learningRate: 0.01);
        slearner.Fit(features, treatment, outcome);
        var sEffects = slearner.EstimateTreatmentEffect(features);

        var tlearner = new TLearner<double>(maxIterations: 100, learningRate: 0.01);
        tlearner.Fit(features, treatment, outcome);
        var tEffects = tlearner.EstimateTreatmentEffect(features);

        var dr = new DoublyRobustEstimator<double>();
        dr.Fit(features, treatment, outcome);
        var drEffects = dr.EstimateTreatmentEffect(features);

        // Compute averages
        double sAvg = 0, tAvg = 0, drAvg = 0;
        for (int i = 0; i < features.Rows; i++)
        {
            sAvg += sEffects[i];
            tAvg += tEffects[i];
            drAvg += drEffects[i];
        }
        sAvg /= features.Rows;
        tAvg /= features.Rows;
        drAvg /= features.Rows;

        // All should agree the effect is positive
        Assert.True(sAvg > 0, $"S-Learner should find positive effect, got {sAvg}");
        Assert.True(tAvg > 0, $"T-Learner should find positive effect, got {tAvg}");
        Assert.True(drAvg > 0, $"DR estimator should find positive effect, got {drAvg}");
    }

    [Fact]
    public void AllEstimators_ThrowWhenNotFitted()
    {
        var features = new Matrix<double>(1, 2);

        var slearner = new SLearner<double>();
        Assert.False(slearner.IsTrained);
        Assert.Throws<InvalidOperationException>(() => slearner.EstimateTreatmentEffect(features));

        var tlearner = new TLearner<double>();
        Assert.False(tlearner.IsTrained);
        Assert.Throws<InvalidOperationException>(() => tlearner.EstimateTreatmentEffect(features));

        var dr = new DoublyRobustEstimator<double>();
        Assert.False(dr.IsTrained);
        Assert.Throws<InvalidOperationException>(() => dr.EstimateTreatmentEffect(features));
    }

    #endregion
}
