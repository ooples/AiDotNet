using System;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Interpretability;

/// <summary>
/// Deep math-correctness integration tests for the Interpretability module.
/// Verifies fairness metrics helper computations (positive rate, TPR, FPR, precision),
/// bias detectors (Demographic Parity, Disparate Impact, Equal Opportunity),
/// and FairnessMetrics data class against hand-computed expected values.
/// </summary>
public class InterpretabilityDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region InterpretabilityMetricsHelper - PositiveRate Tests

    [Fact]
    public void PositiveRate_AllPositive_ReturnsOne()
    {
        // predictions = [1, 1, 1] => 3 positive out of 3 => rate = 1.0
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void PositiveRate_AllNegative_ReturnsZero()
    {
        // predictions = [0, 0, 0] => 0 positive => rate = 0.0
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void PositiveRate_MixedPredictions_ExactComputation()
    {
        // predictions = [1, 0, 1, 0, 1] => 3/5 = 0.6
        // threshold is 0.5, so >= 0.5 is positive
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0, 1.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);
        Assert.Equal(0.6, result, Tolerance);
    }

    [Fact]
    public void PositiveRate_ThresholdAt05()
    {
        // predictions = [0.5, 0.4, 0.6, 0.3]
        // >= 0.5: {0.5, 0.6} => 2 positive out of 4 => rate = 0.5
        var predictions = new Vector<double>(new[] { 0.5, 0.4, 0.6, 0.3 });
        var result = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void PositiveRate_Empty_ReturnsZero()
    {
        var predictions = new Vector<double>(0);
        var result = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region InterpretabilityMetricsHelper - TruePositiveRate Tests

    [Fact]
    public void TPR_PerfectPredictions_ReturnsOne()
    {
        // pred = [1, 0, 1], actual = [1, 0, 1]
        // TP = 2, FN = 0, actual positives = 2
        // TPR = 2/2 = 1.0
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void TPR_AllFalseNegatives_ReturnsZero()
    {
        // pred = [0, 0, 0], actual = [1, 1, 1]
        // TP = 0, FN = 3, actual positives = 3
        // TPR = 0/3 = 0.0
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TPR_HandComputed()
    {
        // pred = [1, 0, 1, 1, 0], actual = [1, 1, 0, 1, 0]
        // Actual positives at: 0, 1, 3
        // At index 0: pred=1, actual=1 => TP
        // At index 1: pred=0, actual=1 => FN
        // At index 3: pred=1, actual=1 => TP
        // TP = 2, actual positives = 3
        // TPR = 2/3
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actuals);
        Assert.Equal(2.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void TPR_NoActualPositives_ReturnsZero()
    {
        // pred = [1, 1], actual = [0, 0]
        // No actual positives => TPR = 0 (by definition)
        var predictions = new Vector<double>(new[] { 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 0.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region InterpretabilityMetricsHelper - FalsePositiveRate Tests

    [Fact]
    public void FPR_NoFalsePositives_ReturnsZero()
    {
        // pred = [1, 0, 0], actual = [1, 0, 0]
        // Actual negatives: 1, 2. Both predicted 0 (correct) => FP = 0
        // FPR = 0/2 = 0.0
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void FPR_AllFalsePositives_ReturnsOne()
    {
        // pred = [1, 1, 1], actual = [0, 0, 0]
        // All actual negatives predicted as positive => FP = 3, TN = 0
        // FPR = 3/3 = 1.0
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void FPR_HandComputed()
    {
        // pred = [1, 0, 1, 1, 0], actual = [1, 0, 0, 1, 0]
        // Actual negatives at: 1, 2, 4
        // At index 1: pred=0 => TN
        // At index 2: pred=1 => FP
        // At index 4: pred=0 => TN
        // FP = 1, actual negatives = 3
        // FPR = 1/3
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actuals);
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    #endregion

    #region InterpretabilityMetricsHelper - Precision Tests

    [Fact]
    public void Precision_PerfectPredictions_ReturnsOne()
    {
        // pred = [1, 0, 1], actual = [1, 0, 1]
        // Predicted positives: 0, 2 => both are TP
        // Precision = 2/2 = 1.0
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Precision_AllFalsePositives_ReturnsZero()
    {
        // pred = [1, 1], actual = [0, 0]
        // All predicted positives are wrong => TP=0, FP=2
        // Precision = 0/2 = 0.0
        var predictions = new Vector<double>(new[] { 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 0.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Precision_HandComputed()
    {
        // pred = [1, 1, 1, 0, 0], actual = [1, 0, 1, 1, 0]
        // Predicted positives at: 0, 1, 2
        // At index 0: actual=1 => TP
        // At index 1: actual=0 => FP
        // At index 2: actual=1 => TP
        // TP = 2, predicted positives = 3
        // Precision = 2/3
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actuals);
        Assert.Equal(2.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void Precision_NoPredictedPositives_ReturnsZero()
    {
        // pred = [0, 0], actual = [1, 0]
        // No predicted positives => Precision = 0
        var predictions = new Vector<double>(new[] { 0.0, 0.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0 });
        var result = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region InterpretabilityMetricsHelper - GetUniqueGroups Tests

    [Fact]
    public void GetUniqueGroups_IdentifiesTwoGroups()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0 });
        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);
        Assert.Equal(2, groups.Count);
    }

    [Fact]
    public void GetUniqueGroups_ThreeGroups()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0, 1.0 });
        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);
        Assert.Equal(3, groups.Count);
    }

    [Fact]
    public void GetGroupIndices_CorrectIndices()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0 });
        var indices = InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, 0.0);
        Assert.Equal(3, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(2, indices);
        Assert.Contains(4, indices);
    }

    [Fact]
    public void GetSubset_ExtractsCorrectValues()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var indices = new System.Collections.Generic.List<int> { 0, 2, 4 };
        var subset = InterpretabilityMetricsHelper<double>.GetSubset(vector, indices);

        Assert.Equal(3, subset.Length);
        Assert.Equal(10.0, subset[0], Tolerance);
        Assert.Equal(30.0, subset[1], Tolerance);
        Assert.Equal(50.0, subset[2], Tolerance);
    }

    #endregion

    #region DemographicParityBiasDetector Tests

    [Fact]
    public void DemographicParity_EqualRates_NoBias()
    {
        // Group 0: predictions [1, 0] => rate = 0.5
        // Group 1: predictions [1, 0] => rate = 0.5
        // SPD = 0.5 - 0.5 = 0 < 0.1 => no bias
        var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Equal(0.0, Convert.ToDouble(result.StatisticalParityDifference), Tolerance);
    }

    [Fact]
    public void DemographicParity_UnequalRates_BiasDetected()
    {
        // Group 0: predictions [1, 1, 1, 1] => rate = 1.0
        // Group 1: predictions [0, 0, 0, 0] => rate = 0.0
        // SPD = 1.0 - 0.0 = 1.0 > 0.1 => bias detected
        var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Equal(1.0, Convert.ToDouble(result.StatisticalParityDifference), Tolerance);
    }

    [Fact]
    public void DemographicParity_ExactSPD_Computation()
    {
        // Group 0: predictions [1, 1, 0] => rate = 2/3
        // Group 1: predictions [1, 0, 0] => rate = 1/3
        // SPD = 2/3 - 1/3 = 1/3 > 0.1 => bias
        var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Equal(1.0 / 3.0, Convert.ToDouble(result.StatisticalParityDifference), Tolerance);
    }

    [Fact]
    public void DemographicParity_BelowThreshold_NoBias()
    {
        // Group 0: predictions [1, 1, 0, 0, 0] => rate = 2/5 = 0.4
        // Group 1: predictions [1, 1, 0, 0, 0] => rate = 2/5 = 0.4
        // SPD = 0.0 < 0.1 => no bias
        var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
    }

    [Fact]
    public void DemographicParity_GroupSizes_Correct()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.Equal(2, result.GroupSizes["0"]);
        Assert.Equal(3, result.GroupSizes["1"]);
    }

    #endregion

    #region DisparateImpactBiasDetector Tests

    [Fact]
    public void DisparateImpact_EqualRates_RatioIsOne()
    {
        // Both groups have rate 0.5 => ratio = 0.5/0.5 = 1.0 >= 0.8 => no bias
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Equal(1.0, Convert.ToDouble(result.DisparateImpactRatio), Tolerance);
    }

    [Fact]
    public void DisparateImpact_ExactRatio_Computation()
    {
        // Group 0: predictions [1, 1, 1, 0] => rate = 3/4 = 0.75
        // Group 1: predictions [1, 0, 0, 0] => rate = 1/4 = 0.25
        // DI ratio = min/max = 0.25/0.75 = 1/3 = 0.333 < 0.8 => bias
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Equal(1.0 / 3.0, Convert.ToDouble(result.DisparateImpactRatio), Tolerance);
    }

    [Fact]
    public void DisparateImpact_80PercentRule_PassingCase()
    {
        // Group 0: predictions [1, 1, 1, 1, 0] => rate = 4/5 = 0.8
        // Group 1: predictions [1, 1, 1, 0, 0] => rate = 3/5 = 0.6
        // DI ratio = 0.6/0.8 = 0.75 < 0.8 => bias (fails 80% rule)
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Equal(0.75, Convert.ToDouble(result.DisparateImpactRatio), Tolerance);
    }

    [Fact]
    public void DisparateImpact_AllZeroPredictions_NosBias()
    {
        // Both groups have rate 0.0 => maxRate = 0 => DI set to 1.0
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Equal(1.0, Convert.ToDouble(result.DisparateImpactRatio), Tolerance);
    }

    [Fact]
    public void DisparateImpact_WithActualLabels_ComputesTPR()
    {
        // Group 0: pred=[1,0], actual=[1,0] => TPR=1/1=1.0
        // Group 1: pred=[0,1], actual=[1,0] => TPR=0/1=0.0
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.Equal(1.0, Convert.ToDouble(result.GroupTruePositiveRates["0"]), Tolerance);
        Assert.Equal(0.0, Convert.ToDouble(result.GroupTruePositiveRates["1"]), Tolerance);
    }

    [Fact]
    public void DisparateImpact_StatisticalParityDifference_IsMaxMinusMin()
    {
        // Group 0: rate = 3/4 = 0.75
        // Group 1: rate = 1/4 = 0.25
        // SPD = 0.75 - 0.25 = 0.5
        var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.Equal(0.5, Convert.ToDouble(result.StatisticalParityDifference), Tolerance);
    }

    #endregion

    #region EqualOpportunityBiasDetector Tests

    [Fact]
    public void EqualOpportunity_EqualTPR_NoBias()
    {
        // Group 0: pred=[1,0], actual=[1,0] => TPR=1/1=1.0
        // Group 1: pred=[1,0], actual=[1,0] => TPR=1/1=1.0
        // EO diff = 1.0 - 1.0 = 0.0 < 0.1 => no bias
        var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.False(result.HasBias);
        Assert.Equal(0.0, Convert.ToDouble(result.EqualOpportunityDifference), Tolerance);
    }

    [Fact]
    public void EqualOpportunity_UnequalTPR_BiasDetected()
    {
        // Group 0: pred=[1,1], actual=[1,1] => TPR=2/2=1.0
        // Group 1: pred=[0,0], actual=[1,1] => TPR=0/2=0.0
        // EO diff = 1.0 - 0.0 = 1.0 > 0.1 => bias
        var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.True(result.HasBias);
        Assert.Equal(1.0, Convert.ToDouble(result.EqualOpportunityDifference), Tolerance);
    }

    [Fact]
    public void EqualOpportunity_ExactEODComputation()
    {
        // Group 0: pred=[1,0,1], actual=[1,1,0] => TP=1, AP=2 => TPR=1/2=0.5
        // Group 1: pred=[1,1,0], actual=[1,0,1] => TP=1, AP=2 => TPR=1/2=0.5
        // EO diff = 0.5 - 0.5 = 0.0 => no bias
        var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.False(result.HasBias);
        Assert.Equal(0.0, Convert.ToDouble(result.EqualOpportunityDifference), Tolerance);
    }

    [Fact]
    public void EqualOpportunity_WithoutActualLabels_NoBias()
    {
        // Without actual labels, EO can't be computed
        var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
    }

    [Fact]
    public void EqualOpportunity_CustomThreshold_RespectsThreshold()
    {
        // Group 0: pred=[1,0], actual=[1,1] => TPR = 1/2 = 0.5
        // Group 1: pred=[1,1], actual=[1,1] => TPR = 2/2 = 1.0
        // EO diff = 1.0 - 0.5 = 0.5
        // Threshold = 0.6 => 0.5 < 0.6 => no bias
        var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.6);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.False(result.HasBias);
        Assert.Equal(0.5, Convert.ToDouble(result.EqualOpportunityDifference), Tolerance);
    }

    #endregion

    #region BiasDetectorBase Tests

    [Fact]
    public void IsBetterBiasScore_LowerIsBetter_ReturnsTrueForLower()
    {
        // DemographicParity has isLowerBiasBetter = true
        var detector = new DemographicParityBiasDetector<double>();
        Assert.True(detector.IsBetterBiasScore(0.05, 0.1)); // 0.05 < 0.1
        Assert.False(detector.IsBetterBiasScore(0.2, 0.1)); // 0.2 > 0.1
    }

    [Fact]
    public void IsLowerBiasBetter_CorrectForDetectors()
    {
        var dp = new DemographicParityBiasDetector<double>();
        var eo = new EqualOpportunityBiasDetector<double>();

        Assert.True(dp.IsLowerBiasBetter);
        Assert.True(eo.IsLowerBiasBetter);
    }

    [Fact]
    public void DisparateImpact_IsLowerBiasBetter_IsFalse()
    {
        // DisparateImpact uses isLowerBiasBetter: false (higher ratio = less bias)
        var di = new DisparateImpactBiasDetector<double>();
        Assert.False(di.IsLowerBiasBetter);
    }

    [Fact]
    public void DetectBias_NullPredictions_ThrowsException()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentNullException>(() =>
            detector.DetectBias(null, sensitiveFeature));
    }

    [Fact]
    public void DetectBias_MismatchedLengths_ThrowsException()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 }); // Different length

        Assert.Throws<ArgumentException>(() =>
            detector.DetectBias(predictions, sensitiveFeature));
    }

    #endregion

    #region FairnessMetrics Data Class Tests

    [Fact]
    public void FairnessMetrics_StoresAllValues()
    {
        var metrics = new FairnessMetrics<double>(
            demographicParity: 0.1,
            equalOpportunity: 0.2,
            equalizedOdds: 0.3,
            predictiveParity: 0.4,
            disparateImpact: 0.85,
            statisticalParityDifference: 0.15
        );

        Assert.Equal(0.1, metrics.DemographicParity, Tolerance);
        Assert.Equal(0.2, metrics.EqualOpportunity, Tolerance);
        Assert.Equal(0.3, metrics.EqualizedOdds, Tolerance);
        Assert.Equal(0.4, metrics.PredictiveParity, Tolerance);
        Assert.Equal(0.85, metrics.DisparateImpact, Tolerance);
        Assert.Equal(0.15, metrics.StatisticalParityDifference, Tolerance);
    }

    [Fact]
    public void FairnessMetrics_AdditionalMetrics_Empty()
    {
        var metrics = new FairnessMetrics<double>(0, 0, 0, 0, 0, 0);
        Assert.NotNull(metrics.AdditionalMetrics);
        Assert.Empty(metrics.AdditionalMetrics);
    }

    #endregion

    #region Cross-Component Integration Tests

    [Fact]
    public void EndToEnd_PerGroupMetrics_ConsistentWithHelper()
    {
        // Manual computation matches detector result
        // Group 0: pred=[1,1,0], actual=[1,0,1]
        //   PositiveRate = 2/3, TPR = 1/2, FPR = 1/1 = 1.0, Precision = 1/2
        // Group 1: pred=[1,0,0], actual=[1,1,0]
        //   PositiveRate = 1/3, TPR = 1/2, FPR = 0/1 = 0.0, Precision = 1/1 = 1.0
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0 });

        // Verify helper directly
        var group0Indices = InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, 0.0);
        var group0Preds = InterpretabilityMetricsHelper<double>.GetSubset(predictions, group0Indices);
        var group0Actuals = InterpretabilityMetricsHelper<double>.GetSubset(actuals, group0Indices);

        Assert.Equal(2.0 / 3.0, InterpretabilityMetricsHelper<double>.ComputePositiveRate(group0Preds), Tolerance);
        Assert.Equal(0.5, InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(group0Preds, group0Actuals), Tolerance);
        Assert.Equal(1.0, InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(group0Preds, group0Actuals), Tolerance);
        Assert.Equal(0.5, InterpretabilityMetricsHelper<double>.ComputePrecision(group0Preds, group0Actuals), Tolerance);

        // Verify via DisparateImpact detector
        var detector = new DisparateImpactBiasDetector<double>();
        var result = detector.DetectBias(predictions, sensitiveFeature, actuals);

        Assert.Equal(2, result.GroupPositiveRates.Count);
        Assert.Equal(2.0 / 3.0, Convert.ToDouble(result.GroupPositiveRates["0"]), Tolerance);
        Assert.Equal(1.0 / 3.0, Convert.ToDouble(result.GroupPositiveRates["1"]), Tolerance);
    }

    [Fact]
    public void EndToEnd_ThreeGroups_BiasDetection()
    {
        // 3 sensitive groups
        // Group 0 (indices 0,1): pred=[1,1] => rate=1.0
        // Group 1 (indices 2,3): pred=[1,0] => rate=0.5
        // Group 2 (indices 4,5): pred=[0,0] => rate=0.0
        // DI ratio = min/max = 0.0/1.0 = 0.0 < 0.8 => bias
        // SPD = max - min = 1.0 - 0.0 = 1.0
        var detector = new DisparateImpactBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Equal(3, result.GroupPositiveRates.Count);
        Assert.Equal(0.0, Convert.ToDouble(result.DisparateImpactRatio), Tolerance);
        Assert.Equal(1.0, Convert.ToDouble(result.StatisticalParityDifference), Tolerance);
    }

    [Fact]
    public void EndToEnd_AllDetectors_SameInput_ConsistentResults()
    {
        // All three detectors on the same data should provide consistent group rates
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var actuals = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 });

        var dp = new DemographicParityBiasDetector<double>();
        var di = new DisparateImpactBiasDetector<double>();
        var eo = new EqualOpportunityBiasDetector<double>();

        var dpResult = dp.DetectBias(predictions, sensitiveFeature, actuals);
        var diResult = di.DetectBias(predictions, sensitiveFeature, actuals);
        var eoResult = eo.DetectBias(predictions, sensitiveFeature, actuals);

        // All should agree on group positive rates
        Assert.Equal(
            Convert.ToDouble(dpResult.GroupPositiveRates["0"]),
            Convert.ToDouble(diResult.GroupPositiveRates["0"]),
            Tolerance);

        Assert.Equal(
            Convert.ToDouble(dpResult.GroupPositiveRates["1"]),
            Convert.ToDouble(diResult.GroupPositiveRates["1"]),
            Tolerance);
    }

    #endregion
}
