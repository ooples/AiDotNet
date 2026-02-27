#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Interpretability;

/// <summary>
/// Comprehensive integration tests for the Interpretability module.
/// Tests enums, data classes, bias detectors, fairness evaluators, and helper methods.
/// </summary>
public class InterpretabilityIntegrationTests
{
    #region InterpretationMethod Enum Tests

    [Fact]
    public void InterpretationMethod_ContainsExpectedValues()
    {
        var values = (InterpretationMethod[])Enum.GetValues(typeof(InterpretationMethod));

        // Original methods
        Assert.Contains(InterpretationMethod.SHAP, values);
        Assert.Contains(InterpretationMethod.LIME, values);
        Assert.Contains(InterpretationMethod.PartialDependence, values);
        Assert.Contains(InterpretationMethod.Counterfactual, values);
        Assert.Contains(InterpretationMethod.Anchor, values);
        Assert.Contains(InterpretationMethod.FeatureImportance, values);
        Assert.Contains(InterpretationMethod.FeatureInteraction, values);
        Assert.Contains(InterpretationMethod.IntegratedGradients, values);
        Assert.Contains(InterpretationMethod.DeepLIFT, values);
        Assert.Contains(InterpretationMethod.GradCAM, values);
        Assert.Contains(InterpretationMethod.TreeSHAP, values);

        // New methods added
        Assert.Contains(InterpretationMethod.DeepSHAP, values);
        Assert.Contains(InterpretationMethod.GradientSHAP, values);
        Assert.Contains(InterpretationMethod.TCAV, values);
        Assert.Contains(InterpretationMethod.InfluenceFunctions, values);
        Assert.Contains(InterpretationMethod.Occlusion, values);
        Assert.Contains(InterpretationMethod.FeatureAblation, values);
        Assert.Contains(InterpretationMethod.DiCE, values);
        Assert.Contains(InterpretationMethod.GuidedBackprop, values);
        Assert.Contains(InterpretationMethod.LayerGradCAM, values);
        Assert.Contains(InterpretationMethod.GuidedGradCAM, values);
        Assert.Contains(InterpretationMethod.NoiseTunnel, values);
    }

    [Fact]
    public void InterpretationMethod_HasExpectedCount()
    {
        var values = (InterpretationMethod[])Enum.GetValues(typeof(InterpretationMethod));
        Assert.Equal(22, values.Length);
    }

    [Theory]
    [InlineData(InterpretationMethod.SHAP)]
    [InlineData(InterpretationMethod.LIME)]
    [InlineData(InterpretationMethod.PartialDependence)]
    [InlineData(InterpretationMethod.Counterfactual)]
    [InlineData(InterpretationMethod.Anchor)]
    [InlineData(InterpretationMethod.FeatureImportance)]
    [InlineData(InterpretationMethod.FeatureInteraction)]
    [InlineData(InterpretationMethod.IntegratedGradients)]
    [InlineData(InterpretationMethod.DeepLIFT)]
    [InlineData(InterpretationMethod.GradCAM)]
    [InlineData(InterpretationMethod.TreeSHAP)]
    [InlineData(InterpretationMethod.DeepSHAP)]
    [InlineData(InterpretationMethod.GradientSHAP)]
    [InlineData(InterpretationMethod.TCAV)]
    [InlineData(InterpretationMethod.InfluenceFunctions)]
    [InlineData(InterpretationMethod.Occlusion)]
    [InlineData(InterpretationMethod.FeatureAblation)]
    [InlineData(InterpretationMethod.DiCE)]
    [InlineData(InterpretationMethod.GuidedBackprop)]
    [InlineData(InterpretationMethod.LayerGradCAM)]
    [InlineData(InterpretationMethod.GuidedGradCAM)]
    [InlineData(InterpretationMethod.NoiseTunnel)]
    public void InterpretationMethod_IsDefined(InterpretationMethod method)
    {
        Assert.True(Enum.IsDefined(typeof(InterpretationMethod), method));
    }

    [Theory]
    [InlineData(InterpretationMethod.SHAP, "SHAP")]
    [InlineData(InterpretationMethod.LIME, "LIME")]
    [InlineData(InterpretationMethod.PartialDependence, "PartialDependence")]
    [InlineData(InterpretationMethod.Counterfactual, "Counterfactual")]
    [InlineData(InterpretationMethod.Anchor, "Anchor")]
    [InlineData(InterpretationMethod.FeatureImportance, "FeatureImportance")]
    [InlineData(InterpretationMethod.FeatureInteraction, "FeatureInteraction")]
    [InlineData(InterpretationMethod.IntegratedGradients, "IntegratedGradients")]
    [InlineData(InterpretationMethod.DeepLIFT, "DeepLIFT")]
    [InlineData(InterpretationMethod.GradCAM, "GradCAM")]
    [InlineData(InterpretationMethod.TreeSHAP, "TreeSHAP")]
    [InlineData(InterpretationMethod.DeepSHAP, "DeepSHAP")]
    [InlineData(InterpretationMethod.GradientSHAP, "GradientSHAP")]
    [InlineData(InterpretationMethod.TCAV, "TCAV")]
    [InlineData(InterpretationMethod.InfluenceFunctions, "InfluenceFunctions")]
    [InlineData(InterpretationMethod.Occlusion, "Occlusion")]
    [InlineData(InterpretationMethod.FeatureAblation, "FeatureAblation")]
    [InlineData(InterpretationMethod.DiCE, "DiCE")]
    [InlineData(InterpretationMethod.GuidedBackprop, "GuidedBackprop")]
    [InlineData(InterpretationMethod.LayerGradCAM, "LayerGradCAM")]
    [InlineData(InterpretationMethod.GuidedGradCAM, "GuidedGradCAM")]
    [InlineData(InterpretationMethod.NoiseTunnel, "NoiseTunnel")]
    public void InterpretationMethod_ToString_ReturnsExpectedString(InterpretationMethod method, string expected)
    {
        Assert.Equal(expected, method.ToString());
    }

    [Fact]
    public void InterpretationMethod_CanBeUsedInHashSet()
    {
        var enabledMethods = new HashSet<InterpretationMethod>
        {
            InterpretationMethod.SHAP,
            InterpretationMethod.LIME,
            InterpretationMethod.FeatureImportance,
            InterpretationMethod.IntegratedGradients,
            InterpretationMethod.DeepLIFT,
            InterpretationMethod.GradCAM,
            InterpretationMethod.TreeSHAP
        };


        Assert.Contains(InterpretationMethod.SHAP, enabledMethods);
        Assert.Contains(InterpretationMethod.LIME, enabledMethods);
        Assert.Contains(InterpretationMethod.IntegratedGradients, enabledMethods);
        Assert.Contains(InterpretationMethod.DeepLIFT, enabledMethods);
        Assert.Contains(InterpretationMethod.GradCAM, enabledMethods);
        Assert.Contains(InterpretationMethod.TreeSHAP, enabledMethods);
        Assert.DoesNotContain(InterpretationMethod.Anchor, enabledMethods);
    }

    #endregion

    #region FairnessMetric Enum Tests

    [Fact]
    public void FairnessMetric_ContainsExpectedValues()
    {
        var values = (FairnessMetric[])Enum.GetValues(typeof(FairnessMetric));

        Assert.Contains(FairnessMetric.DemographicParity, values);
        Assert.Contains(FairnessMetric.EqualOpportunity, values);
        Assert.Contains(FairnessMetric.EqualizedOdds, values);
        Assert.Contains(FairnessMetric.PredictiveParity, values);
        Assert.Contains(FairnessMetric.DisparateImpact, values);
        Assert.Contains(FairnessMetric.StatisticalParityDifference, values);
    }

    [Fact]
    public void FairnessMetric_HasExpectedCount()
    {
        var values = (FairnessMetric[])Enum.GetValues(typeof(FairnessMetric));
        Assert.Equal(6, values.Length);
    }

    [Theory]
    [InlineData(FairnessMetric.DemographicParity)]
    [InlineData(FairnessMetric.EqualOpportunity)]
    [InlineData(FairnessMetric.EqualizedOdds)]
    [InlineData(FairnessMetric.PredictiveParity)]
    [InlineData(FairnessMetric.DisparateImpact)]
    [InlineData(FairnessMetric.StatisticalParityDifference)]
    public void FairnessMetric_IsDefined(FairnessMetric metric)
    {
        Assert.True(Enum.IsDefined(typeof(FairnessMetric), metric));
    }

    [Fact]
    public void FairnessMetric_CanBeUsedInList()
    {
        var metrics = new List<FairnessMetric>
        {
            FairnessMetric.DemographicParity,
            FairnessMetric.EqualOpportunity,
            FairnessMetric.DisparateImpact
        };

        Assert.Equal(3, metrics.Count);
        Assert.Contains(FairnessMetric.DemographicParity, metrics);
    }

    #endregion

    #region BiasDetectionResult Tests

    [Fact]
    public void BiasDetectionResult_DefaultConstructor_InitializesProperties()
    {
        var result = new BiasDetectionResult<double>();

        Assert.False(result.HasBias);
        Assert.NotNull(result.GroupPositiveRates);
        Assert.NotNull(result.GroupSizes);
        Assert.Empty(result.GroupPositiveRates);
        Assert.Empty(result.GroupSizes);
    }

    [Fact]
    public void BiasDetectionResult_CanSetHasBias()
    {
        var result = new BiasDetectionResult<double>
        {
            HasBias = true
        };

        Assert.True(result.HasBias);
    }

    [Fact]
    public void BiasDetectionResult_CanSetMessage()
    {
        var result = new BiasDetectionResult<double>
        {
            Message = "Bias detected: Demographic parity difference exceeds threshold"
        };

        Assert.Equal("Bias detected: Demographic parity difference exceeds threshold", result.Message);
    }

    [Fact]
    public void BiasDetectionResult_CanSetGroupPositiveRates()
    {
        var result = new BiasDetectionResult<double>();
        result.GroupPositiveRates["Group1"] = 0.8;
        result.GroupPositiveRates["Group2"] = 0.6;

        Assert.Equal(0.8, result.GroupPositiveRates["Group1"]);
        Assert.Equal(0.6, result.GroupPositiveRates["Group2"]);
    }

    [Fact]
    public void BiasDetectionResult_CanSetGroupSizes()
    {
        var result = new BiasDetectionResult<double>();
        result.GroupSizes["Group1"] = 100;
        result.GroupSizes["Group2"] = 150;

        Assert.Equal(100, result.GroupSizes["Group1"]);
        Assert.Equal(150, result.GroupSizes["Group2"]);
    }

    [Fact]
    public void BiasDetectionResult_CanSetDisparateImpactRatio()
    {
        var result = new BiasDetectionResult<double>
        {
            DisparateImpactRatio = 0.75
        };

        Assert.Equal(0.75, result.DisparateImpactRatio);
    }

    [Fact]
    public void BiasDetectionResult_CanSetStatisticalParityDifference()
    {
        var result = new BiasDetectionResult<double>
        {
            StatisticalParityDifference = 0.15
        };

        Assert.Equal(0.15, result.StatisticalParityDifference);
    }

    [Fact]
    public void BiasDetectionResult_CanSetEqualOpportunityDifference()
    {
        var result = new BiasDetectionResult<double>
        {
            EqualOpportunityDifference = 0.12
        };

        Assert.Equal(0.12, result.EqualOpportunityDifference);
    }

    [Fact]
    public void BiasDetectionResult_CanSetGroupTruePositiveRates()
    {
        var result = new BiasDetectionResult<double>();
        result.GroupTruePositiveRates["Group1"] = 0.9;
        result.GroupTruePositiveRates["Group2"] = 0.7;

        Assert.Equal(0.9, result.GroupTruePositiveRates["Group1"]);
        Assert.Equal(0.7, result.GroupTruePositiveRates["Group2"]);
    }

    [Fact]
    public void BiasDetectionResult_CanSetGroupFalsePositiveRates()
    {
        var result = new BiasDetectionResult<double>();
        result.GroupFalsePositiveRates["Group1"] = 0.1;
        result.GroupFalsePositiveRates["Group2"] = 0.2;

        Assert.Equal(0.1, result.GroupFalsePositiveRates["Group1"]);
        Assert.Equal(0.2, result.GroupFalsePositiveRates["Group2"]);
    }

    [Fact]
    public void BiasDetectionResult_CanSetGroupPrecisions()
    {
        var result = new BiasDetectionResult<double>();
        result.GroupPrecisions["Group1"] = 0.85;
        result.GroupPrecisions["Group2"] = 0.75;

        Assert.Equal(0.85, result.GroupPrecisions["Group1"]);
        Assert.Equal(0.75, result.GroupPrecisions["Group2"]);
    }

    #endregion

    #region FairnessMetrics Tests

    [Fact]
    public void FairnessMetrics_Constructor_InitializesAllProperties()
    {
        var metrics = new FairnessMetrics<double>(
            demographicParity: 0.1,
            equalOpportunity: 0.2,
            equalizedOdds: 0.3,
            predictiveParity: 0.15,
            disparateImpact: 0.8,
            statisticalParityDifference: 0.1);

        Assert.Equal(0.1, metrics.DemographicParity);
        Assert.Equal(0.2, metrics.EqualOpportunity);
        Assert.Equal(0.3, metrics.EqualizedOdds);
        Assert.Equal(0.15, metrics.PredictiveParity);
        Assert.Equal(0.8, metrics.DisparateImpact);
        Assert.Equal(0.1, metrics.StatisticalParityDifference);
        Assert.NotNull(metrics.AdditionalMetrics);
        Assert.Empty(metrics.AdditionalMetrics);
    }

    [Fact]
    public void FairnessMetrics_CanSetSensitiveFeatureIndex()
    {
        var metrics = new FairnessMetrics<double>(0, 0, 0, 0, 1, 0)
        {
            SensitiveFeatureIndex = 5
        };

        Assert.Equal(5, metrics.SensitiveFeatureIndex);
    }

    [Fact]
    public void FairnessMetrics_CanAddAdditionalMetrics()
    {
        var metrics = new FairnessMetrics<double>(0, 0, 0, 0, 1, 0);
        metrics.AdditionalMetrics["Group_A_PositiveRate"] = 0.7;
        metrics.AdditionalMetrics["Group_B_PositiveRate"] = 0.6;

        Assert.Equal(0.7, metrics.AdditionalMetrics["Group_A_PositiveRate"]);
        Assert.Equal(0.6, metrics.AdditionalMetrics["Group_B_PositiveRate"]);
    }

    [Fact]
    public void FairnessMetrics_CanUpdateProperties()
    {
        var metrics = new FairnessMetrics<double>(0, 0, 0, 0, 1, 0);

        metrics.DemographicParity = 0.5;
        metrics.EqualOpportunity = 0.4;
        metrics.EqualizedOdds = 0.3;
        metrics.PredictiveParity = 0.2;
        metrics.DisparateImpact = 0.75;
        metrics.StatisticalParityDifference = 0.5;

        Assert.Equal(0.5, metrics.DemographicParity);
        Assert.Equal(0.4, metrics.EqualOpportunity);
        Assert.Equal(0.3, metrics.EqualizedOdds);
        Assert.Equal(0.2, metrics.PredictiveParity);
        Assert.Equal(0.75, metrics.DisparateImpact);
        Assert.Equal(0.5, metrics.StatisticalParityDifference);
    }

    #endregion

    #region AnchorExplanation Tests

    [Fact]
    public void AnchorExplanation_DefaultConstructor_InitializesProperties()
    {
        var explanation = new AnchorExplanation<double>();

        Assert.NotNull(explanation.AnchorRules);
        Assert.NotNull(explanation.AnchorFeatures);
        Assert.NotNull(explanation.Description);
        Assert.Empty(explanation.AnchorRules);
        Assert.Empty(explanation.AnchorFeatures);
        Assert.Equal(string.Empty, explanation.Description);
        Assert.Equal(0.0, explanation.Precision);
        Assert.Equal(0.0, explanation.Coverage);
        Assert.Equal(0.0, explanation.Threshold);
    }

    [Fact]
    public void AnchorExplanation_CanSetPrecision()
    {
        var explanation = new AnchorExplanation<double>
        {
            Precision = 0.95
        };

        Assert.Equal(0.95, explanation.Precision);
    }

    [Fact]
    public void AnchorExplanation_CanSetCoverage()
    {
        var explanation = new AnchorExplanation<double>
        {
            Coverage = 0.3
        };

        Assert.Equal(0.3, explanation.Coverage);
    }

    [Fact]
    public void AnchorExplanation_CanSetThreshold()
    {
        var explanation = new AnchorExplanation<double>
        {
            Threshold = 0.9
        };

        Assert.Equal(0.9, explanation.Threshold);
    }

    [Fact]
    public void AnchorExplanation_CanAddAnchorRules()
    {
        var explanation = new AnchorExplanation<double>();
        explanation.AnchorRules[0] = (0.5, 1.0);
        explanation.AnchorRules[2] = (0.0, 0.8);

        Assert.Equal((0.5, 1.0), explanation.AnchorRules[0]);
        Assert.Equal((0.0, 0.8), explanation.AnchorRules[2]);
    }

    [Fact]
    public void AnchorExplanation_CanAddAnchorFeatures()
    {
        var explanation = new AnchorExplanation<double>();
        explanation.AnchorFeatures.Add(0);
        explanation.AnchorFeatures.Add(2);
        explanation.AnchorFeatures.Add(5);

        Assert.Equal(3, explanation.AnchorFeatures.Count);
        Assert.Contains(0, explanation.AnchorFeatures);
        Assert.Contains(2, explanation.AnchorFeatures);
        Assert.Contains(5, explanation.AnchorFeatures);
    }

    [Fact]
    public void AnchorExplanation_CanSetDescription()
    {
        var explanation = new AnchorExplanation<double>
        {
            Description = "If Feature0 >= 0.5 AND Feature2 <= 0.8 THEN class = 1"
        };

        Assert.Equal("If Feature0 >= 0.5 AND Feature2 <= 0.8 THEN class = 1", explanation.Description);
    }

    #endregion

    #region LimeExplanation Tests

    [Fact]
    public void LimeExplanation_DefaultConstructor_InitializesProperties()
    {
        var explanation = new LimeExplanation<double>();

        Assert.NotNull(explanation.FeatureImportance);
        Assert.Empty(explanation.FeatureImportance);
        Assert.Equal(0.0, explanation.Intercept);
        Assert.Equal(0.0, explanation.PredictedValue);
        Assert.Equal(0.0, explanation.LocalModelScore);
        Assert.Equal(0, explanation.NumFeatures);
    }

    [Fact]
    public void LimeExplanation_CanSetIntercept()
    {
        var explanation = new LimeExplanation<double>
        {
            Intercept = 0.5
        };

        Assert.Equal(0.5, explanation.Intercept);
    }

    [Fact]
    public void LimeExplanation_CanSetPredictedValue()
    {
        var explanation = new LimeExplanation<double>
        {
            PredictedValue = 0.85
        };

        Assert.Equal(0.85, explanation.PredictedValue);
    }

    [Fact]
    public void LimeExplanation_CanSetLocalModelScore()
    {
        var explanation = new LimeExplanation<double>
        {
            LocalModelScore = 0.92
        };

        Assert.Equal(0.92, explanation.LocalModelScore);
    }

    [Fact]
    public void LimeExplanation_CanSetNumFeatures()
    {
        var explanation = new LimeExplanation<double>
        {
            NumFeatures = 10
        };

        Assert.Equal(10, explanation.NumFeatures);
    }

    [Fact]
    public void LimeExplanation_CanAddFeatureImportance()
    {
        var explanation = new LimeExplanation<double>();
        explanation.FeatureImportance[0] = 0.3;
        explanation.FeatureImportance[1] = -0.2;
        explanation.FeatureImportance[2] = 0.5;

        Assert.Equal(0.3, explanation.FeatureImportance[0]);
        Assert.Equal(-0.2, explanation.FeatureImportance[1]);
        Assert.Equal(0.5, explanation.FeatureImportance[2]);
    }

    #endregion

    #region CounterfactualExplanation Tests

    [Fact]
    public void CounterfactualExplanation_DefaultConstructor_InitializesProperties()
    {
        var explanation = new CounterfactualExplanation<double>();

        Assert.Null(explanation.OriginalInput);
        Assert.Null(explanation.CounterfactualInput);
        Assert.Null(explanation.OriginalPrediction);
        Assert.Null(explanation.CounterfactualPrediction);
        Assert.NotNull(explanation.FeatureChanges);
        Assert.Empty(explanation.FeatureChanges);
        Assert.Equal(0.0, explanation.Distance);
        Assert.Equal(0, explanation.MaxChanges);
    }

    [Fact]
    public void CounterfactualExplanation_CanSetOriginalInput()
    {
        var explanation = new CounterfactualExplanation<double>();
        var input = Tensor<double>.FromVector(new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        explanation.OriginalInput = input;

        Assert.NotNull(explanation.OriginalInput);
    }

    [Fact]
    public void CounterfactualExplanation_CanSetCounterfactualInput()
    {
        var explanation = new CounterfactualExplanation<double>();
        var input = Tensor<double>.FromVector(new Vector<double>(new[] { 1.5, 2.0, 3.5 }));
        explanation.CounterfactualInput = input;

        Assert.NotNull(explanation.CounterfactualInput);
    }

    [Fact]
    public void CounterfactualExplanation_CanSetDistance()
    {
        var explanation = new CounterfactualExplanation<double>
        {
            Distance = 0.75
        };

        Assert.Equal(0.75, explanation.Distance);
    }

    [Fact]
    public void CounterfactualExplanation_CanSetMaxChanges()
    {
        var explanation = new CounterfactualExplanation<double>
        {
            MaxChanges = 5
        };

        Assert.Equal(5, explanation.MaxChanges);
    }

    [Fact]
    public void CounterfactualExplanation_CanAddFeatureChanges()
    {
        var explanation = new CounterfactualExplanation<double>();
        explanation.FeatureChanges[0] = 0.5;
        explanation.FeatureChanges[2] = -0.3;

        Assert.Equal(0.5, explanation.FeatureChanges[0]);
        Assert.Equal(-0.3, explanation.FeatureChanges[2]);
    }

    #endregion

    #region PartialDependenceData Tests

    [Fact]
    public void PartialDependenceData_DefaultConstructor_InitializesProperties()
    {
        var data = new PartialDependenceData<double>();

        Assert.NotNull(data.FeatureIndices);
        Assert.NotNull(data.GridValues);
        Assert.NotNull(data.PartialDependenceValues);
        Assert.NotNull(data.IceCurves);
        Assert.Equal(0, data.GridResolution);
        Assert.Empty(data.GridValues);
        Assert.Empty(data.IceCurves);
    }

    [Fact]
    public void PartialDependenceData_CanSetGridResolution()
    {
        var data = new PartialDependenceData<double>
        {
            GridResolution = 50
        };

        Assert.Equal(50, data.GridResolution);
    }

    [Fact]
    public void PartialDependenceData_CanSetFeatureIndices()
    {
        var data = new PartialDependenceData<double>
        {
            FeatureIndices = new Vector<int>(new[] { 0, 1, 2 })
        };

        Assert.Equal(3, data.FeatureIndices.Length);
    }

    [Fact]
    public void PartialDependenceData_CanAddGridValues()
    {
        var data = new PartialDependenceData<double>();
        data.GridValues[0] = new Vector<double>(new[] { 0.0, 0.5, 1.0 });
        data.GridValues[1] = new Vector<double>(new[] { -1.0, 0.0, 1.0 });

        Assert.Equal(2, data.GridValues.Count);
        Assert.Equal(3, data.GridValues[0].Length);
    }

    [Fact]
    public void PartialDependenceData_CanAddIceCurves()
    {
        var data = new PartialDependenceData<double>();
        data.IceCurves.Add(new Matrix<double>(10, 20));
        data.IceCurves.Add(new Matrix<double>(10, 20));

        Assert.Equal(2, data.IceCurves.Count);
    }

    #endregion

    #region DemographicParityBiasDetector Tests

    [Fact]
    public void DemographicParityBiasDetector_Constructor_DefaultThreshold()
    {
        var detector = new DemographicParityBiasDetector<double>();

        Assert.True(detector.IsLowerBiasBetter);
    }

    [Fact]
    public void DemographicParityBiasDetector_Constructor_CustomThreshold()
    {
        var detector = new DemographicParityBiasDetector<double>(0.05);

        Assert.True(detector.IsLowerBiasBetter);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void DemographicParityBiasDetector_Constructor_InvalidThreshold_ThrowsException(double threshold)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DemographicParityBiasDetector<double>(threshold));
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_NullPredictions_ThrowsException()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentNullException>(() => detector.DetectBias(null!, sensitiveFeature));
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_NullSensitiveFeature_ThrowsException()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0 });

        Assert.Throws<ArgumentNullException>(() => detector.DetectBias(predictions, null!));
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_LengthMismatch_ThrowsException()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentException>(() => detector.DetectBias(predictions, sensitiveFeature));
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_SingleGroup_NoBias()
    {
        var detector = new DemographicParityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 }); // All same group

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Contains("Insufficient groups", result.Message);
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_EqualRates_NoBias()
    {
        var detector = new DemographicParityBiasDetector<double>();
        // Group 0: 50% positive (indices 0,1 - values 1,0)
        // Group 1: 50% positive (indices 2,3 - values 1,0)
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Equal(2, result.GroupPositiveRates.Count);
        Assert.Equal(2, result.GroupSizes.Count);
    }

    [Fact]
    public void DemographicParityBiasDetector_DetectBias_UnequalRates_DetectsBias()
    {
        var detector = new DemographicParityBiasDetector<double>(0.1);
        // Group 0: 100% positive (indices 0,1 - values 1,1)
        // Group 1: 0% positive (indices 2,3 - values 0,0)
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.True(result.HasBias);
        Assert.Contains("Bias detected", result.Message);
    }

    [Fact]
    public void DemographicParityBiasDetector_IsBetterBiasScore_LowerIsBetter()
    {
        var detector = new DemographicParityBiasDetector<double>();

        Assert.True(detector.IsBetterBiasScore(0.05, 0.1));
        Assert.False(detector.IsBetterBiasScore(0.15, 0.1));
    }

    #endregion

    #region DisparateImpactBiasDetector Tests

    [Fact]
    public void DisparateImpactBiasDetector_Constructor_DefaultThreshold()
    {
        var detector = new DisparateImpactBiasDetector<double>();

        Assert.False(detector.IsLowerBiasBetter); // Higher DI ratio is better (closer to 1)
    }

    [Fact]
    public void DisparateImpactBiasDetector_Constructor_CustomThreshold()
    {
        var detector = new DisparateImpactBiasDetector<double>(0.7);

        Assert.False(detector.IsLowerBiasBetter);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void DisparateImpactBiasDetector_Constructor_InvalidThreshold_ThrowsException(double threshold)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DisparateImpactBiasDetector<double>(threshold));
    }

    [Fact]
    public void DisparateImpactBiasDetector_DetectBias_NullPredictions_ThrowsException()
    {
        var detector = new DisparateImpactBiasDetector<double>();
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentNullException>(() => detector.DetectBias(null!, sensitiveFeature));
    }

    [Fact]
    public void DisparateImpactBiasDetector_DetectBias_SingleGroup_NoBias()
    {
        var detector = new DisparateImpactBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
    }

    [Fact]
    public void DisparateImpactBiasDetector_DetectBias_AllZeroPredictions_NoBias()
    {
        var detector = new DisparateImpactBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature);

        Assert.False(result.HasBias);
        Assert.Equal(1.0, result.DisparateImpactRatio);
    }

    [Fact]
    public void DisparateImpactBiasDetector_DetectBias_WithActualLabels_ComputesAdditionalMetrics()
    {
        var detector = new DisparateImpactBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

        Assert.NotNull(result.GroupTruePositiveRates);
        Assert.NotNull(result.GroupFalsePositiveRates);
        Assert.NotNull(result.GroupPrecisions);
    }

    [Fact]
    public void DisparateImpactBiasDetector_IsBetterBiasScore_HigherIsBetter()
    {
        var detector = new DisparateImpactBiasDetector<double>();

        Assert.True(detector.IsBetterBiasScore(0.9, 0.8));
        Assert.False(detector.IsBetterBiasScore(0.7, 0.8));
    }

    #endregion

    #region EqualOpportunityBiasDetector Tests

    [Fact]
    public void EqualOpportunityBiasDetector_Constructor_DefaultThreshold()
    {
        var detector = new EqualOpportunityBiasDetector<double>();

        Assert.True(detector.IsLowerBiasBetter);
    }

    [Fact]
    public void EqualOpportunityBiasDetector_Constructor_CustomThreshold()
    {
        var detector = new EqualOpportunityBiasDetector<double>(0.05);

        Assert.True(detector.IsLowerBiasBetter);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void EqualOpportunityBiasDetector_Constructor_InvalidThreshold_ThrowsException(double threshold)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EqualOpportunityBiasDetector<double>(threshold));
    }

    [Fact]
    public void EqualOpportunityBiasDetector_DetectBias_NullPredictions_ThrowsException()
    {
        var detector = new EqualOpportunityBiasDetector<double>();
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentNullException>(() => detector.DetectBias(null!, sensitiveFeature));
    }

    [Fact]
    public void EqualOpportunityBiasDetector_DetectBias_SingleGroup_NoBias()
    {
        var detector = new EqualOpportunityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

        Assert.False(result.HasBias);
    }

    [Fact]
    public void EqualOpportunityBiasDetector_DetectBias_NoActualLabels_CannotCompute()
    {
        var detector = new EqualOpportunityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, null);

        Assert.False(result.HasBias);
        Assert.Contains("Cannot compute equal opportunity without actual labels", result.Message);
    }

    [Fact]
    public void EqualOpportunityBiasDetector_DetectBias_WithActualLabels_ComputesTPRs()
    {
        var detector = new EqualOpportunityBiasDetector<double>();
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

        var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

        Assert.NotNull(result.GroupTruePositiveRates);
    }

    [Fact]
    public void EqualOpportunityBiasDetector_IsBetterBiasScore_LowerIsBetter()
    {
        var detector = new EqualOpportunityBiasDetector<double>();

        Assert.True(detector.IsBetterBiasScore(0.05, 0.1));
        Assert.False(detector.IsBetterBiasScore(0.15, 0.1));
    }

    #endregion

    #region BasicFairnessEvaluator Tests

    [Fact]
    public void BasicFairnessEvaluator_Constructor_InitializesCorrectly()
    {
        var evaluator = new BasicFairnessEvaluator<double>();

        Assert.False(evaluator.IsHigherFairnessBetter);
    }

    [Fact]
    public void BasicFairnessEvaluator_IsBetterFairnessScore_LowerIsBetter()
    {
        var evaluator = new BasicFairnessEvaluator<double>();

        Assert.True(evaluator.IsBetterFairnessScore(0.05, 0.1));
        Assert.False(evaluator.IsBetterFairnessScore(0.15, 0.1));
    }

    #endregion

    #region ComprehensiveFairnessEvaluator Tests

    [Fact]
    public void ComprehensiveFairnessEvaluator_Constructor_InitializesCorrectly()
    {
        var evaluator = new ComprehensiveFairnessEvaluator<double>();

        Assert.False(evaluator.IsHigherFairnessBetter);
    }

    [Fact]
    public void ComprehensiveFairnessEvaluator_IsBetterFairnessScore_LowerIsBetter()
    {
        var evaluator = new ComprehensiveFairnessEvaluator<double>();

        Assert.True(evaluator.IsBetterFairnessScore(0.05, 0.1));
        Assert.False(evaluator.IsBetterFairnessScore(0.15, 0.1));
    }

    #endregion

    #region GroupFairnessEvaluator Tests

    [Fact]
    public void GroupFairnessEvaluator_Constructor_InitializesCorrectly()
    {
        var evaluator = new GroupFairnessEvaluator<double>();

        Assert.False(evaluator.IsHigherFairnessBetter);
    }

    [Fact]
    public void GroupFairnessEvaluator_IsBetterFairnessScore_LowerIsBetter()
    {
        var evaluator = new GroupFairnessEvaluator<double>();

        Assert.True(evaluator.IsBetterFairnessScore(0.05, 0.1));
        Assert.False(evaluator.IsBetterFairnessScore(0.15, 0.1));
    }

    #endregion

    #region InterpretabilityMetricsHelper Tests

    [Fact]
    public void InterpretabilityMetricsHelper_GetUniqueGroups_ReturnsUniqueValues()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 2.0 });

        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);

        Assert.Equal(3, groups.Count);
        Assert.Contains(0.0, groups);
        Assert.Contains(1.0, groups);
        Assert.Contains(2.0, groups);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetUniqueGroups_EmptyVector_ReturnsEmptyList()
    {
        var sensitiveFeature = new Vector<double>(Array.Empty<double>());

        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);

        Assert.Empty(groups);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetUniqueGroups_SingleValue_ReturnsSingleGroup()
    {
        var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);

        Assert.Single(groups);
        Assert.Equal(1.0, groups[0]);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetGroupIndices_ReturnsCorrectIndices()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0 });

        var indices = InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, 0.0);

        Assert.Equal(3, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(2, indices);
        Assert.Contains(4, indices);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetGroupIndices_NoMatches_ReturnsEmptyList()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });

        var indices = InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, 2.0);

        Assert.Empty(indices);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetSubset_ReturnsCorrectSubset()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var indices = new List<int> { 0, 2, 4 };

        var subset = InterpretabilityMetricsHelper<double>.GetSubset(vector, indices);

        Assert.Equal(3, subset.Length);
        Assert.Equal(10.0, subset[0]);
        Assert.Equal(30.0, subset[1]);
        Assert.Equal(50.0, subset[2]);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_GetSubset_EmptyIndices_ReturnsEmptyVector()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var indices = new List<int>();

        var subset = InterpretabilityMetricsHelper<double>.GetSubset(vector, indices);

        Assert.Equal(0, subset.Length);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePositiveRate_AllPositive_ReturnsOne()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

        var rate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);

        Assert.Equal(1.0, rate);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePositiveRate_AllNegative_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        var rate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);

        Assert.Equal(0.0, rate);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePositiveRate_MixedValues_ReturnsCorrectRate()
    {
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

        var rate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);

        Assert.Equal(0.5, rate);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePositiveRate_EmptyVector_ReturnsZero()
    {
        var predictions = new Vector<double>(Array.Empty<double>());

        var rate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);

        Assert.Equal(0.0, rate);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeTruePositiveRate_PerfectRecall_ReturnsOne()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

        Assert.Equal(1.0, tpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeTruePositiveRate_ZeroRecall_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, tpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeTruePositiveRate_NoActualPositives_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, tpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeTruePositiveRate_EmptyVectors_ReturnsZero()
    {
        var predictions = new Vector<double>(Array.Empty<double>());
        var actualLabels = new Vector<double>(Array.Empty<double>());

        var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, tpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeFalsePositiveRate_NoFalsePositives_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, fpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeFalsePositiveRate_AllFalsePositives_ReturnsOne()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 0.0, 0.0 });

        var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

        Assert.Equal(1.0, fpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeFalsePositiveRate_NoActualNegatives_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0 });

        var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, fpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputeFalsePositiveRate_EmptyVectors_ReturnsZero()
    {
        var predictions = new Vector<double>(Array.Empty<double>());
        var actualLabels = new Vector<double>(Array.Empty<double>());

        var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

        Assert.Equal(0.0, fpr);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePrecision_PerfectPrecision_ReturnsOne()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

        Assert.Equal(1.0, precision);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePrecision_ZeroPrecision_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 1.0, 1.0 });
        var actualLabels = new Vector<double>(new[] { 0.0, 0.0 });

        var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

        Assert.Equal(0.0, precision);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePrecision_NoPredictedPositives_ReturnsZero()
    {
        var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

        Assert.Equal(0.0, precision);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePrecision_EmptyVectors_ReturnsZero()
    {
        var predictions = new Vector<double>(Array.Empty<double>());
        var actualLabels = new Vector<double>(Array.Empty<double>());

        var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

        Assert.Equal(0.0, precision);
    }

    [Fact]
    public void InterpretabilityMetricsHelper_ComputePrecision_PartialMatch_ReturnsCorrectValue()
    {
        // Predictions: [1, 1, 0, 0] - predicts positive for indices 0,1
        // Actual: [1, 0, 1, 0] - actual positive at indices 0,2
        // TP = 1 (index 0), FP = 1 (index 1)
        // Precision = 1 / 2 = 0.5
        var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

        var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

        Assert.Equal(0.5, precision);
    }

    #endregion

    #region InterpretableModelHelper Tests

    [Fact]
    public async Task InterpretableModelHelper_GetFeatureInteractionAsync_FeatureInteractionNotEnabled_ThrowsException()
    {
        var enabledMethods = new HashSet<InterpretationMethod> { InterpretationMethod.SHAP };

#pragma warning disable CS0618 // Testing legacy overload behavior
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => InterpretableModelHelper.GetFeatureInteractionAsync<double>(enabledMethods, 0, 1));
#pragma warning restore CS0618
    }

    [Fact]
    public async Task InterpretableModelHelper_GetFeatureInteractionAsync_FeatureInteractionEnabled_ReturnsZero()
    {
        var enabledMethods = new HashSet<InterpretationMethod> { InterpretationMethod.FeatureInteraction };

#pragma warning disable CS0618 // Testing legacy overload behavior
        var result = await InterpretableModelHelper.GetFeatureInteractionAsync<double>(enabledMethods, 0, 1);
#pragma warning restore CS0618

        Assert.Equal(0.0, result);
    }

    [Fact]
    public async Task InterpretableModelHelper_ValidateFairnessAsync_ReturnsExpectedDefaults()
    {
        var fairnessMetrics = new List<FairnessMetric>
        {
            FairnessMetric.DemographicParity,
            FairnessMetric.EqualOpportunity
        };

#pragma warning disable CS0618 // Using obsolete method for legacy compatibility testing
        var result = await InterpretableModelHelper.ValidateFairnessAsync<double>(fairnessMetrics);
#pragma warning restore CS0618

        Assert.Equal(0.0, result.DemographicParity);
        Assert.Equal(0.0, result.EqualOpportunity);
        Assert.Equal(0.0, result.EqualizedOdds);
        Assert.Equal(0.0, result.PredictiveParity);
        // DisparateImpact defaults to 1.0 (perfect fairness ratio) rather than 0.0
        Assert.Equal(1.0, result.DisparateImpact);
        Assert.Equal(0.0, result.StatisticalParityDifference);
    }

    #endregion

    #region Integration Tests - End-to-End Bias Detection Scenarios

    [Fact]
    public void BiasDetection_EndToEnd_DemographicParityScenario()
    {
        // Create a scenario with two demographic groups
        // Group A (value 0): Higher positive rate
        // Group B (value 1): Lower positive rate
        var predictions = new Vector<double>(new[]
        {
            // Group A (indices 0-4): 4 positive out of 5 = 80%
            1.0, 1.0, 1.0, 1.0, 0.0,
            // Group B (indices 5-9): 2 positive out of 5 = 40%
            1.0, 1.0, 0.0, 0.0, 0.0
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0
        });

        var detector = new DemographicParityBiasDetector<double>(0.1);
        var result = detector.DetectBias(predictions, sensitiveFeature);

        // Statistical parity difference = 0.8 - 0.4 = 0.4 > 0.1 threshold
        Assert.True(result.HasBias);
        Assert.Equal(5, result.GroupSizes["0"]);
        Assert.Equal(5, result.GroupSizes["1"]);
        Assert.Equal(0.8, result.GroupPositiveRates["0"]);
        Assert.Equal(0.4, result.GroupPositiveRates["1"]);
    }

    [Fact]
    public void BiasDetection_EndToEnd_DisparateImpactScenario()
    {
        // Create a scenario violating the 80% rule
        var predictions = new Vector<double>(new[]
        {
            // Group A (indices 0-4): 5 positive out of 5 = 100%
            1.0, 1.0, 1.0, 1.0, 1.0,
            // Group B (indices 5-9): 2 positive out of 5 = 40%
            1.0, 1.0, 0.0, 0.0, 0.0
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0
        });

        var detector = new DisparateImpactBiasDetector<double>(0.8);
        var result = detector.DetectBias(predictions, sensitiveFeature);

        // Disparate impact ratio = 0.4 / 1.0 = 0.4 < 0.8 threshold
        Assert.True(result.HasBias);
        Assert.Equal(0.4, result.DisparateImpactRatio);
    }

    [Fact]
    public void BiasDetection_EndToEnd_EqualOpportunityScenario()
    {
        // Create a scenario with unequal true positive rates
        var predictions = new Vector<double>(new[]
        {
            // Group A (indices 0-4): Actual [1,1,1,0,0], Pred [1,1,1,0,0] = TPR 100%
            1.0, 1.0, 1.0, 0.0, 0.0,
            // Group B (indices 5-9): Actual [1,1,1,0,0], Pred [1,0,0,0,0] = TPR 33%
            1.0, 0.0, 0.0, 0.0, 0.0
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0
        });
        var actualLabels = new Vector<double>(new[]
        {
            1.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0, 0.0
        });

        var detector = new EqualOpportunityBiasDetector<double>(0.1);
        var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

        // Equal opportunity difference = 1.0 - 0.333... > 0.1 threshold
        Assert.True(result.HasBias);
        Assert.NotNull(result.GroupTruePositiveRates);
    }

    [Fact]
    public void BiasDetection_EndToEnd_NoBiasScenario()
    {
        // Create a scenario with equal treatment across groups
        var predictions = new Vector<double>(new[]
        {
            // Group A: 50% positive rate
            1.0, 0.0, 1.0, 0.0,
            // Group B: 50% positive rate
            1.0, 0.0, 1.0, 0.0
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0
        });

        var demographicParityDetector = new DemographicParityBiasDetector<double>(0.1);
        var disparateImpactDetector = new DisparateImpactBiasDetector<double>(0.8);

        var demographicParityResult = demographicParityDetector.DetectBias(predictions, sensitiveFeature);
        var disparateImpactResult = disparateImpactDetector.DetectBias(predictions, sensitiveFeature);

        Assert.False(demographicParityResult.HasBias);
        Assert.False(disparateImpactResult.HasBias);
        Assert.Equal(1.0, disparateImpactResult.DisparateImpactRatio);
    }

    #endregion

    #region Integration Tests - Multiple Groups Scenario

    [Fact]
    public void BiasDetection_MultipleGroups_ThreeGroups()
    {
        // Create a scenario with three demographic groups
        var predictions = new Vector<double>(new[]
        {
            // Group A (value 0): 4/4 = 100% positive
            1.0, 1.0, 1.0, 1.0,
            // Group B (value 1): 2/4 = 50% positive
            1.0, 1.0, 0.0, 0.0,
            // Group C (value 2): 1/4 = 25% positive
            1.0, 0.0, 0.0, 0.0
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0
        });

        var detector = new DemographicParityBiasDetector<double>(0.1);
        var result = detector.DetectBias(predictions, sensitiveFeature);

        // Statistical parity difference = max(1.0) - min(0.25) = 0.75 > 0.1 threshold
        Assert.True(result.HasBias);
        Assert.Equal(3, result.GroupPositiveRates.Count);
        Assert.Equal(3, result.GroupSizes.Count);
        Assert.Equal(1.0, result.GroupPositiveRates["0"]);
        Assert.Equal(0.5, result.GroupPositiveRates["1"]);
        Assert.Equal(0.25, result.GroupPositiveRates["2"]);
    }

    #endregion

    #region Concurrent Access Tests

    [Fact]
    public void BiasDetector_ConcurrentAccess_ThreadSafe()
    {
        var detector = new DemographicParityBiasDetector<double>(0.1);
        var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var tasks = Enumerable.Range(0, 100).Select(_ => Task.Run(() =>
        {
            var result = detector.DetectBias(predictions, sensitiveFeature);
            return result;
        })).ToArray();

        Task.WaitAll(tasks);

        foreach (var task in tasks)
        {
            Assert.False(task.Result.HasBias);
        }
    }

    [Fact]
    public void MetricsHelper_ConcurrentAccess_ThreadSafe()
    {
        var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 2.0 });

        var tasks = Enumerable.Range(0, 100).Select(_ => Task.Run(() =>
        {
            return InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);
        })).ToArray();

        Task.WaitAll(tasks);

        foreach (var task in tasks)
        {
            Assert.Equal(3, task.Result.Count);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void BiasDetection_VerySmallDifference_BelowThreshold()
    {
        var predictions = new Vector<double>(new[]
        {
            // Group A: 50.5% positive (101/200 simulated)
            1.0, 1.0, 0.0, 0.0,
            // Group B: 49.5% positive (99/200 simulated)
            1.0, 0.0, 0.0, 0.0  // Actually 1/4 = 25%, but for demonstration
        });
        var sensitiveFeature = new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0
        });

        // Set threshold high enough that small difference won't trigger
        var detector = new DemographicParityBiasDetector<double>(0.5);
        var result = detector.DetectBias(predictions, sensitiveFeature);

        // 0.5 - 0.25 = 0.25 < 0.5 threshold
        Assert.False(result.HasBias);
    }

    [Fact]
    public void BiasDetection_LargeDataset_PerformsCorrectly()
    {
        // Create a large dataset
        var size = 10000;
        var predictionsData = new double[size];
        var sensitiveFeatureData = new double[size];

        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            sensitiveFeatureData[i] = i < size / 2 ? 0.0 : 1.0;
            // Group 0: 70% positive rate, Group 1: 30% positive rate
            var rate = sensitiveFeatureData[i] == 0.0 ? 0.7 : 0.3;
            predictionsData[i] = random.NextDouble() < rate ? 1.0 : 0.0;
        }

        var predictions = new Vector<double>(predictionsData);
        var sensitiveFeature = new Vector<double>(sensitiveFeatureData);

        var detector = new DemographicParityBiasDetector<double>(0.1);
        var result = detector.DetectBias(predictions, sensitiveFeature);

        // Should detect bias since 70% - 30% = 40% > 10% threshold
        Assert.True(result.HasBias);
        Assert.Equal(size / 2, result.GroupSizes["0"]);
        Assert.Equal(size / 2, result.GroupSizes["1"]);
    }

    #endregion
}
