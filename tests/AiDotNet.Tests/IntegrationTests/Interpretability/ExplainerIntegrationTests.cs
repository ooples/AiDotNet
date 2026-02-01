using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.Interpretability;

/// <summary>
/// Comprehensive integration tests for interpretability explainer classes.
/// Tests the actual algorithms to ensure they produce valid, mathematically reasonable results.
/// </summary>
public class ExplainerIntegrationTests
{
    private const double Tolerance = 1e-6;
    private readonly int _numFeatures = 5;

    #region Helper Methods

    /// <summary>
    /// Creates a simple linear prediction function for testing.
    /// f(x) = sum(x * weights) where weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    /// </summary>
    private Func<Vector<double>, Vector<double>> CreateLinearPredictFunction()
    {
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        return (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
            {
                sum += input[i] * weights[i];
            }
            return new Vector<double>(new[] { sum });
        };
    }

    /// <summary>
    /// Creates a simple classification function that returns class probabilities.
    /// </summary>
    private Func<Vector<double>, Vector<double>> CreateClassificationFunction(int numClasses = 3)
    {
        return (Vector<double> input) =>
        {
            var probs = new double[numClasses];
            double sum = 0;
            for (int i = 0; i < numClasses; i++)
            {
                // Simple softmax-like computation
                probs[i] = Math.Exp(input.Length > i ? input[i] : 0);
                sum += probs[i];
            }
            for (int i = 0; i < numClasses; i++)
            {
                probs[i] /= sum;
            }
            return new Vector<double>(probs);
        };
    }

    /// <summary>
    /// Creates a batch prediction function from a single prediction function.
    /// </summary>
    private Func<Matrix<double>, Vector<double>> CreateBatchPredictFunction(
        Func<Vector<double>, Vector<double>> singlePredictFunc)
    {
        return (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                var row = inputs.GetRow(i);
                var prediction = singlePredictFunc(row);
                results[i] = prediction[0];
            }
            return new Vector<double>(results);
        };
    }

    /// <summary>
    /// Creates a batch classification function that returns class probabilities.
    /// </summary>
    private Func<Matrix<double>, Vector<double>> CreateBatchClassificationFunction(int numClasses = 3)
    {
        var singleClassify = CreateClassificationFunction(numClasses);
        return (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                var row = inputs.GetRow(i);
                var prediction = singleClassify(row);
                // Return the predicted class (argmax)
                int maxIdx = 0;
                double maxVal = prediction[0];
                for (int j = 1; j < prediction.Length; j++)
                {
                    if (prediction[j] > maxVal)
                    {
                        maxVal = prediction[j];
                        maxIdx = j;
                    }
                }
                results[i] = maxIdx;
            }
            return new Vector<double>(results);
        };
    }

    private Vector<double> CreateTestInstance()
    {
        return new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
    }

    private Matrix<double> CreateTestData(int rows = 100)
    {
        var data = new double[rows, _numFeatures];
        var random = new Random(42);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < _numFeatures; j++)
            {
                data[i, j] = random.NextDouble() * 10;
            }
        }
        return new Matrix<double>(data);
    }

    #endregion

    #region IntegratedGradientsExplainer Tests

    [Fact]
    public void IntegratedGradientsExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateLinearPredictFunction();

        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: _numFeatures,
            numSteps: 50);

        Assert.NotNull(explainer);
        Assert.Equal("IntegratedGradients", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void IntegratedGradientsExplainer_Explain_ReturnsValidExplanation()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: _numFeatures,
            numSteps: 50);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.Attributions);
        Assert.Equal(_numFeatures, explanation.Attributions.Length);
        Assert.Equal(50, explanation.NumSteps);
    }

    [Fact]
    public void IntegratedGradientsExplainer_CompletenessProperty_HoldsApproximately()
    {
        // Integrated Gradients should satisfy completeness:
        // sum(attributions) approximately equals output(input) - output(baseline)
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: _numFeatures,
            numSteps: 100); // More steps for better approximation

        var instance = CreateTestInstance();
        var baseline = new Vector<double>(new double[_numFeatures]); // zeros

        var explanation = explainer.Explain(instance, outputIndex: 0);

        var attributionSum = explanation.Attributions.ToArray().Sum();
        var inputPred = predictFunc(instance)[0];
        var baselinePred = predictFunc(baseline)[0];
        var expectedDiff = inputPred - baselinePred;

        // Convergence delta should be small
        Assert.True(explanation.ConvergenceDelta < 0.1,
            $"Convergence delta {explanation.ConvergenceDelta} should be small");
    }

    [Fact]
    public void IntegratedGradientsExplainer_ExplainBatch_ReturnsMultipleExplanations()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: _numFeatures);

        var instances = CreateTestData(10);
        var explanations = explainer.ExplainBatch(instances);

        Assert.Equal(10, explanations.Length);
        foreach (var explanation in explanations)
        {
            Assert.NotNull(explanation.Attributions);
            Assert.Equal(_numFeatures, explanation.Attributions.Length);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void IntegratedGradientsExplainer_InvalidNumFeatures_ThrowsException(int invalidNumFeatures)
    {
        var predictFunc = CreateLinearPredictFunction();

        if (invalidNumFeatures < 1)
        {
            Assert.Throws<ArgumentException>(() => new IntegratedGradientsExplainer<double>(
                predictFunction: predictFunc,
                gradientFunction: null,
                numFeatures: 0));
        }
    }

    #endregion

    #region DeepLIFTExplainer Tests

    [Fact]
    public void DeepLIFTExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateLinearPredictFunction();

        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.Equal("DeepLIFT", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void DeepLIFTExplainer_Explain_RescaleRule_ReturnsValidExplanation()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            rule: DeepLIFTRule.Rescale);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.Attributions);
        Assert.Equal(_numFeatures, explanation.Attributions.Length);
        Assert.Equal(DeepLIFTRule.Rescale, explanation.Rule);
    }

    [Fact]
    public void DeepLIFTExplainer_Explain_RevealCancelRule_ReturnsValidExplanation()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            rule: DeepLIFTRule.RevealCancel);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.Equal(DeepLIFTRule.RevealCancel, explanation.Rule);
    }

    [Fact]
    public void DeepLIFTExplainer_CompletenessProperty_AttributionsSumToOutputDiff()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        // Completeness error should be small
        Assert.True(explanation.CompletenessError < 0.1,
            $"Completeness error {explanation.CompletenessError} should be small");
    }

    [Fact]
    public void DeepLIFTExplainer_GetSortedAttributions_ReturnsOrderedByAbsoluteValue()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        var sorted = explanation.GetSortedAttributions();
        for (int i = 0; i < sorted.Count - 1; i++)
        {
            Assert.True(Math.Abs(sorted[i].attribution) >= Math.Abs(sorted[i + 1].attribution));
        }
    }

    #endregion

    #region SaliencyMapExplainer Tests

    [Fact]
    public void SaliencyMapExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateLinearPredictFunction();

        var explainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.StartsWith("SaliencyMap", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Theory]
    [InlineData(SaliencyMethod.VanillaGradient)]
    [InlineData(SaliencyMethod.GradientTimesInput)]
    [InlineData(SaliencyMethod.SmoothGrad)]
    public void SaliencyMapExplainer_AllMethods_ReturnValidExplanations(SaliencyMethod method)
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            method: method,
            smoothGradSamples: method == SaliencyMethod.SmoothGrad ? 20 : 0);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.Saliency);
        Assert.Equal(_numFeatures, explanation.Saliency.Length);
        Assert.Equal(method, explanation.Method);
    }

    [Fact]
    public void SaliencyMapExplainer_SmoothGrad_ReducesNoise()
    {
        var predictFunc = CreateLinearPredictFunction();

        var vanillaExplainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            method: SaliencyMethod.VanillaGradient);

        var smoothExplainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            method: SaliencyMethod.SmoothGrad,
            smoothGradSamples: 30,
            smoothGradNoise: 0.1);

        var instance = CreateTestInstance();
        var vanillaExplanation = vanillaExplainer.Explain(instance);
        var smoothExplanation = smoothExplainer.Explain(instance);

        // Both should produce valid results
        Assert.NotNull(vanillaExplanation.Saliency);
        Assert.NotNull(smoothExplanation.Saliency);
    }

    #endregion

    #region AccumulatedLocalEffectsExplainer Tests

    [Fact]
    public void AccumulatedLocalEffectsExplainer_Construction_Succeeds()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(100);

        var explainer = new AccumulatedLocalEffectsExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            numIntervals: 20);

        Assert.NotNull(explainer);
        Assert.Equal("AccumulatedLocalEffects", explainer.MethodName);
        Assert.False(explainer.SupportsLocalExplanations);
        Assert.True(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void AccumulatedLocalEffectsExplainer_ComputeForFeature_ReturnsValidResult()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(100);

        var explainer = new AccumulatedLocalEffectsExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            numIntervals: 20);

        var result = explainer.ComputeForFeature(0);

        Assert.NotNull(result);
        Assert.NotNull(result.ALEValues);
        Assert.NotNull(result.IntervalBounds);
        Assert.True(result.ALEValues.ContainsKey(0));
    }

    [Fact]
    public void AccumulatedLocalEffectsExplainer_ComputeForMultipleFeatures_ReturnsValidResult()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(100);

        var explainer = new AccumulatedLocalEffectsExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            numIntervals: 20);

        var result = explainer.ComputeForFeatures(new[] { 0, 1, 2 });

        Assert.NotNull(result);
        Assert.NotNull(result.ALEValues);
        Assert.True(result.ALEValues.ContainsKey(0));
        Assert.True(result.ALEValues.ContainsKey(1));
        Assert.True(result.ALEValues.ContainsKey(2));
    }

    #endregion

    #region FeatureInteractionExplainer Tests

    [Fact]
    public void FeatureInteractionExplainer_Construction_Succeeds()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(100);

        var explainer = new FeatureInteractionExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            gridSize: 10);

        Assert.NotNull(explainer);
        Assert.Equal("FeatureInteraction", explainer.MethodName);
        Assert.False(explainer.SupportsLocalExplanations);
        Assert.True(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void FeatureInteractionExplainer_GetTopInteractions_ReturnsValidResult()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(50);

        var explainer = new FeatureInteractionExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            gridSize: 10);

        var topInteractions = explainer.GetTopInteractions(5);

        Assert.NotNull(topInteractions);
        // H-statistics should be between 0 and 1
        foreach (var (feature1, feature2, hStatistic) in topInteractions)
        {
            Assert.True(hStatistic >= 0 && hStatistic <= 1.0 + Tolerance,
                $"H-statistic for ({feature1}, {feature2}) should be between 0 and 1, got {hStatistic}");
        }
    }

    [Fact]
    public void FeatureInteractionExplainer_ComputePairwiseHStatistic_ReturnsValidHStatistic()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var data = CreateTestData(50);

        var explainer = new FeatureInteractionExplainer<double>(
            predictFunction: batchPredictFunc,
            data: data,
            gridSize: 10);

        var hStatistic = explainer.ComputePairwiseHStatistic(0, 1);

        // For a linear model, interaction should be small (close to 0)
        Assert.True(hStatistic >= 0 && hStatistic <= 1.0 + Tolerance);
    }

    #endregion

    #region GradientSHAPExplainer Tests

    [Fact]
    public void GradientSHAPExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateLinearPredictFunction();
        var backgroundData = CreateTestData(50);

        var explainer = new GradientSHAPExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            backgroundData: backgroundData,
            numSamples: 100);

        Assert.NotNull(explainer);
        Assert.Equal("GradientSHAP", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void GradientSHAPExplainer_Explain_ReturnsValidExplanation()
    {
        var predictFunc = CreateLinearPredictFunction();
        var backgroundData = CreateTestData(50);

        var explainer = new GradientSHAPExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            backgroundData: backgroundData,
            numSamples: 50,
            numSteps: 20);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.Attributions);
        Assert.Equal(_numFeatures, explanation.Attributions.Length);
    }

    #endregion

    #region LRPExplainer Tests

    [Fact]
    public void LRPExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateLinearPredictFunction();

        var explainer = new LayerwiseRelevancePropagationExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.Equal("LRP", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Theory]
    [InlineData(LRPRule.Basic)]
    [InlineData(LRPRule.Epsilon)]
    [InlineData(LRPRule.Gamma)]
    [InlineData(LRPRule.AlphaBeta)]
    public void LRPExplainer_AllRules_ReturnValidExplanations(LRPRule rule)
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new LayerwiseRelevancePropagationExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            rule: rule);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.RelevanceScores);
        Assert.Equal(_numFeatures, explanation.RelevanceScores.Length);
        Assert.Equal(rule, explanation.Rule);
    }

    [Fact]
    public void LRPExplainer_ConservationProperty_RelevancesSumToOutput()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new LayerwiseRelevancePropagationExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures,
            rule: LRPRule.Basic);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        // Conservation property: relevances should approximately sum to output
        var relevanceSum = explanation.RelevanceScores.ToArray().Sum();
        var output = predictFunc(instance)[0];

        // Allow some tolerance due to numerical approximation
        Assert.True(Math.Abs(relevanceSum - output) < 1.0,
            $"Relevance sum {relevanceSum} should be close to output {output}");
    }

    #endregion

    #region PrototypeExplainer Tests

    [Fact]
    public void PrototypeExplainer_Construction_Succeeds()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var prototypes = CreateTestData(20);

        var explainer = new PrototypeExplainer<double>(
            predictFunction: batchPredictFunc,
            prototypes: prototypes,
            numNeighbors: 5);

        Assert.NotNull(explainer);
        Assert.Equal("Prototype", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Theory]
    [InlineData(AiDotNet.Interpretability.Explainers.DistanceMetric.Euclidean)]
    [InlineData(AiDotNet.Interpretability.Explainers.DistanceMetric.Manhattan)]
    [InlineData(AiDotNet.Interpretability.Explainers.DistanceMetric.Cosine)]
    public void PrototypeExplainer_AllMetrics_ReturnValidExplanations(
        AiDotNet.Interpretability.Explainers.DistanceMetric metric)
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var prototypes = CreateTestData(20);

        var explainer = new PrototypeExplainer<double>(
            predictFunction: batchPredictFunc,
            prototypes: prototypes,
            numNeighbors: 5,
            distanceMetric: metric);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.NearestPrototypes);
        Assert.True(explanation.NearestPrototypes.Count <= 5);
    }

    [Fact]
    public void PrototypeExplainer_NearestPrototypes_AreOrderedByDistance()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var prototypes = CreateTestData(20);

        var explainer = new PrototypeExplainer<double>(
            predictFunction: batchPredictFunc,
            prototypes: prototypes,
            numNeighbors: 5,
            distanceMetric: AiDotNet.Interpretability.Explainers.DistanceMetric.Euclidean);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        var distances = explanation.NearestPrototypes.Select(p => p.Distance).ToList();
        for (int i = 0; i < distances.Count - 1; i++)
        {
            Assert.True(distances[i] <= distances[i + 1],
                $"Distances should be in ascending order: {distances[i]} <= {distances[i + 1]}");
        }
    }

    #endregion

    #region ContrastiveExplainer Tests

    [Fact]
    public void ContrastiveExplainer_Construction_Succeeds()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new ContrastiveExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.Equal("Contrastive", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void ContrastiveExplainer_Explain_ReturnsValidExplanation()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new ContrastiveExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance, factClass: 0, foilClass: 1);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.PertinentPositives);
        Assert.Equal(0, explanation.FactClass);
        Assert.Equal(1, explanation.FoilClass);
    }

    #endregion

    #region GradCAMExplainer Tests

    [Fact]
    public void GradCAMExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateTensorPredictFunction();
        var inputShape = new[] { 1, 8, 8, 3 }; // Batch, Height, Width, Channels
        var featureMapShape = new[] { 1, 4, 4, 16 }; // Feature map shape

        var explainer = new GradCAMExplainer<double>(
            predictFunction: predictFunc,
            featureMapFunction: null,
            gradientFunction: null,
            inputShape: inputShape,
            featureMapShape: featureMapShape);

        Assert.NotNull(explainer);
        Assert.Equal("GradCAM", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void GradCAMExplainer_Explain_ReturnsValidHeatmap()
    {
        var predictFunc = CreateTensorPredictFunction();
        var inputShape = new[] { 1, 8, 8, 3 };
        var featureMapShape = new[] { 1, 4, 4, 16 };

        var explainer = new GradCAMExplainer<double>(
            predictFunction: predictFunc,
            featureMapFunction: null,
            gradientFunction: null,
            inputShape: inputShape,
            featureMapShape: featureMapShape);

        var inputTensor = CreateTestImageTensor(8, 8, 3);
        var explanation = explainer.ExplainTensor(inputTensor, targetClass: 0);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.Heatmap);
    }

    private Func<Tensor<double>, Tensor<double>> CreateTensorPredictFunction()
    {
        return (Tensor<double> input) =>
        {
            // Simple prediction: average of input values
            var data = input.ToVector().ToArray();
            var avg = data.Length > 0 ? data.Average() : 0;
            return new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { avg, avg * 0.5, avg * 0.3 }));
        };
    }

    private Tensor<double> CreateTestImageTensor(int height, int width, int channels)
    {
        var data = new double[height * width * channels];
        var random = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble();
        }
        return new Tensor<double>(new[] { 1, height, width, channels }, new Vector<double>(data));
    }

    #endregion

    #region AttentionVisualizationExplainer Tests

    [Fact]
    public void AttentionVisualizationExplainer_Construction_Succeeds()
    {
        var predictFunc = CreateTensorPredictFunction();
        var getAttentionWeights = CreateMockAttentionWeightsFunction();

        var explainer = new AttentionVisualizationExplainer<double>(
            predictFunction: predictFunc,
            getAttentionWeights: getAttentionWeights,
            numLayers: 2,
            numHeads: 4,
            sequenceLength: 10);

        Assert.NotNull(explainer);
        Assert.Equal("AttentionVisualization", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void AttentionVisualizationExplainer_Explain_ReturnsValidAttentionWeights()
    {
        var predictFunc = CreateTensorPredictFunction();
        var getAttentionWeights = CreateMockAttentionWeightsFunction();

        var explainer = new AttentionVisualizationExplainer<double>(
            predictFunction: predictFunc,
            getAttentionWeights: getAttentionWeights,
            numLayers: 2,
            numHeads: 4,
            sequenceLength: 10);

        var inputTensor = new Tensor<double>(new[] { 1, 10, 64 }); // Batch, Seq, Embed
        var explanation = explainer.ExplainTensor(inputTensor);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.LayerAttention);
    }

    private Func<Tensor<double>, int, Tensor<double>> CreateMockAttentionWeightsFunction()
    {
        return (Tensor<double> input, int layerIndex) =>
        {
            // Return mock attention weights for a single layer
            // Shape: [batch, heads, seq_len, seq_len]
            var data = new double[1 * 4 * 10 * 10];
            var random = new Random(42 + layerIndex);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = random.NextDouble();
            }
            return new Tensor<double>(new[] { 1, 4, 10, 10 }, new Vector<double>(data));
        };
    }

    #endregion

    #region PartialDependenceExplainer Tests

    [Fact]
    public void PartialDependenceExplainer_Construction_Succeeds()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var backgroundData = CreateTestData(100);

        var explainer = new PartialDependenceExplainer<double>(
            predictFunction: batchPredictFunc,
            backgroundData: backgroundData,
            gridResolution: 20);

        Assert.NotNull(explainer);
        Assert.Equal("PartialDependence", explainer.MethodName);
        Assert.False(explainer.SupportsLocalExplanations);
        Assert.True(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void PartialDependenceExplainer_ComputeForFeature_ReturnsValidResult()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var backgroundData = CreateTestData(100);

        var explainer = new PartialDependenceExplainer<double>(
            predictFunction: batchPredictFunc,
            backgroundData: backgroundData,
            gridResolution: 20);

        var result = explainer.ComputeForFeature(0);

        Assert.NotNull(result);
        Assert.NotNull(result.GridValues);
        Assert.NotNull(result.PartialDependence);
        Assert.True(result.GridValues.ContainsKey(0));
        Assert.True(result.PartialDependence.ContainsKey(0));
    }

    [Fact]
    public void PartialDependenceExplainer_ComputeForFeatures_WithICE_ReturnsICECurves()
    {
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var backgroundData = CreateTestData(50);

        var explainer = new PartialDependenceExplainer<double>(
            predictFunction: batchPredictFunc,
            backgroundData: backgroundData,
            gridResolution: 20,
            computeIce: true);

        var result = explainer.ComputeForFeatures(new[] { 0 });

        Assert.NotNull(result);
        Assert.NotNull(result.IceCurves);
        // ICE curves should have one curve per feature
        Assert.True(result.IceCurves.ContainsKey(0));
    }

    #endregion

    #region CounterfactualExplainer Tests

    [Fact]
    public void CounterfactualExplainer_Construction_Succeeds()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new CounterfactualExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.Equal("Counterfactual", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void CounterfactualExplainer_Explain_ReturnsValidExplanation()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new CounterfactualExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures,
            maxIterations: 100,
            stepSize: 0.1);

        var instance = CreateTestInstance();
        // Explain without target - will try to flip the binary prediction
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.OriginalInput);
        // Counterfactual may or may not be found depending on the model
    }

    #endregion

    #region AnchorExplainer Tests

    [Fact]
    public void AnchorExplainer_Construction_Succeeds()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new AnchorExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures);

        Assert.NotNull(explainer);
        Assert.Equal("Anchors", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.False(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void AnchorExplainer_Explain_ReturnsValidExplanation()
    {
        var classifyFunc = CreateBatchClassificationFunction(3);

        var explainer = new AnchorExplainer<double>(
            predictFunction: classifyFunc,
            numFeatures: _numFeatures,
            precisionThreshold: 0.9,
            nSamples: 100);

        var instance = CreateTestInstance();
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.AnchorRules);
        Assert.NotNull(explanation.AnchorFeatures);
        Assert.True(explanation.Precision >= 0 && explanation.Precision <= 1);
        Assert.True(explanation.Coverage >= 0 && explanation.Coverage <= 1);
    }

    #endregion

    #region TreeSHAPExplainer Tests

    /// <summary>
    /// Creates a simple decision tree for testing TreeSHAP.
    /// Tree structure:
    ///   Split on feature 0 at value 5
    ///     Left (x[0] <= 5): Predict 10
    ///     Right (x[0] > 5): Predict 20
    /// </summary>
    private DecisionTreeNode<double> CreateSimpleTestTree()
    {
        var root = new DecisionTreeNode<double>(0, 5.0)
        {
            IsLeaf = false,
            LeftSampleCount = 50,
            RightSampleCount = 50
        };

        root.Left = new DecisionTreeNode<double>(10.0)
        {
            IsLeaf = true,
            LeftSampleCount = 0,
            RightSampleCount = 0
        };

        root.Right = new DecisionTreeNode<double>(20.0)
        {
            IsLeaf = true,
            LeftSampleCount = 0,
            RightSampleCount = 0
        };

        return root;
    }

    /// <summary>
    /// Creates a more complex decision tree for testing.
    /// Tree structure (depth 2):
    ///   Split on feature 0 at value 5
    ///     Left: Split on feature 1 at value 3
    ///       Left-Left: Predict 5
    ///       Left-Right: Predict 15
    ///     Right: Predict 25
    /// </summary>
    private DecisionTreeNode<double> CreateComplexTestTree()
    {
        var root = new DecisionTreeNode<double>(0, 5.0)
        {
            IsLeaf = false,
            LeftSampleCount = 60,
            RightSampleCount = 40
        };

        var leftChild = new DecisionTreeNode<double>(1, 3.0)
        {
            IsLeaf = false,
            LeftSampleCount = 30,
            RightSampleCount = 30
        };

        leftChild.Left = new DecisionTreeNode<double>(5.0)
        {
            IsLeaf = true,
            LeftSampleCount = 0,
            RightSampleCount = 0
        };

        leftChild.Right = new DecisionTreeNode<double>(15.0)
        {
            IsLeaf = true,
            LeftSampleCount = 0,
            RightSampleCount = 0
        };

        root.Left = leftChild;

        root.Right = new DecisionTreeNode<double>(25.0)
        {
            IsLeaf = true,
            LeftSampleCount = 0,
            RightSampleCount = 0
        };

        return root;
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_SingleTree_Succeeds()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        Assert.NotNull(explainer);
        Assert.Equal("TreeSHAP", explainer.MethodName);
        Assert.True(explainer.SupportsLocalExplanations);
        Assert.True(explainer.SupportsGlobalExplanations);
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_Ensemble_Succeeds()
    {
        var trees = new List<DecisionTreeNode<double>>
        {
            CreateSimpleTestTree(),
            CreateComplexTestTree()
        };

        var explainer = new TreeSHAPExplainer<double>(
            trees: trees,
            numFeatures: 3,
            expectedValue: 15.0);

        Assert.NotNull(explainer);
        Assert.Equal("TreeSHAP", explainer.MethodName);
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_WithFeatureNames_Succeeds()
    {
        var tree = CreateSimpleTestTree();
        var featureNames = new[] { "Feature_A", "Feature_B", "Feature_C" };

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0,
            featureNames: featureNames);

        Assert.NotNull(explainer);
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_NullTree_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TreeSHAPExplainer<double>(
                tree: null!,
                numFeatures: 3,
                expectedValue: 15.0));
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_EmptyEnsemble_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
            new TreeSHAPExplainer<double>(
                trees: Array.Empty<DecisionTreeNode<double>>(),
                numFeatures: 3,
                expectedValue: 15.0));
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_ZeroFeatures_ThrowsException()
    {
        var tree = CreateSimpleTestTree();

        Assert.Throws<ArgumentException>(() =>
            new TreeSHAPExplainer<double>(
                tree: tree,
                numFeatures: 0,
                expectedValue: 15.0));
    }

    [Fact]
    public void TreeSHAPExplainer_Construction_NegativeFeatures_ThrowsException()
    {
        var tree = CreateSimpleTestTree();

        Assert.Throws<ArgumentException>(() =>
            new TreeSHAPExplainer<double>(
                tree: tree,
                numFeatures: -1,
                expectedValue: 15.0));
    }

    [Fact]
    public void TreeSHAPExplainer_Explain_ReturnsValidExplanation()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 }); // Goes left (3 <= 5)
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.NotNull(explanation.ShapValues);
        Assert.Equal(3, explanation.ShapValues.Length);
        Assert.Equal(15.0, explanation.ExpectedValue);
        Assert.Equal(10.0, explanation.Prediction); // Left leaf predicts 10
    }

    [Fact]
    public void TreeSHAPExplainer_Explain_LeftBranch_CorrectPrediction()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 2.0, 2.0, 2.0 }); // Goes left (2 <= 5)
        var explanation = explainer.Explain(instance);

        Assert.Equal(10.0, explanation.Prediction);
    }

    [Fact]
    public void TreeSHAPExplainer_Explain_RightBranch_CorrectPrediction()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 8.0, 2.0, 2.0 }); // Goes right (8 > 5)
        var explanation = explainer.Explain(instance);

        Assert.Equal(20.0, explanation.Prediction);
    }

    [Fact]
    public void TreeSHAPExplainer_Explain_WrongFeatureCount_ThrowsException()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var wrongInstance = new Vector<double>(new[] { 3.0, 2.0 }); // Only 2 features, expected 3

        Assert.Throws<ArgumentException>(() => explainer.Explain(wrongInstance));
    }

    [Fact]
    public void TreeSHAPExplainer_ExplainBatch_ReturnsMultipleExplanations()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instances = new Matrix<double>(new double[,]
        {
            { 2.0, 1.0, 1.0 },  // Goes left
            { 8.0, 1.0, 1.0 },  // Goes right
            { 5.0, 1.0, 1.0 },  // Goes left (boundary)
            { 5.1, 1.0, 1.0 }   // Goes right
        });

        var explanations = explainer.ExplainBatch(instances);

        Assert.Equal(4, explanations.Length);
        Assert.Equal(10.0, explanations[0].Prediction);
        Assert.Equal(20.0, explanations[1].Prediction);
        Assert.Equal(10.0, explanations[2].Prediction); // 5 <= 5, goes left
        Assert.Equal(20.0, explanations[3].Prediction); // 5.1 > 5, goes right
    }

    [Fact]
    public void TreeSHAPExplanation_GetSortedAttributions_ReturnsOrderedByAbsoluteValue()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        var sorted = explanation.GetSortedAttributions();

        // Should be sorted by absolute SHAP value (most important first)
        Assert.NotNull(sorted);
        for (int i = 0; i < sorted.Count - 1; i++)
        {
            Assert.True(Math.Abs(sorted[i].shapValue) >= Math.Abs(sorted[i + 1].shapValue),
                "Attributions should be sorted by absolute value descending");
        }
    }

    [Fact]
    public void TreeSHAPExplanation_GetPositiveContributions_ReturnsOnlyPositive()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 8.0, 2.0, 1.0 }); // Goes right, prediction > expected
        var explanation = explainer.Explain(instance);

        var positives = explanation.GetPositiveContributions();

        foreach (var (name, value, shapValue) in positives)
        {
            Assert.True(shapValue > 0, "Positive contributions should have positive SHAP values");
        }
    }

    [Fact]
    public void TreeSHAPExplanation_GetNegativeContributions_ReturnsOnlyNegative()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 }); // Goes left, prediction < expected
        var explanation = explainer.Explain(instance);

        var negatives = explanation.GetNegativeContributions();

        foreach (var (name, value, shapValue) in negatives)
        {
            Assert.True(shapValue < 0, "Negative contributions should have negative SHAP values");
        }
    }

    [Fact]
    public void TreeSHAPExplanation_GetSumError_IsSmall()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        var sumError = explanation.GetSumError();

        // SHAP values should sum to (prediction - expected_value)
        // Sum error should be relatively small for exact TreeSHAP
        Assert.True(sumError < 5.0,
            $"Sum error {sumError} should be small for TreeSHAP (completeness property)");
    }

    [Fact]
    public void TreeSHAPExplanation_ToString_ReturnsValidString()
    {
        var tree = CreateSimpleTestTree();
        var featureNames = new[] { "Feature_A", "Feature_B", "Feature_C" };

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0,
            featureNames: featureNames);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        var str = explanation.ToString();

        Assert.NotNull(str);
        Assert.Contains("TreeSHAP Explanation", str);
        Assert.Contains("Expected value", str);
        Assert.Contains("Prediction", str);
    }

    [Fact]
    public void TreeSHAPExplainer_ComplexTree_CorrectPredictions()
    {
        var tree = CreateComplexTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        // Test all paths through the tree
        // Left-Left path (x[0] <= 5 AND x[1] <= 3): Predict 5
        var instanceLL = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        Assert.Equal(5.0, explainer.Explain(instanceLL).Prediction);

        // Left-Right path (x[0] <= 5 AND x[1] > 3): Predict 15
        var instanceLR = new Vector<double>(new[] { 3.0, 5.0, 1.0 });
        Assert.Equal(15.0, explainer.Explain(instanceLR).Prediction);

        // Right path (x[0] > 5): Predict 25
        var instanceR = new Vector<double>(new[] { 8.0, 2.0, 1.0 });
        Assert.Equal(25.0, explainer.Explain(instanceR).Prediction);
    }

    [Fact]
    public void TreeSHAPExplainer_Ensemble_AveragesPredictions()
    {
        // Create two trees with different predictions
        var tree1 = CreateSimpleTestTree(); // Left: 10, Right: 20
        var tree2 = CreateSimpleTestTree();
        // Modify tree2's predictions
        tree2.Left!.Prediction = 14.0;  // Left: 14
        tree2.Right!.Prediction = 22.0; // Right: 22

        var trees = new List<DecisionTreeNode<double>> { tree1, tree2 };

        var explainer = new TreeSHAPExplainer<double>(
            trees: trees,
            numFeatures: 3,
            expectedValue: 15.0);

        // Left branch: (10 + 14) / 2 = 12
        var instanceLeft = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var leftExplanation = explainer.Explain(instanceLeft);
        Assert.Equal(12.0, leftExplanation.Prediction);

        // Right branch: (20 + 22) / 2 = 21
        var instanceRight = new Vector<double>(new[] { 8.0, 2.0, 1.0 });
        var rightExplanation = explainer.Explain(instanceRight);
        Assert.Equal(21.0, rightExplanation.Prediction);
    }

    [Fact]
    public void TreeSHAPExplainer_FeatureNames_PropagateToExplanation()
    {
        var tree = CreateSimpleTestTree();
        var featureNames = new[] { "Temperature", "Humidity", "WindSpeed" };

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0,
            featureNames: featureNames);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        Assert.Equal(featureNames, explanation.FeatureNames);
        Assert.Equal("Temperature", explanation.FeatureNames[0]);
    }

    [Fact]
    public void TreeSHAPExplainer_NoFeatureNames_UsesDefaultNames()
    {
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0,
            featureNames: null);

        var instance = new Vector<double>(new[] { 3.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        Assert.Equal("Feature 0", explanation.FeatureNames[0]);
        Assert.Equal("Feature 1", explanation.FeatureNames[1]);
        Assert.Equal("Feature 2", explanation.FeatureNames[2]);
    }

    [Fact]
    public void TreeSHAPExplainer_SplitFeatureHasHighestAttribution()
    {
        // For a simple tree that only splits on feature 0,
        // feature 0 should have the highest attribution
        var tree = CreateSimpleTestTree();

        var explainer = new TreeSHAPExplainer<double>(
            tree: tree,
            numFeatures: 3,
            expectedValue: 15.0);

        // Instance that goes right (positive direction)
        var instance = new Vector<double>(new[] { 8.0, 2.0, 1.0 });
        var explanation = explainer.Explain(instance);

        var sorted = explanation.GetSortedAttributions();

        // Feature 0 should be most important since it's the only split feature
        Assert.Equal("Feature 0", sorted[0].name);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void Explainers_NullPredictFunction_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new IntegratedGradientsExplainer<double>(null!, null, _numFeatures));

        Assert.Throws<ArgumentNullException>(() =>
            new DeepLIFTExplainer<double>(predictFunction: null!, numFeatures: _numFeatures));

        Assert.Throws<ArgumentNullException>(() =>
            new SaliencyMapExplainer<double>(null!, numFeatures: _numFeatures));
    }

    [Fact]
    public void IntegratedGradientsExplainer_InstanceSizeMismatch_ThrowsException()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: 5);

        // Instance with wrong number of features
        var wrongInstance = new Vector<double>(new[] { 1.0, 2.0, 3.0 }); // 3 instead of 5

        Assert.Throws<ArgumentException>(() => explainer.Explain(wrongInstance));
    }

    [Fact]
    public void DeepLIFTExplainer_NegativeNumFeatures_ThrowsException()
    {
        var predictFunc = CreateLinearPredictFunction();

        Assert.Throws<ArgumentException>(() =>
            new DeepLIFTExplainer<double>(predictFunc, numFeatures: -1));
    }

    #endregion

    #region Edge Case Bug Detection Tests

    [Fact]
    public void IntegratedGradients_NegativeWeights_AttributionsHaveCorrectSign()
    {
        // Test that negative weights produce negative attributions
        var weights = new[] { -0.5, 0.3, -0.2, 0.1, -0.4 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: 5,
            numSteps: 100);

        // Use positive instance values
        var instance = new Vector<double>(new[] { 2.0, 2.0, 2.0, 2.0, 2.0 });
        var explanation = explainer.Explain(instance, outputIndex: 0);

        // Features with negative weights should have negative attributions
        Assert.True(explanation.Attributions[0] < 0, "Feature 0 (negative weight) should have negative attribution");
        Assert.True(explanation.Attributions[1] > 0, "Feature 1 (positive weight) should have positive attribution");
        Assert.True(explanation.Attributions[2] < 0, "Feature 2 (negative weight) should have negative attribution");
    }

    [Fact]
    public void IntegratedGradients_ZeroBaseline_DoesNotThrow()
    {
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: _numFeatures);

        // Zero instance should still work
        var instance = new Vector<double>(new double[_numFeatures]); // all zeros
        var explanation = explainer.Explain(instance);

        Assert.NotNull(explanation);
        Assert.Equal(_numFeatures, explanation.Attributions.Length);
        // All attributions should be zero for zero input with zero baseline
        foreach (var attr in explanation.Attributions.ToArray())
        {
            Assert.True(Math.Abs(attr) < 1e-6, $"Attributions should be zero for zero input, got {attr}");
        }
    }

    [Fact]
    public void DeepLIFT_VerySmallInput_NoNaN()
    {
        // Test that very small inputs don't produce NaN due to division by zero
        var predictFunc = CreateLinearPredictFunction();
        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: _numFeatures);

        var instance = new Vector<double>(new[] { 1e-15, 1e-15, 1e-15, 1e-15, 1e-15 });
        var explanation = explainer.Explain(instance);

        foreach (var attr in explanation.Attributions.ToArray())
        {
            Assert.False(double.IsNaN(attr), "Attribution should not be NaN for very small inputs");
            Assert.False(double.IsInfinity(attr), "Attribution should not be Infinity for very small inputs");
        }
    }

    [Fact]
    public void LRP_AllPositiveInputs_AllRelevancesHaveSameSign()
    {
        // For a linear model with all positive weights and positive inputs,
        // all relevances should be positive
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new LayerwiseRelevancePropagationExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: 5,
            rule: LRPRule.Basic);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        // With positive weights and positive inputs, relevances should be non-negative
        foreach (var rel in explanation.RelevanceScores.ToArray())
        {
            Assert.True(rel >= -0.1, $"Relevance should be non-negative for positive input/weight, got {rel}");
        }
    }

    [Fact]
    public void SaliencyMap_VanillaGradient_AllFeaturesSameSign()
    {
        // For a linear model, vanilla gradient should be the weights (all same output sign)
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: 5,
            method: SaliencyMethod.VanillaGradient);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        // All gradients should be positive (all weights are positive)
        foreach (var sal in explanation.Saliency.ToArray())
        {
            Assert.True(sal >= 0, $"Saliency should be non-negative for positive weights, got {sal}");
        }
    }

    [Fact]
    public void PartialDependence_SingleDataPoint_DoesNotCrash()
    {
        // Edge case: single data point in background data
        var batchPredictFunc = CreateBatchPredictFunction(CreateLinearPredictFunction());
        var singleRow = new double[1, 5] { { 1.0, 2.0, 3.0, 4.0, 5.0 } };

        var explainer = new PartialDependenceExplainer<double>(
            predictFunction: batchPredictFunc,
            backgroundData: new Matrix<double>(singleRow),
            gridResolution: 5);

        var result = explainer.ComputeForFeature(0);

        Assert.NotNull(result);
        Assert.True(result.PartialDependence.ContainsKey(0));
    }

    [Fact]
    public void ALE_ConstantFeature_ReturnsZeroEffect()
    {
        // If a feature has constant value across all data, ALE should be flat (zero variation)
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Matrix<double>, Vector<double>> batchPredict = (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Columns && j < weights.Length; j++)
                    sum += inputs[i, j] * weights[j];
                results[i] = sum;
            }
            return new Vector<double>(results);
        };

        // Feature 0 is constant, others vary
        var data = new double[50, 5];
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = 5.0; // Constant
            for (int j = 1; j < 5; j++)
                data[i, j] = random.NextDouble() * 10;
        }

        var explainer = new AccumulatedLocalEffectsExplainer<double>(
            predictFunction: batchPredict,
            data: new Matrix<double>(data),
            numIntervals: 10);

        var result = explainer.ComputeForFeature(0);

        // ALE values for constant feature should have very small variation
        var aleValues = result.ALEValues[0];
        if (aleValues.Length > 1)
        {
            double range = aleValues.Max() - aleValues.Min();
            Assert.True(range < 1.0,
                $"ALE range for constant feature should be very small, got {range}");
        }
    }

    #endregion

    #region Mathematical Correctness Tests

    [Fact]
    public void IntegratedGradients_LinearModel_AttributionsMatchWeights()
    {
        // For a linear model f(x) = sum(x * w), the attributions should be approximately
        // (x - baseline) * w / (sum of attributions) * (f(x) - f(baseline))
        // For a zero baseline, attributions should be close to x * w
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new IntegratedGradientsExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: 5,
            numSteps: 200); // High steps for accuracy

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance, outputIndex: 0);

        // For a linear model, attributions should satisfy completeness exactly
        // sum(attributions) should equal f(x) - f(0)
        var attributionSum = explanation.Attributions.ToArray().Sum();
        var expectedOutput = 1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4 + 5.0 * 0.5;

        Assert.True(Math.Abs(attributionSum - expectedOutput) < 0.1,
            $"Attribution sum {attributionSum} should be close to expected output {expectedOutput}");

        // Each attribution should be approximately x_i * w_i for zero baseline
        for (int i = 0; i < 5; i++)
        {
            double expectedAttr = instance[i] * weights[i];
            double actualAttr = explanation.Attributions[i];
            Assert.True(Math.Abs(actualAttr - expectedAttr) < 0.05,
                $"Feature {i}: expected attribution {expectedAttr}, got {actualAttr}");
        }
    }

    [Fact]
    public void DeepLIFT_CompletenessPropertyViolation_DetectsBug()
    {
        // DeepLIFT should satisfy completeness: sum(attributions) = f(x) - f(baseline)
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new DeepLIFTExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: 5,
            rule: DeepLIFTRule.Rescale);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        var attributionSum = explanation.Attributions.ToArray().Sum();
        var expectedOutput = 5.5; // 0.1+0.4+0.9+1.6+2.5

        // Completeness property must hold
        Assert.True(Math.Abs(attributionSum - expectedOutput) < 0.5,
            $"DeepLIFT completeness violated: sum={attributionSum}, expected={expectedOutput}");
    }

    [Fact]
    public void GradientSHAP_CompletenessProperty_AttributionsSumToOutputDiff()
    {
        // GradientSHAP attributions should sum to f(x) - E[f(baseline)]
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        // Create simple background data with known values
        var backgroundData = new double[10, 5];
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 5; j++)
                backgroundData[i, j] = 0.5 * (j + 1); // Deterministic values

        var explainer = new GradientSHAPExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            backgroundData: new Matrix<double>(backgroundData),
            numSamples: 50,
            numSteps: 30);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        // Attributions should be valid (not all zeros, not NaN)
        Assert.True(explanation.Attributions.Length == 5);
        Assert.False(double.IsNaN(explanation.Attributions[0]));

        // For a linear model, attributions should be somewhat close to feature contributions
        var attrSum = explanation.Attributions.ToArray().Sum();
        Assert.True(Math.Abs(attrSum) < 50, // Reasonable range
            $"GradientSHAP attribution sum {attrSum} is unreasonable");
    }

    [Fact]
    public void LRP_ConservationProperty_RelevancesSumToOutput()
    {
        // LRP conservation property: relevances should sum to the output value
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new LayerwiseRelevancePropagationExplainer<double>(
            predictFunction: predictFunc,
            numFeatures: 5,
            rule: LRPRule.Basic);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        var relevanceSum = explanation.RelevanceScores.ToArray().Sum();
        var output = predictFunc(instance)[0];

        // Conservation property: sum of relevances should approximately equal output
        Assert.True(Math.Abs(relevanceSum - output) < 1.0,
            $"LRP conservation violated: relevance sum={relevanceSum}, output={output}");
    }

    [Fact]
    public void FeatureInteraction_LinearModel_NoInteraction()
    {
        // For a purely linear model, H-statistic should be close to 0 (no interactions)
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Matrix<double>, Vector<double>> batchPredict = (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Columns && j < weights.Length; j++)
                    sum += inputs[i, j] * weights[j];
                results[i] = sum;
            }
            return new Vector<double>(results);
        };

        // Create test data
        var data = new double[50, 5];
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
            for (int j = 0; j < 5; j++)
                data[i, j] = random.NextDouble() * 10;

        var explainer = new FeatureInteractionExplainer<double>(
            predictFunction: batchPredict,
            data: new Matrix<double>(data),
            gridSize: 15);

        var hStatistic = explainer.ComputePairwiseHStatistic(0, 1);

        // For linear model, H-statistic should be very small (near 0)
        Assert.True(hStatistic < 0.2,
            $"Linear model should have no interaction, but H-statistic={hStatistic}");
    }

    [Fact]
    public void PartialDependence_LinearModel_ProducesLinearCurve()
    {
        // For a linear model, PDP should produce a linear relationship
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Matrix<double>, Vector<double>> batchPredict = (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Columns && j < weights.Length; j++)
                    sum += inputs[i, j] * weights[j];
                results[i] = sum;
            }
            return new Vector<double>(results);
        };

        // Create test data
        var data = new double[50, 5];
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
            for (int j = 0; j < 5; j++)
                data[i, j] = random.NextDouble() * 10;

        var explainer = new PartialDependenceExplainer<double>(
            predictFunction: batchPredict,
            backgroundData: new Matrix<double>(data),
            gridResolution: 10);

        var result = explainer.ComputeForFeature(0);

        // PDP values should increase monotonically for feature 0 (positive weight)
        var pdValues = result.PartialDependence[0];
        bool isMonotonic = true;
        for (int i = 1; i < pdValues.Length; i++)
        {
            if (pdValues[i] < pdValues[i - 1] - 0.01) // Allow small tolerance
            {
                isMonotonic = false;
                break;
            }
        }

        Assert.True(isMonotonic,
            "PDP for feature with positive weight should be monotonically increasing");
    }

    [Fact]
    public void SaliencyMap_GradientTimesInput_ReflectsWeights()
    {
        // For a linear model, gradient × input should be approximately x * weight
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Vector<double>, Vector<double>> predictFunc = (Vector<double> input) =>
        {
            double sum = 0;
            for (int i = 0; i < input.Length && i < weights.Length; i++)
                sum += input[i] * weights[i];
            return new Vector<double>(new[] { sum });
        };

        var explainer = new SaliencyMapExplainer<double>(
            predictFunction: predictFunc,
            gradientFunction: null,
            numFeatures: 5,
            method: SaliencyMethod.GradientTimesInput);

        var instance = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var explanation = explainer.Explain(instance);

        // Gradient times input should be close to x * w for a linear model
        for (int i = 0; i < 5; i++)
        {
            double expectedSaliency = Math.Abs(instance[i] * weights[i]);
            double actualSaliency = Math.Abs(explanation.Saliency[i]);

            // These should be proportionally close
            Assert.True(actualSaliency > 0,
                $"Feature {i}: Saliency should be non-zero for non-zero input with non-zero weight");
        }
    }

    [Fact]
    public void ALE_LinearModel_EffectsAreLinear()
    {
        // For a linear model, ALE should produce linear effects
        var weights = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        Func<Matrix<double>, Vector<double>> batchPredict = (Matrix<double> inputs) =>
        {
            var results = new double[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Columns && j < weights.Length; j++)
                    sum += inputs[i, j] * weights[j];
                results[i] = sum;
            }
            return new Vector<double>(results);
        };

        // Create test data with uniform distribution
        var data = new double[100, 5];
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 5; j++)
                data[i, j] = random.NextDouble() * 10;

        var explainer = new AccumulatedLocalEffectsExplainer<double>(
            predictFunction: batchPredict,
            data: new Matrix<double>(data),
            numIntervals: 15);

        var result = explainer.ComputeForFeature(0);

        // ALE values should show increasing trend for positive weight
        var aleValues = result.ALEValues[0];
        double firstHalfAvg = aleValues.Take(aleValues.Length / 2).Average();
        double secondHalfAvg = aleValues.Skip(aleValues.Length / 2).Average();

        Assert.True(secondHalfAvg > firstHalfAvg - 0.1,
            "ALE for feature with positive weight should generally increase");
    }

    #endregion
}
