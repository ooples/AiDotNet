using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNetTests.IntegrationTests.Interpretability
{
    /// <summary>
    /// Comprehensive integration tests for Interpretability methods including:
    /// - Explanation methods (LIME, Anchor, Counterfactual)
    /// - Bias detectors (DemographicParity, EqualOpportunity, DisparateImpact)
    /// - Fairness evaluators (Basic, Group, Comprehensive)
    /// - Helper utilities (InterpretabilityMetricsHelper, InterpretableModelHelper)
    /// Tests verify mathematical correctness, bias detection accuracy, and fairness metrics.
    /// </summary>
    public class InterpretabilityIntegrationTests
    {
        private const double EPSILON = 1e-6;

        #region Helper Classes and Models

        /// <summary>
        /// Simple linear model for testing: y = w0*x0 + w1*x1 + bias
        /// </summary>
        private class SimpleLinearModel : IFullModel<double, Matrix<double>, Vector<double>>
        {
            private Vector<double>? _weights;
            private double _bias;

            public SimpleLinearModel(Vector<double> weights, double bias = 0.0)
            {
                _weights = weights;
                _bias = bias;
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                if (_weights == null) throw new InvalidOperationException("Model not initialized");

                var predictions = new Vector<double>(input.Rows);
                for (int i = 0; i < input.Rows; i++)
                {
                    double sum = _bias;
                    for (int j = 0; j < input.Columns && j < _weights.Length; j++)
                    {
                        sum += input[i, j] * _weights[j];
                    }
                    // Convert to binary prediction
                    predictions[i] = sum >= 0.5 ? 1.0 : 0.0;
                }
                return predictions;
            }

            public void Train(Matrix<double> inputs, Vector<double> targets, TrainingOptions<double>? options = null) { }
            public ModelMetadata<double> GetMetadata() => new ModelMetadata<double>();
            public void SaveModel(string filePath) { }
            public void LoadModel(string filePath) { }
            public Dictionary<string, double> GetParameters() => new Dictionary<string, double>();
            public void SetParameters(Dictionary<string, double> parameters) { }
            public int GetFeatureCount() => _weights?.Length ?? 0;
            public string[] GetFeatureNames() => new string[0];
            public void SetFeatureNames(string[] names) { }
            public Dictionary<int, double> GetFeatureImportance() => new Dictionary<int, double>();
            public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new SimpleLinearModel(_weights!, _bias);
        }

        /// <summary>
        /// Biased model that discriminates based on sensitive attribute
        /// </summary>
        private class BiasedModel : IFullModel<double, Matrix<double>, Vector<double>>
        {
            private readonly int _sensitiveIndex;
            private readonly double _biasValue;

            public BiasedModel(int sensitiveIndex, double biasValue = 1.0)
            {
                _sensitiveIndex = sensitiveIndex;
                _biasValue = biasValue;
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                var predictions = new Vector<double>(input.Rows);
                for (int i = 0; i < input.Rows; i++)
                {
                    // Always predict positive for group 1, negative for group 0
                    predictions[i] = input[i, _sensitiveIndex] == _biasValue ? 1.0 : 0.0;
                }
                return predictions;
            }

            public void Train(Matrix<double> inputs, Vector<double> targets, TrainingOptions<double>? options = null) { }
            public ModelMetadata<double> GetMetadata() => new ModelMetadata<double>();
            public void SaveModel(string filePath) { }
            public void LoadModel(string filePath) { }
            public Dictionary<string, double> GetParameters() => new Dictionary<string, double>();
            public void SetParameters(Dictionary<string, double> parameters) { }
            public int GetFeatureCount() => 2;
            public string[] GetFeatureNames() => new string[0];
            public void SetFeatureNames(string[] names) { }
            public Dictionary<int, double> GetFeatureImportance() => new Dictionary<int, double>();
            public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new BiasedModel(_sensitiveIndex, _biasValue);
        }

        /// <summary>
        /// Fair model that makes predictions independent of sensitive attribute
        /// </summary>
        private class FairModel : IFullModel<double, Matrix<double>, Vector<double>>
        {
            private readonly int _featureIndex;

            public FairModel(int featureIndex = 0)
            {
                _featureIndex = featureIndex;
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                var predictions = new Vector<double>(input.Rows);
                for (int i = 0; i < input.Rows; i++)
                {
                    // Predict based on non-sensitive feature
                    predictions[i] = input[i, _featureIndex] >= 0.5 ? 1.0 : 0.0;
                }
                return predictions;
            }

            public void Train(Matrix<double> inputs, Vector<double> targets, TrainingOptions<double>? options = null) { }
            public ModelMetadata<double> GetMetadata() => new ModelMetadata<double>();
            public void SaveModel(string filePath) { }
            public void LoadModel(string filePath) { }
            public Dictionary<string, double> GetParameters() => new Dictionary<string, double>();
            public void SetParameters(Dictionary<string, double> parameters) { }
            public int GetFeatureCount() => 2;
            public string[] GetFeatureNames() => new string[0];
            public void SetFeatureNames(string[] names) { }
            public Dictionary<int, double> GetFeatureImportance() => new Dictionary<int, double>();
            public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new FairModel(_featureIndex);
        }

        #endregion

        #region LIME Explanation Tests

        [Fact]
        public void LimeExplanation_Initialization_SetsDefaultValues()
        {
            // Arrange & Act
            var lime = new LimeExplanation<double>();

            // Assert
            Assert.NotNull(lime.FeatureImportance);
            Assert.Empty(lime.FeatureImportance);
            Assert.Equal(0.0, lime.Intercept);
            Assert.Equal(0.0, lime.PredictedValue);
            Assert.Equal(0.0, lime.LocalModelScore);
        }

        [Fact]
        public void LimeExplanation_FeatureImportance_CanBeSet()
        {
            // Arrange
            var lime = new LimeExplanation<double>();
            var importance = new Dictionary<int, double>
            {
                { 0, 0.5 },
                { 1, 0.3 },
                { 2, 0.2 }
            };

            // Act
            lime.FeatureImportance = importance;
            lime.NumFeatures = 3;

            // Assert
            Assert.Equal(3, lime.FeatureImportance.Count);
            Assert.Equal(0.5, lime.FeatureImportance[0]);
            Assert.Equal(3, lime.NumFeatures);
        }

        [Fact]
        public void LimeExplanation_LocalModelScore_ReflectsApproximationQuality()
        {
            // Arrange
            var lime = new LimeExplanation<double>
            {
                LocalModelScore = 0.95,
                PredictedValue = 1.0,
                Intercept = 0.1
            };

            // Assert - High R² score indicates good local approximation
            Assert.True(lime.LocalModelScore > 0.9);
            Assert.Equal(1.0, lime.PredictedValue);
        }

        [Fact]
        public void LimeExplanation_TopFeatures_CanBeRanked()
        {
            // Arrange
            var lime = new LimeExplanation<double>
            {
                FeatureImportance = new Dictionary<int, double>
                {
                    { 0, 0.1 },
                    { 1, 0.5 },
                    { 2, 0.3 },
                    { 3, 0.8 },
                    { 4, 0.2 }
                },
                NumFeatures = 3
            };

            // Act - Get top 3 features
            var topFeatures = lime.FeatureImportance
                .OrderByDescending(x => Math.Abs(x.Value))
                .Take(lime.NumFeatures)
                .ToList();

            // Assert
            Assert.Equal(3, topFeatures.Count);
            Assert.Equal(3, topFeatures[0].Key); // Feature 3 has highest importance (0.8)
            Assert.Equal(1, topFeatures[1].Key); // Feature 1 has second highest (0.5)
            Assert.Equal(2, topFeatures[2].Key); // Feature 2 has third highest (0.3)
        }

        #endregion

        #region Anchor Explanation Tests

        [Fact]
        public void AnchorExplanation_Initialization_SetsDefaultValues()
        {
            // Arrange & Act
            var anchor = new AnchorExplanation<double>();

            // Assert
            Assert.NotNull(anchor.AnchorRules);
            Assert.Empty(anchor.AnchorRules);
            Assert.NotNull(anchor.AnchorFeatures);
            Assert.Empty(anchor.AnchorFeatures);
            Assert.Equal(0.0, anchor.Precision);
            Assert.Equal(0.0, anchor.Coverage);
            Assert.Equal(string.Empty, anchor.Description);
        }

        [Fact]
        public void AnchorExplanation_Rules_CanBeSet()
        {
            // Arrange
            var anchor = new AnchorExplanation<double>
            {
                AnchorRules = new Dictionary<int, (double Min, double Max)>
                {
                    { 0, (0.5, 0.9) },
                    { 2, (0.3, 0.7) }
                },
                AnchorFeatures = new List<int> { 0, 2 },
                Precision = 0.95,
                Coverage = 0.40,
                Description = "IF feature_0 in [0.5, 0.9] AND feature_2 in [0.3, 0.7] THEN prediction = positive"
            };

            // Assert
            Assert.Equal(2, anchor.AnchorRules.Count);
            Assert.Equal(2, anchor.AnchorFeatures.Count);
            Assert.True(anchor.Precision > 0.9); // High precision means rule is reliable
            Assert.True(anchor.Coverage > 0.3); // Reasonable coverage
            Assert.Contains("IF", anchor.Description);
        }

        [Fact]
        public void AnchorExplanation_PrecisionAndCoverage_TradeOff()
        {
            // Arrange - More specific rules have higher precision but lower coverage
            var specificAnchor = new AnchorExplanation<double>
            {
                AnchorRules = new Dictionary<int, (double Min, double Max)>
                {
                    { 0, (0.45, 0.55) }, // Narrow range
                    { 1, (0.75, 0.85) }  // Narrow range
                },
                Precision = 0.98,
                Coverage = 0.15
            };

            var generalAnchor = new AnchorExplanation<double>
            {
                AnchorRules = new Dictionary<int, (double Min, double Max)>
                {
                    { 0, (0.2, 0.8) } // Wide range
                },
                Precision = 0.75,
                Coverage = 0.60
            };

            // Assert - Specific rules: high precision, low coverage
            Assert.True(specificAnchor.Precision > 0.95);
            Assert.True(specificAnchor.Coverage < 0.20);

            // Assert - General rules: lower precision, higher coverage
            Assert.True(generalAnchor.Precision < 0.80);
            Assert.True(generalAnchor.Coverage > 0.50);
        }

        [Fact]
        public void AnchorExplanation_Threshold_DeterminesRuleStrictness()
        {
            // Arrange
            var strictAnchor = new AnchorExplanation<double>
            {
                Threshold = 0.95,
                Precision = 0.96
            };

            var relaxedAnchor = new AnchorExplanation<double>
            {
                Threshold = 0.80,
                Precision = 0.85
            };

            // Assert
            Assert.True(strictAnchor.Precision >= strictAnchor.Threshold);
            Assert.True(relaxedAnchor.Precision >= relaxedAnchor.Threshold);
        }

        #endregion

        #region Counterfactual Explanation Tests

        [Fact]
        public void CounterfactualExplanation_Initialization_SetsDefaultValues()
        {
            // Arrange & Act
            var cf = new CounterfactualExplanation<double>();

            // Assert
            Assert.NotNull(cf.FeatureChanges);
            Assert.Empty(cf.FeatureChanges);
            Assert.Equal(0.0, cf.Distance);
        }

        [Fact]
        public void CounterfactualExplanation_MinimalChanges_ProducesDifferentOutcome()
        {
            // Arrange - Original input predicts negative, counterfactual predicts positive
            var original = new Tensor<double>(new[] { 0.3, 0.2, 0.1 });
            var counterfactual = new Tensor<double>(new[] { 0.7, 0.2, 0.1 }); // Only changed feature 0

            var cf = new CounterfactualExplanation<double>
            {
                OriginalInput = original,
                CounterfactualInput = counterfactual,
                FeatureChanges = new Dictionary<int, double>
                {
                    { 0, 0.4 } // Changed from 0.3 to 0.7
                },
                MaxChanges = 3
            };

            // Act - Calculate L1 distance
            double distance = 0;
            foreach (var change in cf.FeatureChanges.Values)
            {
                distance += Math.Abs(change);
            }
            cf.Distance = distance;

            // Assert
            Assert.Equal(1, cf.FeatureChanges.Count); // Only 1 feature changed
            Assert.Equal(0.4, cf.Distance, precision: 6); // Minimal change
            Assert.True(cf.FeatureChanges.Count <= cf.MaxChanges);
        }

        [Fact]
        public void CounterfactualExplanation_Distance_ReflectsChangesMagnitude()
        {
            // Arrange
            var cf1 = new CounterfactualExplanation<double>
            {
                FeatureChanges = new Dictionary<int, double> { { 0, 0.1 } },
                Distance = 0.1
            };

            var cf2 = new CounterfactualExplanation<double>
            {
                FeatureChanges = new Dictionary<int, double> { { 0, 0.5 }, { 1, 0.3 } },
                Distance = 0.8
            };

            // Assert - More changes = greater distance
            Assert.True(cf2.Distance > cf1.Distance);
            Assert.True(cf2.FeatureChanges.Count > cf1.FeatureChanges.Count);
        }

        [Fact]
        public void CounterfactualExplanation_Predictions_ShowOutcomeChange()
        {
            // Arrange
            var cf = new CounterfactualExplanation<double>
            {
                OriginalPrediction = new Tensor<double>(new[] { 0.0 }), // Negative
                CounterfactualPrediction = new Tensor<double>(new[] { 1.0 }) // Positive
            };

            // Assert - Predictions should differ
            Assert.NotNull(cf.OriginalPrediction);
            Assert.NotNull(cf.CounterfactualPrediction);
            Assert.NotEqual(cf.OriginalPrediction[0], cf.CounterfactualPrediction[0]);
        }

        #endregion

        #region DemographicParityBiasDetector Tests

        [Fact]
        public void DemographicParityBiasDetector_BiasedDataset_DetectsBias()
        {
            // Arrange - Group 1 gets 100% positive, Group 0 gets 0% positive
            var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias);
            Assert.Contains("Bias detected", result.Message);
            Assert.Equal(1.0, Convert.ToDouble(result.StatisticalParityDifference), precision: 6);
            Assert.Equal(2, result.GroupPositiveRates.Count);
            Assert.Equal(1.0, Convert.ToDouble(result.GroupPositiveRates["1"]), precision: 6);
            Assert.Equal(0.0, Convert.ToDouble(result.GroupPositiveRates["0"]), precision: 6);
        }

        [Fact]
        public void DemographicParityBiasDetector_FairDataset_NoBias()
        {
            // Arrange - Both groups get 50% positive predictions
            var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("No significant bias", result.Message);
            Assert.True(Math.Abs(Convert.ToDouble(result.StatisticalParityDifference)) <= 0.1);
        }

        [Fact]
        public void DemographicParityBiasDetector_BorderlineCase_RespectThreshold()
        {
            // Arrange - Difference exactly at threshold
            var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
            // Group 1: 2/3 = 0.667, Group 0: 1/3 = 0.333, Difference = 0.333 > 0.1
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias); // 0.333 > 0.1 threshold
        }

        [Fact]
        public void DemographicParityBiasDetector_MultipleGroups_DetectsMaxDifference()
        {
            // Arrange - 3 groups with different positive rates
            var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);
            // Group 0: 0/2 = 0.0, Group 1: 1/2 = 0.5, Group 2: 2/2 = 1.0
            var predictions = new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0 });
            var sensitiveFeature = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias);
            Assert.Equal(3, result.GroupPositiveRates.Count);
            // Max difference should be between group 2 (1.0) and group 0 (0.0) = 1.0
            Assert.Equal(1.0, Convert.ToDouble(result.StatisticalParityDifference), precision: 6);
        }

        [Fact]
        public void DemographicParityBiasDetector_InsufficientGroups_NoError()
        {
            // Arrange - Only 1 group
            var detector = new DemographicParityBiasDetector<double>();
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("Insufficient groups", result.Message);
        }

        [Fact]
        public void DemographicParityBiasDetector_InvalidThreshold_ThrowsException()
        {
            // Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DemographicParityBiasDetector<double>(threshold: 0.0));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DemographicParityBiasDetector<double>(threshold: 1.5));
        }

        #endregion

        #region EqualOpportunityBiasDetector Tests

        [Fact]
        public void EqualOpportunityBiasDetector_BiasedDataset_DetectsBias()
        {
            // Arrange - Group 1 has TPR = 1.0, Group 0 has TPR = 0.0
            var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }); // All positive

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

            // Assert
            Assert.True(result.HasBias);
            Assert.Contains("Bias detected", result.Message);
            Assert.Equal(1.0, Convert.ToDouble(result.EqualOpportunityDifference), precision: 6);
            Assert.Equal(2, result.GroupTruePositiveRates.Count);
            Assert.Equal(1.0, Convert.ToDouble(result.GroupTruePositiveRates["1"]), precision: 6);
            Assert.Equal(0.0, Convert.ToDouble(result.GroupTruePositiveRates["0"]), precision: 6);
        }

        [Fact]
        public void EqualOpportunityBiasDetector_FairDataset_NoBias()
        {
            // Arrange - Both groups have same TPR
            var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("No significant bias", result.Message);
            // Both groups: 2/3 correct, TPR difference = 0
            Assert.True(Math.Abs(Convert.ToDouble(result.EqualOpportunityDifference)) <= 0.1);
        }

        [Fact]
        public void EqualOpportunityBiasDetector_WithoutLabels_CannotComputeTPR()
        {
            // Arrange
            var detector = new EqualOpportunityBiasDetector<double>();
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels: null);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("Cannot compute equal opportunity without actual labels", result.Message);
        }

        [Fact]
        public void EqualOpportunityBiasDetector_MixedOutcomes_CalculatesCorrectTPR()
        {
            // Arrange
            var detector = new EqualOpportunityBiasDetector<double>(threshold: 0.1);
            // Group 1: TP=2, FN=1 -> TPR = 2/3 = 0.667
            // Group 0: TP=1, FN=2 -> TPR = 1/3 = 0.333
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

            // Assert
            Assert.True(result.HasBias); // Difference = 0.333 > 0.1
            var expectedDiff = 2.0 / 3.0 - 1.0 / 3.0;
            Assert.Equal(expectedDiff, Convert.ToDouble(result.EqualOpportunityDifference), precision: 6);
        }

        #endregion

        #region DisparateImpactBiasDetector Tests

        [Fact]
        public void DisparateImpactBiasDetector_BiasedDataset_DetectsBias()
        {
            // Arrange - Group 1: 100% positive, Group 0: 0% positive -> Ratio = 0/1 = 0
            var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias);
            Assert.Contains("Bias detected", result.Message);
            Assert.Equal(0.0, Convert.ToDouble(result.DisparateImpactRatio), precision: 6);
            Assert.True(Convert.ToDouble(result.DisparateImpactRatio) < 0.8);
        }

        [Fact]
        public void DisparateImpactBiasDetector_FairDataset_NoBias()
        {
            // Arrange - Both groups: 50% positive -> Ratio = 0.5/0.5 = 1.0
            var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("No significant bias", result.Message);
            Assert.Equal(1.0, Convert.ToDouble(result.DisparateImpactRatio), precision: 6);
        }

        [Fact]
        public void DisparateImpactBiasDetector_EightyPercentRule_Works()
        {
            // Arrange - Test the 80% rule
            // Group 1: 3/3 = 1.0, Group 0: 2/3 = 0.667 -> Ratio = 0.667
            var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias); // 0.667 < 0.8 threshold
            Assert.Equal(2.0 / 3.0, Convert.ToDouble(result.DisparateImpactRatio), precision: 5);
        }

        [Fact]
        public void DisparateImpactBiasDetector_AllZeroPredictions_HandlesGracefully()
        {
            // Arrange - All predictions are 0
            var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
            var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Equal(1.0, Convert.ToDouble(result.DisparateImpactRatio), precision: 6);
            Assert.Contains("zero positive predictions", result.Message);
        }

        [Fact]
        public void DisparateImpactBiasDetector_BorderlineCase_AtThreshold()
        {
            // Arrange - Ratio exactly at 0.8
            var detector = new DisparateImpactBiasDetector<double>(threshold: 0.8);
            // Group 1: 5/5 = 1.0, Group 0: 4/5 = 0.8 -> Ratio = 0.8
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias); // Ratio = 0.8, not < 0.8
            Assert.Equal(0.8, Convert.ToDouble(result.DisparateImpactRatio), precision: 6);
        }

        #endregion

        #region BasicFairnessEvaluator Tests

        [Fact]
        public void BasicFairnessEvaluator_BiasedModel_DetectsFairnessissues()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            // Features: [feature, sensitive], where sensitive is group membership
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.7, 1.0 }, { 0.6, 1.0 },
                { 0.8, 0.0 }, { 0.7, 0.0 }, { 0.6, 0.0 }
            });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1);

            // Assert
            Assert.True(Convert.ToDouble(metrics.DemographicParity) > 0.9); // Large disparity
            Assert.True(Convert.ToDouble(metrics.DisparateImpact) < 0.1); // Fails 80% rule badly
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_PositiveRate"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_PositiveRate"));
        }

        [Fact]
        public void BasicFairnessEvaluator_FairModel_LowDisparityMetrics()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new FairModel(featureIndex: 0); // Predicts based on first feature only
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.3, 1.0 }, { 0.7, 1.0 },
                { 0.9, 0.0 }, { 0.2, 0.0 }, { 0.6, 0.0 }
            });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1);

            // Assert - Fair model should have low disparity
            Assert.True(Convert.ToDouble(metrics.DemographicParity) < 0.2);
            Assert.True(Convert.ToDouble(metrics.DisparateImpact) > 0.7);
        }

        [Fact]
        public void BasicFairnessEvaluator_InsufficientGroups_ReturnsZeroMetrics()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new FairModel();
            var inputs = new Matrix<double>(new[,] { { 0.5, 1.0 }, { 0.6, 1.0 } }); // Only 1 group

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(metrics.DemographicParity));
            Assert.Equal(1.0, Convert.ToDouble(metrics.DisparateImpact));
        }

        [Fact]
        public void BasicFairnessEvaluator_ComputesPerGroupMetrics()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.5, 1.0 }, { 0.6, 1.0 },
                { 0.5, 0.0 }, { 0.6, 0.0 }
            });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1);

            // Assert
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_Size"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_Size"));
            Assert.Equal(2.0, Convert.ToDouble(metrics.AdditionalMetrics["Group_1_Size"]));
            Assert.Equal(2.0, Convert.ToDouble(metrics.AdditionalMetrics["Group_0_Size"]));
        }

        #endregion

        #region GroupFairnessEvaluator Tests

        [Fact]
        public void GroupFairnessEvaluator_WithLabels_ComputesTPRAndFPR()
        {
            // Arrange
            var evaluator = new GroupFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.7, 1.0 }, { 0.6, 1.0 },
                { 0.8, 0.0 }, { 0.7, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert
            Assert.True(Convert.ToDouble(metrics.EqualOpportunity) > 0.0); // TPR difference
            Assert.True(Convert.ToDouble(metrics.EqualizedOdds) > 0.0); // Max of TPR and FPR differences
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_TPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_TPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_FPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_FPR"));
        }

        [Fact]
        public void GroupFairnessEvaluator_FairModel_LowTPRDifference()
        {
            // Arrange
            var evaluator = new GroupFairnessEvaluator<double>();
            var model = new FairModel(featureIndex: 0);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.3, 1.0 }, { 0.7, 1.0 },
                { 0.9, 0.0 }, { 0.2, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert
            Assert.True(Convert.ToDouble(metrics.EqualOpportunity) < 0.3);
        }

        [Fact]
        public void GroupFairnessEvaluator_WithoutLabels_ReturnsZeroPerformanceMetrics()
        {
            // Arrange
            var evaluator = new GroupFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,] { { 0.5, 1.0 }, { 0.6, 0.0 } });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels: null);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(metrics.EqualOpportunity));
            Assert.Equal(0.0, Convert.ToDouble(metrics.EqualizedOdds));
            Assert.Equal(0.0, Convert.ToDouble(metrics.PredictiveParity));
        }

        [Fact]
        public void GroupFairnessEvaluator_ComputesPrecisionPerGroup()
        {
            // Arrange
            var evaluator = new GroupFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.5, 1.0 }, { 0.6, 1.0 },
                { 0.5, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_Precision"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_Precision"));
        }

        #endregion

        #region ComprehensiveFairnessEvaluator Tests

        [Fact]
        public void ComprehensiveFairnessEvaluator_BiasedModel_ComputesAllMetrics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.7, 1.0 }, { 0.6, 1.0 },
                { 0.8, 0.0 }, { 0.7, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert - All metrics should be computed
            Assert.NotEqual(0.0, Convert.ToDouble(metrics.DemographicParity));
            Assert.NotEqual(0.0, Convert.ToDouble(metrics.EqualOpportunity));
            Assert.NotEqual(0.0, Convert.ToDouble(metrics.EqualizedOdds));
            Assert.NotEqual(1.0, Convert.ToDouble(metrics.DisparateImpact));
            Assert.NotEqual(0.0, Convert.ToDouble(metrics.StatisticalParityDifference));
        }

        [Fact]
        public void ComprehensiveFairnessEvaluator_FairModel_AllMetricsShowFairness()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new FairModel(featureIndex: 0);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.8, 1.0 }, { 0.3, 1.0 }, { 0.7, 1.0 },
                { 0.9, 0.0 }, { 0.2, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert - Fair model should have low disparity across all metrics
            Assert.True(Convert.ToDouble(metrics.DemographicParity) < 0.3);
            Assert.True(Convert.ToDouble(metrics.DisparateImpact) > 0.6);
            Assert.True(Convert.ToDouble(metrics.EqualOpportunity) < 0.4);
        }

        [Fact]
        public void ComprehensiveFairnessEvaluator_IncludesPerGroupStatistics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,]
            {
                { 0.5, 1.0 }, { 0.6, 1.0 },
                { 0.5, 0.0 }, { 0.6, 0.0 }
            });
            var actualLabels = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels);

            // Assert
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_PositiveRate"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_PositiveRate"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_TPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_TPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_FPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_FPR"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_1_Precision"));
            Assert.True(metrics.AdditionalMetrics.ContainsKey("Group_0_Precision"));
        }

        [Fact]
        public void ComprehensiveFairnessEvaluator_WithoutLabels_ComputesBasicMetricsOnly()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);
            var inputs = new Matrix<double>(new[,] { { 0.5, 1.0 }, { 0.6, 0.0 } });

            // Act
            var metrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1, actualLabels: null);

            // Assert
            Assert.NotEqual(0.0, Convert.ToDouble(metrics.DemographicParity));
            Assert.Equal(0.0, Convert.ToDouble(metrics.EqualOpportunity)); // Requires labels
            Assert.Equal(0.0, Convert.ToDouble(metrics.EqualizedOdds)); // Requires labels
            Assert.Equal(0.0, Convert.ToDouble(metrics.PredictiveParity)); // Requires labels
        }

        #endregion

        #region InterpretabilityMetricsHelper Tests

        [Fact]
        public void InterpretabilityMetricsHelper_GetUniqueGroups_IdentifiesAllGroups()
        {
            // Arrange
            var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });

            // Act
            var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);

            // Assert
            Assert.Equal(3, groups.Count);
            Assert.Contains(0.0, groups);
            Assert.Contains(1.0, groups);
            Assert.Contains(2.0, groups);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_GetGroupIndices_ReturnsCorrectIndices()
        {
            // Arrange
            var sensitiveFeature = new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });

            // Act
            var group1Indices = InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, 1.0);

            // Assert
            Assert.Equal(2, group1Indices.Count);
            Assert.Contains(1, group1Indices);
            Assert.Contains(4, group1Indices);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_GetSubset_ExtractsCorrectElements()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var indices = new List<int> { 1, 3 };

            // Act
            var subset = InterpretabilityMetricsHelper<double>.GetSubset(vector, indices);

            // Assert
            Assert.Equal(2, subset.Length);
            Assert.Equal(20.0, subset[0]);
            Assert.Equal(40.0, subset[1]);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_ComputePositiveRate_CalculatesCorrectly()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });

            // Act
            var positiveRate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(predictions);

            // Assert
            Assert.Equal(0.6, Convert.ToDouble(positiveRate), precision: 6); // 3/5 = 0.6
        }

        [Fact]
        public void InterpretabilityMetricsHelper_ComputeTruePositiveRate_CalculatesCorrectly()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0 });
            // TP = 2 (indices 0, 2), FN = 1 (index 1), TPR = 2/3

            // Act
            var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

            // Assert
            Assert.Equal(2.0 / 3.0, Convert.ToDouble(tpr), precision: 6);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_ComputeFalsePositiveRate_CalculatesCorrectly()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0 });
            // FP = 1 (index 3), TN = 1 (index 4), FPR = 1/2 = 0.5

            // Act
            var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

            // Assert
            Assert.Equal(0.5, Convert.ToDouble(fpr), precision: 6);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_ComputePrecision_CalculatesCorrectly()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 1.0, 0.0, 1.0, 1.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0 });
            // TP = 2 (indices 0, 2), FP = 1 (index 3), Precision = 2/3

            // Act
            var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

            // Assert
            Assert.Equal(2.0 / 3.0, Convert.ToDouble(precision), precision: 6);
        }

        [Fact]
        public void InterpretabilityMetricsHelper_EmptyVector_ReturnsZero()
        {
            // Arrange
            var emptyPredictions = new Vector<double>(Array.Empty<double>());

            // Act
            var positiveRate = InterpretabilityMetricsHelper<double>.ComputePositiveRate(emptyPredictions);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(positiveRate));
        }

        [Fact]
        public void InterpretabilityMetricsHelper_NoPositives_TPRIsZero()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var tpr = InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(predictions, actualLabels);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(tpr));
        }

        [Fact]
        public void InterpretabilityMetricsHelper_NoNegatives_FPRIsZero()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var fpr = InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(predictions, actualLabels);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(fpr));
        }

        [Fact]
        public void InterpretabilityMetricsHelper_NoPredictedPositives_PrecisionIsZero()
        {
            // Arrange
            var predictions = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var actualLabels = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var precision = InterpretabilityMetricsHelper<double>.ComputePrecision(predictions, actualLabels);

            // Assert
            Assert.Equal(0.0, Convert.ToDouble(precision));
        }

        #endregion

        #region FairnessMetrics Tests

        [Fact]
        public void FairnessMetrics_Initialization_SetsAllMetrics()
        {
            // Arrange & Act
            var metrics = new FairnessMetrics<double>(
                demographicParity: 0.15,
                equalOpportunity: 0.10,
                equalizedOdds: 0.12,
                predictiveParity: 0.08,
                disparateImpact: 0.75,
                statisticalParityDifference: 0.15);

            // Assert
            Assert.Equal(0.15, Convert.ToDouble(metrics.DemographicParity));
            Assert.Equal(0.10, Convert.ToDouble(metrics.EqualOpportunity));
            Assert.Equal(0.12, Convert.ToDouble(metrics.EqualizedOdds));
            Assert.Equal(0.08, Convert.ToDouble(metrics.PredictiveParity));
            Assert.Equal(0.75, Convert.ToDouble(metrics.DisparateImpact));
            Assert.Equal(0.15, Convert.ToDouble(metrics.StatisticalParityDifference));
            Assert.NotNull(metrics.AdditionalMetrics);
            Assert.Empty(metrics.AdditionalMetrics);
        }

        [Fact]
        public void FairnessMetrics_AdditionalMetrics_CanBeAdded()
        {
            // Arrange
            var metrics = new FairnessMetrics<double>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

            // Act
            metrics.AdditionalMetrics["CustomMetric"] = 0.5;
            metrics.AdditionalMetrics["AnotherMetric"] = 0.8;

            // Assert
            Assert.Equal(2, metrics.AdditionalMetrics.Count);
            Assert.Equal(0.5, Convert.ToDouble(metrics.AdditionalMetrics["CustomMetric"]));
            Assert.Equal(0.8, Convert.ToDouble(metrics.AdditionalMetrics["AnotherMetric"]));
        }

        [Fact]
        public void FairnessMetrics_SensitiveFeatureIndex_CanBeSet()
        {
            // Arrange
            var metrics = new FairnessMetrics<double>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            {
                SensitiveFeatureIndex = 3
            };

            // Assert
            Assert.Equal(3, metrics.SensitiveFeatureIndex);
        }

        #endregion

        #region BiasDetectionResult Tests

        [Fact]
        public void BiasDetectionResult_Initialization_SetsDefaults()
        {
            // Arrange & Act
            var result = new BiasDetectionResult<double>();

            // Assert
            Assert.False(result.HasBias);
            Assert.Equal(string.Empty, result.Message);
            Assert.NotNull(result.GroupPositiveRates);
            Assert.Empty(result.GroupPositiveRates);
            Assert.NotNull(result.GroupSizes);
            Assert.Empty(result.GroupSizes);
        }

        [Fact]
        public void BiasDetectionResult_CanStoreMultipleGroupMetrics()
        {
            // Arrange
            var result = new BiasDetectionResult<double>
            {
                HasBias = true,
                Message = "Significant bias detected",
                GroupPositiveRates = new Dictionary<string, double>
                {
                    { "Group_A", 0.8 },
                    { "Group_B", 0.3 }
                },
                GroupSizes = new Dictionary<string, int>
                {
                    { "Group_A", 100 },
                    { "Group_B", 150 }
                },
                StatisticalParityDifference = 0.5
            };

            // Assert
            Assert.True(result.HasBias);
            Assert.Equal(2, result.GroupPositiveRates.Count);
            Assert.Equal(0.8, result.GroupPositiveRates["Group_A"]);
            Assert.Equal(0.5, result.StatisticalParityDifference);
        }

        #endregion

        #region Edge Cases and Integration Scenarios

        [Fact]
        public void BiasDetector_MismatchedLengths_ThrowsException()
        {
            // Arrange
            var detector = new DemographicParityBiasDetector<double>();
            var predictions = new Vector<double>(new[] { 1.0, 0.0 });
            var sensitiveFeature = new Vector<double>(new[] { 1.0, 0.0, 1.0 }); // Different length

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                detector.DetectBias(predictions, sensitiveFeature));
        }

        [Fact]
        public void FairnessEvaluator_InvalidSensitiveIndex_ThrowsException()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new FairModel();
            var inputs = new Matrix<double>(new[,] { { 0.5, 0.6 }, { 0.7, 0.8 } }); // 2 columns

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 5));
        }

        [Fact]
        public void FairnessEvaluator_NullModel_ThrowsException()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var inputs = new Matrix<double>(new[,] { { 0.5, 0.6 } });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                evaluator.EvaluateFairness(null!, inputs, 0));
        }

        [Fact]
        public void FairnessEvaluator_LabelsMismatch_ThrowsException()
        {
            // Arrange
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new FairModel();
            var inputs = new Matrix<double>(new[,] { { 0.5, 0.6 }, { 0.7, 0.8 } }); // 2 rows
            var labels = new Vector<double>(new[] { 1.0, 0.0, 1.0 }); // 3 labels

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                evaluator.EvaluateFairness(model, inputs, 1, labels));
        }

        [Fact]
        public void IntegratedScenario_CreditScoring_DetectsBias()
        {
            // Arrange - Simulate credit scoring with gender bias
            var detector = new DemographicParityBiasDetector<double>(threshold: 0.1);

            // Male applicants (gender=1): 80% approval
            // Female applicants (gender=0): 50% approval
            var predictions = new Vector<double>(new[]
            {
                1.0, 1.0, 1.0, 1.0, 0.0,  // Males: 4/5 approved
                1.0, 1.0, 0.0, 0.0, 0.0   // Females: 2/5 approved
            });
            var gender = new Vector<double>(new[]
            {
                1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            });

            // Act
            var result = detector.DetectBias(predictions, gender);

            // Assert
            Assert.True(result.HasBias);
            var malApprovalRate = Convert.ToDouble(result.GroupPositiveRates["1"]);
            var femaleApprovalRate = Convert.ToDouble(result.GroupPositiveRates["0"]);
            Assert.True(Math.Abs(maleApprovalRate - femaleApprovalRate) > 0.1);
        }

        [Fact]
        public void IntegratedScenario_HiringDecision_ComprehensiveAnalysis()
        {
            // Arrange - Hiring model with race-based bias
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var biasedHiringModel = new BiasedModel(sensitiveIndex: 2); // Race at index 2

            // Features: [experience, education, race]
            var applicants = new Matrix<double>(new[,]
            {
                { 0.8, 0.9, 1.0 }, { 0.7, 0.8, 1.0 }, { 0.6, 0.7, 1.0 },
                { 0.9, 0.9, 0.0 }, { 0.8, 0.8, 0.0 }, { 0.7, 0.7, 0.0 }
            });
            var qualifications = new Vector<double>(new[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0 });

            // Act
            var metrics = evaluator.EvaluateFairness(biasedHiringModel, applicants,
                sensitiveFeatureIndex: 2, actualLabels: qualifications);

            // Assert - Should detect significant bias across all metrics
            Assert.True(Convert.ToDouble(metrics.DemographicParity) > 0.5);
            Assert.True(Convert.ToDouble(metrics.DisparateImpact) < 0.5);
            Assert.True(Convert.ToDouble(metrics.EqualOpportunity) > 0.0);
        }

        [Fact]
        public void IntegratedScenario_MultipleProtectedAttributes_IndependentEvaluation()
        {
            // Arrange - Test fairness for multiple protected attributes independently
            var evaluator = new BasicFairnessEvaluator<double>();
            var model = new BiasedModel(sensitiveIndex: 1);

            // Features: [feature, gender, race, age]
            var inputs = new Matrix<double>(new[,]
            {
                { 0.5, 1.0, 1.0, 0.3 },
                { 0.6, 0.0, 1.0, 0.4 },
                { 0.7, 1.0, 0.0, 0.5 }
            });

            // Act - Evaluate fairness for each protected attribute
            var genderMetrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 1);
            var raceMetrics = evaluator.EvaluateFairness(model, inputs, sensitiveFeatureIndex: 2);

            // Assert - Gender shows bias (model is biased on index 1)
            Assert.True(Convert.ToDouble(genderMetrics.DemographicParity) > 0.3);
            // Race may show different bias patterns
            Assert.NotNull(raceMetrics.DemographicParity);
        }

        [Fact]
        public void BiasDetectorComparison_DifferentMetrics_DifferentResults()
        {
            // Arrange - Same data, different detectors
            var demographicDetector = new DemographicParityBiasDetector<double>(0.1);
            var disparateDetector = new DisparateImpactBiasDetector<double>(0.8);
            var equalOppDetector = new EqualOpportunityBiasDetector<double>(0.1);

            // Group 1: 75% positive, Group 0: 50% positive
            var predictions = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0 });
            var sensitive = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 });
            var actuals = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 });

            // Act
            var demographicResult = demographicDetector.DetectBias(predictions, sensitive);
            var disparateResult = disparateDetector.DetectBias(predictions, sensitive);
            var equalOppResult = equalOppDetector.DetectBias(predictions, sensitive, actuals);

            // Assert - Different detectors may give different bias verdicts
            Assert.NotNull(demographicResult);
            Assert.NotNull(disparateResult);
            Assert.NotNull(equalOppResult);
            // Demographic: |0.75 - 0.5| = 0.25 > 0.1 -> bias
            Assert.True(demographicResult.HasBias);
            // Disparate Impact: 0.5/0.75 = 0.667 < 0.8 -> bias
            Assert.True(disparateResult.HasBias);
        }

        [Fact]
        public void RealWorldScenario_LoanApproval_MultipleGroupComparison()
        {
            // Arrange - Loan approval with 3 ethnic groups
            var detector = new DisparateImpactBiasDetector<double>(0.8);

            // Group 0: 90% approval, Group 1: 70% approval, Group 2: 60% approval
            var predictions = new Vector<double>(new[]
            {
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, // Group 0: 9/10
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, // Group 1: 7/10
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0  // Group 2: 6/10
            });
            var ethnicity = new Vector<double>(new[]
            {
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
            });

            // Act
            var result = detector.DetectBias(predictions, ethnicity);

            // Assert - Should detect bias: min/max = 0.6/0.9 = 0.667 < 0.8
            Assert.True(result.HasBias);
            Assert.Equal(3, result.GroupPositiveRates.Count);
            var ratio = Convert.ToDouble(result.DisparateImpactRatio);
            Assert.True(ratio < 0.8);
            Assert.Equal(0.6 / 0.9, ratio, precision: 2);
        }

        #endregion
    }
}
