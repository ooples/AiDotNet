using System;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Interpretability
{
    /// <summary>
    /// Unit tests for the FairnessEvaluator class.
    /// </summary>
    public class FairnessEvaluatorTests
    {
        [Fact]
        public void EvaluateFairness_WithInvalidSensitiveFeatureIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));
            Matrix<double> inputs = new Matrix<double>(4, 2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                evaluator.EvaluateFairness(model, inputs, 5));
        }

        [Fact]
        public void EvaluateFairness_WithBalancedPredictions_ReturnsZeroMetrics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();

            // Create a simple model that predicts based on first feature
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            // Create balanced inputs where both groups get 50% positive predictions
            // Sensitive feature is in column 1
            Matrix<double> inputs = new Matrix<double>(8, 2);
            // Group 0 (sensitive=0): features that will predict 0.5 (half positive, half negative)
            inputs[0, 0] = 0.4; inputs[0, 1] = 0; // predict 0.4
            inputs[1, 0] = 0.4; inputs[1, 1] = 0; // predict 0.4
            inputs[2, 0] = 0.6; inputs[2, 1] = 0; // predict 0.6
            inputs[3, 0] = 0.6; inputs[3, 1] = 0; // predict 0.6
            // Group 1 (sensitive=1): same distribution
            inputs[4, 0] = 0.4; inputs[4, 1] = 1; // predict 0.4
            inputs[5, 0] = 0.4; inputs[5, 1] = 1; // predict 0.4
            inputs[6, 0] = 0.6; inputs[6, 1] = 1; // predict 0.6
            inputs[7, 0] = 0.6; inputs[7, 1] = 1; // predict 0.6

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(1, result.SensitiveFeatureIndex);
            Assert.Equal(0.0, result.DemographicParity, 2);
            Assert.Equal(1.0, result.DisparateImpact, 2);
            Assert.Equal(0.0, result.StatisticalParityDifference, 2);
        }

        [Fact]
        public void EvaluateFairness_WithUnbalancedPredictions_DetectsBias()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();

            // Create a model that predicts based on first feature
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            // Create unbalanced inputs where groups get different prediction rates
            Matrix<double> inputs = new Matrix<double>(8, 2);
            // Group 0 (sensitive=0): all high values (100% positive predictions)
            inputs[0, 0] = 1.0; inputs[0, 1] = 0;
            inputs[1, 0] = 1.0; inputs[1, 1] = 0;
            inputs[2, 0] = 1.0; inputs[2, 1] = 0;
            inputs[3, 0] = 1.0; inputs[3, 1] = 0;
            // Group 1 (sensitive=1): all low values (0% positive predictions)
            inputs[4, 0] = 0.0; inputs[4, 1] = 1;
            inputs[5, 0] = 0.0; inputs[5, 1] = 1;
            inputs[6, 0] = 0.0; inputs[6, 1] = 1;
            inputs[7, 0] = 0.0; inputs[7, 1] = 1;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.DemographicParity > 0.9); // Large difference
            Assert.True(result.DisparateImpact < 0.1); // Very low ratio
            Assert.True(result.StatisticalParityDifference > 0.9); // Large difference
        }

        [Fact]
        public void EvaluateFairness_WithActualLabels_ComputesAllMetrics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            Matrix<double> inputs = new Matrix<double>(8, 2);
            // Set up predictions to create different TPR/precision between groups
            // Group 0: predictions [1,1,0,0], actuals [1,1,1,0] -> TPR=2/3, Precision=2/2=1.0
            inputs[0, 0] = 0.6; inputs[0, 1] = 0; // predict 1, actual=1 → TP
            inputs[1, 0] = 0.6; inputs[1, 1] = 0; // predict 1, actual=1 → TP
            inputs[2, 0] = 0.4; inputs[2, 1] = 0; // predict 0, actual=1 → FN
            inputs[3, 0] = 0.4; inputs[3, 1] = 0; // predict 0, actual=0 → TN
            // Group 1: predictions [1,1,0,0], actuals [1,0,0,0] -> TPR=1/1=1.0, Precision=1/2=0.5
            inputs[4, 0] = 0.6; inputs[4, 1] = 1; // predict 1, actual=1 → TP
            inputs[5, 0] = 0.6; inputs[5, 1] = 1; // predict 1, actual=0 → FP
            inputs[6, 0] = 0.4; inputs[6, 1] = 1; // predict 0, actual=0 → TN
            inputs[7, 0] = 0.4; inputs[7, 1] = 1; // predict 0, actual=0 → TN

            // Group 0: actuals [1,1,1,0], predictions [1,1,0,0] -> TPR=2/3, Precision=2/2=1.0
            // Group 1: actuals [1,0,0,0], predictions [1,1,0,0] -> TPR=1/1=1.0, Precision=1/2=0.5
            Vector<double> actualLabels = new Vector<double>(new double[] { 1, 1, 1, 0, 1, 0, 0, 0 });

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1, actualLabels);

            // Assert
            Assert.NotNull(result);
            // EqualOpportunity = |TPR0 - TPR1| = |1.0 - 0.5| = 0.5 (different TPRs)
            Assert.NotEqual(0.0, result.EqualOpportunity);
            // EqualizedOdds = max(|TPR diff|, |FPR diff|) = max(0.5, |1.0-0|) = 1.0
            // FPR0 = 1/1 = 1.0 (1 FP out of 1 actual negative), FPR1 = 0/2 = 0 (0 FP out of 2 actual negatives)
            Assert.NotEqual(0.0, result.EqualizedOdds);
            // PredictiveParity = |Precision0 - Precision1| = |0.75 - 1.0| = 0.25 (different precisions)
            Assert.NotEqual(0.0, result.PredictiveParity);
        }

        [Fact]
        public void EvaluateFairness_WithSingleGroup_ReturnsZeroMetrics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            Matrix<double> inputs = new Matrix<double>(4, 2);
            // All rows have same sensitive feature value
            for (int i = 0; i < 4; i++)
            {
                inputs[i, 0] = 0.5;
                inputs[i, 1] = 0; // All same group
            }

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(0.0, result.DemographicParity);
            Assert.Equal(1.0, result.DisparateImpact); // Should be 1.0 (no disparity)
            Assert.Equal(0.0, result.StatisticalParityDifference);
        }

        [Fact]
        public void EvaluateFairness_AdditionalMetrics_ContainGroupStatistics()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            Matrix<double> inputs = new Matrix<double>(6, 2);
            inputs[0, 0] = 0.6; inputs[0, 1] = 0;
            inputs[1, 0] = 0.6; inputs[1, 1] = 0;
            inputs[2, 0] = 0.4; inputs[2, 1] = 0;
            inputs[3, 0] = 0.6; inputs[3, 1] = 1;
            inputs[4, 0] = 0.4; inputs[4, 1] = 1;
            inputs[5, 0] = 0.4; inputs[5, 1] = 1;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result.AdditionalMetrics);
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_0_PositiveRate"));
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_1_PositiveRate"));
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_0_Size"));
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_1_Size"));
        }

        [Fact]
        public void EvaluateFairness_WithThreeGroups_HandlesMultipleGroups()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            Matrix<double> inputs = new Matrix<double>(9, 2);
            // Group 0
            inputs[0, 0] = 0.6; inputs[0, 1] = 0;
            inputs[1, 0] = 0.6; inputs[1, 1] = 0;
            inputs[2, 0] = 0.4; inputs[2, 1] = 0;
            // Group 1
            inputs[3, 0] = 0.6; inputs[3, 1] = 1;
            inputs[4, 0] = 0.6; inputs[4, 1] = 1;
            inputs[5, 0] = 0.6; inputs[5, 1] = 1;
            // Group 2
            inputs[6, 0] = 0.4; inputs[6, 1] = 2;
            inputs[7, 0] = 0.4; inputs[7, 1] = 2;
            inputs[8, 0] = 0.4; inputs[8, 1] = 2;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_0_Size"));
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_1_Size"));
            Assert.True(result.AdditionalMetrics.ContainsKey("Group_2_Size"));
        }

        [Fact]
        public void EvaluateFairness_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<float>();
            var model = new VectorModel<float>(new Vector<float>(new float[] { 1f, 0f }));

            Matrix<float> inputs = new Matrix<float>(4, 2);
            inputs[0, 0] = 0.6f; inputs[0, 1] = 0f;
            inputs[1, 0] = 0.6f; inputs[1, 1] = 0f;
            inputs[2, 0] = 0.6f; inputs[2, 1] = 1f;
            inputs[3, 0] = 0.6f; inputs[3, 1] = 1f;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(0.0f, result.DemographicParity, 2);
            Assert.Equal(1.0f, result.DisparateImpact, 2);
        }

        [Fact]
        public void EvaluateFairness_ComputesDemographicParity_Correctly()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            // Group 0: 75% positive (3 out of 4)
            // Group 1: 25% positive (1 out of 4)
            // Demographic Parity = 75% - 25% = 50%
            Matrix<double> inputs = new Matrix<double>(8, 2);
            inputs[0, 0] = 0.6; inputs[0, 1] = 0;
            inputs[1, 0] = 0.6; inputs[1, 1] = 0;
            inputs[2, 0] = 0.6; inputs[2, 1] = 0;
            inputs[3, 0] = 0.4; inputs[3, 1] = 0;
            inputs[4, 0] = 0.6; inputs[4, 1] = 1;
            inputs[5, 0] = 0.4; inputs[5, 1] = 1;
            inputs[6, 0] = 0.4; inputs[6, 1] = 1;
            inputs[7, 0] = 0.4; inputs[7, 1] = 1;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.Equal(0.5, result.DemographicParity, 2);
            Assert.Equal(0.5, result.StatisticalParityDifference, 2);
        }

        [Fact]
        public void EvaluateFairness_ComputesDisparateImpact_Correctly()
        {
            // Arrange
            var evaluator = new ComprehensiveFairnessEvaluator<double>();
            var model = new VectorModel<double>(new Vector<double>(new double[] { 1, 0 }));

            // Group 0: 50% positive (2 out of 4)
            // Group 1: 100% positive (4 out of 4)
            // Disparate Impact = 50% / 100% = 0.5
            Matrix<double> inputs = new Matrix<double>(8, 2);
            inputs[0, 0] = 0.6; inputs[0, 1] = 0;
            inputs[1, 0] = 0.6; inputs[1, 1] = 0;
            inputs[2, 0] = 0.4; inputs[2, 1] = 0;
            inputs[3, 0] = 0.4; inputs[3, 1] = 0;
            inputs[4, 0] = 0.6; inputs[4, 1] = 1;
            inputs[5, 0] = 0.6; inputs[5, 1] = 1;
            inputs[6, 0] = 0.6; inputs[6, 1] = 1;
            inputs[7, 0] = 0.6; inputs[7, 1] = 1;

            // Act
            var result = evaluator.EvaluateFairness(model, inputs, 1);

            // Assert
            Assert.Equal(0.5, result.DisparateImpact, 2);
        }
    }
}
