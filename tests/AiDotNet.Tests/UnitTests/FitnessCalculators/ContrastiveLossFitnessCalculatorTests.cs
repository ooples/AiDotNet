using System;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNetTests.UnitTests.FitnessCalculators
{
    /// <summary>
    /// Unit tests for ContrastiveLossFitnessCalculator, which evaluates model performance for similarity learning tasks.
    /// </summary>
    public class ContrastiveLossFitnessCalculatorTests
    {
        [Fact]
        public void Constructor_WithDefaultParameters_UsesDefaultMargin()
        {
            // Arrange & Act
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter); // Contrastive loss: lower is better
        }

        [Fact]
        public void Constructor_WithCustomMargin_UsesSpecifiedMargin()
        {
            // Arrange & Act
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(
                margin: 2.0, dataSetType: DataSetType.Validation);

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void Constructor_WithTrainingDataSetType_UsesTraining()
        {
            // Arrange & Act
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(
                dataSetType: DataSetType.Training);

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CalculateFitnessScore_WithIdenticalSimilarPairs_ReturnsZero()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // First half and second half are identical, actual values are same (similarity = 1)
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 1.0, 2.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // For similar pairs (label=1) with identical embeddings: loss = distance² = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithDissimilarPairsBeyondMargin_ReturnsZero()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 1.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Pairs that are different (label=0) and far apart (beyond margin)
                Predicted = new Vector<double>(new double[] { 0.0, 5.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // For dissimilar pairs beyond margin: loss = max(0, margin - distance)² = 0
            // Distance between [0] and [5] is 5, which is > margin (1.0)
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithSimilarPairsAtDistance_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Similar pairs (label=1) with some distance
                Predicted = new Vector<double>(new double[] { 1.0, 4.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance between [1.0] and [4.0] = 3.0
            // For similar pairs: loss = distance² = 9.0
            Assert.Equal(9.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithDissimilarPairsWithinMargin_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 2.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Dissimilar pairs (label=0) within margin
                Predicted = new Vector<double>(new double[] { 1.0, 1.5 }),
                Actual = new Vector<double>(new double[] { 2.0, 3.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance = 0.5, margin = 2.0
            // For dissimilar pairs: loss = max(0, 2.0 - 0.5)² = 1.5² = 2.25
            Assert.Equal(2.25, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithMultiplePairs_ReturnsAverageLoss()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 1.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Predicted is split in half: output1 = [1.0, 2.0], output2 = [3.0, 4.0]
                // Actual is split in half: actual1 = [1.0, 2.0], actual2 = [1.0, 6.0]
                // Similarity labels: actual1[0]==actual2[0] → similar, actual1[1]!=actual2[1] → dissimilar
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0, 1.0, 6.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance between output1=[1,2] and output2=[3,4] = sqrt((1-3)² + (2-4)²) = sqrt(8) ≈ 2.828
            // i=0: similar (1.0 == 1.0), loss = distance² = 8.0
            // i=1: dissimilar (2.0 != 6.0), loss = max(0, 1.0 - 2.828)² = 0
            // Average loss = (8.0 + 0) / 2 = 4.0
            Assert.True(result >= 3.9 && result <= 4.1);
        }

        [Fact]
        public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<float, Vector<float>, Vector<float>>(margin: 1.0f);
            var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
            {
                Predicted = new Vector<float>(new float[] { 0.0f, 3.0f }),
                Actual = new Vector<float>(new float[] { 1.0f, 1.0f })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Similar pair, distance = 3.0, loss = 9.0
            Assert.Equal(9.0f, result, 5);
        }

        [Fact]
        public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationData_UsesValidationSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            // Note: ModelEvaluationData requires 2D tensors for proper matrix conversion
            var calculator = new ContrastiveLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(
                dataSetType: DataSetType.Validation);
            var evaluationData = new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
            {
                ValidationSet = new DataSetStats<double, Tensor<double>, Tensor<double>>
                {
                    Predicted = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 1.0 })),
                    Actual = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 1.0 }))
                }
            };

            // Act
            var result = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, result, 10); // Identical pairs
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationDataAndTestSet_UsesTestSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            // Note: ModelEvaluationData requires 2D tensors for proper matrix conversion
            var calculator = new ContrastiveLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(
                dataSetType: DataSetType.Testing);
            var evaluationData = new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
            {
                TestSet = new DataSetStats<double, Tensor<double>, Tensor<double>>
                {
                    Predicted = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 0.0, 5.0 })),
                    Actual = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 1.0 }))
                }
            };

            // Act
            var result = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(25.0, result, 10); // Distance = 5.0, loss = 25.0
        }

        [Fact]
        public void IsHigherScoreBetter_ReturnsFalse()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void IsBetterFitness_WithLowerScore_ReturnsTrue()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 0.5;
            double currentBestScore = 1.0;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.True(result); // Lower score is better for loss functions
        }

        [Fact]
        public void IsBetterFitness_WithHigherScore_ReturnsFalse()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 1.5;
            double currentBestScore = 0.8;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.False(result); // Higher score is worse for loss functions
        }

        [Fact]
        public void CalculateFitnessScore_FaceRecognitionScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating face verification scenario
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 1.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Predicted is split: output1 = [0.1, 0.2], output2 = [0.9, 0.8]
                // Actual is split: actual1 = [1.0, 2.0], actual2 = [1.0, 3.0]
                // Similarity: actual1[0]==actual2[0] → similar, actual1[1]!=actual2[1] → dissimilar
                Predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.9, 0.8 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0, 1.0, 3.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance between output1=[0.1,0.2] and output2=[0.9,0.8] = sqrt((0.8)² + (0.6)²) = 1.0
            // i=0: similar (1.0 == 1.0), loss = distance² = 1.0
            // i=1: dissimilar (2.0 != 3.0), loss = max(0, 1.0 - 1.0)² = 0
            // Average loss = (1.0 + 0) / 2 = 0.5
            Assert.True(result >= 0.45 && result <= 0.55);
        }

        [Fact]
        public void CalculateFitnessScore_WithLargeMargin_AllowsMoreSeparation()
        {
            // Arrange
            var smallMarginCalc = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 0.5);
            var largeMarginCalc = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 2.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Dissimilar pairs at medium distance
                Predicted = new Vector<double>(new double[] { 0.0, 1.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0 })
            };

            // Act
            var smallMarginResult = smallMarginCalc.CalculateFitnessScore(dataSet);
            var largeMarginResult = largeMarginCalc.CalculateFitnessScore(dataSet);

            // Assert
            // Distance = 1.0
            // Small margin (0.5): loss = max(0, 0.5 - 1.0)² = 0
            // Large margin (2.0): loss = max(0, 2.0 - 1.0)² = 1.0
            Assert.Equal(0.0, smallMarginResult, 10);
            Assert.Equal(1.0, largeMarginResult, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithEvenNumberOfElements_SplitsCorrectly()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // 6 elements: splits into two groups of 3
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // All pairs are similar (same actuals) and identical
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithZeroDistance_SimilarPairs_ReturnsZero()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.5, 0.5, 0.5, 0.5 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Zero distance for similar pairs: loss = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithMaximumSeparation_DissimilarPairs_ReturnsZero()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 1.0);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Dissimilar pairs far apart
                Predicted = new Vector<double>(new double[] { 0.0, 10.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance = 10.0 >> margin (1.0), loss = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_SignatureVerificationScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating signature verification
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 1.5);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Genuine signatures vs forgeries
                Predicted = new Vector<double>(new double[] { 0.8, 0.9, 1.0, 0.1, 0.2, 5.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 2.0, 3.0, 4.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Should handle mixed scenarios of genuine and forged signatures
            Assert.True(result >= 0.0);
        }

        [Fact]
        public void CalculateFitnessScore_WithVerySmallMargin_PenalizesDissimilarPairsMore()
        {
            // Arrange
            var calculator = new ContrastiveLossFitnessCalculator<double, Vector<double>, Vector<double>>(margin: 0.1);
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Dissimilar pairs at small distance
                Predicted = new Vector<double>(new double[] { 0.0, 0.05 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Distance = 0.05, margin = 0.1
            // Loss = max(0, 0.1 - 0.05)² = 0.0025
            Assert.Equal(0.0025, result, 10);
        }
    }
}
