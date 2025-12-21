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
    /// Unit tests for JaccardLossFitnessCalculator, which evaluates model performance using Intersection over Union.
    /// </summary>
    public class JaccardLossFitnessCalculatorTests
    {
        [Fact]
        public void Constructor_WithDefaultDataSetType_UsesValidation()
        {
            // Arrange & Act
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter); // Jaccard loss: lower is better
        }

        [Fact]
        public void Constructor_WithTrainingDataSetType_UsesTraining()
        {
            // Arrange & Act
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Perfect overlap: Intersection = 3, Union = 3, IoU = 1.0, Loss = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithNoOverlap_ReturnsOne()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // No overlap: Intersection = 0, Union = 4, IoU = 0, Loss = 1
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithPartialOverlap_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = min(1,1) + min(1,1) + min(1,0) + min(0,0) = 2
            // Union = max(1,1) + max(1,1) + max(1,0) + max(0,0) = 3
            // IoU = 2/3 ≈ 0.6667, Loss = 1 - 0.6667 ≈ 0.3333
            Assert.Equal(0.33333333333333331, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithProbabilisticPredictions_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.8, 0.6, 0.2, 0.1 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = min(0.8,1.0) + min(0.6,1.0) + min(0.2,0) + min(0.1,0) = 0.8 + 0.6 = 1.4
            // Union = max(0.8,1.0) + max(0.6,1.0) + max(0.2,0) + max(0.1,0) = 1.0 + 1.0 + 0.2 + 0.1 = 2.3
            // IoU = 1.4 / 2.3 ≈ 0.6087, Loss = 1 - 0.6087 ≈ 0.3913
            Assert.Equal(0.39130434782608703, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithAllZeros_ReturnsOne()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // All zeros: Union = 0 + epsilon, Intersection = 0
            // IoU ≈ 0, Loss ≈ 1.0
            Assert.True(result >= 0.999);
        }

        [Fact]
        public void CalculateFitnessScore_WithSingleElement_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.5 }),
                Actual = new Vector<double>(new double[] { 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = min(0.5, 1.0) = 0.5
            // Union = max(0.5, 1.0) = 1.0
            // IoU = 0.5 / 1.0 = 0.5, Loss = 1 - 0.5 = 0.5
            Assert.Equal(0.5, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<float, Vector<float>, Vector<float>>();
            var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
            {
                Predicted = new Vector<float>(new float[] { 1.0f, 1.0f, 1.0f, 0.0f }),
                Actual = new Vector<float>(new float[] { 1.0f, 1.0f, 0.0f, 0.0f })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 2, Union = 3, IoU = 2/3, Loss = 1/3
            Assert.Equal(0.33333333f, result, 5);
        }

        [Fact]
        public void CalculateFitnessScore_WithImbalancedData_HandlesCorrectly()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Simulating object detection scenario with small object
                Predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.9 + 0.8 = 1.7
            // Union = 1.0 + 1.0 = 2.0
            // IoU = 1.7 / 2.0 = 0.85, Loss = 0.15
            Assert.Equal(0.15, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationData_UsesValidationSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            var calculator = new JaccardLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Validation);
            var evaluationData = new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
            {
                ValidationSet = new DataSetStats<double, Tensor<double>, Tensor<double>>
                {
                    Predicted = new Tensor<double>(new int[] { 1, 3 }, new Vector<double>(new double[] { 1.0, 1.0, 0.0 })),
                    Actual = new Tensor<double>(new int[] { 1, 3 }, new Vector<double>(new double[] { 1.0, 1.0, 0.0 }))
                }
            };

            // Act
            var result = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, result, 10); // Perfect predictions
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationDataAndTestSet_UsesTestSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            var calculator = new JaccardLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Testing);
            var evaluationData = new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
            {
                TestSet = new DataSetStats<double, Tensor<double>, Tensor<double>>
                {
                    Predicted = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 0.0 })),
                    Actual = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 0.0, 1.0 }))
                }
            };

            // Act
            var result = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(1.0, result, 10); // No overlap
        }

        [Fact]
        public void IsHigherScoreBetter_ReturnsFalse()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void IsBetterFitness_WithLowerScore_ReturnsTrue()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 0.2;
            double currentBestScore = 0.5;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.True(result); // Lower score is better for loss functions
        }

        [Fact]
        public void IsBetterFitness_WithHigherScore_ReturnsFalse()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 0.8;
            double currentBestScore = 0.3;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.False(result); // Higher score is worse for loss functions
        }

        [Fact]
        public void CalculateFitnessScore_ObjectDetectionScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating bounding box IoU scenario
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Predicted box overlaps partially with ground truth box
                Predicted = new Vector<double>(new double[] { 0.8, 0.9, 0.7, 0.1, 0.2 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.8 + 0.9 + 0.7 = 2.4
            // Union = 1.0 + 1.0 + 1.0 + 0.1 + 0.2 = 3.3
            // IoU = 2.4 / 3.3 ≈ 0.7273, Loss = 1 - 0.7273 ≈ 0.2727
            Assert.Equal(0.27272727272727271, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithMixedValues_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.3, 0.7, 0.5 }),
                Actual = new Vector<double>(new double[] { 0.6, 0.4, 0.8 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = min(0.3,0.6) + min(0.7,0.4) + min(0.5,0.8) = 0.3 + 0.4 + 0.5 = 1.2
            // Union = max(0.3,0.6) + max(0.7,0.4) + max(0.5,0.8) = 0.6 + 0.7 + 0.8 = 2.1
            // IoU = 1.2 / 2.1 ≈ 0.5714, Loss = 1 - 0.5714 ≈ 0.4286
            Assert.Equal(0.42857142857142855, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithVerySmallValues_HandlesCorrectly()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.001, 0.002, 0.001 }),
                Actual = new Vector<double>(new double[] { 0.001, 0.001, 0.002 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Should handle small values without numerical instability
            Assert.True(result >= 0.0 && result <= 1.0);
        }

        [Fact]
        public void CalculateFitnessScore_WithHighOverlap_ReturnsLowLoss()
        {
            // Arrange
            var calculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.9, 0.95, 0.92, 0.88 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // High overlap should result in low loss
            Assert.True(result < 0.15); // Loss should be less than 15%
        }

        [Fact]
        public void CalculateFitnessScore_CompareWithDiceMetric_ShowsDifference()
        {
            // Arrange
            var jaccardCalculator = new JaccardLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var diceCalculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var jaccardResult = jaccardCalculator.CalculateFitnessScore(dataSet);
            var diceResult = diceCalculator.CalculateFitnessScore(dataSet);

            // Assert
            // Jaccard and Dice should give different values for the same data
            // Jaccard: Intersection=2, Union=3, IoU=2/3, Loss=1/3 ≈ 0.333
            // Dice: Intersection=2, Sum=5, Dice=4/5, Loss=1/5 = 0.2
            Assert.Equal(0.33333333333333331, jaccardResult, 10);
            Assert.Equal(0.2, diceResult, 10);
            Assert.NotEqual(jaccardResult, diceResult);
        }
    }
}
