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
    /// Unit tests for DiceLossFitnessCalculator, which evaluates model performance for segmentation tasks.
    /// </summary>
    public class DiceLossFitnessCalculatorTests
    {
        [Fact]
        public void Constructor_WithDefaultDataSetType_UsesValidation()
        {
            // Arrange & Act
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter); // Dice loss: lower is better
        }

        [Fact]
        public void Constructor_WithTrainingDataSetType_UsesTraining()
        {
            // Arrange & Act
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CalculateFitnessScore_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Perfect overlap: Dice = (2 * 3) / (3 + 3) = 1.0, Loss = 1 - 1.0 = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithNoOverlap_ReturnsOne()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // No overlap: intersection = 0, Dice = 0, Loss = 1 - 0 = 1
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithPartialOverlap_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 1*1 + 1*1 + 1*0 + 0*0 = 2
            // Sum predicted = 3, Sum actual = 2
            // Dice = (2 * 2) / (3 + 2) = 4/5 = 0.8
            // Loss = 1 - 0.8 = 0.2
            Assert.Equal(0.2, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithProbabilisticPredictions_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.8, 0.6, 0.2, 0.1 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.8*1 + 0.6*1 + 0.2*0 + 0.1*0 = 1.4
            // Sum predicted = 1.7, Sum actual = 2.0
            // Dice = (2 * 1.4) / (1.7 + 2.0) = 2.8 / 3.7 ≈ 0.7568
            // Loss = 1 - 0.7568 ≈ 0.2432
            Assert.Equal(0.24324324324324326, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithAllZeros_ReturnsOne()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // All zeros case: handled by epsilon to prevent division by zero
            // Should return close to 1.0 (worst case)
            Assert.True(result >= 0.999);
        }

        [Fact]
        public void CalculateFitnessScore_WithSingleElement_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.5 }),
                Actual = new Vector<double>(new double[] { 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.5*1 = 0.5
            // Sum predicted = 0.5, Sum actual = 1.0
            // Dice = (2 * 0.5) / (0.5 + 1.0) = 1.0 / 1.5 ≈ 0.6667
            // Loss = 1 - 0.6667 ≈ 0.3333
            Assert.Equal(0.33333333333333331, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<float, Vector<float>, Vector<float>>();
            var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
            {
                Predicted = new Vector<float>(new float[] { 1.0f, 1.0f, 1.0f, 0.0f }),
                Actual = new Vector<float>(new float[] { 1.0f, 1.0f, 0.0f, 0.0f })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 2, Sum predicted = 3, Sum actual = 2
            // Dice = (2 * 2) / (3 + 2) = 0.8, Loss = 0.2
            Assert.Equal(0.2f, result, 5);
        }

        [Fact]
        public void CalculateFitnessScore_WithImbalancedData_HandlesCorrectly()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Simulating rare positive class (medical imaging scenario)
                Predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.9 + 0.8 = 1.7
            // Sum predicted = 1.7, Sum actual = 2.0
            // Dice = (2 * 1.7) / (1.7 + 2.0) = 3.4 / 3.7 ≈ 0.9189
            // Loss = 1 - 0.9189 ≈ 0.0811
            Assert.Equal(0.08108108108108109, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationData_UsesValidationSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            // Note: ModelEvaluationData requires 2D tensors for proper matrix conversion
            var calculator = new DiceLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Validation);
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
            // Note: ModelEvaluationData requires 2D tensors for proper matrix conversion
            var calculator = new DiceLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Testing);
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
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void IsBetterFitness_WithLowerScore_ReturnsTrue()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 0.8;
            double currentBestScore = 0.3;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.False(result); // Higher score is worse for loss functions
        }

        [Fact]
        public void CalculateFitnessScore_MedicalImagingScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating tumor segmentation scenario
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Model predicts tumor pixels with confidence scores
                Predicted = new Vector<double>(new double[] { 0.95, 0.88, 0.75, 0.10, 0.05 }),
                // Ground truth: first 3 pixels are tumor, last 2 are not
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Intersection = 0.95 + 0.88 + 0.75 = 2.58
            // Sum predicted = 2.73, Sum actual = 3.0
            // Dice = (2 * 2.58) / (2.73 + 3.0) = 5.16 / 5.73 ≈ 0.9005
            // Loss = 1 - 0.9005 ≈ 0.0995
            Assert.Equal(0.09947643979057592, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithVerySmallValues_HandlesCorrectly()
        {
            // Arrange
            var calculator = new DiceLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
    }
}
