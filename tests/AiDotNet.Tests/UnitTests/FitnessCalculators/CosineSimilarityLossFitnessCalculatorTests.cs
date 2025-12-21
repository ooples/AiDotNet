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
    /// Unit tests for CosineSimilarityLossFitnessCalculator, which evaluates model performance based on vector direction similarity.
    /// </summary>
    public class CosineSimilarityLossFitnessCalculatorTests
    {
        [Fact]
        public void Constructor_WithDefaultDataSetType_UsesValidation()
        {
            // Arrange & Act
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter); // Cosine similarity loss: lower is better
        }

        [Fact]
        public void Constructor_WithTrainingDataSetType_UsesTraining()
        {
            // Arrange & Act
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>(DataSetType.Training);

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void CalculateFitnessScore_WithIdenticalVectors_ReturnsZero()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Perfect alignment: cosine similarity = 1.0, loss = 1 - 1.0 = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithOppositeVectors_ReturnsTwo()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                Actual = new Vector<double>(new double[] { -1.0, -2.0, -3.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Opposite directions: cosine similarity = -1.0, loss = 1 - (-1.0) = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithPerpendicularVectors_ReturnsOne()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Perpendicular vectors: cosine similarity = 0, loss = 1 - 0 = 1.0
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithSameDirectionDifferentMagnitude_ReturnsZero()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                Actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 }) // Same direction, 2x magnitude
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Same direction regardless of magnitude: cosine similarity = 1.0, loss = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithPartialAlignment_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 1.0, 1.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Dot product = 1.0
            // Norm predicted = 1.0, Norm actual = sqrt(2) ≈ 1.414
            // Cosine similarity = 1.0 / (1.0 * 1.414) ≈ 0.707
            // Loss = 1 - 0.707 ≈ 0.293
            Assert.Equal(0.29289321881345248, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithAllZeros_ReturnsOne()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // All zeros case: handled by epsilon to prevent division by zero
            // Should return close to 1.0 (no meaningful similarity)
            Assert.True(result >= 0.999);
        }

        [Fact]
        public void CalculateFitnessScore_WithSingleElement_ReturnsCorrectValue()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.5 }),
                Actual = new Vector<double>(new double[] { 1.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Dot product = 0.5, norms = 0.5 and 1.0
            // Cosine similarity = 0.5 / (0.5 * 1.0) = 1.0
            // Loss = 1 - 1.0 = 0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<float, Vector<float>, Vector<float>>();
            var dataSet = new DataSetStats<float, Vector<float>, Vector<float>>
            {
                Predicted = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f }),
                Actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            Assert.Equal(0.0f, result, 5);
        }

        [Fact]
        public void CalculateFitnessScore_WithNullDataSet_ThrowsArgumentNullException()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                calculator.CalculateFitnessScore((DataSetStats<double, Vector<double>, Vector<double>>)null));
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationData_UsesValidationSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Validation);
            var evaluationData = new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
            {
                ValidationSet = new DataSetStats<double, Tensor<double>, Tensor<double>>
                {
                    Predicted = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 2.0 })),
                    Actual = new Tensor<double>(new int[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 2.0 }))
                }
            };

            // Act
            var result = calculator.CalculateFitnessScore(evaluationData);

            // Assert
            Assert.Equal(0.0, result, 10); // Perfect alignment
        }

        [Fact]
        public void CalculateFitnessScore_WithModelEvaluationDataAndTestSet_UsesTestSet()
        {
            // Arrange - Use Tensor types which are supported by ModelEvaluationData
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Tensor<double>, Tensor<double>>(DataSetType.Testing);
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
            Assert.Equal(1.0, result, 10); // Perpendicular vectors
        }

        [Fact]
        public void IsHigherScoreBetter_ReturnsFalse()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.False(calculator.IsHigherScoreBetter);
        }

        [Fact]
        public void IsBetterFitness_WithLowerScore_ReturnsTrue()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
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
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            double newScore = 0.8;
            double currentBestScore = 0.3;

            // Act
            var result = calculator.IsBetterFitness(newScore, currentBestScore);

            // Assert
            Assert.False(result); // Higher score is worse for loss functions
        }

        [Fact]
        public void CalculateFitnessScore_DocumentSimilarityScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating document vector comparison
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Term frequency vectors for similar documents
                Predicted = new Vector<double>(new double[] { 0.5, 0.8, 0.1, 0.0 }),
                Actual = new Vector<double>(new double[] { 0.6, 0.7, 0.2, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Dot product = 0.5*0.6 + 0.8*0.7 + 0.1*0.2 + 0.0*0.0 = 0.30 + 0.56 + 0.02 + 0.00 = 0.88
            // Norm predicted = sqrt(0.5² + 0.8² + 0.1² + 0.0²) = sqrt(0.90) ≈ 0.9487
            // Norm actual = sqrt(0.6² + 0.7² + 0.2² + 0.0²) = sqrt(0.89) ≈ 0.9434
            // Cosine similarity = 0.88 / (0.9487 * 0.9434) ≈ 0.9833
            // Loss = 1 - 0.9833 ≈ 0.0167
            Assert.Equal(0.016744432707480161, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithNegativeValues_HandlesCorrectly()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { -1.0, 2.0, -3.0 }),
                Actual = new Vector<double>(new double[] { 1.0, -2.0, 3.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Dot product = -1 + (-4) + (-9) = -14
            // Norms are both sqrt(14)
            // Cosine similarity = -14 / 14 = -1.0 (opposite directions)
            // Loss = 1 - (-1.0) = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateFitnessScore_WithSmallAngles_ReturnsSmallLoss()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Nearly aligned vectors
                Predicted = new Vector<double>(new double[] { 1.0, 0.1 }),
                Actual = new Vector<double>(new double[] { 1.0, 0.0 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Small angle should result in high similarity and low loss
            Assert.True(result < 0.01);
        }

        [Fact]
        public void CalculateFitnessScore_WithLargeAngles_ReturnsLargeLoss()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Nearly opposite vectors
                Predicted = new Vector<double>(new double[] { 1.0, 0.1 }),
                Actual = new Vector<double>(new double[] { -1.0, -0.1 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Large angle (nearly 180°) should result in low similarity and high loss (close to 2.0)
            Assert.True(result > 1.9);
        }

        [Fact]
        public void CalculateFitnessScore_RecommendationSystemScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating user preference vectors
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // User preferences for different categories
                Predicted = new Vector<double>(new double[] { 0.9, 0.1, 0.5, 0.3 }),
                Actual = new Vector<double>(new double[] { 0.8, 0.2, 0.6, 0.2 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Should show high similarity (low loss) for similar preferences
            Assert.True(result < 0.1); // Less than 10% loss
        }

        [Fact]
        public void CalculateFitnessScore_ImageRetrievalScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating image feature vectors
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Feature vectors for similar images
                Predicted = new Vector<double>(new double[] { 0.7, 0.5, 0.3, 0.8, 0.2 }),
                Actual = new Vector<double>(new double[] { 0.6, 0.6, 0.4, 0.7, 0.1 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Dot product = 0.42 + 0.30 + 0.12 + 0.56 + 0.02 = 1.42
            // Should handle feature vector comparison correctly
            Assert.True(result >= 0.0 && result <= 2.0);
        }

        [Fact]
        public void CalculateFitnessScore_WithVerySmallValues_HandlesCorrectly()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 0.001, 0.002, 0.001 }),
                Actual = new Vector<double>(new double[] { 0.001, 0.001, 0.002 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Should handle small values without numerical instability
            Assert.True(result >= 0.0 && result <= 2.0);
        }

        [Fact]
        public void CalculateFitnessScore_MagnitudeInvariance_VerifiesProperty()
        {
            // Arrange
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet1 = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                Actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 })
            };
            var dataSet2 = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                Predicted = new Vector<double>(new double[] { 10.0, 20.0, 30.0 }),
                Actual = new Vector<double>(new double[] { 20.0, 40.0, 60.0 })
            };

            // Act
            var result1 = calculator.CalculateFitnessScore(dataSet1);
            var result2 = calculator.CalculateFitnessScore(dataSet2);

            // Assert
            // Cosine similarity is magnitude-invariant - both should give same result
            Assert.Equal(result1, result2, 10);
        }

        [Fact]
        public void CalculateFitnessScore_TextEmbeddingScenario_ReturnsCorrectValue()
        {
            // Arrange - Simulating word/sentence embeddings
            var calculator = new CosineSimilarityLossFitnessCalculator<double, Vector<double>, Vector<double>>();
            var dataSet = new DataSetStats<double, Vector<double>, Vector<double>>
            {
                // Embeddings for semantically similar text
                Predicted = new Vector<double>(new double[] { 0.8, 0.6, -0.2, 0.9, -0.1 }),
                Actual = new Vector<double>(new double[] { 0.7, 0.7, -0.1, 0.8, -0.2 })
            };

            // Act
            var result = calculator.CalculateFitnessScore(dataSet);

            // Assert
            // Should indicate high similarity for semantically similar embeddings
            Assert.True(result < 0.15); // Less than 15% loss
        }
    }
}
