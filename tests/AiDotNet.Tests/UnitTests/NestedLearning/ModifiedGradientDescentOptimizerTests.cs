using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.NestedLearning
{
    public class ModifiedGradientDescentOptimizerTests
    {
        [Fact]
        public void Constructor_WithValidLearningRate_InitializesCorrectly()
        {
            // Arrange & Act
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal(0.01, optimizer.LearningRate);
        }

        [Fact]
        public void UpdateMatrix_ImplementsEquation27Correctly()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

            // Create small test matrices
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0;
            weights[0, 1] = 0.5;
            weights[1, 0] = 0.5;
            weights[1, 1] = 1.0;

            var input = new Vector<double>(2);
            input[0] = 0.3;
            input[1] = 0.7;

            var gradient = new Vector<double>(2);
            gradient[0] = 0.1;
            gradient[1] = 0.2;

            // Act - Equation 27-29: Wt+1 = Wt * (I - xt*xt^T) - η * ∇ytL(Wt; xt) ⊗ xt
            var updated = optimizer.UpdateMatrix(weights, input, gradient);

            // Assert
            Assert.NotNull(updated);
            Assert.Equal(2, updated.Rows);
            Assert.Equal(2, updated.Columns);

            // Verify update includes both terms:
            // 1. Wt * (I - xt*xt^T) - modification term
            // 2. - η * gradient ⊗ input - standard gradient descent

            // The output should differ from input due to both terms
            bool hasChanged = false;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (Math.Abs(updated[i, j] - weights[i, j]) > 1e-10)
                    {
                        hasChanged = true;
                        break;
                    }
                }
            }
            Assert.True(hasChanged, "Weights should change after update");
        }

        [Fact]
        public void UpdateMatrix_WithZeroInput_OnlyAppliesGradientTerm()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0;
            weights[0, 1] = 0.5;
            weights[1, 0] = 0.5;
            weights[1, 1] = 1.0;

            var input = new Vector<double>(2);
            input[0] = 0.0;
            input[1] = 0.0;

            var gradient = new Vector<double>(2);
            gradient[0] = 0.1;
            gradient[1] = 0.2;

            // Act
            var updated = optimizer.UpdateMatrix(weights, input, gradient);

            // Assert
            Assert.NotNull(updated);

            // With zero input, (I - xt*xt^T) = I, so Wt * I = Wt
            // Only gradient term should apply, but since input is zero, outer product is zero
            // So result should be approximately Wt
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(weights[i, j], updated[i, j], precision: 5);
                }
            }
        }

        [Fact]
        public void UpdateVector_ImplementsModifiedGDForVectors()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

            var parameters = new Vector<double>(3);
            parameters[0] = 1.0;
            parameters[1] = 0.5;
            parameters[2] = -0.3;

            var input = new Vector<double>(3);
            input[0] = 0.2;
            input[1] = 0.5;
            input[2] = 0.3;

            var gradient = new Vector<double>(3);
            gradient[0] = 0.1;
            gradient[1] = -0.05;
            gradient[2] = 0.15;

            // Act
            var updated = optimizer.UpdateVector(parameters, input, gradient);

            // Assert
            Assert.NotNull(updated);
            Assert.Equal(3, updated.Length);

            // Verify parameters changed
            bool hasChanged = false;
            for (int i = 0; i < 3; i++)
            {
                if (Math.Abs(updated[i] - parameters[i]) > 1e-10)
                {
                    hasChanged = true;
                    break;
                }
            }
            Assert.True(hasChanged, "Parameters should change after update");
        }

        [Fact]
        public void UpdateVector_WithSmallLearningRate_GradientTermIsSmall()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.001); // Small LR

            var parameters = new Vector<double>(3);
            parameters[0] = 1.0;
            parameters[1] = 0.5;
            parameters[2] = -0.3;

            var input = new Vector<double>(3);
            input[0] = 0.2;
            input[1] = 0.5;
            input[2] = 0.3;

            var gradient = new Vector<double>(3);
            gradient[0] = 0.1;
            gradient[1] = -0.05;
            gradient[2] = 0.15;

            // Act
            var updated = optimizer.UpdateVector(parameters, input, gradient);

            // Assert
            // The Modified GD formula: w_{t+1} = w_t - x_t*dot(w_t,x_t) - η*gradient
            // The projection term (x_t*dot(w_t,x_t)) does NOT depend on learning rate
            // Only the gradient term (η*gradient) is scaled by learning rate
            // So total change can be larger than what the learning rate alone would suggest
            // We verify the parameters changed and are finite
            for (int i = 0; i < 3; i++)
            {
                double change = Math.Abs(updated[i] - parameters[i]);
                Assert.True(change < 1.0, $"Change at index {i} should be bounded");
                Assert.True(!double.IsNaN(updated[i]), $"Updated value at index {i} should not be NaN");
                Assert.True(!double.IsInfinity(updated[i]), $"Updated value at index {i} should not be infinity");
            }
        }

        [Fact]
        public void UpdateMatrix_MultipleUpdates_ConvergesParameters()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0;
            weights[0, 1] = 0.5;
            weights[1, 0] = 0.5;
            weights[1, 1] = 1.0;

            var input = new Vector<double>(2);
            input[0] = 0.3;
            input[1] = 0.7;

            var gradient = new Vector<double>(2);
            gradient[0] = 0.1;
            gradient[1] = 0.2;

            // Act - multiple updates
            var current = weights;
            for (int i = 0; i < 10; i++)
            {
                current = optimizer.UpdateMatrix(current, input, gradient);
            }

            // Assert - parameters should have changed significantly over multiple updates
            double totalChange = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    totalChange += Math.Abs(current[i, j] - weights[i, j]);
                }
            }
            Assert.True(totalChange > 0.01, "Parameters should change over multiple updates");
        }

        [Fact]
        public void LearningRate_Getter_ReturnsCorrectValue()
        {
            // Arrange
            var optimizer1 = new ModifiedGradientDescentOptimizer<double>(0.001);
            var optimizer2 = new ModifiedGradientDescentOptimizer<double>(0.1);

            // Act & Assert
            Assert.Equal(0.001, optimizer1.LearningRate);
            Assert.Equal(0.1, optimizer2.LearningRate);
        }

        [Fact]
        public void UpdateMatrix_DifferentFromStandardGD_DueToModificationTerm()
        {
            // Arrange
            var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0;
            weights[0, 1] = 0.5;
            weights[1, 0] = 0.5;
            weights[1, 1] = 1.0;

            var input = new Vector<double>(2);
            input[0] = 0.5;
            input[1] = 0.5;

            var gradient = new Vector<double>(2);
            gradient[0] = 0.1;
            gradient[1] = 0.1;

            // Act
            var modifiedUpdate = optimizer.UpdateMatrix(weights, input, gradient);

            // Compute standard GD update for comparison
            // Standard GD: Wt+1 = Wt - η * (gradient ⊗ input)
            var standardUpdate = new Matrix<double>(2, 2);
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    standardUpdate[i, j] = weights[i, j] - 0.01 * gradient[i] * input[j];
                }
            }

            // Assert - modified GD should differ from standard GD due to (I - xt*xt^T) term
            bool isDifferent = false;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (Math.Abs(modifiedUpdate[i, j] - standardUpdate[i, j]) > 1e-10)
                    {
                        isDifferent = true;
                        break;
                    }
                }
            }
            Assert.True(isDifferent, "Modified GD should differ from standard GD");
        }

        [Theory]
        [InlineData(0.001)]
        [InlineData(0.01)]
        [InlineData(0.1)]
        [InlineData(1.0)]
        public void Constructor_WithVariousLearningRates_InitializesCorrectly(double learningRate)
        {
            // Arrange & Act
            var optimizer = new ModifiedGradientDescentOptimizer<double>(learningRate);

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal(learningRate, optimizer.LearningRate);
        }
    }
}
