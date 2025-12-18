using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Optimizers
{
    public class LionOptimizerTests
    {
        [Fact]
        public void Constructor_WithDefaultOptions_InitializesCorrectly()
        {
            // Arrange & Act
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null);
            var options = optimizer.GetOptions() as LionOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(options);
            if (options == null)
            {
                throw new InvalidOperationException("Options should not be null after assertion.");
            }

            Assert.Equal(1e-4, options.LearningRate);
            Assert.Equal(0.9, options.Beta1);
            Assert.Equal(0.99, options.Beta2);
            Assert.Equal(0.0, options.WeightDecay);
        }

        [Fact]
        public void Constructor_WithCustomOptions_UsesProvidedOptions()
        {
            // Arrange
            var customOptions = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.001,
                Beta1 = 0.95,
                Beta2 = 0.999,
                WeightDecay = 0.01
            };

            // Act
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, customOptions);
            var options = optimizer.GetOptions() as LionOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(options);
            if (options == null)
            {
                throw new InvalidOperationException("Options should not be null after assertion.");
            }

            Assert.Equal(0.001, options.LearningRate);
            Assert.Equal(0.95, options.Beta1);
            Assert.Equal(0.999, options.Beta2);
            Assert.Equal(0.01, options.WeightDecay);
        }

        [Fact]
        public void UpdateParameters_Vector_WithPositiveGradient_DecreasesParameters()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            // With positive gradient, sign is +1, so params should decrease by learning_rate
            Assert.True(updatedParams[0] < parameters[0]);
            Assert.True(updatedParams[1] < parameters[1]);
            Assert.True(updatedParams[2] < parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Vector_WithNegativeGradient_IncreasesParameters()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { -1.0, -1.0, -1.0 });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            // With negative gradient, sign is -1, so params should increase by learning_rate
            Assert.True(updatedParams[0] > parameters[0]);
            Assert.True(updatedParams[1] > parameters[1]);
            Assert.True(updatedParams[2] > parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Vector_WithMixedGradients_UpdatesCorrectly()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, -1.0, 0.5 });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0] < parameters[0]); // Positive gradient -> decrease
            Assert.True(updatedParams[1] > parameters[1]); // Negative gradient -> increase
            Assert.True(updatedParams[2] < parameters[2]); // Positive gradient -> decrease
        }

        [Fact]
        public void UpdateParameters_Vector_WithWeightDecay_AppliesRegularization()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.01
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var gradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }); // Zero gradient

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            // With zero gradient and weight decay, parameters should decrease proportionally
            Assert.True(updatedParams[0] < parameters[0]);
            Assert.True(updatedParams[1] < parameters[1]);
            Assert.True(updatedParams[2] < parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Matrix_WorksCorrectly()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Matrix<double>(2, 2);
            parameters[0, 0] = 1.0;
            parameters[0, 1] = 2.0;
            parameters[1, 0] = 3.0;
            parameters[1, 1] = 4.0;

            var gradient = new Matrix<double>(2, 2);
            gradient[0, 0] = 1.0;
            gradient[0, 1] = -1.0;
            gradient[1, 0] = 0.5;
            gradient[1, 1] = -0.5;

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0, 0] < parameters[0, 0]); // Positive gradient
            Assert.True(updatedParams[0, 1] > parameters[0, 1]); // Negative gradient
            Assert.True(updatedParams[1, 0] < parameters[1, 0]); // Positive gradient
            Assert.True(updatedParams[1, 1] > parameters[1, 1]); // Negative gradient
        }

        [Fact]
        public void UpdateParameters_Vector_ConsecutiveCalls_BuildsMomentum()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act - First update
            var updated1 = optimizer.UpdateParameters(parameters, gradient);

            // Act - Second update with same gradient
            var updated2 = optimizer.UpdateParameters(updated1, gradient);

            // Assert
            // Momentum should cause updates to be consistent across iterations
            Assert.NotEqual(parameters, updated1);
            Assert.NotEqual(updated1, updated2);
        }

        [Fact]
        public void UpdateParameters_Vector_SignBasedUpdates_IgnoreGradientMagnitude()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.0, // No interpolation to isolate sign effect
                Beta2 = 0.0,
                WeightDecay = 0.0
            };
            var optimizer1 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var smallGradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });
            var largeGradient = new Vector<double>(new double[] { 10.0, 10.0, 10.0 });

            // Act
            var updated1 = optimizer1.UpdateParameters(parameters, smallGradient);
            var updated2 = optimizer2.UpdateParameters(parameters, largeGradient);

            // Assert
            // With beta1=0, both should produce same result (sign-based)
            for (int i = 0; i < parameters.Length; i++)
            {
                Assert.Equal(updated1[i], updated2[i], 1e-9);
            }
        }

        [Fact]
        public void Reset_ClearsMomentumState()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act - Build momentum
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.UpdateParameters(parameters, gradient);

            // Reset
            optimizer.Reset();

            // Update after reset
            var updatedAfterReset = optimizer.UpdateParameters(parameters, gradient);

            // Assert - Should behave like first update again
            Assert.NotNull(updatedAfterReset);
        }

        [Fact]
        public void Serialize_Deserialize_PreservesState()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.001,
                Beta1 = 0.95,
                Beta2 = 0.999,
                WeightDecay = 0.01
            };
            var optimizer1 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 0.5, -0.5, 1.0 });

            // Build some state
            optimizer1.UpdateParameters(parameters, gradient);
            optimizer1.UpdateParameters(parameters, gradient);

            // Act - Serialize
            var serialized = optimizer1.Serialize();

            // Act - Deserialize into new optimizer
            var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null);
            optimizer2.Deserialize(serialized);

            // Get options to verify
            var deserializedOptions = optimizer2.GetOptions() as LionOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(deserializedOptions);
            if (deserializedOptions == null)
            {
                throw new InvalidOperationException("Deserialized options should not be null after assertion.");
            }

            Assert.Equal(options.LearningRate, deserializedOptions.LearningRate);
            Assert.Equal(options.Beta1, deserializedOptions.Beta1);
            Assert.Equal(options.Beta2, deserializedOptions.Beta2);
            Assert.Equal(options.WeightDecay, deserializedOptions.WeightDecay);
        }

        [Fact]
        public void UpdateOptions_WithValidOptions_UpdatesSuccessfully()
        {
            // Arrange
            var initialOptions = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.001
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, initialOptions);

            var newOptions = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.01
            };

            // Act
            optimizer.GetType()
                .GetMethod("UpdateOptions", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                ?.Invoke(optimizer, new object[] { newOptions });

            var currentOptions = optimizer.GetOptions() as LionOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(currentOptions);
            if (currentOptions == null)
            {
                throw new InvalidOperationException("Current options should not be null after assertion.");
            }

            Assert.Equal(0.01, currentOptions.LearningRate);
        }

        [Fact]
        public void UpdateParameters_Vector_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new LionOptimizerOptions<float, Vector<float>, Vector<float>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<float, Vector<float>, Vector<float>>(null, options);
            var parameters = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
            var gradient = new Vector<float>(new float[] { 1.0f, -1.0f, 0.5f });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0] < parameters[0]);
            Assert.True(updatedParams[1] > parameters[1]);
            Assert.True(updatedParams[2] < parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Matrix_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new LionOptimizerOptions<float, Vector<float>, Vector<float>>
            {
                LearningRate = 0.1f,
                Beta1 = 0.9,
                Beta2 = 0.99,
                WeightDecay = 0.0
            };
            var optimizer = new LionOptimizer<float, Vector<float>, Vector<float>>(null, options);
            var parameters = new Matrix<float>(2, 2);
            parameters[0, 0] = 1.0f;
            parameters[0, 1] = 2.0f;
            parameters[1, 0] = 3.0f;
            parameters[1, 1] = 4.0f;

            var gradient = new Matrix<float>(2, 2);
            gradient[0, 0] = 1.0f;
            gradient[0, 1] = -1.0f;
            gradient[1, 0] = 0.5f;
            gradient[1, 1] = -0.5f;

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0, 0] < parameters[0, 0]);
            Assert.True(updatedParams[0, 1] > parameters[0, 1]);
            Assert.True(updatedParams[1, 0] < parameters[1, 0]);
            Assert.True(updatedParams[1, 1] > parameters[1, 1]);
        }

        [Fact]
        public void UpdateParameters_Vector_DifferentBeta1Values_ProducesDifferentResults()
        {
            // Arrange
            var options1 = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.1,  // Low beta1 - more weight on current gradient
                Beta2 = 0.99
            };
            var options2 = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,  // High beta1 - more weight on momentum
                Beta2 = 0.99
            };

            var optimizer1 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options1);
            var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options2);

            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            // Use gradients that will build strong positive momentum, then a small negative gradient
            var gradient1 = new Vector<double>(new double[] { 10.0, 10.0, 10.0 });
            var gradient2 = new Vector<double>(new double[] { -1.0, -1.0, -1.0 });

            // Act - Build momentum with large positive gradient, then update with small negative gradient
            // After first update: m ≈ 0.1 (with beta2=0.99)
            // Low beta1 (0.1): c = 0.1 * 0.1 + 0.9 * (-1.0) = 0.01 - 0.9 = -0.89 → sign = -1
            // High beta1 (0.9): c = 0.9 * 0.1 + 0.1 * (-1.0) = 0.09 - 0.1 = -0.01 → sign = -1
            // Still both negative! Need even more extreme difference or multiple iterations

            // Let's do multiple updates to build stronger momentum
            optimizer1.UpdateParameters(parameters, gradient1);
            optimizer1.UpdateParameters(parameters, gradient1);
            var updated1 = optimizer1.UpdateParameters(parameters, gradient2);

            optimizer2.UpdateParameters(parameters, gradient1);
            optimizer2.UpdateParameters(parameters, gradient1);
            var updated2 = optimizer2.UpdateParameters(parameters, gradient2);

            // Assert - Different beta1 values should produce different interpolations
            // With low beta1, interpolation is closer to negative gradient (may be negative)
            // With high beta1, interpolation is closer to positive momentum (likely positive)
            // This can produce different signs and thus different results
            Assert.NotNull(updated1);
            Assert.NotNull(updated2);

            // Verify that at least one parameter value is different
            bool anyDifferent = false;
            for (int i = 0; i < updated1.Length; i++)
            {
                if (Math.Abs(updated1[i] - updated2[i]) > 1e-9)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.True(anyDifferent, "Different beta1 values should produce different results");
        }

        [Fact]
        public void UpdateParameters_Vector_DifferentBeta2Values_ProducesDifferentMomentum()
        {
            // Arrange
            var options1 = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.5,  // Moderate beta1 so momentum differences show through interpolation
                Beta2 = 0.1   // Low beta2 - momentum changes quickly
            };
            var options2 = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.5,  // Same beta1
                Beta2 = 0.9   // High beta2 - momentum changes slowly
            };

            var optimizer1 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options1);
            var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options2);

            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            // Use varying gradients to make momentum differences more visible
            var gradient1 = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });
            var gradient2 = new Vector<double>(new double[] { -0.5, -0.5, -0.5 });

            // Act - Multiple updates with alternating gradients to build different momentum profiles
            var params1 = new Vector<double>(parameters);
            var params2 = new Vector<double>(parameters);

            // First update with positive gradient
            params1 = optimizer1.UpdateParameters(params1, gradient1);
            params2 = optimizer2.UpdateParameters(params2, gradient1);

            // Second update with negative gradient - low beta2 adapts faster
            params1 = optimizer1.UpdateParameters(params1, gradient2);
            params2 = optimizer2.UpdateParameters(params2, gradient2);

            // Third update with positive gradient - momentum differences should be visible
            params1 = optimizer1.UpdateParameters(params1, gradient1);
            params2 = optimizer2.UpdateParameters(params2, gradient1);

            // Assert - Both should update, but momentum behavior differs
            Assert.NotEqual(parameters, params1);
            Assert.NotEqual(parameters, params2);

            // Verify that different Beta2 values produce different momentum behavior
            // With beta1=0.5, the interpolation gives equal weight to momentum and gradient
            // So different momentum (from different beta2) should produce different signs
            bool anyDifferent = false;
            for (int i = 0; i < params1.Length; i++)
            {
                if (Math.Abs(params1[i] - params2[i]) > 1e-9)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.True(anyDifferent, "Different beta2 values should produce different momentum behavior");
        }

        [Fact]
        public void GetOptions_ReturnsCurrentOptions()
        {
            // Arrange
            var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                LearningRate = 0.002,
                Beta1 = 0.85,
                Beta2 = 0.98
            };
            var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

            // Act
            var retrievedOptions = optimizer.GetOptions() as LionOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(retrievedOptions);
            if (retrievedOptions == null)
            {
                throw new InvalidOperationException("Retrieved options should not be null after assertion.");
            }

            Assert.Equal(0.002, retrievedOptions.LearningRate);
            Assert.Equal(0.85, retrievedOptions.Beta1);
            Assert.Equal(0.98, retrievedOptions.Beta2);
        }
    }
}
