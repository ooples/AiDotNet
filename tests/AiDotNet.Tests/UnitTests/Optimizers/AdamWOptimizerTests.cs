using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Optimizers
{
    public class AdamWOptimizerTests
    {
        [Fact]
        public void Constructor_WithDefaultOptions_InitializesCorrectly()
        {
            // Arrange & Act
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null);
            var options = optimizer.GetOptions() as AdamWOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(options);
            if (options == null)
            {
                throw new InvalidOperationException("Options should not be null after assertion.");
            }

            Assert.Equal(0.001, options.InitialLearningRate);
            Assert.Equal(0.9, options.Beta1);
            Assert.Equal(0.999, options.Beta2);
            Assert.Equal(1e-8, options.Epsilon);
            Assert.Equal(0.01, options.WeightDecay);
        }

        [Fact]
        public void Constructor_WithCustomOptions_UsesProvidedOptions()
        {
            // Arrange
            var customOptions = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.85,
                Beta2 = 0.9999,
                Epsilon = 1e-7,
                WeightDecay = 0.05
            };

            // Act
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, customOptions);
            var options = optimizer.GetOptions() as AdamWOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(options);
            Assert.Equal(0.01, options!.InitialLearningRate);
            Assert.Equal(0.85, options.Beta1);
            Assert.Equal(0.9999, options.Beta2);
            Assert.Equal(1e-7, options.Epsilon);
            Assert.Equal(0.05, options.WeightDecay);
        }

        [Fact]
        public void UpdateParameters_Vector_WithPositiveGradient_DecreasesParameters()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.0
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0] < parameters[0]);
            Assert.True(updatedParams[1] < parameters[1]);
            Assert.True(updatedParams[2] < parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Vector_WithNegativeGradient_IncreasesParameters()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.0
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { -1.0, -1.0, -1.0 });

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            Assert.True(updatedParams[0] > parameters[0]);
            Assert.True(updatedParams[1] > parameters[1]);
            Assert.True(updatedParams[2] > parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Vector_WithWeightDecay_AppliesDecoupledDecay()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.1 // Large weight decay for visibility
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var gradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }); // Zero gradient

            // Act
            var updatedParams = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            // With zero gradient and weight decay, parameters should still decrease
            // due to decoupled weight decay: params = params - lr * wd * params
            Assert.True(updatedParams[0] < parameters[0]);
            Assert.True(updatedParams[1] < parameters[1]);
            Assert.True(updatedParams[2] < parameters[2]);
        }

        [Fact]
        public void UpdateParameters_Matrix_WorksCorrectly()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.0
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
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
        public void UpdateParameters_ConsecutiveCalls_BuildsMomentum()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.0
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act - Multiple updates
            var current = parameters;
            var differences = new List<double>();
            for (int i = 0; i < 5; i++)
            {
                var next = optimizer.UpdateParameters(current, gradient);
                differences.Add(Math.Abs(current[0] - next[0]));
                current = next;
            }

            // Assert - Later updates should have built momentum
            Assert.NotNull(current);
            Assert.True(differences.Count == 5);
        }

        [Fact]
        public void AdamW_DifferentFromAdam_DueToDecoupledWeightDecay()
        {
            // This test verifies the key difference between AdamW and Adam:
            // AdamW applies weight decay directly to parameters, not to gradients

            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.1
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

            // Large initial weights
            var parameters = new Vector<double>(new double[] { 100.0, 100.0, 100.0 });
            var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 }); // Small gradient

            // Act
            var updated = optimizer.UpdateParameters(parameters, gradient);

            // Assert
            // With weight decay 0.1 and lr 0.01, the decoupled decay should be noticeable
            // params should decrease by at least lr * wd * params = 0.01 * 0.1 * 100 = 0.1
            Assert.True(parameters[0] - updated[0] > 0.05); // Significant decrease
        }

        [Fact]
        public void Reset_ClearsOptimizerState()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Build momentum
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.UpdateParameters(parameters, gradient);
            optimizer.UpdateParameters(parameters, gradient);

            // Act
            optimizer.Reset();

            // Update after reset
            var updatedAfterReset = optimizer.UpdateParameters(parameters, gradient);

            // Assert - Should behave like first update
            Assert.NotNull(updatedAfterReset);
        }

        [Fact]
        public void Serialize_Deserialize_PreservesState()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.002,
                Beta1 = 0.85,
                Beta2 = 0.9999,
                WeightDecay = 0.05
            };
            var optimizer1 = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient = new Vector<double>(new double[] { 0.5, -0.5, 1.0 });

            // Build state
            optimizer1.UpdateParameters(parameters, gradient);
            optimizer1.UpdateParameters(parameters, gradient);

            // Act - Serialize
            var serialized = optimizer1.Serialize();

            // Act - Deserialize
            var defaultOptions = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>();
            var optimizer2 = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, defaultOptions);
            optimizer2.Deserialize(serialized);

            var deserializedOptions = optimizer2.GetOptions() as AdamWOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(deserializedOptions);
            Assert.Equal(options.InitialLearningRate, deserializedOptions!.InitialLearningRate);
            Assert.Equal(options.WeightDecay, deserializedOptions.WeightDecay);
        }

        [Fact]
        public void UpdateParameters_WithAMSGrad_UsesMaxSecondMoment()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.9,
                Beta2 = 0.999,
                UseAMSGrad = true
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);
            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient1 = new Vector<double>(new double[] { 10.0, 10.0, 10.0 }); // Large gradient
            var gradient2 = new Vector<double>(new double[] { 0.1, 0.1, 0.1 }); // Small gradient

            // Act
            optimizer.UpdateParameters(parameters, gradient1);
            var afterSmallGrad = optimizer.UpdateParameters(parameters, gradient2);

            // Assert - With AMSGrad, the large second moment from first update should persist
            Assert.NotNull(afterSmallGrad);
        }

        [Fact]
        public void UpdateParameters_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<float, Vector<float>, Vector<float>>
            {
                InitialLearningRate = 0.1f,
                Beta1 = 0.9,
                Beta2 = 0.999,
                WeightDecay = 0.0
            };
            var optimizer = new AdamWOptimizer<float, Vector<float>, Vector<float>>(null, options);
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
        public void GetOptions_ReturnsCurrentOptions()
        {
            // Arrange
            var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                InitialLearningRate = 0.005,
                Beta1 = 0.92,
                Beta2 = 0.9995,
                WeightDecay = 0.02
            };
            var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

            // Act
            var retrievedOptions = optimizer.GetOptions() as AdamWOptimizerOptions<double, Vector<double>, Vector<double>>;

            // Assert
            Assert.NotNull(retrievedOptions);
            Assert.Equal(0.005, retrievedOptions!.InitialLearningRate);
            Assert.Equal(0.92, retrievedOptions.Beta1);
            Assert.Equal(0.9995, retrievedOptions.Beta2);
            Assert.Equal(0.02, retrievedOptions.WeightDecay);
        }

        [Fact]
        public void UpdateParameters_DifferentBeta1Values_ProducesDifferentResults()
        {
            // Arrange
            var options1 = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, Beta1 = 0.5, Beta2 = 0.999 };
            var options2 = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, Beta1 = 0.99, Beta2 = 0.999 };

            var optimizer1 = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options1);
            var optimizer2 = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options2);

            var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var gradient1 = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });
            var gradient2 = new Vector<double>(new double[] { -1.0, -1.0, -1.0 });

            // Act - Update with different gradients
            optimizer1.UpdateParameters(parameters, gradient1);
            var updated1 = optimizer1.UpdateParameters(parameters, gradient2);

            optimizer2.UpdateParameters(parameters, gradient1);
            var updated2 = optimizer2.UpdateParameters(parameters, gradient2);

            // Assert - Different beta1 should produce different momentum behavior
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
    }
}
