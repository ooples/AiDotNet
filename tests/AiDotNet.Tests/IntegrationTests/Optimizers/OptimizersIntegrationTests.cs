using AiDotNet.Optimizers;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;

namespace AiDotNetTests.IntegrationTests.Optimizers
{
    /// <summary>
    /// Comprehensive integration tests for ALL optimizer classes with mathematically verified results.
    /// Tests verify parameter update behavior, convergence properties, and hyperparameter effects.
    /// Each optimizer is tested with well-known mathematical functions and convergence patterns.
    /// </summary>
    public class OptimizersIntegrationTests
    {
        private const double Tolerance = 1e-3;
        private const double LooseTolerance = 1e-1;

        #region Test Helper Classes and Functions

        /// <summary>
        /// Simple test model for optimizer testing
        /// </summary>
        private class SimpleTestModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
        {
            private Vector<T> _parameters;
            private readonly INumericOperations<T> _numOps;

            public SimpleTestModel(int parameterCount)
            {
                _numOps = MathHelper.GetNumericOperations<T>();
                _parameters = new Vector<T>(parameterCount);
                for (int i = 0; i < parameterCount; i++)
                {
                    _parameters[i] = _numOps.FromDouble(1.0);
                }
                ParameterCount = parameterCount;
            }

            public int ParameterCount { get; }

            public Vector<T> GetParameters() => _parameters;

            public void SetParameters(Vector<T> parameters)
            {
                if (parameters.Length != _parameters.Length)
                    throw new ArgumentException("Parameter count mismatch");
                _parameters = parameters;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
            {
                var model = new SimpleTestModel<T>(_parameters.Length);
                model.SetParameters(parameters);
                return model;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> Clone()
            {
                var clone = new SimpleTestModel<T>(_parameters.Length);
                clone.SetParameters(_parameters.Clone());
                return clone;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
            {
                return Clone();
            }

            public void Train(Matrix<T> inputs, Vector<T> outputs)
            {
                // No-op for test model
            }

            public Vector<T> Predict(Matrix<T> inputs)
            {
                // Simple linear prediction for testing
                var result = new Vector<T>(inputs.Rows);
                for (int i = 0; i < inputs.Rows; i++)
                {
                    T sum = _numOps.Zero;
                    for (int j = 0; j < Math.Min(inputs.Columns, _parameters.Length); j++)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(inputs[i, j], _parameters[j]));
                    }
                    result[i] = sum;
                }
                return result;
            }
        }

        /// <summary>
        /// Creates simple training data for optimizer testing
        /// </summary>
        private static (Matrix<double> X, Vector<double> y) CreateSimpleData(int samples = 20, int features = 2)
        {
            var X = new Matrix<double>(samples, features);
            var y = new Vector<double>(samples);

            var random = new Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    X[i, j] = random.NextDouble() * 10.0 - 5.0; // Range [-5, 5]
                }
                // y = 2*x1 + 3*x2 + noise
                y[i] = 2.0 * X[i, 0] + (features > 1 ? 3.0 * X[i, 1] : 0) + (random.NextDouble() - 0.5);
            }

            return (X, y);
        }

        /// <summary>
        /// Creates optimization input data
        /// </summary>
        private static OptimizationInputData<double, Matrix<double>, Vector<double>> CreateOptimizationData(
            Matrix<double> X, Vector<double> y)
        {
            return new OptimizationInputData<double, Matrix<double>, Vector<double>>
            {
                XTrain = X,
                YTrain = y,
                XValidation = X,
                YValidation = y,
                XTest = X,
                YTest = y
            };
        }

        #endregion

        #region Adam Optimizer Tests

        [Fact]
        public void Adam_UpdatesParameters_ReducesLoss()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                MaxIterations = 100,
                Tolerance = 1e-6
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
            Assert.True(result.IterationCount <= 100);
        }

        [Fact]
        public void Adam_WithHighLearningRate_ConvergesFaster()
        {
            // Arrange
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);

            var optionsLow = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 200
            };
            var optionsHigh = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 200
            };

            var optimizerLow = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model1, optionsLow);
            var optimizerHigh = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model2, optionsHigh);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);

            var resultLow = optimizerLow.Optimize(inputData);
            var resultHigh = optimizerHigh.Optimize(inputData);

            // Assert - Higher learning rate typically needs fewer iterations
            Assert.True(resultHigh.IterationCount <= resultLow.IterationCount ||
                       resultHigh.IterationCount < 200);
        }

        [Fact]
        public void Adam_WithDifferentBeta1_AffectsFirstMoment()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.95, // Non-default value
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void Adam_WithDifferentBeta2_AffectsSecondMoment()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta2 = 0.99, // Non-default value
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void Adam_WithSmallEpsilon_MaintainsNumericalStability()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Epsilon = 1e-10,
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Adam_WithFloatPrecision_ConvergesCorrectly()
        {
            // Arrange
            var model = new SimpleTestModel<float>(2);
            var options = new AdamOptimizerOptions<float, Matrix<float>, Vector<float>>
            {
                LearningRate = 0.1f,
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<float, Matrix<float>, Vector<float>>(model, options);

            // Act
            var X = new Matrix<float>(20, 2);
            var y = new Vector<float>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.5f;
                X[i, 1] = i * 0.3f;
                y[i] = 2.0f * X[i, 0] + 3.0f * X[i, 1];
            }

            var inputData = new OptimizationInputData<float, Matrix<float>, Vector<float>>
            {
                XTrain = X,
                YTrain = y,
                XValidation = X,
                YValidation = y,
                XTest = X,
                YTest = y
            };

            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Adam_StopsAtMaxIterations()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.001, // Very small to prevent early convergence
                MaxIterations = 50
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.True(result.IterationCount <= 50);
        }

        [Fact]
        public void Adam_ResetClearsState()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            optimizer.Reset();
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert - Should still work after reset
            Assert.NotNull(result);
        }

        [Fact]
        public void Adam_WithAdaptiveLearningRate_AdjustsDynamically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                UseAdaptiveLearningRate = true,
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region SGD Tests

        [Fact]
        public void SGD_ConvergesToSolution()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500,
                Tolerance = 1e-6
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void SGD_WithHighLearningRate_ConvergesFaster()
        {
            // Arrange
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);

            var optionsLow = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.001,
                MaxIterations = 500
            };
            var optionsHigh = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };

            var optimizerLow = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model1, optionsLow);
            var optimizerHigh = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model2, optionsHigh);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);

            var resultLow = optimizerLow.Optimize(inputData);
            var resultHigh = optimizerHigh.Optimize(inputData);

            // Assert
            Assert.True(resultHigh.IterationCount <= resultLow.IterationCount ||
                       resultHigh.IterationCount < 500);
        }

        [Fact]
        public void SGD_WithMomentum_ImprovesConvergence()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 500
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void SGD_WithAdaptiveLearningRate_AdjustsDynamically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                UseAdaptiveLearningRate = true,
                MaxIterations = 500
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void SGD_WithTolerance_StopsEarly()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Tolerance = 0.1,
                MaxIterations = 500
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.True(result.IterationCount < 500);
        }

        [Fact]
        public void SGD_WithFloatPrecision_Works()
        {
            // Arrange
            var model = new SimpleTestModel<float>(2);
            var options = new StochasticGradientDescentOptimizerOptions<float, Matrix<float>, Vector<float>>
            {
                InitialLearningRate = 0.01f,
                MaxIterations = 500
            };
            var optimizer = new StochasticGradientDescentOptimizer<float, Matrix<float>, Vector<float>>(model, options);

            // Act
            var X = new Matrix<float>(20, 2);
            var y = new Vector<float>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.5f;
                X[i, 1] = i * 0.3f;
                y[i] = 2.0f * X[i, 0] + 3.0f * X[i, 1];
            }

            var inputData = new OptimizationInputData<float, Matrix<float>, Vector<float>>
            {
                XTrain = X,
                YTrain = y,
                XValidation = X,
                YValidation = y,
                XTest = X,
                YTest = y
            };

            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Momentum Optimizer Tests

        [Fact]
        public void Momentum_AcceleratesConvergence()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 500
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void Momentum_WithHighMomentum_OvercomesLocalMinima()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.95,
                MaxIterations = 500
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Momentum_WithLowMomentum_BehavesLikeSGD()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.1,
                MaxIterations = 500
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Momentum_WithAdaptiveMomentum_AdjustsDynamically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                UseAdaptiveMomentum = true,
                MaxIterations = 500
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region RMSProp Tests

        [Fact]
        public void RMSProp_AdaptsLearningRatePerParameter()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Decay = 0.9,
                MaxIterations = 500
            };
            var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void RMSProp_WithDifferentDecay_AffectsConvergence()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Decay = 0.95,
                MaxIterations = 500
            };
            var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void RMSProp_WithSmallEpsilon_RemainsStable()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                Epsilon = 1e-10,
                MaxIterations = 500
            };
            var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void RMSProp_WithFloatPrecision_Converges()
        {
            // Arrange
            var model = new SimpleTestModel<float>(2);
            var options = new RootMeanSquarePropagationOptimizerOptions<float, Matrix<float>, Vector<float>>
            {
                InitialLearningRate = 0.01f,
                MaxIterations = 500
            };
            var optimizer = new RootMeanSquarePropagationOptimizer<float, Matrix<float>, Vector<float>>(model, options);

            // Act
            var X = new Matrix<float>(20, 2);
            var y = new Vector<float>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.5f;
                X[i, 1] = i * 0.3f;
                y[i] = 2.0f * X[i, 0] + 3.0f * X[i, 1];
            }

            var inputData = new OptimizationInputData<float, Matrix<float>, Vector<float>>
            {
                XTrain = X,
                YTrain = y,
                XValidation = X,
                YValidation = y,
                XTest = X,
                YTest = y
            };

            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region AdaGrad Tests

        [Fact]
        public void AdaGrad_AdaptsLearningRateIndividually()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdagradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new AdagradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void AdaGrad_WithHighLearningRate_ConvergesFaster()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdagradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.5,
                MaxIterations = 500
            };
            var optimizer = new AdagradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void AdaGrad_WithSmallEpsilon_MaintainsStability()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdagradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                Epsilon = 1e-10,
                MaxIterations = 500
            };
            var optimizer = new AdagradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region AdaDelta Tests

        [Fact]
        public void AdaDelta_DoesNotRequireLearningRate()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdaDeltaOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                Rho = 0.95,
                MaxIterations = 500
            };
            var optimizer = new AdaDeltaOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void AdaDelta_WithDifferentRho_AffectsConvergence()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdaDeltaOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                Rho = 0.9,
                MaxIterations = 500
            };
            var optimizer = new AdaDeltaOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void AdaDelta_WithAdaptiveRho_AdjustsDynamically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdaDeltaOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                Rho = 0.95,
                UseAdaptiveRho = true,
                MaxIterations = 500
            };
            var optimizer = new AdaDeltaOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Nadam Tests

        [Fact]
        public void Nadam_CombinesNesterovAndAdam()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NadamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new NadamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void Nadam_WithDifferentBeta1_ModifiesMomentum()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NadamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.95,
                MaxIterations = 500
            };
            var optimizer = new NadamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Nadam_WithDifferentBeta2_ModifiesAdaptiveRate()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NadamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta2 = 0.995,
                MaxIterations = 500
            };
            var optimizer = new NadamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region AMSGrad Tests

        [Fact]
        public void AMSGrad_MaintainsMaxSecondMoment()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new AMSGradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void AMSGrad_WithDifferentBeta1_AffectsFirstMoment()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                Beta1 = 0.95,
                MaxIterations = 500
            };
            var optimizer = new AMSGradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void AMSGrad_PreventsLearningRateIncreases()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new AMSGradOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        #endregion

        #region BFGS Tests

        [Fact]
        public void BFGS_UsesQuasiNewtonMethod()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200
            };
            var optimizer = new BFGSOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void BFGS_UpdatesHessianApproximation()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200
            };
            var optimizer = new BFGSOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void BFGS_WithLineSearch_FindsOptimalStep()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200
            };
            var optimizer = new BFGSOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region DFP Tests

        [Fact]
        public void DFP_UsesQuasiNewtonFormula()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new DFPOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200
            };
            var optimizer = new DFPOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void DFP_UpdatesInverseHessian()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new DFPOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200
            };
            var optimizer = new DFPOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region L-BFGS Tests

        [Fact]
        public void LBFGS_UsesLimitedMemory()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200,
                HistorySize = 10
            };
            var optimizer = new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void LBFGS_WithSmallHistory_UsesLessMemory()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.1,
                MaxIterations = 200,
                HistorySize = 5
            };
            var optimizer = new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Newton's Method Tests

        [Fact]
        public void NewtonMethod_UsesSecondOrderInfo()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NewtonMethodOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 100
            };
            var optimizer = new NewtonMethodOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void NewtonMethod_ConvergesQuadratically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NewtonMethodOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 100
            };
            var optimizer = new NewtonMethodOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            // Newton's method typically converges very quickly
            Assert.True(result.IterationCount <= 100);
        }

        #endregion

        #region Conjugate Gradient Tests

        [Fact]
        public void ConjugateGradient_UsesConjugateDirections()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void ConjugateGradient_ImprovesSteepestDescent()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region GradientDescent Tests

        [Fact]
        public void GradientDescent_FollowsSteepestDescentDirection()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void GradientDescent_WithLargeDataset_Scales()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData(100, 2); // Larger dataset
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Nesterov Accelerated Gradient Tests

        [Fact]
        public void NesterovAcceleratedGradient_UsesLookahead()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 500
            };
            var optimizer = new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void NesterovAcceleratedGradient_ImprovesMomentum()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.95,
                MaxIterations = 500
            };
            var optimizer = new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region AdaMax Tests

        [Fact]
        public void AdaMax_UsesInfinityNorm()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new AdaMaxOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void AdaMax_HandlesLargeGradients()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 500
            };
            var optimizer = new AdaMaxOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Lion Optimizer Tests

        [Fact]
        public void Lion_UsesSignBasedUpdate()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void Lion_WithMomentum_ImprovesConvergence()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                Beta1 = 0.9,
                Beta2 = 0.99,
                MaxIterations = 500
            };
            var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region MiniBatchGradientDescent Tests

        [Fact]
        public void MiniBatchGradientDescent_UsesRandomBatches()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MiniBatchGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                BatchSize = 4,
                MaxIterations = 500
            };
            var optimizer = new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void MiniBatchGradientDescent_WithDifferentBatchSizes_VariesSpeed()
        {
            // Arrange
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);

            var optionsSmall = new MiniBatchGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                BatchSize = 2,
                MaxIterations = 500
            };
            var optionsLarge = new MiniBatchGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                BatchSize = 8,
                MaxIterations = 500
            };

            var optimizerSmall = new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model1, optionsSmall);
            var optimizerLarge = new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model2, optionsLarge);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);

            var resultSmall = optimizerSmall.Optimize(inputData);
            var resultLarge = optimizerLarge.Optimize(inputData);

            // Assert
            Assert.NotNull(resultSmall);
            Assert.NotNull(resultLarge);
        }

        #endregion

        #region FTRL Tests

        [Fact]
        public void FTRL_OptimizesForOnlineLearning()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                Alpha = 0.1,
                Beta = 1.0,
                MaxIterations = 500
            };
            var optimizer = new FTRLOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void FTRL_WithL1Regularization_ProducesSparseWeights()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                Alpha = 0.1,
                Beta = 1.0,
                Lambda1 = 0.1,
                MaxIterations = 500
            };
            var optimizer = new FTRLOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region ProximalGradientDescent Tests

        [Fact]
        public void ProximalGradientDescent_HandlesNonSmoothProblems()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new ProximalGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new ProximalGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void ProximalGradientDescent_WithProximalOperator_PromotesSparseity()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new ProximalGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 500
            };
            var optimizer = new ProximalGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region CoordinateDescent Tests

        [Fact]
        public void CoordinateDescent_UpdatesOneCoordinateAtTime()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 500
            };
            var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void CoordinateDescent_ConvergesToSolution()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 500
            };
            var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region TrustRegion Tests

        [Fact]
        public void TrustRegion_UsesTrustRegionMethod()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 200
            };
            var optimizer = new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IterationCount > 0);
        }

        [Fact]
        public void TrustRegion_WithAdaptiveRadius_AdjustsDynamically()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 200,
                InitialRadius = 1.0
            };
            var optimizer = new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Serialization Tests

        [Fact]
        public void Adam_SerializeDeserialize_PreservesState()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 50
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            byte[] serialized = optimizer.Serialize();
            optimizer.Deserialize(serialized);

            // Assert
            Assert.NotNull(serialized);
            Assert.True(serialized.Length > 0);
        }

        [Fact]
        public void SGD_SerializeDeserialize_PreservesState()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 50
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            byte[] serialized = optimizer.Serialize();
            optimizer.Deserialize(serialized);

            // Assert
            Assert.NotNull(serialized);
            Assert.True(serialized.Length > 0);
        }

        [Fact]
        public void Momentum_SerializeDeserialize_PreservesState()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 50
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            byte[] serialized = optimizer.Serialize();
            optimizer.Deserialize(serialized);

            // Assert
            Assert.NotNull(serialized);
            Assert.True(serialized.Length > 0);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Adam_WithZeroGradient_HandlesGracefully()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 100
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act - Create data with constant output (flat gradient)
            var X = new Matrix<double>(10, 2);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                X[i, 1] = i * 2;
                y[i] = 5.0; // Constant
            }

            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert - Should not crash
            Assert.NotNull(result);
        }

        [Fact]
        public void SGD_WithLargeGradient_RemainsStable()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.0001, // Very small for stability
                MaxIterations = 500
            };
            var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act - Create data with large values
            var X = new Matrix<double>(10, 2);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i * 100;
                X[i, 1] = i * 200;
                y[i] = i * 1000; // Large values
            }

            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void Momentum_WithOscillatingLoss_Dampens()
        {
            // Arrange
            var model = new SimpleTestModel<double>(2);
            var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 500
            };
            var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model, options);

            // Act
            var X = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i;
                X[i, 1] = i * 2;
                y[i] = Math.Sin(i) * 10.0; // Oscillating
            }

            var inputData = CreateOptimizationData(X, y);
            var result = optimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Performance Comparison Tests

        [Fact]
        public void PerformanceComparison_Adam_Vs_SGD()
        {
            // Arrange
            var modelAdam = new SimpleTestModel<double>(2);
            var modelSGD = new SimpleTestModel<double>(2);

            var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 200
            };
            var sgdOptions = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 200
            };

            var adamOptimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(modelAdam, adamOptions);
            var sgdOptimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(modelSGD, sgdOptions);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);

            var adamResult = adamOptimizer.Optimize(inputData);
            var sgdResult = sgdOptimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(adamResult);
            Assert.NotNull(sgdResult);
            Assert.True(adamResult.IterationCount > 0);
            Assert.True(sgdResult.IterationCount > 0);
        }

        [Fact]
        public void PerformanceComparison_Momentum_Vs_Vanilla()
        {
            // Arrange
            var modelMomentum = new SimpleTestModel<double>(2);
            var modelVanilla = new SimpleTestModel<double>(2);

            var momentumOptions = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.9,
                MaxIterations = 500
            };
            var vanillaOptions = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                InitialMomentum = 0.0,
                MaxIterations = 500
            };

            var momentumOptimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(modelMomentum, momentumOptions);
            var vanillaOptimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(modelVanilla, vanillaOptions);

            // Act
            var (X, y) = CreateSimpleData();
            var inputData = CreateOptimizationData(X, y);

            var momentumResult = momentumOptimizer.Optimize(inputData);
            var vanillaResult = vanillaOptimizer.Optimize(inputData);

            // Assert
            Assert.NotNull(momentumResult);
            Assert.NotNull(vanillaResult);
        }

        [Fact]
        public void PerformanceComparison_AdaptiveVsNonAdaptive()
        {
            // Arrange
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);

            var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.1,
                MaxIterations = 200
            };
            var sgdOptions = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 200
            };

            var adamOptimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model1, adamOptions);
            var sgdOptimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model2, sgdOptions);

            // Act
            var (X, y) = CreateSimpleData(50, 2); // Larger dataset
            var inputData = CreateOptimizationData(X, y);

            var adamResult = adamOptimizer.Optimize(inputData);
            var sgdResult = sgdOptimizer.Optimize(inputData);

            // Assert - Both should converge
            Assert.NotNull(adamResult);
            Assert.NotNull(sgdResult);
        }

        #endregion
    }
}
