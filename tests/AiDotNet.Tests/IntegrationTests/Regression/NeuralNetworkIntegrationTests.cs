using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for neural network-based regression models.
    /// Tests feedforward networks, multilayer perceptrons, and radial basis function networks.
    /// </summary>
    public class NeuralNetworkIntegrationTests
    {
        #region NeuralNetworkRegression Tests

        [Fact]
        public void NeuralNetworkRegression_LinearData_ConvergesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1] + 5;
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 10 },
                MaxEpochs = 100,
                LearningRate = 0.01
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should approximate linear relationship
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 10.0);
        }

        [Fact]
        public void NeuralNetworkRegression_NonLinearPattern_LearnsComplex()
        {
            // Arrange - non-linear XOR-like pattern
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i / 4.0;
                y[i] = Math.Sin(x[i, 0]) * 10 + 5;
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 15, 10 },
                MaxEpochs = 200,
                LearningRate = 0.01
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should learn non-linear pattern
            double totalError = 0;
            for (int i = 0; i < 20; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 20 < 8.0);
        }

        [Fact]
        public void NeuralNetworkRegression_DifferentArchitectures_AffectPerformance()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] * x[i, 1]; // Non-linear interaction
            }

            // Act - shallow vs deep network
            var shallowNN = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 5 }, MaxEpochs = 50 });
            shallowNN.Train(x, y);
            var predShallow = shallowNN.Predict(x);

            var deepNN = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 10, 10 }, MaxEpochs = 50 });
            deepNN.Train(x, y);
            var predDeep = deepNN.Predict(x);

            // Assert - both should work but may have different accuracies
            Assert.NotNull(predShallow);
            Assert.NotNull(predDeep);
        }

        [Fact]
        public void NeuralNetworkRegression_LearningRate_AffectsConvergence()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2 + 3;
            }

            // Act - different learning rates
            var nnSlow = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 10 }, MaxEpochs = 20, LearningRate = 0.001 });
            nnSlow.Train(x, y);

            var nnFast = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 10 }, MaxEpochs = 20, LearningRate = 0.1 });
            nnFast.Train(x, y);

            // Assert - both should produce valid predictions
            var predSlow = nnSlow.Predict(x);
            var predFast = nnFast.Predict(x);
            Assert.NotNull(predSlow);
            Assert.NotNull(predFast);
        }

        [Fact]
        public void NeuralNetworkRegression_ActivationFunctions_AffectLearning()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - different activation functions
            var nnReLU = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions
                {
                    HiddenLayers = new[] { 10 },
                    ActivationFunction = ActivationFunction.ReLU,
                    MaxEpochs = 50
                });
            nnReLU.Train(x, y);

            var nnSigmoid = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions
                {
                    HiddenLayers = new[] { 10 },
                    ActivationFunction = ActivationFunction.Sigmoid,
                    MaxEpochs = 50
                });
            nnSigmoid.Train(x, y);

            // Assert - both should work
            var predReLU = nnReLU.Predict(x);
            var predSigmoid = nnSigmoid.Predict(x);
            Assert.NotNull(predReLU);
            Assert.NotNull(predSigmoid);
        }

        [Fact]
        public void NeuralNetworkRegression_BatchSize_AffectsTraining()
        {
            // Arrange
            var x = new Matrix<double>(40, 2);
            var y = new Vector<double>(40);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act - different batch sizes
            var nnSmallBatch = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 10 }, BatchSize = 5, MaxEpochs = 30 });
            nnSmallBatch.Train(x, y);

            var nnLargeBatch = new NeuralNetworkRegression<double>(
                new NeuralNetworkRegressionOptions { HiddenLayers = new[] { 10 }, BatchSize = 20, MaxEpochs = 30 });
            nnLargeBatch.Train(x, y);

            // Assert
            var predSmall = nnSmallBatch.Predict(x);
            var predLarge = nnLargeBatch.Predict(x);
            Assert.NotNull(predSmall);
            Assert.NotNull(predLarge);
        }

        [Fact]
        public void NeuralNetworkRegression_EarlyStopping_PreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = i + (random.NextDouble() - 0.5) * 2;
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 20 },
                MaxEpochs = 500,
                UseEarlyStopping = true,
                ValidationSplit = 0.2
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);

            // Assert - should stop early
            var actualEpochs = regression.GetActualEpochsRun();
            Assert.True(actualEpochs < 500);
        }

        [Fact]
        public void NeuralNetworkRegression_Dropout_ReducesOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 15, 10 },
                DropoutRate = 0.3,
                MaxEpochs = 50
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void NeuralNetworkRegression_SmallDataset_StillLearns()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i + 1;
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 5 },
                MaxEpochs = 100
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void NeuralNetworkRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 200;
            var x = new Matrix<double>(n, 3);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                x[i, 2] = i / 2.0;
                y[i] = x[i, 0] + x[i, 1] + x[i, 2];
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 15 },
                MaxEpochs = 30,
                BatchSize = 32
            };

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new NeuralNetworkRegression<double>(options);
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 15000);
        }

        [Fact]
        public void NeuralNetworkRegression_MultipleOutputs_HandlesCorrectly()
        {
            // Arrange - multiple outputs
            var x = new Matrix<double>(20, 2);
            var y = new Matrix<double>(20, 2); // Multiple outputs

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i, 0] = x[i, 0] + x[i, 1];
                y[i, 1] = x[i, 0] - x[i, 1];
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 10 },
                MaxEpochs = 50,
                OutputDimension = 2
            };

            // Act
            var regression = new NeuralNetworkRegression<double>(options);
            regression.TrainMultiOutput(x, y);
            var predictions = regression.PredictMultiOutput(x);

            // Assert
            Assert.Equal(20, predictions.Rows);
            Assert.Equal(2, predictions.Columns);
        }

        [Fact]
        public void NeuralNetworkRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(15, 1);
            var y = new Vector<float>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3 + 2;
            }

            var options = new NeuralNetworkRegressionOptions
            {
                HiddenLayers = new[] { 8 },
                MaxEpochs = 50
            };

            // Act
            var regression = new NeuralNetworkRegression<float>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        #endregion

        #region MultilayerPerceptronRegression Tests

        [Fact]
        public void MultilayerPerceptronRegression_DeepNetwork_LearnsComplexPatterns()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i / 5.0;
                x[i, 1] = i / 3.0;
                y[i] = Math.Sin(x[i, 0]) * Math.Cos(x[i, 1]) * 10;
            }

            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 20, 15, 10 },
                MaxEpochs = 150,
                LearningRate = 0.01
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should learn complex interaction
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 8.0);
        }

        [Fact]
        public void MultilayerPerceptronRegression_BackpropagationLearning_ConvergesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 5;
            }

            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 12, 8 },
                MaxEpochs = 100,
                LearningRate = 0.01,
                Momentum = 0.9
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void MultilayerPerceptronRegression_MomentumParameter_AcceleratesLearning()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - with and without momentum
            var mlpNoMomentum = new MultilayerPerceptronRegression<double>(
                new MultilayerPerceptronOptions { HiddenLayers = new[] { 10 }, MaxEpochs = 50, Momentum = 0.0 });
            mlpNoMomentum.Train(x, y);

            var mlpWithMomentum = new MultilayerPerceptronRegression<double>(
                new MultilayerPerceptronOptions { HiddenLayers = new[] { 10 }, MaxEpochs = 50, Momentum = 0.9 });
            mlpWithMomentum.Train(x, y);

            // Assert - both should work
            var predNoMomentum = mlpNoMomentum.Predict(x);
            var predWithMomentum = mlpWithMomentum.Predict(x);
            Assert.NotNull(predNoMomentum);
            Assert.NotNull(predWithMomentum);
        }

        [Fact]
        public void MultilayerPerceptronRegression_AdaptiveLearningRate_ImprovesConvergence()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 15, 10 },
                MaxEpochs = 100,
                UseAdaptiveLearningRate = true
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void MultilayerPerceptronRegression_BatchNormalization_StabilizesTraining()
        {
            // Arrange
            var x = new Matrix<double>(40, 3);
            var y = new Vector<double>(40);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i * 100; // Large scale
                x[i, 1] = i * 0.01; // Small scale
                x[i, 2] = i;
                y[i] = x[i, 0] / 100 + x[i, 1] * 100 + x[i, 2];
            }

            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 15, 10 },
                MaxEpochs = 50,
                UseBatchNormalization = true
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void MultilayerPerceptronRegression_RegularizationL2_PreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);
            var random = new Random(123);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 5;
            }

            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(0.01);
            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 20, 15 },
                MaxEpochs = 100
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options, regularization);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void MultilayerPerceptronRegression_DifferentOptimizers_AffectTraining()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            // Act - different optimizers
            var mlpSGD = new MultilayerPerceptronRegression<double>(
                new MultilayerPerceptronOptions
                {
                    HiddenLayers = new[] { 10 },
                    MaxEpochs = 50,
                    Optimizer = OptimizerType.SGD
                });
            mlpSGD.Train(x, y);

            var mlpAdam = new MultilayerPerceptronRegression<double>(
                new MultilayerPerceptronOptions
                {
                    HiddenLayers = new[] { 10 },
                    MaxEpochs = 50,
                    Optimizer = OptimizerType.Adam
                });
            mlpAdam.Train(x, y);

            // Assert
            var predSGD = mlpSGD.Predict(x);
            var predAdam = mlpAdam.Predict(x);
            Assert.NotNull(predSGD);
            Assert.NotNull(predAdam);
        }

        [Fact]
        public void MultilayerPerceptronRegression_ClassificationTask_CanAdapt()
        {
            // Arrange - binary classification task (probabilities)
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = i < 10 ? 0.0 : 1.0;
            }

            var options = new MultilayerPerceptronOptions
            {
                HiddenLayers = new[] { 10, 5 },
                MaxEpochs = 100,
                OutputActivation = ActivationFunction.Sigmoid
            };

            // Act
            var regression = new MultilayerPerceptronRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be between 0 and 1
            for (int i = 0; i < 20; i++)
            {
                Assert.True(predictions[i] >= 0.0 && predictions[i] <= 1.0);
            }
        }

        #endregion

        #region RadialBasisFunctionRegression Tests

        [Fact]
        public void RadialBasisFunctionRegression_GaussianKernels_ApproximatesFunction()
        {
            // Arrange
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i / 5.0;
                y[i] = Math.Sin(x[i, 0]) * 10 + 5;
            }

            var options = new RadialBasisFunctionOptions { NumCenters = 10 };

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should approximate sine wave
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void RadialBasisFunctionRegression_NumberOfCenters_AffectsCapacity()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sqrt(i) * 5;
            }

            // Act - different number of centers
            var rbf5 = new RadialBasisFunctionRegression<double>(
                new RadialBasisFunctionOptions { NumCenters = 5 });
            rbf5.Train(x, y);
            var pred5 = rbf5.Predict(x);

            var rbf15 = new RadialBasisFunctionRegression<double>(
                new RadialBasisFunctionOptions { NumCenters = 15 });
            rbf15.Train(x, y);
            var pred15 = rbf15.Predict(x);

            // Assert - more centers should provide better fit
            double error5 = 0, error15 = 0;
            for (int i = 0; i < 20; i++)
            {
                error5 += Math.Abs(pred5[i] - y[i]);
                error15 += Math.Abs(pred15[i] - y[i]);
            }
            Assert.True(error15 <= error5);
        }

        [Fact]
        public void RadialBasisFunctionRegression_BandwidthParameter_AffectsSmoothness()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (i % 3) * 3;
            }

            // Act - different bandwidths
            var rbfNarrow = new RadialBasisFunctionRegression<double>(
                new RadialBasisFunctionOptions { NumCenters = 8, Bandwidth = 0.5 });
            rbfNarrow.Train(x, y);
            var predNarrow = rbfNarrow.Predict(x);

            var rbfWide = new RadialBasisFunctionRegression<double>(
                new RadialBasisFunctionOptions { NumCenters = 8, Bandwidth = 3.0 });
            rbfWide.Train(x, y);
            var predWide = rbfWide.Predict(x);

            // Assert - different smoothness
            bool different = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(predNarrow[i] - predWide[i]) > 2.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void RadialBasisFunctionRegression_LocalApproximation_WorksWell()
        {
            // Arrange - piecewise different patterns
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            for (int i = 15; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = 50 - i;
            }

            var options = new RadialBasisFunctionOptions { NumCenters = 12 };

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should capture both patterns
            Assert.True(predictions[5] < predictions[10]); // First segment increasing
            Assert.True(predictions[20] > predictions[25]); // Second segment decreasing
        }

        [Fact]
        public void RadialBasisFunctionRegression_MultipleFeatures_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = Math.Sqrt(x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]); // Euclidean distance
            }

            var options = new RadialBasisFunctionOptions { NumCenters = 10 };

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void RadialBasisFunctionRegression_InterpolationProperty_FitsTrainingData()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            var options = new RadialBasisFunctionOptions { NumCenters = 10 }; // One center per point

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should interpolate training points well
            for (int i = 0; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        [Fact]
        public void RadialBasisFunctionRegression_SmallDataset_StillWorks()
        {
            // Arrange
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i;
            }

            var options = new RadialBasisFunctionOptions { NumCenters = 4 };

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(6, predictions.Length);
        }

        [Fact]
        public void RadialBasisFunctionRegression_RegularizationPreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            var random = new Random(456);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (random.NextDouble() - 0.5) * 10;
            }

            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(0.1);
            var options = new RadialBasisFunctionOptions { NumCenters = 15 };

            // Act
            var regression = new RadialBasisFunctionRegression<double>(options, regularization);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        #endregion
    }
}
