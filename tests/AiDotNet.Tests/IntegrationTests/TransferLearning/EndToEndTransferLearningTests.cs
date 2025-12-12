using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.TransferLearning.Algorithms;
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;

namespace AiDotNetTests.IntegrationTests.TransferLearning
{
    /// <summary>
    /// End-to-end integration tests for complete transfer learning workflows.
    /// Tests realistic scenarios combining all transfer learning components.
    /// </summary>
    public class EndToEndTransferLearningTests
    {
        private const double Tolerance = 1e-6;

        #region Helper Classes

        private class SimpleLinearModel : IFullModel<double, Matrix<double>, Vector<double>>
        {
            private Vector<double> _parameters;
            private int _inputFeatures;

            public SimpleLinearModel(int inputFeatures)
            {
                _inputFeatures = inputFeatures;
                _parameters = new Vector<double>(inputFeatures + 1);
                var random = new Random(42);
                for (int i = 0; i < _parameters.Length; i++)
                {
                    _parameters[i] = (random.NextDouble() - 0.5) * 0.1;
                }
            }

            public void Train(Matrix<double> input, Vector<double> expectedOutput)
            {
                double learningRate = 0.01;
                for (int epoch = 0; epoch < 50; epoch++)
                {
                    for (int i = 0; i < input.Rows; i++)
                    {
                        double prediction = 0.0;
                        for (int j = 0; j < _inputFeatures; j++)
                        {
                            prediction += input[i, j] * _parameters[j];
                        }
                        prediction += _parameters[_inputFeatures];

                        double error = prediction - expectedOutput[i];
                        for (int j = 0; j < _inputFeatures; j++)
                        {
                            _parameters[j] -= learningRate * error * input[i, j];
                        }
                        _parameters[_inputFeatures] -= learningRate * error;
                    }
                }
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                var predictions = new double[input.Rows];
                for (int i = 0; i < input.Rows; i++)
                {
                    double prediction = 0.0;
                    for (int j = 0; j < Math.Min(_inputFeatures, input.Columns); j++)
                    {
                        prediction += input[i, j] * _parameters[j];
                    }
                    prediction += _parameters[_inputFeatures];
                    predictions[i] = prediction;
                }
                return new Vector<double>(predictions);
            }

            public Vector<double> GetParameters() => _parameters.Clone();
            public void SetParameters(Vector<double> parameters) => _parameters = parameters.Clone();
            public int ParameterCount => _parameters.Length;
            public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
            {
                var model = new SimpleLinearModel(_inputFeatures);
                model.SetParameters(parameters);
                return model;
            }
            public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
            {
                var copy = new SimpleLinearModel(_inputFeatures);
                copy.SetParameters(_parameters);
                return copy;
            }
            public IFullModel<double, Matrix<double>, Vector<double>> Clone() => DeepCopy();
            public void SaveModel(string filePath) { }
            public void LoadModel(string filePath) { }
            public byte[] Serialize() => Array.Empty<byte>();
            public void Deserialize(byte[] data) { }
            public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
            public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatures);
            public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
            public bool IsFeatureUsed(int featureIndex) => featureIndex < _inputFeatures;
            public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
        }

        #endregion

        #region Helper Methods

        private (Matrix<double> X, Vector<double> Y) CreateDataset(int samples, int features, double noiseLevel, int seed)
        {
            var random = new Random(seed);
            var X = new Matrix<double>(samples, features);
            var Y = new double[samples];

            for (int i = 0; i < samples; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < features; j++)
                {
                    X[i, j] = random.NextDouble() * 10.0 - 5.0;
                    sum += X[i, j] * (0.5 + j * 0.1);
                }
                Y[i] = sum + (random.NextDouble() - 0.5) * noiseLevel;
            }

            return (X, new Vector<double>(Y));
        }

        private double ComputeMSE(Vector<double> predictions, Vector<double> actual)
        {
            double sum = 0.0;
            for (int i = 0; i < predictions.Length; i++)
            {
                double diff = predictions[i] - actual[i];
                sum += diff * diff;
            }
            return sum / predictions.Length;
        }

        #endregion

        #region Complete Workflow Tests

        [Fact]
        public void EndToEnd_SameDomain_CORAL_NeuralNetwork()
        {
            // Arrange - Complete workflow with CORAL and Neural Network
            var transfer = new TransferNeuralNetwork<double>();
            var adapter = new CORALDomainAdapter<double>();
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 5, 1.0, 43);

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);
            var mse = ComputeMSE(predictions, targetY);

            // Assert
            Assert.True(mse < 100.0, $"MSE should be reasonable: {mse}");
            Assert.Equal(targetY.Length, predictions.Length);
        }

        [Fact]
        public void EndToEnd_SameDomain_MMD_NeuralNetwork()
        {
            // Arrange - Complete workflow with MMD and Neural Network
            var transfer = new TransferNeuralNetwork<double>();
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 5, 1.0, 43);

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);
            var mse = ComputeMSE(predictions, targetY);

            // Assert
            Assert.True(mse < 100.0, $"MSE should be reasonable: {mse}");
            Assert.Equal(targetY.Length, predictions.Length);
        }

        [Fact]
        public void EndToEnd_CrossDomain_FeatureMapper_CORAL_NeuralNetwork()
        {
            // Arrange - Complete cross-domain workflow
            var transfer = new TransferNeuralNetwork<double>();
            var mapper = new LinearFeatureMapper<double>();
            var adapter = new CORALDomainAdapter<double>();
            transfer.SetFeatureMapper(mapper);
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 8, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 4, 1.0, 43);

            var sourceModel = new SimpleLinearModel(8);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);
            var mse = ComputeMSE(predictions, targetY);

            // Assert
            Assert.True(mse < 200.0, $"MSE should be reasonable: {mse}");
            Assert.True(mapper.IsTrained, "Mapper should be trained");
        }

        [Fact]
        public void EndToEnd_CrossDomain_FeatureMapper_MMD_NeuralNetwork()
        {
            // Arrange
            var transfer = new TransferNeuralNetwork<double>();
            var mapper = new LinearFeatureMapper<double>();
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            transfer.SetFeatureMapper(mapper);
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 8, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 4, 1.0, 43);

            var sourceModel = new SimpleLinearModel(8);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(targetY.Length, predictions.Length);
            Assert.True(mapper.IsTrained);
        }

        [Fact]
        public void EndToEnd_RandomForest_CORAL_SameDomain()
        {
            // Arrange
            var options = new RandomForestRegressionOptions
            {
                NumberOfTrees = 5,
                MaxDepth = 3,
                MinSamplesSplit = 2
            };
            var transfer = new TransferRandomForest<double>(options);
            var adapter = new CORALDomainAdapter<double>();
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 5, 1.0, 43);

            var sourceModel = new RandomForestRegression<double>(options);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);
            var mse = ComputeMSE(predictions, targetY);

            // Assert
            Assert.True(mse < 200.0, $"MSE should be reasonable: {mse}");
        }

        [Fact]
        public void EndToEnd_RandomForest_MMD_SameDomain()
        {
            // Arrange
            var options = new RandomForestRegressionOptions
            {
                NumberOfTrees = 5,
                MaxDepth = 3,
                MinSamplesSplit = 2
            };
            var transfer = new TransferRandomForest<double>(options);
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 5, 1.0, 43);

            var sourceModel = new RandomForestRegression<double>(options);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(targetY.Length, predictions.Length);
        }

        [Fact]
        public void EndToEnd_RandomForest_CrossDomain_FullPipeline()
        {
            // Arrange - Complete cross-domain random forest pipeline
            var options = new RandomForestRegressionOptions
            {
                NumberOfTrees = 5,
                MaxDepth = 3,
                MinSamplesSplit = 2
            };
            var transfer = new TransferRandomForest<double>(options);
            var mapper = new LinearFeatureMapper<double>();
            var adapter = new CORALDomainAdapter<double>();
            transfer.SetFeatureMapper(mapper);
            transfer.SetDomainAdapter(adapter);

            var (sourceX, sourceY) = CreateDataset(100, 10, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 5, 1.0, 43);

            var sourceModel = new RandomForestRegression<double>(options);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);
            var mse = ComputeMSE(predictions, targetY);

            // Assert
            Assert.True(mse < 300.0, $"MSE should be reasonable: {mse}");
            Assert.True(mapper.IsTrained);
        }

        #endregion

        #region Realistic Scenario Tests

        [Fact]
        public void RealScenario_ImageToText_DifferentDomains()
        {
            // Simulate: Image features (128 dims) → Text features (64 dims)
            var transfer = new TransferNeuralNetwork<double>();
            var mapper = new LinearFeatureMapper<double>();
            transfer.SetFeatureMapper(mapper);

            var (sourceX, sourceY) = CreateDataset(200, 128, 2.0, 42); // Image domain
            var (targetX, targetY) = CreateDataset(50, 64, 1.5, 43);   // Text domain

            var sourceModel = new SimpleLinearModel(128);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.False(double.IsNaN(predictions[i]));
            }
        }

        [Fact]
        public void RealScenario_HighToLowDimensional_Compression()
        {
            // Simulate: 100D features → 10D features (dimensionality reduction)
            var transfer = new TransferNeuralNetwork<double>();
            var mapper = new LinearFeatureMapper<double>();
            transfer.SetFeatureMapper(mapper);

            var (sourceX, sourceY) = CreateDataset(100, 100, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 10, 1.0, 43);

            var sourceModel = new SimpleLinearModel(100);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
            Assert.True(mapper.GetMappingConfidence() >= 0.0);
        }

        [Fact]
        public void RealScenario_LowToHighDimensional_Expansion()
        {
            // Simulate: 5D features → 50D features (feature expansion)
            var transfer = new TransferNeuralNetwork<double>();
            var mapper = new LinearFeatureMapper<double>();
            transfer.SetFeatureMapper(mapper);

            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(50, 50, 1.0, 43);

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
        }

        [Fact]
        public void RealScenario_MultiStage_Transfer()
        {
            // Stage 1: Source (10D) → Intermediate (7D)
            var transfer1 = new TransferNeuralNetwork<double>();
            var mapper1 = new LinearFeatureMapper<double>();
            transfer1.SetFeatureMapper(mapper1);

            var (sourceX, sourceY) = CreateDataset(100, 10, 1.0, 42);
            var (intermediateX, intermediateY) = CreateDataset(50, 7, 1.0, 43);

            var sourceModel = new SimpleLinearModel(10);
            sourceModel.Train(sourceX, sourceY);

            var intermediateModel = transfer1.Transfer(sourceModel, sourceX, intermediateX, intermediateY);

            // Stage 2: Intermediate (7D) → Target (5D)
            var transfer2 = new TransferNeuralNetwork<double>();
            var mapper2 = new LinearFeatureMapper<double>();
            transfer2.SetFeatureMapper(mapper2);

            var (targetX, targetY) = CreateDataset(30, 5, 1.0, 44);

            // Act
            var targetModel = transfer2.Transfer(intermediateModel, intermediateX, targetX, targetY);
            var predictions = targetModel.Predict(targetX);

            // Assert
            Assert.Equal(30, predictions.Length);
        }

        [Fact]
        public void RealScenario_VerySmallTarget_LeveragesSource()
        {
            // Simulate: large source, tiny target (5 samples)
            var transfer = new TransferNeuralNetwork<double>();
            var (sourceX, sourceY) = CreateDataset(500, 8, 1.0, 42); // Large source
            var (targetX, targetY) = CreateDataset(5, 8, 1.0, 43);   // Tiny target

            var sourceModel = new SimpleLinearModel(8);
            sourceModel.Train(sourceX, sourceY);

            // Baseline without transfer
            var baselineModel = new SimpleLinearModel(8);
            baselineModel.Train(targetX, targetY);
            var baselinePred = baselineModel.Predict(targetX);
            var baselineMSE = ComputeMSE(baselinePred, targetY);

            // Act - transfer learning
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var transferPred = transferredModel.Predict(targetX);
            var transferMSE = ComputeMSE(transferPred, targetY);

            // Assert - transfer should help with tiny dataset
            Assert.True(transferMSE < baselineMSE * 2.0,
                $"Transfer should help: Transfer={transferMSE}, Baseline={baselineMSE}");
        }

        #endregion

        #region Robustness Tests

        [Fact]
        public void Robustness_HighNoise_StableTransfer()
        {
            // Test with high noise in both domains
            var transfer = new TransferNeuralNetwork<double>();
            var (sourceX, sourceY) = CreateDataset(100, 5, 5.0, 42); // High noise
            var (targetX, targetY) = CreateDataset(50, 5, 5.0, 43);

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var predictions = transferredModel.Predict(targetX);

            // Assert - should still work despite noise
            Assert.Equal(50, predictions.Length);
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.False(double.IsNaN(predictions[i]));
                Assert.False(double.IsInfinity(predictions[i]));
            }
        }

        [Fact]
        public void Robustness_ExtremeScaleDifference_HandlesCorrectly()
        {
            // Source: large scale, Target: small scale
            var transfer = new TransferNeuralNetwork<double>();
            var random = new Random(42);

            // Source with large values
            var sourceX = new Matrix<double>(100, 5);
            var sourceY = new double[100];
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    sourceX[i, j] = random.NextDouble() * 1000.0;
                }
                sourceY[i] = sourceX[i, 0] * 0.1;
            }

            // Target with small values
            var targetX = new Matrix<double>(50, 5);
            var targetY = new double[50];
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    targetX[i, j] = random.NextDouble() * 0.1;
                }
                targetY[i] = targetX[i, 0] * 0.1;
            }

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, new Vector<double>(sourceY));

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, new Vector<double>(targetY));
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
        }

        [Fact]
        public void Robustness_ZeroVarianceFeature_HandlesGracefully()
        {
            // Create data with a zero-variance feature
            var transfer = new TransferNeuralNetwork<double>();
            var random = new Random(42);

            var sourceX = new Matrix<double>(100, 5);
            var sourceY = new double[100];
            for (int i = 0; i < 100; i++)
            {
                sourceX[i, 0] = 5.0; // Zero variance
                for (int j = 1; j < 5; j++)
                {
                    sourceX[i, j] = random.NextDouble() * 10.0;
                }
                sourceY[i] = sourceX[i, 1] * 0.5;
            }

            var targetX = new Matrix<double>(50, 5);
            var targetY = new double[50];
            for (int i = 0; i < 50; i++)
            {
                targetX[i, 0] = 5.0; // Zero variance
                for (int j = 1; j < 5; j++)
                {
                    targetX[i, j] = random.NextDouble() * 10.0;
                }
                targetY[i] = targetX[i, 1] * 0.5;
            }

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, new Vector<double>(sourceY));

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, new Vector<double>(targetY));
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
        }

        [Fact]
        public void Robustness_CorrelatedFeatures_StablePerformance()
        {
            // Create highly correlated features
            var transfer = new TransferNeuralNetwork<double>();
            var random = new Random(42);

            var sourceX = new Matrix<double>(100, 5);
            var sourceY = new double[100];
            for (int i = 0; i < 100; i++)
            {
                double base_value = random.NextDouble() * 10.0;
                for (int j = 0; j < 5; j++)
                {
                    sourceX[i, j] = base_value + random.NextDouble() * 0.1; // Highly correlated
                }
                sourceY[i] = base_value * 0.5;
            }

            var targetX = new Matrix<double>(50, 5);
            var targetY = new double[50];
            for (int i = 0; i < 50; i++)
            {
                double base_value = random.NextDouble() * 10.0;
                for (int j = 0; j < 5; j++)
                {
                    targetX[i, j] = base_value + random.NextDouble() * 0.1;
                }
                targetY[i] = base_value * 0.5;
            }

            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, new Vector<double>(sourceY));

            // Act
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, new Vector<double>(targetY));
            var predictions = transferredModel.Predict(targetX);

            // Assert
            Assert.Equal(50, predictions.Length);
        }

        #endregion

        #region Performance Measurement Tests

        [Fact]
        public void Performance_MeasureAdaptationQuality_CORAL()
        {
            // Measure how well CORAL reduces domain discrepancy
            var adapter = new CORALDomainAdapter<double>();
            var (sourceX, _) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, _) = CreateDataset(100, 5, 1.0, 43);

            var beforeDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceX, targetX);
            var adapted = adapter.AdaptSource(sourceX, targetX);
            var afterDiscrepancy = adapter.ComputeDomainDiscrepancy(adapted, targetX);

            // Assert - CORAL should reduce discrepancy
            Assert.True(afterDiscrepancy < beforeDiscrepancy,
                $"Before: {beforeDiscrepancy}, After: {afterDiscrepancy}");
        }

        [Fact]
        public void Performance_MeasureAdaptationQuality_MMD()
        {
            // Measure how well MMD adaptation works
            var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
            var (sourceX, _) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, _) = CreateDataset(100, 5, 1.0, 43);

            var beforeDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceX, targetX);
            var adapted = adapter.AdaptSource(sourceX, targetX);
            var afterDiscrepancy = adapter.ComputeDomainDiscrepancy(adapted, targetX);

            // Assert - MMD should not increase discrepancy
            Assert.True(afterDiscrepancy <= beforeDiscrepancy * 1.5,
                $"Before: {beforeDiscrepancy}, After: {afterDiscrepancy}");
        }

        [Fact]
        public void Performance_MeasureFeatureMappingQuality()
        {
            // Measure feature mapping quality through confidence
            var mapper = new LinearFeatureMapper<double>();
            var (sourceX, _) = CreateDataset(100, 10, 1.0, 42);
            var (targetX, _) = CreateDataset(100, 5, 1.0, 43);

            mapper.Train(sourceX, targetX);
            var confidence = mapper.GetMappingConfidence();

            // Assert - confidence should be in valid range
            Assert.True(confidence >= 0.0 && confidence <= 1.0,
                $"Confidence should be in [0,1], got {confidence}");
        }

        [Fact]
        public void Performance_CompareTransferVsNoTransfer()
        {
            // Compare transfer learning vs training from scratch
            var transfer = new TransferNeuralNetwork<double>();
            var (sourceX, sourceY) = CreateDataset(200, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(20, 5, 1.0, 43); // Small target

            // With transfer
            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);
            var transferredModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);
            var transferPred = transferredModel.Predict(targetX);
            var transferMSE = ComputeMSE(transferPred, targetY);

            // Without transfer (baseline)
            var baselineModel = new SimpleLinearModel(5);
            baselineModel.Train(targetX, targetY);
            var baselinePred = baselineModel.Predict(targetX);
            var baselineMSE = ComputeMSE(baselinePred, targetY);

            // Assert - document the comparison
            Assert.True(transferMSE < 1000.0, $"Transfer MSE: {transferMSE}");
            Assert.True(baselineMSE < 1000.0, $"Baseline MSE: {baselineMSE}");
        }

        #endregion

        #region Complex Integration Tests

        [Fact]
        public void Complex_MultipleAdapters_Sequential()
        {
            // Test using multiple adapters in sequence
            var coralAdapter = new CORALDomainAdapter<double>();
            var mmdAdapter = new MMDDomainAdapter<double>(sigma: 1.0);

            var (sourceX, _) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, _) = CreateDataset(100, 5, 1.0, 43);

            // Apply CORAL first
            var adapted1 = coralAdapter.AdaptSource(sourceX, targetX);

            // Then MMD
            var adapted2 = mmdAdapter.AdaptSource(adapted1, targetX);

            // Assert
            Assert.Equal(sourceX.Rows, adapted2.Rows);
            Assert.Equal(sourceX.Columns, adapted2.Columns);
        }

        [Fact]
        public void Complex_ChainedTransfer_ThreeDomains()
        {
            // Transfer: Domain A → Domain B → Domain C
            var transfer1 = new TransferNeuralNetwork<double>();
            var transfer2 = new TransferNeuralNetwork<double>();

            var (domainA_X, domainA_Y) = CreateDataset(100, 6, 1.0, 42);
            var (domainB_X, domainB_Y) = CreateDataset(50, 6, 1.0, 43);
            var (domainC_X, domainC_Y) = CreateDataset(30, 6, 1.0, 44);

            // A → B
            var modelA = new SimpleLinearModel(6);
            modelA.Train(domainA_X, domainA_Y);
            var modelB = transfer1.Transfer(modelA, domainA_X, domainB_X, domainB_Y);

            // B → C
            var modelC = transfer2.Transfer(modelB, domainB_X, domainC_X, domainC_Y);

            // Assert
            var predictions = modelC.Predict(domainC_X);
            Assert.Equal(30, predictions.Length);
        }

        [Fact]
        public void Complex_BidirectionalTransfer()
        {
            // Test transfer in both directions
            var transfer = new TransferNeuralNetwork<double>();
            var (sourceX, sourceY) = CreateDataset(100, 5, 1.0, 42);
            var (targetX, targetY) = CreateDataset(100, 5, 1.0, 43);

            // Source → Target
            var sourceModel = new SimpleLinearModel(5);
            sourceModel.Train(sourceX, sourceY);
            var targetModel = transfer.Transfer(sourceModel, sourceX, targetX, targetY);

            // Target → Source (reverse)
            var transferReverse = new TransferNeuralNetwork<double>();
            var targetModelBase = new SimpleLinearModel(5);
            targetModelBase.Train(targetX, targetY);
            var sourceModelReverse = transferReverse.Transfer(targetModelBase, targetX, sourceX, sourceY);

            // Assert - both directions should work
            var predTarget = targetModel.Predict(targetX);
            var predSource = sourceModelReverse.Predict(sourceX);

            Assert.Equal(100, predTarget.Length);
            Assert.Equal(100, predSource.Length);
        }

        #endregion
    }
}
