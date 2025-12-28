using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.TransferLearning.Algorithms;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TransferLearning;

/// <summary>
/// Integration tests for Transfer Learning Algorithm classes (TransferNeuralNetwork and TransferRandomForest).
/// These tests verify the transfer learning functionality works correctly.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class TransferLearningAlgorithmsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region TransferNeuralNetwork Tests

    [Fact]
    public void TransferNeuralNetwork_Transfer_SameDomain_ReturnsTrainedModel()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0 },
            { 8.0, 9.0, 10.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
        Assert.True(sourceModel.TrainCallCount > 0, "Model should have been trained at least once during transfer");
    }

    [Fact]
    public void TransferNeuralNetwork_Transfer_CrossDomain_WithFeatureMapper_ReturnsTrainedModel()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var featureMapper = new LinearFeatureMapper<double>();

        // Source has 3 features, target has 5 features (different dimensions)
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 3.0, 4.0, 5.0, 6.0 },
            { 3.0, 4.0, 5.0, 6.0, 7.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        transfer.SetFeatureMapper(featureMapper);

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
        Assert.True(featureMapper.IsTrained, "Feature mapper should be trained during cross-domain transfer");
    }

    [Fact]
    public void TransferNeuralNetwork_Transfer_CrossDomain_WithoutFeatureMapper_ThrowsException()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var sourceModel = new MockFullModel<double>(featureCount: 3);

        // Different dimensions without feature mapper
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 3.0, 4.0, 5.0, 6.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            transfer.Transfer(sourceModel, sourceData, targetData, targetLabels));
    }

    [Fact]
    public void TransferNeuralNetwork_SetFeatureMapper_StoresMapper()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var mapper = new LinearFeatureMapper<double>();

        // Act
        transfer.SetFeatureMapper(mapper);

        // Assert - Can use the mapper in transfer (indirectly tests it was stored)
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 2.0, 3.0, 4.0, 5.0 },
            { 3.0, 4.0, 5.0, 6.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);
        Assert.NotNull(result);
    }

    [Fact]
    public void TransferNeuralNetwork_SetDomainAdapter_DoesNotThrow()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var adapter = new AiDotNet.TransferLearning.DomainAdaptation.CORALDomainAdapter<double>();

        // Act & Assert - Verify method completes without exception
        var exception = Record.Exception(() => transfer.SetDomainAdapter(adapter));
        Assert.Null(exception);
    }

    #endregion

    #region TransferRandomForest Tests

    [Fact]
    public void TransferRandomForest_Transfer_SameDomain_ReturnsTrainedModel()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            MinSamplesSplit = 2
        };
        var transfer = new TransferRandomForest<double>(options);
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0 },
            { 8.0, 9.0, 10.0 },
            { 11.0, 12.0, 13.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void TransferRandomForest_Transfer_CrossDomain_WithFeatureMapper_ReturnsWrappedModel()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 3,
            MaxDepth = 2,
            MinSamplesSplit = 2
        };
        var transfer = new TransferRandomForest<double>(options);
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var featureMapper = new LinearFeatureMapper<double>();

        // Source has 3 features, target has 5 features
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 3.0, 4.0, 5.0, 6.0 },
            { 3.0, 4.0, 5.0, 6.0, 7.0 },
            { 4.0, 5.0, 6.0, 7.0, 8.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });

        transfer.SetFeatureMapper(featureMapper);

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
        Assert.True(featureMapper.IsTrained, "Feature mapper should be trained");
    }

    [Fact]
    public void TransferRandomForest_Transfer_CrossDomain_WithoutFeatureMapper_ThrowsException()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 3,
            MaxDepth = 2
        };
        var transfer = new TransferRandomForest<double>(options);
        var sourceModel = new MockFullModel<double>(featureCount: 3);

        // Different dimensions without feature mapper
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 3.0, 4.0, 5.0, 6.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            transfer.Transfer(sourceModel, sourceData, targetData, targetLabels));
    }

    [Fact]
    public void TransferRandomForest_WithRegularization_CreatesModel()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 3,
            MaxDepth = 2,
            MinSamplesSplit = 2
        };
        var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(
            new RegularizationOptions { Strength = 0.1 });
        var transfer = new TransferRandomForest<double>(options, regularization);
        var sourceModel = new MockFullModel<double>(featureCount: 3);

        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0 },
            { 8.0, 9.0, 10.0 },
            { 11.0, 12.0, 13.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TransferNeuralNetwork_Transfer_MinimalData_HandlesGracefully()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var sourceModel = new MockFullModel<double>(featureCount: 2);
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void TransferRandomForest_Transfer_MinimalData_HandlesGracefully()
    {
        // Arrange - Random Forest needs at least 2 samples for tree building (MinSamplesSplit=2)
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 1,
            MaxDepth = 1,
            MinSamplesSplit = 2
        };
        var transfer = new TransferRandomForest<double>(options);
        var sourceModel = new MockFullModel<double>(featureCount: 3);
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void TransferNeuralNetwork_Transfer_LargeDimensionMismatch_HandlesWithMapper()
    {
        // Arrange
        var transfer = new TransferNeuralNetwork<double>();
        var sourceModel = new MockFullModel<double>(featureCount: 2);
        var featureMapper = new LinearFeatureMapper<double>();

        // Source: 2 features, Target: 10 features (5x expansion)
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var targetData = Matrix<double>.CreateRandom(3, 10, 0, 10);
        var targetLabels = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        transfer.SetFeatureMapper(featureMapper);

        // Act
        var result = transfer.Transfer(sourceModel, sourceData, targetData, targetLabels);

        // Assert
        Assert.NotNull(result);
    }

    #endregion

    #region Mock Classes

    /// <summary>
    /// Mock implementation of IFullModel for testing transfer learning.
    /// </summary>
    private class MockFullModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
    {
        private readonly int _featureCount;
        private Vector<T> _parameters;
        private HashSet<int> _activeFeatures;
        private readonly INumericOperations<T> _numOps;

        public int TrainCallCount { get; private set; }

        public MockFullModel(int featureCount)
        {
            _featureCount = featureCount;
            _numOps = MathHelper.GetNumericOperations<T>();
            _parameters = new Vector<T>(featureCount);
            _activeFeatures = new HashSet<int>(Enumerable.Range(0, featureCount));
        }

        public void Train(Matrix<T> input, Vector<T> expectedOutput)
        {
            TrainCallCount++;
            // Simple mock training - just store feature count
            _parameters = new Vector<T>(input.Columns);
        }

        public Vector<T> Predict(Matrix<T> input)
        {
            // Return simple predictions
            var result = new Vector<T>(input.Rows);
            for (int i = 0; i < input.Rows; i++)
            {
                result[i] = _numOps.FromDouble(i + 1.0);
            }
            return result;
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>();
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _activeFeatures;
        }

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            _activeFeatures = new HashSet<int>(featureIndices);
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return _activeFeatures.Contains(featureIndex);
        }

        public Vector<T> GetParameters()
        {
            return _parameters ?? new Vector<T>(_featureCount);
        }

        public void SetParameters(Vector<T> parameters)
        {
            _parameters = parameters;
        }

        public int ParameterCount => _featureCount;

        public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
        {
            var copy = new MockFullModel<T>(_featureCount);
            copy._parameters = parameters;
            return copy;
        }

        public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
        {
            var copy = new MockFullModel<T>(_featureCount);
            if (_parameters != null)
            {
                copy._parameters = new Vector<T>(_parameters.Length);
                for (int i = 0; i < _parameters.Length; i++)
                {
                    copy._parameters[i] = _parameters[i];
                }
            }
            copy._activeFeatures = new HashSet<int>(_activeFeatures);
            return copy;
        }

        /// <summary>
        /// Clone delegates to DeepCopy because this mock has no reference-type fields
        /// that would require distinguishing between shallow and deep copies.
        /// Both IFullModel methods are implemented identically for simplicity.
        /// </summary>
        public IFullModel<T, Matrix<T>, Vector<T>> Clone()
        {
            return DeepCopy();
        }

        public byte[] Serialize()
        {
            return Array.Empty<byte>();
        }

        public void Deserialize(byte[] data)
        {
            // No-op for mock
        }

        public void SaveModel(string filePath)
        {
            // No-op for mock
        }

        public void LoadModel(string filePath)
        {
            // No-op for mock
        }

        public void SaveState(Stream stream)
        {
            // No-op for mock
        }

        public void LoadState(Stream stream)
        {
            // No-op for mock
        }

        public Dictionary<string, T> GetFeatureImportance()
        {
            var importance = new Dictionary<string, T>();
            for (int i = 0; i < _featureCount; i++)
            {
                importance[$"Feature_{i}"] = _numOps.FromDouble(1.0 / _featureCount);
            }
            return importance;
        }

        public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

        public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
        {
            return new Vector<T>(_featureCount);
        }

        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            // No-op for mock
        }

        public bool SupportsJitCompilation => false;

        public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            throw new NotSupportedException("Mock model does not support JIT compilation");
        }
    }

    #endregion
}
