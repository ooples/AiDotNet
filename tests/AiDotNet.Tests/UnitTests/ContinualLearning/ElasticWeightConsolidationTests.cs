using AiDotNet.Autodiff;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the Elastic Weight Consolidation strategy.
/// </summary>
public class ElasticWeightConsolidationTests
{
    [Fact]
    public void Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var lambda = 1000.0;

        // Act
        var ewc = new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
            lossFunction,
            lambda);

        // Assert
        Assert.NotNull(ewc);
    }

    [Fact]
    public void Constructor_NullLossFunction_ThrowsArgumentNullException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
                null!,
                1000.0));

        Assert.Contains("lossFunction", exception.Message);
    }

    [Fact]
    public void ComputeRegularizationLoss_WithNoPreviousTask_ReturnsZero()
    {
        // Arrange
        var lossFunction = new MeanSquaredErrorLoss<double>();
        var ewc = new ElasticWeightConsolidation<double, Matrix<double>, Vector<double>>(
            lossFunction,
            1000.0);

        var model = new MockModel(parameterCount: 10);

        // Act
        var regularizationLoss = ewc.ComputeRegularizationLoss(model);

        // Assert
        Assert.Equal(0.0, regularizationLoss, precision: 10);
    }

    /// <summary>
    /// Mock model for testing.
    /// </summary>
    private class MockModel : IFullModel<double, Matrix<double>, Vector<double>>
    {
        private Vector<double> _parameters;

        public MockModel(int parameterCount)
        {
            _parameters = new Vector<double>(parameterCount);
            for (int i = 0; i < parameterCount; i++)
            {
                _parameters[i] = i * 0.1;
            }
        }

        public Vector<double> Predict(Matrix<double> input)
        {
            return new Vector<double>(input.Rows);
        }

        public void Train(Matrix<double> input, Vector<double> expectedOutput)
        {
            // Mock training
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                Name = "MockModel",
                ModelType = ModelType.None,
                FeatureCount = 0,
                Complexity = 1
            };
        }

        public Dictionary<string, double> GetFeatureImportance()
        {
            return new Dictionary<string, double>
            {
                { "feature_0", 0.5 },
                { "feature_1", 0.3 },
                { "feature_2", 0.2 }
            };
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return new[] { 0, 1, 2 };
        }

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            // Mock implementation - no-op
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return featureIndex >= 0 && featureIndex < 3;
        }

        public Vector<double> GetParameters()
        {
            return _parameters;
        }

        public void SetParameters(Vector<double> parameters)
        {
            if (parameters == null)
                throw new ArgumentNullException(nameof(parameters));
            _parameters = new Vector<double>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                _parameters[i] = parameters[i];
            }
        }

        public int ParameterCount => _parameters.Length;

        public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var newModel = new MockModel(_parameters.Length);
            newModel.SetParameters(parameters);
            return newModel;
        }

        public byte[] Serialize()
        {
            var data = new byte[_parameters.Length * sizeof(double)];
            Buffer.BlockCopy(_parameters.ToArray(), 0, data, 0, data.Length);
            return data;
        }

        public void Deserialize(byte[] data)
        {
            var values = new double[data.Length / sizeof(double)];
            Buffer.BlockCopy(data, 0, values, 0, data.Length);
            _parameters = new Vector<double>(values);
        }

        public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
        {
            var copy = new MockModel(_parameters.Length);
            copy.SetParameters(_parameters);
            return copy;
        }

        public IFullModel<double, Matrix<double>, Vector<double>> Clone()
        {
            return DeepCopy();
        }

        public void SaveModel(string filePath)
        {
            File.WriteAllBytes(filePath, Serialize());
        }

        public void LoadModel(string filePath)
        {
            Deserialize(File.ReadAllBytes(filePath));
        }

        // IFullModel<T> - DefaultLossFunction
        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        // ICheckpointableModel - SaveState and LoadState
        public void SaveState(Stream stream)
        {
            var data = Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }

        public void LoadState(Stream stream)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            Deserialize(ms.ToArray());
        }

        // IGradientComputable<double, Matrix<double>, Vector<double>>
        public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
        {
            // Mock implementation - return zero gradients
            return new Vector<double>(_parameters.Length);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            // Mock implementation - simple gradient descent
            for (int i = 0; i < _parameters.Length && i < gradients.Length; i++)
            {
                _parameters[i] -= learningRate * gradients[i];
            }
        }

        // IJitCompilable<double>
        public bool SupportsJitCompilation => false;

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            throw new NotSupportedException("MockModel does not support JIT compilation.");
        }
    }
}
