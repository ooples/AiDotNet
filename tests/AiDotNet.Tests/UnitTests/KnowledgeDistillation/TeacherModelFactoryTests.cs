using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Teachers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the TeacherModelFactory class.
/// </summary>
public class TeacherModelFactoryTests
{
    [Fact]
    public void CreateTeacher_NeuralNetwork_WithModel_ReturnsTeacherModelWrapper()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.NeuralNetwork,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<TeacherModelWrapper<double>>(teacher);
        Assert.Equal(5, teacher.OutputDimension);
    }

    [Fact]
    public void CreateTeacher_NeuralNetwork_WithoutModel_ThrowsArgumentException()
    {
        // Arrange, Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            TeacherModelFactory<double>.CreateTeacher(TeacherModelType.NeuralNetwork));

        Assert.Contains("Model is required", exception.Message);
    }

    [Fact]
    public void CreateTeacher_Ensemble_WithModels_ReturnsEnsembleTeacher()
    {
        // Arrange
        var model1 = new MockFullModel(inputDim: 10, outputDim: 5);
        var model2 = new MockFullModel(inputDim: 10, outputDim: 5);
        var teachers = new[]
        {
            new TeacherModelWrapper<double>(model1.Predict, outputDimension: 5),
            new TeacherModelWrapper<double>(model2.Predict, outputDimension: 5)
        };

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Ensemble,
            ensembleModels: teachers);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<EnsembleTeacherModel<double>>(teacher);
    }

    [Fact]
    public void CreateTeacher_Ensemble_WithoutModels_ThrowsArgumentException()
    {
        // Arrange, Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            TeacherModelFactory<double>.CreateTeacher(TeacherModelType.Ensemble));

        Assert.Contains("Ensemble models are required", exception.Message);
    }

    [Fact]
    public void CreateTeacher_Pretrained_WithModel_ReturnsPretrainedTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Pretrained,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<PretrainedTeacherModel<double>>(teacher);
    }

    [Fact]
    public void CreateTeacher_Transformer_WithModel_ReturnsTransformerTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Transformer,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<TransformerTeacherModel<double>>(teacher);
    }

    [Fact]
    public void CreateTeacher_MultiModal_WithModels_ReturnsMultiModalTeacher()
    {
        // Arrange
        var model1 = new MockFullModel(inputDim: 10, outputDim: 5);
        var model2 = new MockFullModel(inputDim: 10, outputDim: 5);
        var teachers = new[]
        {
            new TeacherModelWrapper<double>(model1.Predict, outputDimension: 5),
            new TeacherModelWrapper<double>(model2.Predict, outputDimension: 5)
        };

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.MultiModal,
            ensembleModels: teachers);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<MultiModalTeacherModel<double>>(teacher);
    }

    [Fact]
    public void CreateTeacher_Adaptive_WithModel_ReturnsAdaptiveTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Adaptive,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<AdaptiveTeacherModel<double>>(teacher);
        Assert.Equal(5, teacher.OutputDimension);
    }

    [Fact]
    public void CreateTeacher_Online_WithModel_ReturnsOnlineTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Online,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<OnlineTeacherModel<double>>(teacher);
    }

    [Fact]
    public void CreateTeacher_Curriculum_WithModel_ReturnsCurriculumTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Curriculum,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<CurriculumTeacherModel<double>>(teacher);
        Assert.Equal(5, teacher.OutputDimension);
    }

    [Fact]
    public void CreateTeacher_Self_WithoutModel_ReturnsSelfTeacher()
    {
        // Arrange & Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Self,
            outputDimension: 10);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<SelfTeacherModel<double>>(teacher);
        Assert.Equal(10, teacher.OutputDimension);
    }

    [Fact]
    public void CreateTeacher_Quantized_WithModel_ReturnsQuantizedTeacher()
    {
        // Arrange
        var mockModel = new MockFullModel(inputDim: 10, outputDim: 5);

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Quantized,
            model: mockModel,
            outputDimension: 5);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<QuantizedTeacherModel<double>>(teacher);
        Assert.Equal(5, teacher.OutputDimension);
    }

    [Fact]
    public void CreateTeacher_Distributed_WithModels_ReturnsDistributedTeacher()
    {
        // Arrange
        var model1 = new MockFullModel(inputDim: 10, outputDim: 5);
        var model2 = new MockFullModel(inputDim: 10, outputDim: 5);
        var teachers = new[]
        {
            new TeacherModelWrapper<double>(model1.Predict, outputDimension: 5),
            new TeacherModelWrapper<double>(model2.Predict, outputDimension: 5)
        };

        // Act
        var teacher = TeacherModelFactory<double>.CreateTeacher(
            TeacherModelType.Distributed,
            ensembleModels: teachers);

        // Assert
        Assert.NotNull(teacher);
        Assert.IsType<DistributedTeacherModel<double>>(teacher);
    }

    // Mock model for testing
    private class MockFullModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private readonly Random _random = RandomHelper.CreateSeededRandom(42);
        private readonly HashSet<int> _activeFeatures = new();

        public MockFullModel(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;

            // Initialize with all features active
            for (int i = 0; i < inputDim; i++)
                _activeFeatures.Add(i);
        }

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(_outputDim);
            double sum = 0;
            for (int i = 0; i < _outputDim; i++)
            {
                output[i] = _random.NextDouble();
                sum += output[i];
            }
            for (int i = 0; i < _outputDim; i++)
                output[i] /= sum;
            return output;
        }

        public void Train(Vector<double> input, Vector<double> target) { }

        public ModelMetadata<double> GetModelMetadata()
        {
            var metadata = new ModelMetadata<double>
            {
                FeatureCount = _inputDim,
                Description = "Mock model for testing"
            };
            metadata.SetProperty("OutputDimension", _outputDim);
            return metadata;
        }

        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        // IFeatureAware implementation
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            _activeFeatures.Clear();
            foreach (var idx in featureIndices)
                _activeFeatures.Add(idx);
        }

        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);

        // IFeatureImportance implementation
        public Dictionary<string, double> GetFeatureImportance()
        {
            var importance = new Dictionary<string, double>();
            for (int i = 0; i < _inputDim; i++)
                importance[$"feature_{i}"] = 1.0 / _inputDim;
            return importance;
        }

        // ICloneable implementation
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy()
        {
            var copy = new MockFullModel(_inputDim, _outputDim);
            copy.SetActiveFeatureIndices(_activeFeatures);
            return copy;
        }

        public IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            return new MockFullModel(_inputDim, _outputDim);
        }

        // IGradientComputable implementation
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
        {
            // Return dummy gradients for testing
            return new Vector<double>(_inputDim * _outputDim);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            // Placeholder - do nothing for mock
        }

        // DefaultLossFunction property
        public ILossFunction<double> DefaultLossFunction =>
            throw new InvalidOperationException("Mock model does not have a default loss function");

        // IParameterizable implementation
        public Vector<double> GetParameters()
        {
            // Return dummy parameters for testing
            return new Vector<double>(_inputDim * _outputDim);
        }

        public void SetParameters(Vector<double> parameters)
        {
            // Placeholder - do nothing for mock
        }

        public int ParameterCount => _inputDim * _outputDim;

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var copy = new MockFullModel(_inputDim, _outputDim);
            copy.SetParameters(parameters);
            return copy;
        }

        // IJitCompilable implementation
        public bool SupportsJitCompilation => true;

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            // Create a computation graph for the mock model
            var inputShape = new int[] { 1, _inputDim };
            var inputTensor = new Tensor<double>(inputShape);
            var inputNode = TensorOperations<double>.Variable(inputTensor, "input");
            inputNodes.Add(inputNode);

            // Simple computation: sum of input elements normalized
            var sumNode = TensorOperations<double>.Sum(inputNode);
            return sumNode;
        }
    }
}
