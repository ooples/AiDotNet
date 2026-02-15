using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Test helper utilities for Continual Learning strategy tests.
/// </summary>
public static class ContinualLearningTestHelper
{
    /// <summary>
    /// Creates a mock neural network for testing continual learning strategies.
    /// </summary>
    public static MockNeuralNetwork<double> CreateMockNetwork(int numParameters = 100)
    {
        return new MockNeuralNetwork<double>(numParameters);
    }

    /// <summary>
    /// Creates test task data for continual learning.
    /// </summary>
    public static (Tensor<double> inputs, Tensor<double> targets) CreateTaskData(
        int numSamples = 50,
        int numFeatures = 10,
        int numClasses = 3)
    {
        var random = RandomHelper.CreateSeededRandom(42);

        var inputData = new Vector<double>(numSamples * numFeatures);
        for (int i = 0; i < numSamples * numFeatures; i++)
        {
            inputData[i] = random.NextDouble();
        }
        var inputs = new Tensor<double>([numSamples, numFeatures], inputData);

        var targetData = new Vector<double>(numSamples * numClasses);
        for (int i = 0; i < numSamples; i++)
        {
            var classIdx = random.Next(numClasses);
            targetData[i * numClasses + classIdx] = 1.0;
        }
        var targets = new Tensor<double>([numSamples, numClasses], targetData);

        return (inputs, targets);
    }

    /// <summary>
    /// Creates test gradients for gradient modification tests.
    /// </summary>
    public static Vector<double> CreateTestGradients(int numParameters = 100)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var gradients = new Vector<double>(numParameters);

        for (int i = 0; i < numParameters; i++)
        {
            gradients[i] = random.NextDouble() * 2 - 1; // Range [-1, 1]
        }

        return gradients;
    }
}

/// <summary>
/// Mock implementation of INeuralNetwork for testing continual learning strategies.
/// </summary>
public class MockNeuralNetwork<T> : INeuralNetwork<T>
{
    private readonly int _numParameters;
    private readonly Vector<T> _parameters;
    private Vector<T> _gradients;
    private readonly List<ILayer<T>> _layers;
    private readonly INumericOperations<T> _ops;
    private bool _isTraining;
    private List<int> _activeFeatureIndices;
    private Tensor<T>? _lastInput;

    public MockNeuralNetwork(int numParameters = 100, int seed = 42)
    {
        _numParameters = numParameters;
        _ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _parameters = new Vector<T>(numParameters);
        _gradients = new Vector<T>(numParameters);
        _layers = [];
        _isTraining = true;
        _activeFeatureIndices = Enumerable.Range(0, 10).ToList();

        var random = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < numParameters; i++)
        {
            _parameters[i] = _ops.FromDouble(random.NextDouble() * 2 - 1);
            _gradients[i] = _ops.Zero;
        }

        // Create mock layers
        _layers.Add(new MockLayer<T>(_ops, numParameters / 2));
        _layers.Add(new MockLayer<T>(_ops, numParameters / 2));
    }

    public IReadOnlyList<ILayer<T>> Layers => _layers;

    // ILayeredModel<T> implementation
    public int LayerCount => _layers.Count;

    public LayerInfo<T> GetLayerInfo(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        }

        int parameterOffset = 0;
        for (int i = 0; i < layerIndex; i++)
        {
            parameterOffset += _layers[i].ParameterCount;
        }

        var layer = _layers[layerIndex];
        return new LayerInfo<T>
        {
            Index = layerIndex,
            Name = layer.LayerName,
            Category = LayerCategory.Dense,
            Layer = layer,
            ParameterOffset = parameterOffset,
            ParameterCount = layer.ParameterCount,
            InputShape = layer.GetInputShape(),
            OutputShape = layer.GetOutputShape(),
            IsTrainable = layer.SupportsTraining,
            EstimatedFlops = 2L * layer.ParameterCount,
            EstimatedActivationMemory = layer.GetOutputShape().Aggregate(1L, (a, b) => a * b) * 4
        };
    }

    public IReadOnlyList<LayerInfo<T>> GetAllLayerInfo()
    {
        var result = new List<LayerInfo<T>>(_layers.Count);
        for (int i = 0; i < _layers.Count; i++)
        {
            result.Add(GetLayerInfo(i));
        }
        return result;
    }

    public bool ValidatePartitionPoint(int afterLayerIndex)
    {
        if (afterLayerIndex < 0 || afterLayerIndex >= _layers.Count - 1)
        {
            return false;
        }
        return true;
    }

    public SubModel<T> ExtractSubModel(int startLayer, int endLayer)
    {
        if (startLayer < 0 || startLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(startLayer));
        }
        if (endLayer < 0 || endLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(endLayer));
        }
        if (startLayer > endLayer)
        {
            throw new ArgumentOutOfRangeException(nameof(startLayer));
        }

        var subLayers = new List<ILayer<T>>();
        var subInfos = new List<LayerInfo<T>>();
        int localOffset = 0;

        for (int i = startLayer; i <= endLayer; i++)
        {
            var layer = _layers[i];
            subLayers.Add(layer);
            subInfos.Add(new LayerInfo<T>
            {
                Index = i - startLayer,
                Name = layer.LayerName,
                Category = LayerCategory.Dense,
                Layer = layer,
                ParameterOffset = localOffset,
                ParameterCount = layer.ParameterCount,
                InputShape = layer.GetInputShape(),
                OutputShape = layer.GetOutputShape(),
                IsTrainable = layer.SupportsTraining,
                EstimatedFlops = 2L * layer.ParameterCount,
                EstimatedActivationMemory = 40
            });
            localOffset += layer.ParameterCount;
        }

        return new SubModel<T>(subLayers, subInfos, startLayer, endLayer);
    }

    // INeuralNetwork<T> specific members
    public void UpdateParameters(Vector<T> parameters)
    {
        for (int i = 0; i < Math.Min(parameters.Length, _parameters.Length); i++)
        {
            _parameters[i] = parameters[i];
        }
    }

    public void SetTrainingMode(bool isTrainingMode)
    {
        _isTraining = isTrainingMode;
        foreach (var layer in _layers)
        {
            layer.SetTrainingMode(isTrainingMode);
        }
    }

    public Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        _lastInput = input;
        return Predict(input);
    }

    public Tensor<T> Backpropagate(Tensor<T> outputGradients)
    {
        // Simple mock backprop - set random gradients
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < _gradients.Length; i++)
        {
            _gradients[i] = _ops.FromDouble(random.NextDouble() * 0.1 - 0.05);
        }

        // Return input-shaped gradients
        if (_lastInput != null)
        {
            var inputGradData = new Vector<T>(_lastInput.Length);
            for (int i = 0; i < inputGradData.Length; i++)
            {
                inputGradData[i] = _ops.Zero;
            }
            return new Tensor<T>(_lastInput.Shape, inputGradData);
        }

        return outputGradients;
    }

    public Vector<T> GetParameterGradients()
    {
        return _gradients;
    }

    // IFullModel<T> - DefaultLossFunction
    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    // IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>>
    public Tensor<T> Predict(Tensor<T> input)
    {
        // Simple mock prediction
        var numSamples = input.Shape[0];
        var outputSize = 3; // Default 3 classes
        var data = new Vector<T>(numSamples * outputSize);

        for (int i = 0; i < numSamples * outputSize; i++)
        {
            data[i] = _ops.FromDouble(0.33);
        }

        return new Tensor<T>([numSamples, outputSize], data);
    }

    public void Train(Tensor<T> inputs, Tensor<T> targets)
    {
        // Mock training - no-op
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>();
    }

    // IModelSerializer
    public byte[] Serialize()
    {
        return [];
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

    // ICheckpointableModel
    public void SaveState(Stream stream)
    {
        // No-op for mock
    }

    public void LoadState(Stream stream)
    {
        // No-op for mock
    }

    // IParameterizable<T, Tensor<T>, Tensor<T>>
    public Vector<T> GetParameters()
    {
        return _parameters;
    }

    public void SetParameters(Vector<T> parameters)
    {
        for (int i = 0; i < Math.Min(parameters.Length, _parameters.Length); i++)
        {
            _parameters[i] = parameters[i];
        }
    }

    public int ParameterCount => _numParameters;

    public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var newNetwork = new MockNeuralNetwork<T>(_numParameters);
        newNetwork.SetParameters(parameters);
        return newNetwork;
    }

    // IFeatureAware
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return _activeFeatureIndices;
    }

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = featureIndices.ToList();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return _activeFeatureIndices.Contains(featureIndex);
    }

    // IFeatureImportance<T>
    public Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < 10; i++)
        {
            importance[$"feature_{i}"] = _ops.FromDouble(0.1);
        }
        return importance;
    }

    // ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new MockNeuralNetwork<T>(_numParameters);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return DeepCopy();
    }

    // IGradientComputable<T, Tensor<T>, Tensor<T>>
    public Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Simple mock - return random gradients
        var random = RandomHelper.CreateSeededRandom(42);
        var grads = new Vector<T>(_numParameters);
        for (int i = 0; i < _numParameters; i++)
        {
            grads[i] = _ops.FromDouble(random.NextDouble() * 0.1 - 0.05);
        }
        return grads;
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            var update = _ops.Multiply(gradients[i], learningRate);
            _parameters[i] = _ops.Subtract(_parameters[i], update);
        }
    }

    // IJitCompilable<T>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("Mock network does not support JIT compilation");
    }

    public bool SupportsJitCompilation => false;
}

/// <summary>
/// Mock implementation of ILayer for testing.
/// </summary>
public class MockLayer<T> : ILayer<T>
{
    private readonly INumericOperations<T> _ops;
    private readonly int _parameterCount;
    private readonly Vector<T> _parameters;
    private Vector<T> _gradients;
    private bool _isTraining;

    public MockLayer(INumericOperations<T> ops, int parameterCount)
    {
        _ops = ops;
        _parameterCount = parameterCount;
        _parameters = new Vector<T>(parameterCount);
        _gradients = new Vector<T>(parameterCount);
        _isTraining = true;

        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = ops.FromDouble(random.NextDouble() * 0.1);
            _gradients[i] = ops.FromDouble(0);
        }
    }

    // ILayer<T> members
    public int[] GetInputShape() => [10];

    public int[] GetOutputShape() => [10];

    public Tensor<T>? GetWeights() => null;

    public Tensor<T>? GetBiases() => null;

    public Tensor<T> Forward(Tensor<T> input)
    {
        return input; // Identity for testing
    }

    public Tensor<T> ForwardWithPrecisionCheck(Tensor<T> input)
    {
        // For testing, just delegate to Forward
        return Forward(input);
    }

    public string LayerName => "MockLayer";

    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Set some gradients for testing
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < _gradients.Length; i++)
        {
            _gradients[i] = _ops.FromDouble(random.NextDouble() * 0.01);
        }
        return outputGradient;
    }

    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _parameters.Length; i++)
        {
            var update = _ops.Multiply(_gradients[i], learningRate);
            _parameters[i] = _ops.Subtract(_parameters[i], update);
        }
    }

    public void UpdateParameters(Vector<T> parameters)
    {
        for (int i = 0; i < Math.Min(parameters.Length, _parameters.Length); i++)
        {
            _parameters[i] = parameters[i];
        }
    }

    public int ParameterCount => _parameterCount;

    public void Serialize(BinaryWriter writer)
    {
        // No-op for mock
    }

    public void Deserialize(BinaryReader reader)
    {
        // No-op for mock
    }

    public IEnumerable<ActivationFunction> GetActivationTypes()
    {
        return [ActivationFunction.ReLU];
    }

    public Vector<T> GetParameters()
    {
        return _parameters;
    }

    public bool SupportsTraining => true;

    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    public Vector<T> GetParameterGradients()
    {
        return _gradients;
    }

    public void ClearGradients()
    {
        for (int i = 0; i < _gradients.Length; i++)
        {
            _gradients[i] = _ops.Zero;
        }
    }

    public void SetParameters(Vector<T> parameters)
    {
        for (int i = 0; i < Math.Min(parameters.Length, _parameters.Length); i++)
        {
            _parameters[i] = parameters[i];
        }
    }

    public void ResetState()
    {
        // No state to reset for mock
    }

    // IJitCompilable<T>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("Mock layer does not support JIT compilation");
    }

    public bool SupportsJitCompilation => false;

    // IGpuExecutable<T>
    public bool CanExecuteOnGpu => false;

    public IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        throw new NotSupportedException("Mock layer does not support GPU execution");
    }

    // IDiagnosticsProvider
    public Dictionary<string, string> GetDiagnostics()
    {
        return new Dictionary<string, string>
        {
            ["layer.type"] = "MockLayer",
            ["parameter.count"] = _parameterCount.ToString()
        };
    }

    // IWeightLoadable<T>
    public IEnumerable<string> GetParameterNames()
    {
        return Enumerable.Empty<string>();
    }

    public bool TryGetParameter(string name, out Tensor<T>? tensor)
    {
        tensor = null;
        return false;
    }

    public bool SetParameter(string name, Tensor<T> value)
    {
        return false;
    }

    public int[]? GetParameterShape(string name)
    {
        return null;
    }

    public int NamedParameterCount => 0;

    public WeightLoadValidation ValidateWeights(IEnumerable<string> weightNames, Func<string, string?>? mapping = null)
    {
        return new WeightLoadValidation();
    }

    public WeightLoadResult LoadWeights(Dictionary<string, Tensor<T>> weights, Func<string, string?>? mapping = null, bool strict = false)
    {
        return new WeightLoadResult { Success = true };
    }

    // GPU Training interface members (mock implementations)
    public bool SupportsGpuTraining => false;

    public void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        // No-op for mock - GPU training not supported
    }

    public void UploadWeightsToGpu()
    {
        // No-op for mock - GPU training not supported
    }

    public void DownloadWeightsFromGpu()
    {
        // No-op for mock - GPU training not supported
    }

    public void ZeroGradientsGpu()
    {
        // No-op for mock - GPU training not supported
    }
}
