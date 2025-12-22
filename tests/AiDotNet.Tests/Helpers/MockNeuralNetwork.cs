using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Mock neural network for testing continual learning strategies and other
/// components that require an INeuralNetwork implementation.
/// </summary>
public class MockNeuralNetwork : INeuralNetwork<double>
{
    private Vector<double> _parameters;
    private Vector<double> _parameterGradients;
    private bool _isTrainingMode;
    private Tensor<double>? _lastInput;
    private Tensor<double>? _lastOutput;
    private readonly int _outputSize;

    public int ForwardCallCount { get; private set; }
    public int BackpropagateCallCount { get; private set; }
    public int TrainCallCount { get; private set; }
    public int PredictCallCount { get; private set; }
    public bool IsTrainingMode => _isTrainingMode;

    /// <summary>
    /// Creates a mock neural network with the specified parameter and output sizes.
    /// </summary>
    /// <param name="parameterCount">Number of parameters in the network.</param>
    /// <param name="outputSize">Size of the output tensor's last dimension (num classes).</param>
    public MockNeuralNetwork(int parameterCount = 10, int outputSize = 3)
    {
        _parameters = new Vector<double>(parameterCount);
        _parameterGradients = new Vector<double>(parameterCount);
        _outputSize = outputSize;
        _isTrainingMode = false;

        // Initialize with small values
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = 0.1 * i;
            _parameterGradients[i] = 0.0;
        }
    }

    // INeuralNetwork specific methods

    public void UpdateParameters(Vector<double> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        }
        _parameters = parameters.Clone();
    }

    public void SetTrainingMode(bool isTrainingMode)
    {
        _isTrainingMode = isTrainingMode;
    }

    public Tensor<double> ForwardWithMemory(Tensor<double> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }
        ForwardCallCount++;
        _lastInput = input.Clone();

        // Generate output with shape [batchSize, outputSize]
        int batchSize = input.Shape[0];
        var outputShape = new int[] { batchSize, _outputSize };
        var output = new Tensor<double>(outputShape);

        // Fill with mock predictions (simple function of parameters)
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _outputSize; c++)
            {
                // Generate logits based on parameters
                double logit = _parameters[c % _parameters.Length] + 0.1 * c;
                output[b * _outputSize + c] = logit;
            }
        }

        _lastOutput = output.Clone();
        return output;
    }

    public Tensor<double> Backpropagate(Tensor<double> outputGradients)
    {
        if (outputGradients is null)
        {
            throw new ArgumentNullException(nameof(outputGradients));
        }
        BackpropagateCallCount++;

        // Compute mock parameter gradients based on output gradients
        double gradSum = 0;
        for (int i = 0; i < outputGradients.Length; i++)
        {
            gradSum += outputGradients[i];
        }

        // Update parameter gradients
        for (int i = 0; i < _parameterGradients.Length; i++)
        {
            _parameterGradients[i] = gradSum / _parameterGradients.Length + 0.01 * i;
        }

        // Return input gradients (same shape as last input)
        if (_lastInput != null)
        {
            var inputGradients = new Tensor<double>(_lastInput.Shape);
            for (int i = 0; i < inputGradients.Length; i++)
            {
                inputGradients[i] = gradSum / inputGradients.Length;
            }
            return inputGradients;
        }

        return new Tensor<double>(new int[] { 1 });
    }

    public Vector<double> GetParameterGradients()
    {
        return _parameterGradients.Clone();
    }

    // IFullModel<T, Tensor<T>, Tensor<T>> implementation

    public Tensor<double> Predict(Tensor<double> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }
        PredictCallCount++;

        // Generate output with shape [batchSize, outputSize]
        int batchSize = input.Shape[0];
        var outputShape = new int[] { batchSize, _outputSize };
        var output = new Tensor<double>(outputShape);

        // Fill with mock predictions
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _outputSize; c++)
            {
                double logit = _parameters[c % _parameters.Length] + 0.1 * c;
                output[b * _outputSize + c] = logit;
            }
        }

        return output;
    }

    public void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
        TrainCallCount++;
        // Simple mock training - slightly adjust parameters
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] += 0.01;
        }
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            Name = "MockNeuralNetwork",
            ModelType = Enums.ModelType.None,
            FeatureCount = _parameters.Length,
            Complexity = 1
        };
    }

    // IParameterizable implementation
    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        }
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new MockNeuralNetwork(_parameters.Length, _outputSize);
        newModel.SetParameters(parameters);
        return newModel;
    }

    // IModelSerializer implementation
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }

    // ICheckpointableModel implementation
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }

    // ICloneable implementation
    public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
    {
        var copy = new MockNeuralNetwork(_parameters.Length, _outputSize);
        copy.SetParameters(_parameters);
        return copy;
    }

    IFullModel<double, Tensor<double>, Tensor<double>> ICloneable<IFullModel<double, Tensor<double>, Tensor<double>>>.Clone()
    {
        return DeepCopy();
    }

    // IFeatureAware implementation
    public int InputFeatureCount => 10;
    public int OutputFeatureCount => _outputSize;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, InputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < InputFeatureCount;

    // IFeatureImportance implementation
    public Dictionary<string, double> GetFeatureImportance()
    {
        var importance = new Dictionary<string, double>();
        for (int i = 0; i < _parameters.Length; i++)
        {
            importance[$"feature_{i}"] = 1.0 / _parameters.Length;
        }
        return importance;
    }

    // IGradientComputable implementation
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
    {
        // Return mock gradients
        var gradients = new Vector<double>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            gradients[i] = 0.1 * (i + 1);
        }
        return gradients;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    // IJitCompilable implementation
    public bool SupportsJitCompilation => false;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation not supported in mock model");
    }

    /// <summary>
    /// Sets the parameter gradients directly (useful for testing gradient modification).
    /// </summary>
    public void SetParameterGradients(Vector<double> gradients)
    {
        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }
        if (gradients.Length != _parameterGradients.Length)
        {
            throw new ArgumentException($"Gradient count mismatch: expected {_parameterGradients.Length}, got {gradients.Length}");
        }
        _parameterGradients = gradients.Clone();
    }
}
