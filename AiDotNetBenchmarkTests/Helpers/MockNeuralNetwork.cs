using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Benchmarks.Helpers;

/// <summary>
/// Mock neural network for benchmark testing that implements IFullModel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class MockNeuralNetwork<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private Vector<T> _parameters;
    private readonly int _inputFeatureCount;
    private readonly int _outputFeatureCount;

    public MockNeuralNetwork(int inputFeatureCount, int outputFeatureCount)
    {
        _inputFeatureCount = inputFeatureCount;
        _outputFeatureCount = outputFeatureCount;

        // Initialize parameters with zeros
        int paramCount = inputFeatureCount * outputFeatureCount + outputFeatureCount;
        _parameters = new Vector<T>(paramCount);
    }

    public Vector<T> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = new MockNeuralNetwork<T, TInput, TOutput>(_inputFeatureCount, _outputFeatureCount);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public void Train(TInput input, TOutput expectedOutput)
    {
        // Mock training - no-op for benchmarks
    }

    public TOutput Predict(TInput input)
    {
        // Return a default output - this is just for benchmarking, not actual computation
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)new Vector<T>(_outputFeatureCount);
        }

        return default!;
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "MockNeuralNetwork",
            FeatureCount = _inputFeatureCount,
            Complexity = _parameters.Length
        };
    }

    // IModelSerializer implementation
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }

    // ICheckpointableModel implementation
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }

    // IFeatureAware implementation
    public int InputFeatureCount => _inputFeatureCount;
    public int OutputFeatureCount => _outputFeatureCount;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputFeatureCount;

    // IFeatureImportance implementation
    public Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        return importance;
    }

    // ICloneable implementation
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var copy = new MockNeuralNetwork<T, TInput, TOutput>(_inputFeatureCount, _outputFeatureCount);
        copy.SetParameters(_parameters);
        return copy;
    }

    IFullModel<T, TInput, TOutput> ICloneable<IFullModel<T, TInput, TOutput>>.Clone()
    {
        return DeepCopy();
    }

    // IGradientComputable implementation
    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        // Return non-zero gradients for benchmarking
        var gradients = new Vector<T>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            gradients[i] = NumOps.FromDouble(0.1 * (i + 1));
        }
        return gradients;
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Mock implementation
    }

    // IJitCompilable implementation
    public bool SupportsJitCompilation => false;

    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation not supported in mock model");
    }
}
