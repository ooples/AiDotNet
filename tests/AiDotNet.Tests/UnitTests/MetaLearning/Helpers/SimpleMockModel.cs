using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Simple mock model for testing that tracks parameter updates.
/// </summary>
public class SimpleMockModel : IFullModel<double, Tensor<double>, Tensor<double>>
{
    private Vector<double> _parameters;
    public int TrainCallCount { get; private set; }
    public int PredictCallCount { get; private set; }

    public SimpleMockModel(int parameterCount)
    {
        _parameters = new Vector<double>(parameterCount);
        // Initialize with small random values
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = 0.1 * i;
        }
        TrainCallCount = 0;
        PredictCallCount = 0;
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        }
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new SimpleMockModel(_parameters.Length);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
        TrainCallCount++;
        // Simple update: add a small value to each parameter to simulate training
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] += 0.01;
        }
    }

    public Tensor<double> Predict(Tensor<double> input)
    {
        PredictCallCount++;
        // Return a tensor of the same shape as input filled with zeros
        return new Tensor<double>(input.Shape);
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>();
    }

    // IModelSerializer implementation
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }

    public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
    {
        var copy = new SimpleMockModel(_parameters.Length);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<double, Tensor<double>, Tensor<double>> Clone()
    {
        return DeepCopy();
    }

    // IFeatureAware implementation
    public int InputFeatureCount => 10;
    public int OutputFeatureCount => 1;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, InputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < InputFeatureCount;

    // IFeatureImportance implementation
    public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
}
