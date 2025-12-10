using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Shared mock model for testing IFullModel implementations.
/// Can be used across all test classes that need a predictable model.
/// </summary>
public class MockFullModel : IFullModel<double, Matrix<double>, Vector<double>>
{
    private Vector<double> _parameters;
    private readonly Func<Matrix<double>, Vector<double>> _predictFunc;

    public MockFullModel(Func<Matrix<double>, Vector<double>> predictFunc, int parameterCount = 5)
    {
        _predictFunc = predictFunc ?? throw new ArgumentNullException(nameof(predictFunc));
        _parameters = new Vector<double>(parameterCount);
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = 0.1 * i;
        }
    }

    public Vector<double> Predict(Matrix<double> input)
    {
        return _predictFunc(input);
    }

    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        // Mock training - no-op
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            Name = "MockFullModel",
            ModelType = Enums.ModelType.None,
            FeatureCount = _parameters.Length,
            Complexity = 1
        };
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new MockFullModel(_predictFunc, parameters.Length);
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

    // IFeatureAware implementation
    public int InputFeatureCount => _parameters.Length;
    public int OutputFeatureCount => 1;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _parameters.Length);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _parameters.Length;

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

    // ICloneable implementation
    public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new MockFullModel(_predictFunc, _parameters.Length);
        copy.SetParameters(_parameters);
        return copy;
    }

    IFullModel<double, Matrix<double>, Vector<double>> ICloneable<IFullModel<double, Matrix<double>, Vector<double>>>.Clone()
    {
        return DeepCopy();
    }

    // IGradientComputable implementation
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
    {
        return new Vector<double>(ParameterCount);
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
}
