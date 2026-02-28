using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNetTestConsole.Examples.MetaLearning;

/// <summary>
/// A simple linear model for meta-learning demonstrations. Maps Matrix input to Vector output
/// via a learned weight matrix + bias, with analytic gradient computation.
/// </summary>
internal class SimpleMetaModel : IFullModel<double, Matrix<double>, Vector<double>>, ICloneable
{
    private Vector<double> _parameters;
    private readonly int _inputFeatures;

    public SimpleMetaModel(int inputFeatures)
    {
        _inputFeatures = inputFeatures;
        // Initialize with small random-looking values
        _parameters = new Vector<double>(inputFeatures + 1);
        var rng = new Random(42);
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] = (rng.NextDouble() - 0.5) * 0.2;
        }
    }

    public Vector<double> Predict(Matrix<double> input)
    {
        var output = new Vector<double>(input.Rows);
        for (int r = 0; r < input.Rows; r++)
        {
            double sum = _parameters[_inputFeatures]; // bias
            for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
            {
                sum += input[r, c] * _parameters[c];
            }
            output[r] = sum;
        }
        return output;
    }

    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, 0.01);
    }

    public ModelMetadata<double> GetModelMetadata() => new()
    {
        Name = "SimpleMetaModel",
        FeatureCount = _inputFeatures,
        Complexity = _parameters.Length
    };

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters.Length != _parameters.Length)
            throw new ArgumentException($"Expected {_parameters.Length} parameters, got {parameters.Length}");
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var model = new SimpleMetaModel(_inputFeatures);
        model.SetParameters(parameters);
        return model;
    }

    public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new SimpleMetaModel(_inputFeatures);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<double, Matrix<double>, Vector<double>> Clone() => DeepCopy();
    object ICloneable.Clone() => DeepCopy();

    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
    {
        var gradients = new Vector<double>(_parameters.Length);
        int count = Math.Min(input.Rows, target.Length);
        if (count == 0) return gradients;

        var predictions = Predict(input);
        double scale = 2.0 / count;
        for (int r = 0; r < count; r++)
        {
            double error = predictions[r] - target[r];
            for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
            {
                gradients[c] += scale * error * input[r, c];
            }
            gradients[_inputFeatures] += scale * error; // bias gradient
        }
        return gradients;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        int length = Math.Min(gradients.Length, _parameters.Length);
        for (int i = 0; i < length; i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    public byte[] Serialize() => Encoding.UTF8.GetBytes(
        string.Join(",", _parameters.Select(p => p.ToString("R", CultureInfo.InvariantCulture))));

    public void Deserialize(byte[] data)
    {
        var parts = Encoding.UTF8.GetString(data).Split(',');
        for (int i = 0; i < Math.Min(parts.Length, _parameters.Length); i++)
            _parameters[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
    }

    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());
    public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

    public void SaveState(Stream stream)
    {
        using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
        writer.Write(string.Join(",", _parameters.Select(p => p.ToString("R", CultureInfo.InvariantCulture))));
        writer.Flush();
    }

    public void LoadState(Stream stream)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen: true);
        var parts = reader.ReadToEnd().Split(',');
        for (int i = 0; i < Math.Min(parts.Length, _parameters.Length); i++)
            _parameters[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
    }

    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatures);
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputFeatures;

    public Dictionary<string, double> GetFeatureImportance()
    {
        var importance = new Dictionary<string, double>();
        for (int i = 0; i < _inputFeatures; i++)
            importance[$"feature_{i}"] = Math.Abs(_parameters[i]);
        return importance;
    }

    public bool SupportsJitCompilation => false;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        => throw new NotSupportedException();
}
