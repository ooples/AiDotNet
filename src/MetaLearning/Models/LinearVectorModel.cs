using System.Globalization;
using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// A simple linear model mapping Matrix input to Vector output, useful for meta-learning examples and testing.
/// </summary>
/// <remarks>
/// <para>Computes y = X * w + b where w is a weight vector and b is a bias scalar.
/// Provides gradient computation via MSE loss for use with gradient-based meta-learners.</para>
/// </remarks>
public class LinearVectorModel : IFullModel<double, Matrix<double>, Vector<double>>, ICloneable
{
    private Vector<double> _parameters;
    private readonly int _inputDim;
    private readonly double _learningRate;

    /// <summary>
    /// Creates a new linear model with the given input dimension.
    /// </summary>
    /// <param name="inputDim">Number of input features.</param>
    /// <param name="learningRate">Learning rate for gradient descent in <see cref="Train"/>. Default is 0.01.</param>
    public LinearVectorModel(int inputDim, double learningRate = 0.01)
    {
        Guard.Positive(inputDim);
        if (learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        _inputDim = inputDim;
        _learningRate = learningRate;
        _parameters = new Vector<double>(inputDim + 1);
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] = 0.01 * (i + 1);
        }
    }

    /// <inheritdoc/>
    public Vector<double> Predict(Matrix<double> input)
    {
        Guard.NotNull(input);
        var output = new Vector<double>(input.Rows);
        if (input.Columns < _inputDim)
            throw new ArgumentException(
                $"Input has {input.Columns} columns but model expects at least {_inputDim}.", nameof(input));

        for (int r = 0; r < input.Rows; r++)
        {
            double sum = _parameters[_inputDim];
            for (int c = 0; c < _inputDim; c++)
            {
                sum += input[r, c] * _parameters[c];
            }

            output[r] = sum;
        }

        return output;
    }

    /// <inheritdoc/>
    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, _learningRate);
    }

    /// <inheritdoc/>
    public ModelMetadata<double> GetModelMetadata() => new()
    {
        Name = "LinearVectorModel",
        FeatureCount = _inputDim,
        Complexity = _parameters.Length
    };

    /// <inheritdoc/>
    public Vector<double> GetParameters() => _parameters.Clone();

    /// <inheritdoc/>
    public void SetParameters(Vector<double> parameters)
    {
        Guard.NotNull(parameters);
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}",
                nameof(parameters));
        }

        _parameters = parameters.Clone();
    }

    /// <inheritdoc/>
    public int ParameterCount => _parameters.Length;

    /// <inheritdoc/>
    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var model = new LinearVectorModel(_inputDim, _learningRate);
        model.SetParameters(parameters);
        return model;
    }

    /// <inheritdoc/>
    public virtual IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new LinearVectorModel(_inputDim, _learningRate);
        copy.SetParameters(_parameters);
        return copy;
    }

    /// <inheritdoc/>
    public virtual IFullModel<double, Matrix<double>, Vector<double>> Clone() => DeepCopy();

    object ICloneable.Clone() => DeepCopy();

    /// <inheritdoc/>
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    /// <inheritdoc/>
    public Vector<double> ComputeGradients(
        Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
    {
        Guard.NotNull(input);
        Guard.NotNull(target);

        var gradients = new Vector<double>(_parameters.Length);
        int count = Math.Min(input.Rows, target.Length);
        if (count == 0)
        {
            return gradients;
        }

        var predictions = Predict(input);

        // Use the provided loss function if available, otherwise fall back to MSE
        var loss = lossFunction ?? DefaultLossFunction;
        var lossGradient = loss.CalculateDerivative(predictions, target);

        double scale = 1.0 / count;
        for (int r = 0; r < count; r++)
        {
            double error = lossGradient[r];
            for (int c = 0; c < _inputDim && c < input.Columns; c++)
            {
                gradients[c] += scale * error * input[r, c];
            }

            gradients[_inputDim] += scale * error;
        }

        return gradients;
    }

    /// <inheritdoc/>
    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        Guard.NotNull(gradients);
        if (gradients.Length != _parameters.Length)
            throw new ArgumentException(
                $"Gradient length mismatch: expected {_parameters.Length}, got {gradients.Length}.",
                nameof(gradients));
        if (learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");

        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    /// <inheritdoc/>
    public byte[] Serialize() => Encoding.UTF8.GetBytes(SerializeParameters());

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        Guard.NotNull(data);
        DeserializeParameters(Encoding.UTF8.GetString(data));
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path is required.", nameof(filePath));
        }

        File.WriteAllText(filePath, SerializeParameters());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path is required.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Model file not found.", filePath);
        }

        DeserializeParameters(File.ReadAllText(filePath));
    }

    /// <inheritdoc/>
    public void SaveState(Stream stream)
    {
        Guard.NotNull(stream);
        using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
        writer.Write(SerializeParameters());
        writer.Flush();
    }

    /// <inheritdoc/>
    public void LoadState(Stream stream)
    {
        Guard.NotNull(stream);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true,
            bufferSize: 1024, leaveOpen: true);
        DeserializeParameters(reader.ReadToEnd());
    }

    /// <inheritdoc/>
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputDim);

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
    }

    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputDim;

    /// <inheritdoc/>
    public Dictionary<string, double> GetFeatureImportance()
    {
        var importance = new Dictionary<string, double>();
        if (_inputDim == 0)
        {
            return importance;
        }

        // Use absolute weight magnitude as feature importance
        for (int i = 0; i < _inputDim; i++)
        {
            importance[$"feature_{i}"] = Math.Abs(_parameters[i]);
        }

        return importance;
    }

    /// <inheritdoc/>
    public bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported.");
    }

    private string SerializeParameters()
    {
        return string.Join(",", _parameters.Select(p => p.ToString("R", CultureInfo.InvariantCulture)));
    }

    private void DeserializeParameters(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            _parameters = new Vector<double>(_parameters.Length);
            return;
        }

        var parts = content.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        int expectedCount = _inputDim + 1;
        if (parts.Length != expectedCount)
        {
            throw new InvalidDataException(
                $"Parameter count mismatch: expected {expectedCount}, got {parts.Length}");
        }

        var vector = new Vector<double>(parts.Length);
        for (int i = 0; i < parts.Length; i++)
        {
            vector[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
        }

        _parameters = vector;
    }
}
