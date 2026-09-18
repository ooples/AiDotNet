using AiDotNet.Attributes;
using System.Globalization;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// A linear embedding model mapping each row of a Matrix input to one embedding row: <c>h_r = W x_r + b</c>.
/// </summary>
/// <remarks>
/// <para>
/// The per-example counterpart of <see cref="LinearVectorModel"/>. Classifier and metric meta-learners -
/// prototypical, matching and relation networks, ANIL's body under its head - consume one embedding per
/// example, which a model emitting one scalar per row cannot supply. The output is <c>[rows, embeddingDim]</c>.
/// </para>
/// <para>
/// Its gradient is exact for any loss. <c>dL/dh</c> comes from differentiating the loss's
/// <see cref="ILossFunction{T}.ComputeTapeLoss"/> on the autodiff tape, so a meta-learner can pass a loss
/// that closes over its own head or metric and train the embedding through it, rather than regressing the
/// embedding onto the labels.
/// </para>
/// <para><b>For Beginners:</b> Where <see cref="LinearVectorModel"/> turns each example into one number, this
/// model turns each example into a short list of numbers (an embedding). Few-shot classifiers compare
/// examples by their embeddings, so they need this richer output.</para>
/// </remarks>
/// <example>
/// <code>
/// var model = new LinearEmbeddingModel(inputDim: 3, embeddingDim: 4);
/// var input = new Matrix&lt;double&gt;(10, 3);          // 10 examples, 3 features
/// Tensor&lt;double&gt; embeddings = model.Predict(input); // shape [10, 4]
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Tensor<>))]
[ResearchPaper("Pattern Recognition and Machine Learning", "https://www.springer.com/gp/book/9780387310732")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LinearEmbeddingModel : ModelBase<double, Matrix<double>, Tensor<double>>, ICloneable
{
    /// <summary>W (embeddingDim x inputDim, row-major) followed by b (embeddingDim).</summary>
    [FittedParameter]
    private Vector<double> _parameters;
    private readonly int _inputDim;
    private readonly int _embeddingDim;
    private readonly double _learningRate;

    /// <summary>
    /// Creates a linear embedding model.
    /// </summary>
    /// <param name="inputDim">Number of input features.</param>
    /// <param name="embeddingDim">Width of each example's embedding.</param>
    /// <param name="learningRate">Learning rate for gradient descent in <see cref="Train"/>. Default is 0.01.</param>
    public LinearEmbeddingModel(int inputDim, int embeddingDim, double learningRate = 0.01)
    {
        Guard.Positive(inputDim);
        Guard.Positive(embeddingDim);
        if (learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        _inputDim = inputDim;
        _embeddingDim = embeddingDim;
        _learningRate = learningRate;
        _parameters = new Vector<double>(embeddingDim * inputDim + embeddingDim);

        // Distinct values, so every embedding coordinate responds differently to the input from the start.
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] = 0.01 * (i + 1);
        }
    }

    /// <summary>Gets the width of each example's embedding.</summary>
    public int EmbeddingDimension => _embeddingDim;

    /// <inheritdoc/>
    public override Tensor<double> Predict(Matrix<double> input)
    {
        Guard.NotNull(input);
        if (input.Columns < _inputDim)
            throw new ArgumentException(
                $"Input has {input.Columns} columns but model expects at least {_inputDim}.", nameof(input));

        int biasOffset = _embeddingDim * _inputDim;
        var output = new Tensor<double>(new[] { input.Rows, _embeddingDim });
        for (int r = 0; r < input.Rows; r++)
        {
            for (int e = 0; e < _embeddingDim; e++)
            {
                double sum = _parameters[biasOffset + e];
                int row = e * _inputDim;
                for (int c = 0; c < _inputDim; c++)
                {
                    sum += input[r, c] * _parameters[row + c];
                }

                output[r * _embeddingDim + e] = sum;
            }
        }

        return output;
    }

    /// <inheritdoc/>
    public override void Train(Matrix<double> input, Tensor<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, _learningRate);
    }

    /// <inheritdoc/>
    public override ModelMetadata<double> GetModelMetadata() => new()
    {
        Name = "LinearEmbeddingModel",
        FeatureCount = _inputDim,
        Complexity = _parameters.Length
    };

    /// <inheritdoc/>
    public override IFullModel<double, Matrix<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var model = new LinearEmbeddingModel(_inputDim, _embeddingDim, _learningRate);
        model.SetParameters(parameters);
        return model;
    }

    object ICloneable.Clone() => DeepCopy();

    /// <inheritdoc/>
    public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    /// <inheritdoc/>
    /// <remarks>
    /// <c>dL/dh</c> is the loss's own tape derivative with respect to the whole <c>[rows, embeddingDim]</c>
    /// output, so the target can be anything the loss understands - class labels for a loss that closes over
    /// a metric, or embeddings for a regression loss. The chain rule through <c>h = W x + b</c> is exact.
    /// </remarks>
    public override Vector<double> ComputeGradients(
        Matrix<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
    {
        Guard.NotNull(input);
        Guard.NotNull(target);

        var gradients = new Vector<double>(_parameters.Length);
        if (input.Rows == 0)
        {
            return gradients;
        }

        var predictions = Predict(input);
        var loss = lossFunction ?? DefaultLossFunction;
        var outputGradient = loss.ComputeGradient(predictions, target);

        int biasOffset = _embeddingDim * _inputDim;
        for (int r = 0; r < input.Rows; r++)
        {
            for (int e = 0; e < _embeddingDim; e++)
            {
                double g = outputGradient[r * _embeddingDim + e];
                int row = e * _inputDim;
                for (int c = 0; c < _inputDim; c++)
                {
                    gradients[row + c] += g * input[r, c];
                }

                gradients[biasOffset + e] += g;
            }
        }

        return gradients;
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<double> gradients, double learningRate)
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
    public override void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path is required.", nameof(filePath));
        }

        File.WriteAllText(filePath, SerializeParameters());
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
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
    public override void SaveState(Stream stream)
    {
        Guard.NotNull(stream);
        using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
        writer.Write(SerializeParameters());
        writer.Flush();
    }

    /// <inheritdoc/>
    public override void LoadState(Stream stream)
    {
        Guard.NotNull(stream);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true,
            bufferSize: 1024, leaveOpen: true);
        DeserializeParameters(reader.ReadToEnd());
    }

    /// <inheritdoc/>
    public override IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputDim);

    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
    }

    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputDim;

    /// <inheritdoc/>
    public override Dictionary<string, double> GetFeatureImportance()
    {
        // A feature's importance is the size of its column of W: how strongly it moves the embedding.
        var importance = new Dictionary<string, double>();
        for (int c = 0; c < _inputDim; c++)
        {
            double norm = 0;
            for (int e = 0; e < _embeddingDim; e++)
            {
                double w = _parameters[e * _inputDim + c];
                norm += w * w;
            }

            importance[$"feature_{c}"] = Math.Sqrt(norm);
        }

        return importance;
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
        int expectedCount = _embeddingDim * _inputDim + _embeddingDim;
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
