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
/// var trainX = new Matrix&lt;double&gt;(10, 3);           // 10 examples, 3 features
/// var trainY = new Tensor&lt;double&gt;(new[] { 10, 4 }); // their 4-dimensional embeddings
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Tensor&lt;double&gt;&gt;()
///     .ConfigureModel(new LinearEmbeddingModel(inputDim: 3, embeddingDim: 4))
///     .Build(trainX, trainY);
/// Tensor&lt;double&gt; embeddings = result.Predict(trainX);  // shape [10, 4]
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
        // Distinct values, so every embedding coordinate responds differently to the input from the start.
        _parameters = AiDotNetEngine.Current.TensorMultiplyScalar(
            AiDotNetEngine.Current.TensorArange<double>(1, embeddingDim * inputDim + embeddingDim + 1, 1), 0.01).ToVector();
    }

    /// <summary>W as an [embeddingDim, inputDim] tensor (row e holds embedding coordinate e's weights).</summary>
    private Tensor<double> Weights()
        => Tensor<double>.FromVector(_parameters.GetSubVector(0, _embeddingDim * _inputDim)).Reshape(_embeddingDim, _inputDim);

    /// <summary>The model's <c>_inputDim</c> feature columns of <paramref name="input"/>.</summary>
    private Tensor<double> Features(Matrix<double> input)
        => AiDotNetEngine.Current.TensorNarrow(Tensor<double>.FromMatrix(input), 1, 0, _inputDim);

    /// <summary>Gets the width of each example's embedding.</summary>
    public int EmbeddingDimension => _embeddingDim;

    /// <inheritdoc/>
    public override Tensor<double> Predict(Matrix<double> input)
    {
        Guard.NotNull(input);
        if (input.Columns < _inputDim)
            throw new ArgumentException(
                $"Input has {input.Columns} columns but model expects at least {_inputDim}.", nameof(input));

        if (input.Rows == 0) return new Tensor<double>(new[] { 0, _embeddingDim });

        // Y = X W^T + b
        var engine = AiDotNetEngine.Current;
        var bias = Tensor<double>.FromVector(_parameters.GetSubVector(_embeddingDim * _inputDim, _embeddingDim))
            .Reshape(1, _embeddingDim);
        var linear = engine.TensorMatMul(Features(input), engine.TensorTranspose(Weights()));
        return engine.TensorAdd(linear, engine.TensorBroadcastTo(bias, new[] { input.Rows, _embeddingDim }));
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

        // dL/dW = G^T X and dL/db = the column sums of G
        var engine = AiDotNetEngine.Current;
        var g = outputGradient.Reshape(input.Rows, _embeddingDim);
        var weightGradient = engine.TensorMatMul(engine.TensorTranspose(g), Features(input));
        var biasGradient = engine.ReduceSum(g, new[] { 0 }, false);
        return engine.Concat(new[]
        {
            weightGradient.Reshape(_embeddingDim * _inputDim),
            biasGradient.Reshape(_embeddingDim),
        }, 0).ToVector();
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

        // In place, so a caller holding the parameter vector sees the step.
        AiDotNetEngine.Current.Subtract(_parameters, AiDotNetEngine.Current.Multiply(gradients, learningRate))
            .AsSpan().CopyTo(_parameters.AsWritableSpan());
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
        var engine = AiDotNetEngine.Current;
        var weights = Weights();
        var columnNorms = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(weights, weights), new[] { 0 }, false))
            .ToVector().ToArray();
        for (int c = 0; c < columnNorms.Length; c++)
        {
            importance[$"feature_{c}"] = columnNorms[c];
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

        _parameters = new Vector<double>(parts.Select(part => double.Parse(part, CultureInfo.InvariantCulture)));
    }
}
