using AiDotNet.Attributes;
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
/// <para><b>For Beginners:</b> This is a simple linear model used mainly for testing and
/// demonstrating meta-learning algorithms. It takes a matrix of inputs and produces predictions
/// using a straightforward linear formula (y = inputs * weights + bias). Its simplicity makes
/// it ideal for verifying that meta-learning algorithms like MAML work correctly before
/// scaling up to more complex models.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a simple linear model for meta-learning testing
/// var input = new Matrix&lt;double&gt;(10, 5);  // 10 samples, 5 features
/// var labels = new Vector&lt;double&gt;(10);
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LinearVectorModel(inputDim: 5, learningRate: 0.01))
///     .Build(input, labels);
/// Vector&lt;double&gt; predictions = result.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Pattern Recognition and Machine Learning", "https://www.springer.com/gp/book/9780387310732")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LinearVectorModel : ModelBase<double, Matrix<double>, Vector<double>>, ICloneable
{
    [FittedParameter]
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
        // w_i = 0.01 (i + 1), bias included
        _parameters = AiDotNetEngine.Current.TensorMultiplyScalar(
            AiDotNetEngine.Current.TensorArange<double>(1, inputDim + 2, 1), 0.01).ToVector();
    }

    /// <inheritdoc/>
    public override Vector<double> Predict(Matrix<double> input)
    {
        Guard.NotNull(input);
        var output = new Vector<double>(input.Rows);
        if (input.Columns < _inputDim)
            throw new ArgumentException(
                $"Input has {input.Columns} columns but model expects at least {_inputDim}.", nameof(input));

        if (input.Rows == 0) return output;

        // y = X[:, :d] w + b
        var engine = AiDotNetEngine.Current;
        var weights = Tensor<double>.FromVector(_parameters.GetSubVector(0, _inputDim)).Reshape(_inputDim, 1);
        var linear = engine.TensorMatMul(Features(input, input.Rows), weights).Reshape(input.Rows);
        return engine.TensorAddScalar(linear, _parameters[_inputDim]).ToVector();
    }

    /// <summary>The first <paramref name="rows"/> rows of the model's <c>_inputDim</c> feature columns.</summary>
    private Tensor<double> Features(Matrix<double> input, int rows)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNarrow(engine.TensorNarrow(Tensor<double>.FromMatrix(input), 0, 0, rows), 1, 0, _inputDim);
    }

    /// <inheritdoc/>
    public override void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, _learningRate);
    }

    /// <inheritdoc/>
    public override ModelMetadata<double> GetModelMetadata() => new()
    {
        Name = "LinearVectorModel",
        FeatureCount = _inputDim,
        Complexity = _parameters.Length
    };

    // GetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    // SetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    // ParameterCount restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    /// <inheritdoc/>
    public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var model = new LinearVectorModel(_inputDim, _learningRate);
        model.SetParameters(parameters);
        return model;
    }

    object ICloneable.Clone() => DeepCopy();

    /// <inheritdoc/>
    public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    /// <inheritdoc/>
    public override Vector<double> ComputeGradients(
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

        // dL/dprediction already carries the loss's own reduction (MSE's 1/n). The chain rule through
        // y = X w + b adds nothing more: it used to divide by the row count a second time, so the model
        // reported 1/n of its loss's gradient and every learner stepping on it moved n times too slowly.
        var lossGradient = loss.ComputeGradient(predictions, target);

        // dL/dw = X^T e and dL/db = sum(e) over the first count rows
        var engine = AiDotNetEngine.Current;
        var errors = lossGradient.GetSubVector(0, count);
        var weightGradient = engine.TensorMatMul(
            engine.TensorTranspose(Features(input, count)), Tensor<double>.FromVector(errors).Reshape(count, 1));
        return engine.Concat(new[]
        {
            weightGradient.Reshape(_inputDim),
            new Tensor<double>(new[] { engine.Sum(errors) }, new[] { 1 }),
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
        var importance = new Dictionary<string, double>();
        if (_inputDim == 0)
        {
            return importance;
        }

        // Use absolute weight magnitude as feature importance
        var magnitudes = AiDotNetEngine.Current.Abs(_parameters.GetSubVector(0, _inputDim)).ToArray();
        for (int i = 0; i < magnitudes.Length; i++)
        {
            importance[$"feature_{i}"] = magnitudes[i];
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
        int expectedCount = _inputDim + 1;
        if (parts.Length != expectedCount)
        {
            throw new InvalidDataException(
                $"Parameter count mismatch: expected {expectedCount}, got {parts.Length}");
        }

        _parameters = new Vector<double>(parts.Select(part => double.Parse(part, CultureInfo.InvariantCulture)));
    }
}
