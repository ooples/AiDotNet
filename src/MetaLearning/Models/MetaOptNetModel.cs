using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// The linear classifier MetaOptNet's convex base learner produced for one task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Holds its own copy of the embedding network and one weight row per class. <see cref="Predict"/> returns each
/// example's score for every class, <c>[rows, NumClasses]</c>, scaled by the learned logit scale of eq. 12, for
/// Tensor and Matrix outputs, and the highest-scoring class for a Vector output. It used to flatten the batch into
/// one vector, so a batch of four examples produced one score per class instead of four.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Differentiable Convex Optimization",
    "https://arxiv.org/abs/1904.03758",
    Year = 2019,
    Authors = "Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, Stefano Soatto")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaOptNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Matrix<T> _classifierWeights;
    private readonly T _temperature;
    private readonly MetaOptNetOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the MetaOptNetModel.
    /// </summary>
    /// <param name="featureEncoder">The embedding network; the model keeps its own copy.</param>
    /// <param name="classifierWeights">One weight row per class, <c>[NumClasses, EmbeddingDimension]</c>.</param>
    /// <param name="temperature">The learned scale that multiplies the logits (eq. 12).</param>
    /// <param name="options">The MetaOptNet options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <exception cref="ArgumentException">The weights do not hold one row per class.</exception>
    public MetaOptNetModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Matrix<T> classifierWeights,
        T temperature,
        MetaOptNetOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(featureEncoder);
        Guard.NotNull(classifierWeights);
        Guard.NotNull(options);
        if (classifierWeights.Rows != options.NumClasses)
        {
            throw new ArgumentException(
                $"Classifier weights rows ({classifierWeights.Rows}) must match NumClasses ({options.NumClasses}).",
                nameof(classifierWeights));
        }

        _featureEncoder = featureEncoder.DeepCopy();
        _classifierWeights = classifierWeights;
        _temperature = temperature;
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>Gets the classifier the base learner produced, one row per class.</summary>
    public Matrix<T> ClassifierWeights => _classifierWeights;

    /// <summary>Gets the scale that multiplies the logits.</summary>
    public T Temperature => _temperature;

    /// <summary>Gets the number of classes.</summary>
    public int NumClasses => _options.NumClasses;

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var rows = ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input));
        var embeddings = _options.NormalizeEmbeddings
            ? PrototypeMetric<T>.Normalized(rows, normalize: true)
            : rows;
        if (embeddings.Shape[1] != _options.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"The embedding network emits {embeddings.Shape[1]}-wide embeddings per example but "
                + $"EmbeddingDimension is {_options.EmbeddingDimension}.");
        }

        var weights = new Tensor<T>(new[] { _options.NumClasses, _options.EmbeddingDimension });
        for (int c = 0; c < _options.NumClasses; c++)
            for (int j = 0; j < _options.EmbeddingDimension; j++)
                weights[c * _options.EmbeddingDimension + j] = _classifierWeights[c, j];

        var scores = engine.TensorMultiplyScalar(
            engine.TensorMatMul(embeddings, engine.TensorTranspose(weights)), _temperature);

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            int n = scores.Shape[0];
            var predicted = new Vector<T>(n);
            for (int r = 0; r < n; r++)
            {
                int best = 0;
                for (int c = 1; c < _options.NumClasses; c++)
                {
                    if (NumOps.GreaterThan(scores[r * _options.NumClasses + c], scores[r * _options.NumClasses + best])) best = c;
                }

                predicted[r] = NumOps.FromDouble(best);
            }

            return (TOutput)(object)predicted;
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the MetaOptNet algorithm to train; the base learner is solved per task.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("MetaOptNet's classifier comes from the convex solver, not from an update.");
    }

    /// <inheritdoc/>
    /// <remarks>The embedding network's parameters, then the classifier's rows.</remarks>
    public Vector<T> GetParameters()
    {
        var encoderParams = InterfaceGuard.Parameterizable(_featureEncoder).GetParameters();
        int classifierSize = _classifierWeights.Rows * _classifierWeights.Columns;
        var combined = new Vector<T>(encoderParams.Length + classifierSize);
        for (int i = 0; i < encoderParams.Length; i++) combined[i] = encoderParams[i];
        int index = encoderParams.Length;
        for (int r = 0; r < _classifierWeights.Rows; r++)
            for (int c = 0; c < _classifierWeights.Columns; c++) combined[index++] = _classifierWeights[r, c];
        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
