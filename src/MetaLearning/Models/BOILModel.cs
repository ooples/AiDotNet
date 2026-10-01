using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// BOIL model for few-shot classification with body-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of BOIL for one task: its own copy of the body, carrying the parameters the inner loop adapted,
/// and the head, which the inner loop does not change. The body embeds every row and the head scores each embedding,
/// so a batch of <c>n</c> examples yields <c>[n, NumClasses]</c> scores.
/// </para>
/// <para><b>For Beginners:</b> After BOIL adapts its feature extractor to a new task, this model stores that adapted
/// extractor together with the shared classifier head, and uses both to score new examples.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BOIL: Towards Representation Change for Few-shot Learning",
    "https://arxiv.org/abs/2008.08882",
    Year = 2021,
    Authors = "Oh, J., Yoo, H., Kim, C., & Yun, S.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class BOILModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _body;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _adaptedBodyParams;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _headWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T>? _headBias;
    private readonly BOILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the BOILModel.
    /// </summary>
    /// <param name="baseModel">The body whose architecture the adapted parameters belong to; this model copies it.</param>
    /// <param name="adaptedBodyParams">The adapted body parameters for this task.</param>
    /// <param name="headWeights">The head weights, <c>[NumClasses x FeatureDimension]</c> row-major.</param>
    /// <param name="headBias">The head bias, <c>[NumClasses]</c>, or null for none.</param>
    /// <param name="options">The BOIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <remarks>
    /// The model keeps its own copy of the body. It used to write the adapted parameters into the base model it was
    /// handed on every Predict - and Adapt handed it the meta-model itself, so predicting with an adapted model
    /// overwrote the meta-learned body.
    /// </remarks>
    public BOILModel(
        IFullModel<T, TInput, TOutput> baseModel,
        Vector<T> adaptedBodyParams,
        Vector<T> headWeights,
        Vector<T>? headBias,
        BOILOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(baseModel);
        Guard.NotNull(adaptedBodyParams);
        Guard.NotNull(headWeights);
        Guard.NotNull(options);
        _body = baseModel.DeepCopy();
        InterfaceGuard.Parameterizable(_body).SetParameters(adaptedBodyParams);
        _adaptedBodyParams = adaptedBodyParams;
        _headWeights = headWeights;
        _headBias = headBias;
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted body parameters.
    /// </summary>
    public Vector<T> AdaptedBodyParams => _adaptedBodyParams;

    /// <summary>
    /// Gets the head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the head bias (may be null if not used).
    /// </summary>
    public Vector<T>? HeadBias => _headBias;

    /// <summary>
    /// Gets the number of classes this model is adapted for.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Gets the feature dimension expected by the head.
    /// </summary>
    public int FeatureDimension => _options.FeatureDimension;

    /// <inheritdoc/>
    /// <remarks>One score row per example, <c>[rows, NumClasses]</c>.</remarks>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(_body.Predict(input));
        if (embeddings.Shape[1] != _options.FeatureDimension)
        {
            throw new InvalidOperationException(
                $"The body emits {embeddings.Shape[1]}-wide embeddings per example but the head reads "
                + $"FeatureDimension = {_options.FeatureDimension}.");
        }

        var weights = Tensor<T>.FromVector(_headWeights).Reshape(_options.NumClasses, _options.FeatureDimension);
        var bias = _headBias is { Length: > 0 } ? Tensor<T>.FromVector(_headBias) : null;
        var scores = EmbeddingClassificationLoss<T>.LinearHead(embeddings, weights, bias);
        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the BOIL algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("BOIL model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined body + head parameters
        int totalSize = _adaptedBodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < _adaptedBodyParams.Length; i++)
        {
            combined[idx++] = _adaptedBodyParams[i];
        }
        for (int i = 0; i < _headWeights.Length; i++)
        {
            combined[idx++] = _headWeights[i];
        }
        if (_headBias != null)
        {
            for (int i = 0; i < _headBias.Length; i++)
            {
                combined[idx++] = _headBias[i];
            }
        }

        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
