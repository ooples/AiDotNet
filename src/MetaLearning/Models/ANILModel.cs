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
/// ANIL model for few-shot classification with head-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of ANIL after inner-loop adaptation: the body (feature extractor) and the
/// head adapted to one task. It classifies every example: the body embeds each row, and the head scores each
/// embedding, so a batch of <c>n</c> examples yields <c>[n, NumClasses]</c> scores.
/// </para>
/// <para><b>For Beginners:</b> After ANIL adapts to a new task by training only the classification head on
/// support examples, this model stores the body and that adapted head, and uses both to score new examples.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML",
    "https://arxiv.org/abs/1909.09157",
    Year = 2020,
    Authors = "Raghu, A., Raghu, M., Bengio, S., & Vinyals, O.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ANILModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _featureExtractor;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _headWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T>? _headBias;
    private readonly ANILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the ANILModel.
    /// </summary>
    /// <param name="featureExtractor">The feature extractor (body), owned by this model.</param>
    /// <param name="headWeights">The adapted head weights, <c>[NumClasses x FeatureDimension]</c> row-major.</param>
    /// <param name="headBias">The adapted head bias, <c>[NumClasses]</c>, or null for none.</param>
    /// <param name="options">The ANIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public ANILModel(
        IFullModel<T, TInput, TOutput> featureExtractor,
        Vector<T> headWeights,
        Vector<T>? headBias,
        ANILOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(featureExtractor);
        _featureExtractor = featureExtractor;
        Guard.NotNull(headWeights);
        _headWeights = headWeights;
        _headBias = headBias;
        Guard.NotNull(options);
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the adapted head bias (may be null if not used).
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
    /// <remarks>
    /// One score row per example, <c>[rows, NumClasses]</c>. It used to flatten the whole batch into one feature
    /// vector and return a single row of scores for it.
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(_featureExtractor.Predict(input));
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
        throw new NotSupportedException("Use the ANIL algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("ANIL model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined body + head parameters
        var bodyParams = InterfaceGuard.Parameterizable(_featureExtractor).GetParameters();
        int totalSize = bodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < bodyParams.Length; i++)
        {
            combined[idx++] = bodyParams[i];
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
