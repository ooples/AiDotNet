using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// The state LFT adapted to one task: the encoder as the inner loop left it, and the metric head that scores each
/// example against the task's classes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LFT (Tseng et al. 2020) classifies with a metric head over the encoder's features. Adaptation used to return the
/// bare encoder, so predicting gave one feature vector per example where the task asks for one score per class -
/// the trained head, which is the only part of the model that knows how many classes there are, was discarded at
/// the moment it was needed.
/// </para>
/// <para>
/// The feature-wise transformation is deliberately absent here: the paper removes those layers before the model is
/// used, so applying them at inference would inject the very noise they exist to simulate during training.
/// </para>
/// <para><b>For Beginners:</b> after adapting to a task, this model turns each example into features and then asks
/// the head "how much does this look like each class?", giving one probability per class.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation",
    "https://arxiv.org/abs/2001.08735",
    Year = 2020,
    Authors = "Hung-Yu Tseng, Hsin-Ying Lee, Jia-Bin Huang, Ming-Hsuan Yang")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Evaluation)]
public partial class LFTAdaptedModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    /// <summary>The task-adapted metric head, flat as <c>[outputDim * featureDim weights | outputDim biases]</c>.</summary>
    /// <remarks>
    /// Trained by the algorithm's closed-form step rather than by the optimizer or the tape, so it is persistent
    /// non-trainable state here, exactly as it is on the learner.
    /// </remarks>
    [AiDotNet.Attributes.Buffer]
    private Vector<T> _metricHead;

    private readonly int _featureDimension;
    private readonly int _outputDimension;

    /// <summary>Gets the metric head this model scores with.</summary>
    public Vector<T> MetricHead => _metricHead;

    /// <summary>Initializes the adapted model.</summary>
    /// <param name="encoder">The task-adapted feature encoder.</param>
    /// <param name="metricHead">theta_m as adaptation left it; the model keeps its own copy.</param>
    /// <param name="featureDimension">The head's input width, the encoder's per-example output width.</param>
    /// <param name="outputDimension">The head's output width, the task's class count.</param>
    /// <exception cref="ArgumentNullException">Thrown when the encoder or the head is null.</exception>
    public LFTAdaptedModel(
        IFullModel<T, TInput, TOutput> encoder,
        Vector<T> metricHead,
        int featureDimension,
        int outputDimension)
        : base(encoder)
    {
        if (metricHead is null) throw new ArgumentNullException(nameof(metricHead));
        _metricHead = new Vector<T>(metricHead.Length);
        for (int i = 0; i < metricHead.Length; i++) _metricHead[i] = metricHead[i];
        _featureDimension = featureDimension;
        _outputDimension = outputDimension;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// One score row per example, <c>[rows, classes]</c>: encode, resize to the head's input width, and apply the
    /// head's softmax. The encoder is never modified, so this is safe to call repeatedly.
    /// </remarks>
    public override TOutput Predict(TInput input)
    {
        int batch = MbPAConversions<T>.GetBatchSize(input);
        var scores = new Tensor<T>(new[] { batch, _outputDimension });
        for (int row = 0; row < batch; row++)
        {
            var single = MbPAConversions<T>.SliceExample(input, row);
            var features = MbPAConversions<T>.ResizeTo(
                ExtractFeaturesFromBaseModel(single, _featureDimension), _featureDimension);
            var probabilities = MbPAOutputNetwork<T>.Forward(
                _metricHead, features, _featureDimension, _outputDimension, MbPAOutputDistribution.Categorical);
            for (int c = 0; c < _outputDimension; c++) scores[row * _outputDimension + c] = probabilities[c];
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The model's own registered parameters are the metric head, so a parameter vector handed back here is that
    /// head: the encoder comes from the wrapped base model, as it does for every other adapted meta-model.
    /// </remarks>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        => new LFTAdaptedModel<T, TInput, TOutput>(BaseModel, parameters, _featureDimension, _outputDimension);
}
