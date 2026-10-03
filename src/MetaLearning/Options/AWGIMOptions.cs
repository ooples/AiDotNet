using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration for AWGIM, Attentive Weights Generation for few-shot learning via Information
/// Maximization (Guo &amp; Cheung, CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The defaults are the authors' (github.com/Yiluan/AWGIM, main.py): a 128-wide latent space, 4-head
/// attention, a two-layer weight decoder, dropout 0.3, AdamW at 2e-4 with weight decay 1e-6 and a
/// staircase decay of 0.2 every 15,000 steps, gradients clipped at 0.1 by value and by norm, and loss
/// weights α1 = 1 (support classification), α2 = α3 = 0.001 (the two reconstruction terms).
/// </para>
/// <para>
/// <b>For Beginners:</b> AWGIM writes a classifier for every query example on the spot. It looks at
/// the whole support set to understand the task, looks at how the query relates to each support
/// example, and turns both into the weights that classify that query. Two extra losses make the
/// written weights carry as much information as possible about the task and the query.
/// </para>
/// </remarks>
public class AWGIMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>Initializes a new instance with the feature encoder whose embeddings AWGIM classifies.</summary>
    /// <param name="metaModel">The feature encoder: one embedding row per example.</param>
    public AWGIMOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #region Required

    /// <summary>Gets or sets the feature encoder, which maps each example to one embedding row.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Standard meta-learning components

    /// <summary>Gets or sets the classification loss. Default: cross-entropy over logits.</summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>Not used: AWGIM runs its own AdamW meta-update (see <see cref="OuterLearningRate"/>).</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>Not used: AWGIM generates weights by attention and has no inner optimization loop.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    #endregion

    #region IMetaLearnerOptions

    /// <summary>
    /// Not an AWGIM hyperparameter: the method has no inner gradient loop. Present because the meta-learner
    /// contract declares it; the value is not read.
    /// </summary>
    public double InnerLearningRate { get; set; } = 0.0;

    /// <summary>Gets or sets the AdamW learning rate of the meta-update.</summary>
    /// <value>Default 2e-4, the authors'.</value>
    public double OuterLearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Not an AWGIM hyperparameter: weights are generated in one attention pass. Present because the
    /// meta-learner contract declares it; the value is not read.
    /// </summary>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>Gets or sets the number of tasks per meta-update.</summary>
    /// <value>Default 64, the authors' batch size.</value>
    public int MetaBatchSize { get; set; } = 64;

    /// <summary>Gets or sets the number of meta-training iterations.</summary>
    /// <value>Default 50,000 (500 epochs of 100 episodes), the authors'.</value>
    public int NumMetaIterations { get; set; } = 50000;

    /// <summary>Not applicable: there is no inner loop to differentiate through.</summary>
    public bool UseFirstOrder { get; set; } = false;

    /// <summary>Gets or sets the per-element gradient clipping threshold (clip by value).</summary>
    /// <value>Default 0.1, the authors'. Null disables value clipping.</value>
    public double? GradientClipThreshold { get; set; } = 0.1;

    /// <summary>Gets or sets the gradient-norm clipping threshold, applied after value clipping.</summary>
    /// <value>Default 0.1, the authors'. Null disables norm clipping.</value>
    public double? GradientNormClipThreshold { get; set; } = 0.1;

    /// <summary>Gets or sets the random seed.</summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>Gets or sets the number of evaluation tasks.</summary>
    public int EvaluationTasks { get; set; } = 600;

    /// <summary>Gets or sets how often to evaluate, in meta-iterations.</summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>Gets or sets whether checkpointing is enabled.</summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>Gets or sets the checkpoint frequency.</summary>
    public int CheckpointFrequency { get; set; } = 500;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the number of classes (ways) per task.</summary>
    public int NumClasses { get; set; } = 5;

    /// <summary>Gets or sets the width of the feature encoder's per-example embedding (p).</summary>
    /// <value>Default 640, the width of the LEO embeddings the paper uses.</value>
    public int EmbeddingDimension { get; set; } = 640;

    /// <summary>Gets or sets the latent width (d) of the encoders and the attention paths.</summary>
    /// <value>Default 128, the authors'. Must be divisible by <see cref="NumHeads"/>.</value>
    public int LatentDimension { get; set; } = 128;

    /// <summary>Gets or sets the number of attention heads in every attention block.</summary>
    /// <value>Default 4, the authors'.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of layers in the weight decoder and the reconstruction networks.</summary>
    /// <value>Default 2 (one hidden layer of width 2d), the authors' mlp_size.</value>
    public int DecoderLayers { get; set; } = 2;

    /// <summary>Gets or sets the dropout rate applied to MLP inputs and to the features being classified.</summary>
    /// <value>Default 0.3, the authors'.</value>
    public double DropoutRate { get; set; } = 0.3;

    #endregion

    #region Objective

    /// <summary>Gets or sets α1, the weight of classifying the support set with each query's weights.</summary>
    /// <value>Default 1, the authors'.</value>
    public double SupportClassificationWeight { get; set; } = 1.0;

    /// <summary>Gets or sets α2, the weight of reconstructing the contextual code from the generated weights.</summary>
    /// <value>Default 0.001, the authors'.</value>
    public double ContextReconstructionWeight { get; set; } = 0.001;

    /// <summary>Gets or sets α3, the weight of reconstructing the attentive (query) code from the generated weights.</summary>
    /// <value>Default 0.001, the authors'.</value>
    public double QueryReconstructionWeight { get; set; } = 0.001;

    /// <summary>Gets or sets the decoupled (AdamW) weight decay.</summary>
    /// <value>Default 1e-6, the authors'.</value>
    public double WeightDecay { get; set; } = 1e-6;

    /// <summary>Gets or sets the number of meta-updates between learning-rate decays.</summary>
    /// <value>Default 15,000, the authors'. Zero or less disables the decay.</value>
    public int LearningRateDecaySteps { get; set; } = 15000;

    /// <summary>Gets or sets the staircase learning-rate decay factor.</summary>
    /// <value>Default 0.2, the authors'.</value>
    public double LearningRateDecayRate { get; set; } = 0.2;

    #endregion

    /// <summary>Checks that the configuration describes a buildable AWGIM.</summary>
    public bool IsValid()
    {
        return MetaModel != null
            && OuterLearningRate > 0
            && MetaBatchSize > 0
            && NumMetaIterations > 0
            && NumClasses > 1
            && EmbeddingDimension > 0
            && LatentDimension > 0
            && NumHeads > 0
            && LatentDimension % NumHeads == 0
            && DecoderLayers >= 1
            && DropoutRate >= 0 && DropoutRate < 1
            && SupportClassificationWeight >= 0
            && ContextReconstructionWeight >= 0
            && QueryReconstructionWeight >= 0
            && WeightDecay >= 0
            && LearningRateDecayRate > 0
            && (GradientClipThreshold is null || GradientClipThreshold > 0)
            && (GradientNormClipThreshold is null || GradientNormClipThreshold > 0);
    }

    /// <summary>Creates a copy of these options that shares the feature encoder.</summary>
    public IMetaLearnerOptions<T> Clone()
    {
        return new AWGIMOptions<T, TInput, TOutput>(MetaModel)
        {
            Seed = Seed,
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            UseFirstOrder = UseFirstOrder,
            GradientClipThreshold = GradientClipThreshold,
            GradientNormClipThreshold = GradientNormClipThreshold,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            NumClasses = NumClasses,
            EmbeddingDimension = EmbeddingDimension,
            LatentDimension = LatentDimension,
            NumHeads = NumHeads,
            DecoderLayers = DecoderLayers,
            DropoutRate = DropoutRate,
            SupportClassificationWeight = SupportClassificationWeight,
            ContextReconstructionWeight = ContextReconstructionWeight,
            QueryReconstructionWeight = QueryReconstructionWeight,
            WeightDecay = WeightDecay,
            LearningRateDecaySteps = LearningRateDecaySteps,
            LearningRateDecayRate = LearningRateDecayRate,
        };
    }
}
