using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for TIM (Transductive Information Maximization) (Boudiaf et al., NeurIPS 2020).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// TIM is a transductive method that maximizes mutual information between query features
/// and their predicted labels. By using ALL query examples jointly (not independently),
/// it exploits the structure of the query set for better classification.
/// </para>
/// <para><b>For Beginners:</b> TIM uses query examples to help classify each other:
///
/// **Inductive vs Transductive:**
/// - Inductive: Classify each query example independently
/// - Transductive: Use ALL query examples together (they provide info about each other)
///
/// **How TIM works:**
/// 1. Compute initial prototypes from support set (like ProtoNets)
/// 2. For each query, compute soft assignments to classes
/// 3. Iteratively refine assignments by maximizing mutual information:
///    - Conditional entropy: Each query should be confidently assigned to one class
///    - Marginal entropy: Classes should have balanced assignments overall
///    - KL divergence: Soft assignments shouldn't deviate too far from initial predictions
///
/// **Analogy:**
/// Imagine sorting a pile of photos into groups:
/// - Inductive: Sort each photo individually
/// - TIM: Sort all photos simultaneously, using the fact that groups should be balanced
///   and each photo should clearly belong to one group
/// </para>
/// <para>
/// Reference: Boudiaf, M., Ziko, I., Rony, J., Dolz, J., Piantanida, P., &amp; Ben Ayed, I. (2020).
/// Information Maximization for Few-Shot Learning. NeurIPS 2020.
/// </para>
/// </remarks>
public class TIMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>Gets or sets the feature extractor model.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Standard Meta-Learning Properties

    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.AdaptationSteps"/>
    public int AdaptationSteps { get; set; } = 1;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.MetaBatchSize"/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.NumMetaIterations"/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.GradientClipThreshold"/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <summary>Gets or sets the random seed.</summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationTasks"/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationFrequency"/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EnableCheckpointing"/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.CheckpointFrequency"/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.UseFirstOrder"/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Gets or sets the loss function.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Gets or sets the outer loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Gets or sets the inner loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    #endregion

    #region TIM-Specific Properties

    /// <summary>
    /// Gets or sets the number of transductive refinement iterations.
    /// </summary>
    /// <value>Default is 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many rounds of "mutual improvement" the query
    /// predictions go through. More iterations = more refined predictions, but
    /// diminishing returns after about 50.
    /// </para>
    /// </remarks>
    public int TransductiveIterations { get; set; } = 50;

    /// <summary>
    /// Gets or sets the weight for the conditional entropy term.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How strongly to encourage confident predictions.
    /// Higher = more confident (but potentially overconfident) assignments.
    /// </para>
    /// </remarks>
    public double ConditionalEntropyWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight for the marginal entropy term.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How strongly to encourage balanced class assignments.
    /// Prevents the model from assigning all queries to one class.
    /// </para>
    /// </remarks>
    public double MarginalEntropyWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the temperature for softmax computation.
    /// </summary>
    /// <value>Default is 15.0.</value>
    public double Temperature { get; set; } = 15.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of TIMOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public TIMOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               TransductiveIterations > 0 &&
               Temperature > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new TIMOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            TransductiveIterations = TransductiveIterations,
            ConditionalEntropyWeight = ConditionalEntropyWeight,
            MarginalEntropyWeight = MarginalEntropyWeight, Temperature = Temperature
        };
    }

    #endregion
}
