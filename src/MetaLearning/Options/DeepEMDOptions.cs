using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for DeepEMD (Zhang et al., CVPR 2020) few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// DeepEMD uses the Earth Mover's Distance (optimal transport) as a metric for
/// comparing image representations in few-shot learning. Unlike ProtoNets which
/// compares single vectors, DeepEMD compares sets of local features using optimal
/// transport, capturing fine-grained structural similarity.
/// </para>
/// <para><b>For Beginners:</b> DeepEMD uses a clever way to compare examples:
///
/// **The Problem with simple metrics:**
/// ProtoNets averages all features into one vector. This loses local details like
/// "the left part of image A looks like the right part of image B."
///
/// **DeepEMD's solution: Earth Mover's Distance**
/// Think of features as piles of dirt. EMD measures the minimum "work" needed to
/// reshape one pile into another. This captures structural alignment:
/// - Each local feature (patch) from one image can match any patch from another
/// - The cost of matching = distance between the patches
/// - EMD finds the optimal matching that minimizes total distance
///
/// **Three modes:**
/// - FCN: Feature maps from the last conv layer (spatial patches)
/// - Grid: Divide feature maps into a regular grid
/// - Sampling: Sample representative local features
/// </para>
/// <para>
/// Reference: Zhang, C., Cai, Y., Lin, G., &amp; Shen, C. (2020).
/// DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers. CVPR 2020.
/// </para>
/// </remarks>
public class DeepEMDOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the feature extractor model.
    /// </summary>
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

    #region DeepEMD-Specific Properties

    /// <summary>
    /// Gets or sets the number of local features (nodes) per example for EMD computation.
    /// </summary>
    /// <value>Default is 25 (5x5 spatial grid).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many "parts" to split each example into
    /// before computing the optimal transport distance. More parts = finer-grained
    /// comparison but more computation. 25 (5x5 grid) is a good default.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 25;

    /// <summary>
    /// Gets or sets the number of Sinkhorn iterations for approximate EMD.
    /// </summary>
    /// <value>Default is 10.</value>
    /// <remarks>
    /// <para>
    /// EMD is computed approximately using the Sinkhorn algorithm (entropic regularization).
    /// More iterations = more accurate approximation but slower computation.
    /// </para>
    /// <para><b>For Beginners:</b> The Sinkhorn algorithm is an efficient way to compute
    /// Earth Mover's Distance. More iterations means a more accurate answer but takes longer.
    /// 10 iterations is usually enough for good results.
    /// </para>
    /// </remarks>
    public int SinkhornIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the regularization parameter for Sinkhorn algorithm.
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the smoothness of the optimal transport plan.
    /// Smaller values = closer to exact EMD but potentially unstable.
    /// Larger values = smoother approximation but less precise.
    /// </para>
    /// </remarks>
    public double SinkhornRegularization { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the temperature for the EMD-based classification logits.
    /// </summary>
    /// <value>Default is 12.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how "sharp" the classification decisions are.
    /// Higher temperature = more confident predictions. This scales the negative EMD
    /// distances before softmax.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 12.5;

    /// <summary>
    /// Gets or sets the EMD mode: "fcn", "grid", or "sampling".
    /// </summary>
    /// <value>Default is "fcn".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to extract local features:
    /// - "fcn": Use spatial positions from the last convolutional feature map
    /// - "grid": Divide features into a regular grid
    /// - "sampling": Randomly sample representative features
    /// </para>
    /// </remarks>
    public string EMDMode { get; set; } = "fcn";

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of DeepEMDOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public DeepEMDOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
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
               NumNodes > 0 &&
               SinkhornIterations > 0 &&
               SinkhornRegularization > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new DeepEMDOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            NumNodes = NumNodes, SinkhornIterations = SinkhornIterations,
            SinkhornRegularization = SinkhornRegularization,
            Temperature = Temperature, EMDMode = EMDMode
        };
    }

    #endregion
}
