using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the OML (Online Meta-Learning) algorithm.
/// </summary>
/// <remarks>
/// <para>
/// OML (Javed &amp; White, 2019) partitions the model into a Representation Learning
/// Network (RLN) and a Prediction Learning Network (PLN). Only the PLN parameters are
/// adapted in the inner loop, while the RLN is meta-learned to produce sparse,
/// non-interfering representations that enable continual learning.
/// </para>
/// </remarks>
public class OMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    public double InnerLearningRate { get; set; } = 0.01;
    public double OuterLearningRate { get; set; } = 0.001;
    public int AdaptationSteps { get; set; } = 5;
    public int MetaBatchSize { get; set; } = 4;
    public int NumMetaIterations { get; set; } = 1000;
    public double? GradientClipThreshold { get; set; } = 10.0;
    public int? RandomSeed { get => Seed; set => Seed = value; }
    public int EvaluationTasks { get; set; } = 100;
    public int EvaluationFrequency { get; set; } = 100;
    public bool EnableCheckpointing { get; set; } = false;
    public int CheckpointFrequency { get; set; } = 500;
    public bool UseFirstOrder { get; set; } = true;
    public ILossFunction<T>? LossFunction { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Fraction of total model parameters that form the PLN (prediction head).
    /// These parameters are adapted in the inner loop. The remaining (1-fraction) form
    /// the RLN (representation backbone). Default: 0.3.
    /// </summary>
    public double PlnFraction { get; set; } = 0.3;

    /// <summary>
    /// L1 sparsity penalty on the RLN parameter activations to encourage
    /// non-interfering, sparse representations. Default: 0.01.
    /// </summary>
    public double SparsityPenalty { get; set; } = 0.01;

    /// <summary>
    /// L2 regularization weight on RLN parameter changes to prevent catastrophic
    /// forgetting of the learned representation. Default: 0.001.
    /// </summary>
    public double RepresentationRegWeight { get; set; } = 0.001;

    public OMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new OMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        PlnFraction = PlnFraction, SparsityPenalty = SparsityPenalty,
        RepresentationRegWeight = RepresentationRegWeight
    };
}
