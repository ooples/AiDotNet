using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for In-Context RL: meta-RL via in-context adaptation
/// without explicit gradient updates at test time.
/// </summary>
/// <remarks>
/// <para>
/// In-Context RL trains a model to perform RL adaptation purely through its forward pass,
/// without any gradient updates at test time. The model conditions on a context buffer
/// of past (input, output, loss) triplets and learns to improve its predictions based
/// on this growing context. During meta-training, the context is built sequentially.
/// </para>
/// </remarks>
public class InContextRLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Maximum size of the in-context buffer. Default: 32.</summary>
    public int ContextBufferSize { get; set; } = 32;

    /// <summary>Dimensionality of context embeddings per entry. Default: 32.</summary>
    public int ContextEmbeddingDim { get; set; } = 32;

    /// <summary>Weight for the context prediction loss (auxiliary). Default: 0.1.</summary>
    public double ContextPredictionWeight { get; set; } = 0.1;

    public InContextRLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && ContextBufferSize > 0;
    public IMetaLearnerOptions<T> Clone() => new InContextRLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ContextBufferSize = ContextBufferSize, ContextEmbeddingDim = ContextEmbeddingDim,
        ContextPredictionWeight = ContextPredictionWeight
    };
}
