using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-LoRA Bank (2024).
/// </summary>
/// <remarks>
/// <para>
/// Meta-LoRA Bank maintains a bank of diverse LoRA modules. For a new task, the algorithm
/// selects and combines the most relevant modules using a task-conditioned gating mechanism.
/// Meta-learning optimizes both the LoRA modules in the bank and the gating network.
/// </para>
/// </remarks>
public class MetaLoRABankOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of LoRA modules in the bank. Default: 8.
    /// </summary>
    public int BankSize { get; set; } = 8;

    /// <summary>
    /// Rank of each individual LoRA module. Default: 4.
    /// </summary>
    public int Rank { get; set; } = 4;

    /// <summary>
    /// Number of top modules to select per task (top-K gating). Default: 3.
    /// </summary>
    public int TopK { get; set; } = 3;

    /// <summary>
    /// Softmax temperature for module selection gating. Lower = sharper selection. Default: 1.0.
    /// </summary>
    public double GatingTemperature { get; set; } = 1.0;

    /// <summary>
    /// Load balancing regularization to encourage uniform module utilization. Default: 0.01.
    /// </summary>
    public double LoadBalanceRegularization { get; set; } = 0.01;

    public MetaLoRABankOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && BankSize > 0 && Rank > 0 && TopK > 0 && TopK <= BankSize;
    public IMetaLearnerOptions<T> Clone() => new MetaLoRABankOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        BankSize = BankSize, Rank = Rank, TopK = TopK, GatingTemperature = GatingTemperature,
        LoadBalanceRegularization = LoadBalanceRegularization
    };
}
