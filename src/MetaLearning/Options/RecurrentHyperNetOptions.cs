using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>Configuration options for Recurrent HyperNetwork meta-learning.</summary>
/// <remarks>
/// <para>
/// A GRU-like recurrent cell processes gradient information at each adaptation step,
/// maintaining hidden state that captures the optimization trajectory. The recurrent
/// output modulates per-parameter learning rates, enabling adaptive step sizes that
/// evolve through the inner loop.
/// </para>
/// </remarks>
public class RecurrentHyperNetOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimension of the GRU hidden state. The recurrent cell processes
    /// compressed gradient features and maintains state across adaptation steps.
    /// Default: 32.
    /// </summary>
    public int HiddenStateDim { get; set; } = 32;

    /// <summary>
    /// Initial forget gate bias. Higher values encourage more memory retention
    /// in early training (standard LSTM/GRU convention). Default: 1.0.
    /// </summary>
    public double ForgetBias { get; set; } = 1.0;

    /// <summary>
    /// L2 regularization weight on recurrent cell state magnitude to prevent
    /// unbounded hidden state growth. Default: 0.001.
    /// </summary>
    public double CellRegWeight { get; set; } = 0.001;

    /// <summary>
    /// Gradient compression dimension for feeding into the recurrent cell.
    /// Default: 32.
    /// </summary>
    public int InputDim { get; set; } = 32;

    public RecurrentHyperNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
                              && HiddenStateDim > 0 && InputDim > 0;
    public IMetaLearnerOptions<T> Clone() => new RecurrentHyperNetOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        HiddenStateDim = HiddenStateDim, ForgetBias = ForgetBias,
        CellRegWeight = CellRegWeight, InputDim = InputDim
    };
}
