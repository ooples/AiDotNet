using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo v3 adapts momentum contrastive learning specifically for
/// Vision Transformers (ViT). It simplifies the framework by removing the memory queue and
/// using in-batch negatives with a symmetric loss.</para>
///
/// <para><b>Key changes from MoCo v1/v2:</b></para>
/// <list type="bullet">
/// <item><b>No memory queue:</b> Uses in-batch negatives like SimCLR</item>
/// <item><b>Symmetric loss:</b> Both views serve as queries and keys</item>
/// <item><b>Prediction head:</b> Adds a predictor MLP on one branch</item>
/// <item><b>ViT optimizations:</b> Random patch projection, no BN in MLP heads</item>
/// </list>
///
/// <para><b>Training stability for ViT:</b></para>
/// <list type="bullet">
/// <item>Uses lower learning rates and careful initialization</item>
/// <item>Gradient clipping and careful warmup</item>
/// <item>Momentum encoder still provides stable targets</item>
/// </list>
///
/// <para><b>Reference:</b> Chen et al., "An Empirical Study of Training Self-Supervised Vision
/// Transformers" (ICCV 2021)</para>
///
/// <para><b>Best for:</b> Vision Transformers (ViT), modern architectures.</para>
/// <para><b>Pros:</b> Optimized for ViT, simpler than MoCo v1/v2.</para>
/// <para><b>Cons:</b> Best suited for transformer architectures.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("An Empirical Study of Training Self-Supervised Vision Transformers", "https://arxiv.org/abs/2104.02057", Year = 2021, Authors = "Xinlei Chen, Saining Xie, Kaiming He")]
public partial class MoCoV3<T> : SelfSupervisedLearningMethodBase<T>
{
    [FrozenParameter]
    private readonly IMomentumEncoder<T> _momentumEncoder;

    [FrozenParameter]
    private readonly IProjectorHead<T>? _momentumProjector;

    [TrainableParameter(Optional = true)]
    private readonly IProjectorHead<T>? _predictor;
    private readonly InfoNCELoss<T> _loss;
    private readonly double _temperature;
    private readonly SelfSupervisedLearningAugmentationPolicies<T> _augmentation;

    /// <inheritdoc />
    public override string Name => "MoCo v3";

    /// <inheritdoc />
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.Contrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false; // MoCo v3 doesn't use queue

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Initializes a new instance of the MoCoV3 class.
    /// </summary>
    /// <param name="encoder">The online encoder network (ViT recommended).</param>
    /// <param name="momentumEncoder">The momentum encoder.</param>
    /// <param name="projector">Projection head for online encoder.</param>
    /// <param name="momentumProjector">Projection head for momentum encoder.</param>
    /// <param name="predictor">Predictor head (applied to online branch only).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MoCoV3(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> momentumEncoder,
        IProjectorHead<T> projector,
        IProjectorHead<T> momentumProjector,
        IProjectorHead<T>? predictor = null,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(encoder, projector, config ?? CreateMoCoV3Config())
    {
        Guard.NotNull(momentumEncoder);
        _momentumEncoder = momentumEncoder;
        Guard.NotNull(momentumProjector);
        _momentumProjector = momentumProjector;
        _predictor = predictor;
        // The momentum projector tracks the online one by EMA, element for element, so the two must match in size.
        if (projector is not null && momentumProjector is not null && projector.ParameterCount != momentumProjector.ParameterCount)
            throw new ArgumentException(
                $"The momentum projector has {momentumProjector.ParameterCount} parameters; the projector it tracks has {projector.ParameterCount}.",
                nameof(momentumProjector));

        var temperature = _config.Temperature ?? 0.2; // MoCo v3 uses higher temperature
        _temperature = temperature;
        _loss = new InfoNCELoss<T>(temperature);
        _augmentation = new SelfSupervisedLearningAugmentationPolicies<T>(_config.Seed);
    }

    /// <summary>
    /// Creates a MoCo v3 instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder (ViT recommended).</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for momentum.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 256).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 4096).</param>
    /// <returns>A configured MoCo v3 instance.</returns>
    public static MoCoV3<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 256,
        int hiddenDim = 4096)
    {
        Guard.NotNull(encoder);
        Guard.NotNull(createEncoderCopy);

        var projector = new MLPProjector<T>(encoderOutputDim, hiddenDim, projectionDim);
        var momentumProjector = new MLPProjector<T>(encoderOutputDim, hiddenDim, projectionDim);
        momentumProjector.SetParameters(projector.GetParameters());

        var predictor = new MLPProjector<T>(projectionDim, hiddenDim / 4, projectionDim);

        var encoderCopy = createEncoderCopy(encoder);
        var momentumEncoder = new MomentumEncoder<T>(encoderCopy, 0.99);

        return new MoCoV3<T>(encoder, momentumEncoder, projector, momentumProjector, predictor);
    }

    private static SelfSupervisedLearningConfig<T> CreateMoCoV3Config()
    {
        return new SelfSupervisedLearningConfig<T>
        {
            Temperature = 0.2,
            LearningRate = 1.5e-4, // Lower LR for ViT
            WarmupEpochs = 40,
            UseCosineDecay = true,
            MoCo = new MoCoConfig
            {
                Momentum = 0.99,
                UseMLPProjector = true
            }
        };
    }

    /// <inheritdoc />
    /// <remarks>
    /// Chen, Xie and He 2021, Alg. 1: each view's query (encoder, projector, predictor) is contrasted in-batch with
    /// the other view's key from the momentum branch, L = ctr(q1, k2) + ctr(q2, k1), each scaled by 2t as in the
    /// official code. Keys carry no gradient.
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var q1 = Query(view1);
        var q2 = Query(view2);
        var k1 = Engine.StopGradient(Key(view1));
        var k2 = Engine.StopGradient(Key(view2));
        var total = Engine.TensorAdd(_loss.ComputeLoss(q1, k2), _loss.ComputeLoss(q2, k1));
        return Engine.TensorMultiplyScalar(total, NumOps.FromDouble(2 * _temperature));
    }

    private Tensor<T> Query(Tensor<T> view)
    {
        var z = EncodeForTraining(view);
        if (_projector is not null) z = _projector.Project(z);
        return _predictor is null ? z : _predictor.Project(z);
    }

    private Tensor<T> Key(Tensor<T> view)
    {
        var k = _momentumEncoder.Encode(view);
        return _momentumProjector is null ? k : _momentumProjector.Project(k);
    }

    /// <inheritdoc />
    /// <remarks>The predictor is part of the online network.</remarks>
    protected override IReadOnlyList<Tensor<T>> GetOnlineTrainableTensors()
    {
        var tensors = new List<Tensor<T>>(base.GetOnlineTrainableTensors());
        if (_predictor is not null) tensors.AddRange(_predictor.GetTrainableTensors());
        return tensors;
    }

    /// <inheritdoc />
    /// <remarks>The momentum encoder and projector follow the online ones by EMA.</remarks>
    protected override void AfterOptimizerStep()
    {
        _momentumEncoder.UpdateFromMainEncoder(_encoder);
        if (_projector is null || _momentumProjector is null) return;
        double m = _momentumEncoder.Momentum;
        var online = _projector.GetParameters();
        var target = _momentumProjector.GetParameters();
        var updated = new Vector<T>(target.Length);
        for (int i = 0; i < target.Length; i++)
            updated[i] = NumOps.Add(NumOps.Multiply(target[i], NumOps.FromDouble(m)),
                NumOps.Multiply(online[i], NumOps.FromDouble(1.0 - m)));
        _momentumProjector.SetParameters(updated);
    }

    /// <inheritdoc />
    protected override (Tensor<T> View1, Tensor<T> View2) AugmentViews(
        SelfSupervisedLearningAugmentationPolicies<T> policies, Tensor<T> batch) => _augmentation.ApplyBYOL(batch);

    /// <inheritdoc />
    /// <remarks>The paper trains ViT backbones with AdamW (Sec. 4).</remarks>
    protected override SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.AdamW;

    /// <inheritdoc />
    public override void OnEpochStart(int epochNumber)
    {
        base.OnEpochStart(epochNumber);

        // Update momentum schedule (cosine from base to 1.0)
        var mocoConfig = _config.MoCo ?? new MoCoConfig();
        var baseMomentum = mocoConfig.Momentum ?? 0.99;
        var totalEpochs = _config.PretrainingEpochs ?? 300;

        var newMomentum = MomentumEncoder<T>.ScheduleMomentum(
            baseMomentum, 1.0, epochNumber, totalEpochs);

        _momentumEncoder.SetMomentum(newMomentum);
    

}
}
