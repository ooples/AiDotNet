using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// BYOL: Bootstrap Your Own Latent - Self-supervised learning without negative samples.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BYOL is a breakthrough method that learns representations
/// without requiring negative samples. It uses an online network that learns to predict
/// the output of a target network, which is updated as an exponential moving average (EMA)
/// of the online network.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>No negatives:</b> Unlike SimCLR or MoCo, BYOL doesn't need negative samples</item>
/// <item><b>Asymmetric architecture:</b> Online has a predictor, target doesn't</item>
/// <item><b>EMA target:</b> Target network is a slow-moving average of online network</item>
/// <item><b>Symmetric loss:</b> Both views serve as online and target</item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Online: encoder → projector → predictor → p
/// Target: encoder → projector → z (stop-gradient)
/// Loss: MSE(normalize(p), normalize(z))
/// </code>
///
/// <para><b>Why it doesn't collapse:</b> The combination of the predictor (asymmetry),
/// EMA updates (target moves slowly), and batch normalization prevents trivial solutions.</para>
///
/// <para><b>Reference:</b> Grill et al., "Bootstrap Your Own Latent - A New Approach to
/// Self-Supervised Learning" (NeurIPS 2020)</para>
///
/// <para><b>Best for:</b> Avoiding negative sample mining, asymmetric networks.</para>
/// <para><b>Pros:</b> No negative samples needed, robust to batch size.</para>
/// <para><b>Cons:</b> Requires careful design to prevent collapse.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning", "https://arxiv.org/abs/2006.07733", Year = 2020, Authors = "Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko")]
public partial class BYOL<T> : SelfSupervisedLearningMethodBase<T>
{
    [FrozenParameter]
    private readonly IMomentumEncoder<T> _targetEncoder;

    [ParameterAlias(nameof(_projector))]
    private readonly SymmetricProjector<T> _onlineProjector;

    [FrozenParameter]
    private readonly SymmetricProjector<T> _targetProjector;
    private readonly BYOLLoss<T> _loss;
    private readonly SelfSupervisedLearningAugmentationPolicies<T> _augmentation;
    private readonly double _baseMomentum;

    /// <inheritdoc />
    public override string Name => "BYOL";

    /// <inheritdoc />
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.NonContrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Initializes a new instance of the BYOL class.
    /// </summary>
    /// <param name="encoder">The online encoder network.</param>
    /// <param name="targetEncoder">The target encoder (momentum-updated copy).</param>
    /// <param name="onlineProjector">Symmetric projector with predictor for online network.</param>
    /// <param name="targetProjector">Symmetric projector (no predictor) for target network.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public BYOL(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> targetEncoder,
        SymmetricProjector<T> onlineProjector,
        SymmetricProjector<T> targetProjector,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(encoder, onlineProjector, config ?? new SelfSupervisedLearningConfig<T>())
    {
        Guard.NotNull(targetEncoder);
        _targetEncoder = targetEncoder;
        Guard.NotNull(onlineProjector);
        _onlineProjector = onlineProjector;
        Guard.NotNull(targetProjector);
        _targetProjector = targetProjector;

        if (!_onlineProjector.HasPredictor)
            throw new ArgumentException("Online projector must have a predictor head", nameof(onlineProjector));

        var byolConfig = _config.BYOL ?? new BYOLConfig();
        _baseMomentum = byolConfig.BaseMomentum ?? 0.996;

        _loss = new BYOLLoss<T>();
        _augmentation = new SelfSupervisedLearningAugmentationPolicies<T>(_config.Seed);
    }

    private void UpdateTargetProjector()
    {
        // EMA of the target projector toward the online one. The online projector also carries the predictor, which
        // the target lacks; both flatten the projector first, so the target follows that leading part only. Blending
        // the whole online vector into the shorter target one mismatched their lengths.
        double momentum = _targetEncoder.Momentum;
        var onlineParams = _onlineProjector.GetParameters();
        var targetParams = _targetProjector.GetParameters();
        var updated = new Vector<T>(targetParams.Length);
        for (int i = 0; i < targetParams.Length; i++)
        {
            updated[i] = NumOps.Add(
                NumOps.Multiply(targetParams[i], NumOps.FromDouble(momentum)),
                NumOps.Multiply(onlineParams[i], NumOps.FromDouble(1.0 - momentum)));
        }

        _targetProjector.SetParameters(updated);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Grill et al. 2020, Eq. 2: the online network predicts the target network's projection of the other view,
    /// L = ||normalize(q(z(v1))) - normalize(sg(z'(v2)))||^2, symmetrized by swapping the views. The target branch
    /// (momentum encoder and target projector) receives no gradient; it follows by EMA after each step.
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var p1 = _onlineProjector.Predict(_onlineProjector.Project(EncodeForTraining(view1)));
        var p2 = _onlineProjector.Predict(_onlineProjector.Project(EncodeForTraining(view2)));
        var t1 = Engine.StopGradient(_targetProjector.Project(_targetEncoder.Encode(view1)));
        var t2 = Engine.StopGradient(_targetProjector.Project(_targetEncoder.Encode(view2)));
        return Engine.TensorAdd(_loss.ComputeLoss(p1, t2), _loss.ComputeLoss(p2, t1));
    }

    /// <inheritdoc />
    /// <remarks>The target network follows the online one by EMA at the scheduled momentum.</remarks>
    protected override void AfterOptimizerStep()
    {
        _targetEncoder.UpdateFromMainEncoder(_encoder);
        UpdateTargetProjector();
    }

    /// <inheritdoc />
    protected override (Tensor<T> View1, Tensor<T> View2) AugmentViews(
        SelfSupervisedLearningAugmentationPolicies<T> policies, Tensor<T> batch) => _augmentation.ApplyBYOL(batch);

    /// <inheritdoc />
    /// <remarks>The paper trains with LARS (Sec. 3.3).</remarks>
    protected override SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.LARS;

    /// <inheritdoc />
    public override void OnEpochStart(int epochNumber)
    {
        base.OnEpochStart(epochNumber);

        // Update momentum schedule (cosine from base to 1.0)
        var byolConfig = _config.BYOL ?? new BYOLConfig();
        var totalEpochs = _config.PretrainingEpochs ?? 300;

        var newMomentum = MomentumEncoder<T>.ScheduleMomentum(
            _baseMomentum, 1.0, epochNumber, totalEpochs);

        _targetEncoder.SetMomentum(newMomentum);
    }

    /// <summary>
    /// Creates a BYOL instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for target.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 256).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 4096).</param>
    /// <returns>A configured BYOL instance.</returns>
    public static BYOL<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 256,
        int hiddenDim = 4096)
    {
        // Create projectors
        var onlineProjector = new SymmetricProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, predictorHiddenDim: hiddenDim);
        var targetProjector = new SymmetricProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, predictorHiddenDim: 0);

        // Copy projector parameters (without predictor)
        CopyProjectorToTarget(onlineProjector, targetProjector);

        // Create target encoder
        var encoderCopy = createEncoderCopy(encoder);
        var targetEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new BYOL<T>(encoder, targetEncoder, onlineProjector, targetProjector);
    }

    private static void CopyProjectorToTarget(
        SymmetricProjector<T> online, SymmetricProjector<T> target)
    {
        // Copy only the projector parameters (target doesn't have predictor)
        var onlineParams = online.GetParameters();
        int targetParamCount = checked((int)target.ParameterCount);

        var targetParams = new T[targetParamCount];
        for (int i = 0; i < targetParamCount; i++)
        {
            targetParams[i] = onlineParams[i];
        }

        target.SetParameters(new Vector<T>(targetParams));
    

}
}
