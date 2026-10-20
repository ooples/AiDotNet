using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo is a contrastive learning method that uses a momentum encoder
/// and a memory queue to provide a large pool of consistent negative samples without requiring
/// huge batch sizes.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>Momentum Encoder:</b> A slowly-updating copy of the encoder for consistent keys</item>
/// <item><b>Memory Queue:</b> Stores past embeddings as negative samples (65536 by default)</item>
/// <item><b>Asymmetric design:</b> Query from main encoder, keys from momentum encoder</item>
/// </list>
///
/// <para><b>How MoCo works:</b></para>
/// <list type="number">
/// <item>Pass query image through online encoder → query q</item>
/// <item>Pass key image through momentum encoder → positive key k+</item>
/// <item>Get negative keys k- from memory queue</item>
/// <item>Compute InfoNCE loss: pull q closer to k+, push away from k-</item>
/// <item>Update momentum encoder with EMA</item>
/// <item>Enqueue new keys, dequeue oldest</item>
/// </list>
///
/// <para><b>Reference:</b> He et al., "Momentum Contrast for Unsupervised Visual Representation
/// Learning" (CVPR 2020)</para>
///
/// <para><b>Best for:</b> Limited GPU memory, consistent negative samples.</para>
/// <para><b>Pros:</b> Memory efficient, consistent negative samples, good performance.</para>
/// <para><b>Cons:</b> More complex than SimCLR, requires a momentum encoder.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Momentum Contrast for Unsupervised Visual Representation Learning", "https://arxiv.org/abs/1911.05722", Year = 2020, Authors = "Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick")]
public partial class MoCo<T> : SelfSupervisedLearningMethodBase<T>
{
    [FrozenParameter]
    private readonly IMomentumEncoder<T> _momentumEncoder;
    [FrozenParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private readonly IProjectorHead<T>? _momentumProjector;
    private readonly IMemoryBank<T> _memoryBank;
    private readonly InfoNCELoss<T> _loss;
    private readonly SelfSupervisedLearningAugmentationPolicies<T> _augmentation;
    private readonly double _baseMomentum;
    private readonly double _temperature;

    // The keys of the step in flight, enqueued once the optimizer has stepped.
    [Scratch]
    private Tensor<T>? _pendingKeys;

    /// <inheritdoc />
    public override string Name => "MoCo";

    /// <inheritdoc />
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.Contrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => true;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Gets the memory bank used for negative samples.
    /// </summary>
    public IMemoryBank<T> MemoryBank => _memoryBank;

    /// <summary>
    /// Initializes a new instance of the MoCo class.
    /// </summary>
    /// <param name="encoder">The online encoder network.</param>
    /// <param name="momentumEncoder">The momentum encoder (copy of main encoder).</param>
    /// <param name="projector">Optional projection head for online encoder.</param>
    /// <param name="momentumProjector">Optional projection head for momentum encoder.</param>
    /// <param name="embeddingDim">Dimension of embeddings for memory bank.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MoCo(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> momentumEncoder,
        IProjectorHead<T>? projector = null,
        IProjectorHead<T>? momentumProjector = null,
        int embeddingDim = 128,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(encoder, projector, config ?? new SelfSupervisedLearningConfig<T>())
    {
        Guard.NotNull(momentumEncoder);
        _momentumEncoder = momentumEncoder;
        _momentumProjector = momentumProjector;

        var mocoConfig = _config.MoCo ?? new MoCoConfig();
        var queueSize = mocoConfig.QueueSize ?? 65536;
        _baseMomentum = mocoConfig.Momentum ?? 0.999;

        _memoryBank = new MemoryBank<T>(queueSize, embeddingDim, _config.Seed);

        var temperature = _config.Temperature ?? 0.07;
        _temperature = temperature;
        _loss = new InfoNCELoss<T>(temperature);
        _augmentation = new SelfSupervisedLearningAugmentationPolicies<T>(_config.Seed);
    }

    /// <summary>
    /// Creates a MoCo instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for momentum.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 128).</param>
    /// <param name="queueSize">Size of the memory queue (default: 65536).</param>
    /// <returns>A configured MoCo instance.</returns>
    public static MoCo<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 128,
        int queueSize = 65536)
    {
        Guard.NotNull(encoder);
        Guard.NotNull(createEncoderCopy);

        var projector = new LinearProjector<T>(encoderOutputDim, projectionDim);
        var momentumProjector = new LinearProjector<T>(encoderOutputDim, projectionDim);
        momentumProjector.SetParameters(projector.GetParameters());

        var encoderCopy = createEncoderCopy(encoder);
        var momentumEncoder = new MomentumEncoder<T>(encoderCopy, 0.999);

        var config = new SelfSupervisedLearningConfig<T>
        {
            MoCo = new MoCoConfig { QueueSize = queueSize }
        };

        return new MoCo<T>(encoder, momentumEncoder, projector, momentumProjector, projectionDim, config);
    }

    /// <inheritdoc />
    /// <remarks>
    /// He et al. 2020, Eq. 1: a query from the online encoder is classified against its key (the momentum encoder's
    /// view of the same image) among the queue's keys, l = -log exp(q.k+/t) / (exp(q.k+/t) + sum exp(q.k-/t)), with
    /// both L2-normalized. Keys carry no gradient: the momentum branch follows the online one by EMA. Until the queue
    /// holds keys, the batch's other keys are the negatives.
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var queries = ObjectiveOps.L2NormalizeRows(ProjectOnline(EncodeForTraining(view1)));
        var keys = Engine.StopGradient(ObjectiveOps.L2NormalizeRows(EncodeKeys(view2)));
        _pendingKeys = keys;

        if (_memoryBank.CurrentSize == 0)
            return _loss.ComputeLoss(queries, keys);

        int n = queries.Shape[0];
        var positive = Engine.ReduceSum(Engine.TensorMultiply(queries, keys), new[] { 1 }, keepDims: true); // [N, 1]
        var negatives = Engine.TensorMatMul(queries, _memoryBank.GetAll().Transpose(new[] { 1, 0 }));     // [N, K]
        var logits = Engine.TensorDivideScalar(Engine.Concat(new[] { positive, negatives }, 1), NumOps.FromDouble(_temperature));
        var logProbs = ObjectiveOps.LogSoftmax(logits, axis: 1);
        var positiveLogProb = Engine.TensorSlice(logProbs, new[] { 0, 0 }, new[] { n, 1 });
        return Engine.TensorNegate(Engine.TensorDivideScalar(
            Engine.ReduceSum(positiveLogProb, null, keepDims: false), NumOps.FromDouble(n)));
    }

    private Tensor<T> ProjectOnline(Tensor<T> features) => _projector is null ? features : _projector.Project(features);

    /// <summary>The momentum branch's keys for a view: its encoder, then its projector when it has one.</summary>
    protected Tensor<T> EncodeKeys(Tensor<T> view)
    {
        var features = _momentumEncoder.Encode(view);
        return _momentumProjector is null ? features : _momentumProjector.Project(features);
    }

    /// <inheritdoc />
    /// <remarks>The momentum encoder and projector follow the online ones by EMA; the step's keys join the queue.</remarks>
    protected override void AfterOptimizerStep()
    {
        _momentumEncoder.UpdateFromMainEncoder(_encoder);
        if (_projector is not null && _momentumProjector is not null)
        {
            double m = _momentumEncoder.Momentum;
            var online = _projector.GetParameters();
            var target = _momentumProjector.GetParameters();
            if (online.Length == target.Length)
            {
                var updated = new Vector<T>(target.Length);
                for (int i = 0; i < target.Length; i++)
                    updated[i] = NumOps.Add(NumOps.Multiply(target[i], NumOps.FromDouble(m)),
                        NumOps.Multiply(online[i], NumOps.FromDouble(1.0 - m)));
                _momentumProjector.SetParameters(updated);
            }
        }

        if (_pendingKeys is { } keys)
        {
            _memoryBank.Enqueue(keys);
            _pendingKeys = null;
        }
    }

    /// <inheritdoc />
    protected override (Tensor<T> View1, Tensor<T> View2) AugmentViews(
        SelfSupervisedLearningAugmentationPolicies<T> policies, Tensor<T> batch) => _augmentation.ApplyMoCoV2(batch);

    /// <inheritdoc />
    /// <remarks>The paper trains with SGD (momentum 0.9, learning rate 0.03, Sec. 4).</remarks>
    protected override SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.SGD;

    /// <inheritdoc />
    protected override double DefaultLearningRate => 0.03;

    private void UpdateEncoderParameters(T learningRate)
    {
        var encoderGrads = new Vector<T>(_encoder.GetParameterGradients());
        var encoderParams = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(encoderParams, Engine.Multiply(encoderGrads, learningRate)));

        if (_projector is not null)
        {
            var projGrads = new Vector<T>(_projector.GetParameterGradients());
            var projParams = _projector.GetParameters();
            _projector.SetParameters(Engine.Subtract(projParams, Engine.Multiply(projGrads, learningRate)));
        }
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _memoryBank.Clear();
    }

}
