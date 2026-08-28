using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch.
/// </summary>
/// <remarks>
/// <para>
/// Batch normalization helps stabilize and accelerate training by normalizing layer inputs.
/// It works by normalizing each feature to have zero mean and unit variance across the batch,
/// then applying learnable scale (gamma) and shift (beta) parameters.
/// </para>
/// <para>
/// Benefits include:
/// - Faster training convergence
/// - Reduced sensitivity to weight initialization
/// - Ability to use higher learning rates
/// - Acts as a form of regularization
/// </para>
/// <para><b>For Beginners:</b> Batch normalization is like standardizing test scores in a classroom.
/// 
/// Imagine a class where each student (input) has a raw test score. Batch normalization:
/// 1. Calculates the average score and how spread out the scores are
/// 2. Converts each score to show how many standard deviations it is from the average
/// 3. Applies adjustable scaling and shifting to the standardized scores
/// 
/// This helps neural networks learn more efficiently by:
/// - Keeping input values in a consistent range
/// - Reducing the "internal covariate shift" problem
/// - Making the network less sensitive to poor weight initialization
/// - Allowing higher learning rates without divergence
/// 
/// In practice, this means your network will typically train faster and perform better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, HasTrainingMode = true, IsStateful = true, TestInputShape = "1, 4", TestConstructorArgs = "")]
// Rescales values using batch statistics; never resizes, at any rank.
[ElementWiseShape(Note = "Normalises using batch statistics; every dimension is carried through.")]
[AutoParameters]
public partial class BatchNormalizationLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>Normalization rescales values; it never changes any axis.</remarks>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Identity;

    /// <summary>
    /// A small constant added to the variance for numerical stability.
    /// </summary>
    /// <remarks>
    /// This prevents division by zero when normalizing features with very small variance.
    /// Typical values are around 1e-5 to 1e-3.
    /// </remarks>
    private readonly T _epsilon;

    /// <summary>
    /// The momentum for updating running statistics.
    /// </summary>
    /// <remarks>
    /// Controls how much weight is given to the current batch versus previous batches
    /// when updating running statistics. Values closer to 1.0 give more weight to past
    /// statistics (slower updates).
    /// </remarks>
    private readonly T _momentum;

    /// <summary>
    /// The scale parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as gamma. This learnable parameter allows the network to scale
    /// each normalized feature. Initialized to ones.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams,
        Shape = "InputShape[0]")]

    private Tensor<T> _gamma;

    /// <summary>
    /// The shift parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as beta. This learnable parameter allows the network to shift
    /// each normalized feature. Initialized to zeros.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams,
        Shape = "InputShape[0]")]
    private Tensor<T> _beta;

    /// <summary>
    /// The running mean used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to zeros.
    /// </remarks>
    [Buffer(Name = "running_mean", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _runningMean;

    /// <summary>
    /// The running variance used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to ones.
    /// </remarks>
    [Buffer(Name = "running_variance", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _runningVariance;

    /// <summary>
    /// The input from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Tracks whether the last forward pass input was rank-1, so backward can preserve rank.
    /// </summary>
    private bool _inputWas1D;

    /// <summary>
    /// Stores the original input shape from forward pass so backward can restore it.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The batch mean from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _lastMean;

    /// <summary>
    /// The batch variance from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _lastVariance;

    /// <summary>
    /// The gradient of the loss with respect to gamma.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update gamma.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _gammaGradient;

    /// <summary>
    /// The gradient of the loss with respect to beta.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update beta.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _betaGradient;

    // GPU-resident cached tensors for GPU training pipeline
    [Scratch]
    private Tensor<T>? _lastInputGpu;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Batch normalization behaves differently during training versus inference:
    /// - During training: Uses statistics from the current batch
    /// - During inference: Uses running statistics collected during training
    /// </para>
    /// <para>
    /// This property always returns true because the layer needs to track its training state.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the network that this layer behaves differently during training versus testing.
    /// 
    /// During training, batch normalization uses statistics (mean and variance) calculated from
    /// the current batch of data. During testing or inference, it uses the average statistics
    /// collected during training.
    /// 
    /// This property being true means:
    /// - The layer needs to know whether it's in training or inference mode
    /// - The layer has parameters that can be updated during training
    /// - The layer's behavior will change depending on the mode
    /// 
    /// This is important because it affects how the network processes data and how
    /// the layer's internal statistics are updated.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Gets the gamma (scale) parameters of the batch normalization layer.
    /// </summary>
    /// <returns>The gamma tensor used for scaling normalized values.</returns>
    public Tensor<T> GetGamma()
    {
        return _gamma;
    }

    /// <summary>
    /// Gets the beta (shift) parameters of the batch normalization layer.
    /// </summary>
    /// <returns>The beta tensor used for shifting scaled values.</returns>
    public Tensor<T> GetBeta()
    {
        return _beta;
    }

    /// <summary>
    /// Initializes gamma (scale) parameters to zero.
    /// </summary>
    /// <remarks>
    /// This is used for zero-init residual in ResNet, where the last BatchNorm in each
    /// residual block has gamma initialized to zero. This makes the residual blocks
    /// start as identity mappings, which can improve training.
    /// </remarks>
    private bool _zeroInitGammaPending;

    public void ZeroInitGamma()
    {
        // ResNet zero-init residual (He et al. 2019). Defer when shape is
        // still lazy ([-1]); the resolution step picks up _zeroInitGammaPending.
        if (InputShape.Length == 0 || InputShape[0] <= 0)
        {
            _zeroInitGammaPending = true;
            return;
        }
        // Zero out the existing _gamma in place rather than replacing the field
        // with a fresh tensor. Replacement would orphan any existing trainable-
        // parameter registration (RegisterTrainableParameter holds the original
        // ref) and break the parameter buffer's view alignment if a buffer was
        // already built around the old tensor.
        if (_gamma is { Length: > 0 })
        {
            var span = _gamma.Data.Span;
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.Zero;
            return;
        }

        // Lazy / placeholder _gamma path. Re-run the standard initialization
        // sequence so we end up with a fully wired layer:
        //   - all four state tensors (_gamma, _beta, _runningMean,
        //     _runningVariance) sized to InputShape[0] with their canonical
        //     defaults
        //   - _gamma + _beta registered with RegisterTrainableParameter so
        //     the parameter buffer + weight registry pick them up
        // Only then zero _gamma in place. Earlier code created a fresh
        // _gamma tensor and skipped registration / _beta init, so a layer
        // that hit ZeroInitGamma before its first forward ended up with
        // _gamma trainable but unregistered, _beta still at the placeholder
        // length 0, and the running stats absent — which silently produced
        // identity-like normalization once Forward ran.
        InitializeNormalizationParameters();
        if (_gamma.Length > 0)
        {
            var span = _gamma.Data.Span;
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Allocates _gamma / _beta / _runningMean / _runningVariance to match
    /// the resolved InputShape and registers gamma + beta as trainable.
    /// Idempotent: tensors already at the right length are reused so the
    /// existing RegisterTrainableParameter registrations stay valid.
    /// </summary>
    /// <summary>
    /// Sizes gamma, beta and the running statistics from the resolved channel count, and registers
    /// the running statistics as buffers.
    /// </summary>
    /// <remarks>
    /// The tape swap rebinds gamma and beta directly, so a trained layer could reach serialization
    /// with correctly-sized affine parameters and running statistics that had never been allocated
    /// or registered — it wrote 48 values where a freshly constructed clone expected 96, and the
    /// checkpoint would not load. Materializing here means whatever reads the parameter surface
    /// sees the layer's full persistent state, affine and statistics alike.
    /// </remarks>
    protected override void EnsureInitialized()
    {
        // Guarded: an unresolved layer still carries the -1 sentinel and would allocate against it.
        if (IsShapeResolved && InputShape is { Length: > 0 } && InputShape[0] > 0)
            InitializeNormalizationParameters();

        base.EnsureInitialized();
    }

    private void InitializeNormalizationParameters()
    {
        int channels = InputShape[0];

        // The running statistics are sized and registered INDEPENDENTLY of gamma. They used to
        // share gamma's reinit guard, which made them a casualty of the tape-buffer swap: that
        // swap rebinds _gamma to a correctly-sized view, so the next call here saw gamma already
        // at `channels`, took no branch, and left the running stats unallocated and unregistered.
        // A trained model then serialized 48 values where the freshly-constructed clone expected
        // 96, and the checkpoint would not load. Splitting the conditions removes the coupling;
        // RegisterBuffer is name-keyed and idempotent, so registering on every call is free.
        if (_gamma is null || _gamma.Length != channels)
        {
            _gamma = Tensor<T>.CreateDefault([channels], NumOps.One);
            _beta = Tensor<T>.CreateDefault([channels], NumOps.Zero);
            RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
            RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
        }

        if (_runningMean is null || _runningMean.Length != channels)
        {
            _runningMean = TensorAllocator.RentPinned<T>([channels]);
            _runningMean.Fill(NumOps.Zero);
            _runningVariance = TensorAllocator.RentPinned<T>([channels]);
            _runningVariance.Fill(NumOps.One);
        }

        RegisterRunningStatisticBuffers();
    }

    /// <summary>
    /// Registers BatchNorm's running statistics as persistent, non-trainable
    /// buffers. Lazy initialization can occur inside a training arena, so leaving
    /// these tensors unregistered allows arena/tape cleanup to recycle storage that
    /// must survive into the subsequent inference-mode generator pass.
    /// </summary>
    private void RegisterRunningStatisticBuffers()
    {
        RegisterBuffer(_runningMean, "running_mean");
        RegisterBuffer(_runningVariance, "running_variance");
    }

    /// <summary>
    /// Gets the running mean of the batch normalization layer.
    /// </summary>
    /// <returns>The running mean tensor used during inference.</returns>
    public Tensor<T> GetRunningMean()
    {
        return _runningMean;
    }

    /// <summary>
    /// Gets the running variance of the batch normalization layer.
    /// </summary>
    /// <returns>The running variance tensor used during inference.</returns>
    public Tensor<T> GetRunningVariance()
    {
        return _runningVariance;
    }
    /// <summary>
    /// Gets the epsilon value used for numerical stability.
    /// </summary>
    /// <returns>The epsilon value.</returns>
    public T GetEpsilon()
    {
        return _epsilon;
    }

    /// <summary>
    /// Gets the momentum value for running statistics.
    /// </summary>
    /// <returns>The momentum value.</returns>
    public T GetMomentum()
    {
        return _momentum;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Epsilon"] = NumOps.ToDouble(_epsilon).ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            ["Momentum"] = NumOps.ToDouble(_momentum).ToString("R", System.Globalization.CultureInfo.InvariantCulture)
        };
    }

    /// <inheritdoc />
    /// <remarks>Per-channel centering cancels a preceding per-channel bias exactly.</remarks>
    public override bool MakesUpstreamBiasRedundant => true;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BatchNormalizationLayer class.
    /// </summary>
    /// <param name="numFeatures">The number of features (neurons) to normalize.</param>
    /// <param name="epsilon">A small constant added to the variance for numerical stability (default: 1e-5).</param>
    /// <param name="momentum">The momentum for updating running statistics (default: 0.9).</param>
    /// <remarks>
    /// <para>
    /// The epsilon parameter prevents division by zero when normalizing features with very small variance.
    /// </para>
    /// <para>
    /// The momentum parameter controls how much the running statistics are updated during training:
    /// - Values closer to 1.0 give more weight to past batches (slower updates)
    /// - Values closer to 0.0 give more weight to the current batch (faster updates)
    /// </para>
    /// <para>
    /// A typical value is 0.9, which means each new batch contributes about 10% to the running statistics.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a batch normalization layer with the specified settings.
    /// 
    /// When creating a BatchNormalizationLayer:
    /// - numFeatures: How many features (neurons) this layer will normalize
    /// - epsilon: A small number (like 0.00001) to prevent division by zero
    /// - momentum: How quickly running statistics are updated (0.9 means 90% old + 10% new)
    /// 
    /// For example, in a neural network for image classification:
    /// ```csharp
    /// // Create a batch normalization layer for 128 features
    /// var batchNormLayer = new BatchNormalizationLayer<float>();
    /// ```
    /// 
    /// The layer initializes with:
    /// - Scale parameters (gamma) set to 1.0
    /// - Shift parameters (beta) set to 0.0
    /// - Running statistics (mean and variance) initialized to 0.0 and 1.0
    /// </para>
    /// </remarks>
    /// <summary>
    /// How this layer should read a rank-3 input. <see cref="BatchNormDataLayout.Infer"/> by default,
    /// which is the historical behaviour.
    /// </summary>
    /// <remarks>
    /// Rank 3 is genuinely ambiguous -- [C, H, W] unbatched image versus [B, C, T] batched
    /// channels-first sequence -- and the layer cannot tell them apart from the shape alone. Rather
    /// than guess, callers that know say so. Opt-in: leaving this at Infer changes nothing.
    /// </remarks>
    public BatchNormDataLayout Layout { get; set; } = BatchNormDataLayout.Infer;

    public BatchNormalizationLayer(double epsilon = NumericalStabilityHelper.LargeEpsilon, double momentum = 0.9)
        : base(new[] { -1 }, new[] { -1 })
    {
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _momentum = NumOps.FromDouble(momentum);
        // Lazy: gamma/beta/running stats sized on first forward from input channel count
        // (input.Shape[1] for rank>=2 channels-first NCHW, input.Length for rank-1 input).
        _gamma = new Tensor<T>([0]);
        _beta = new Tensor<T>([0]);
        _runningMean = new Tensor<T>([0]);
        _runningVariance = new Tensor<T>([0]);
    }

    /// <summary>Construction state: the 'numFeatures' the layer was built with.</summary>
    private readonly int _numFeatures;

    /// <summary>
    /// AiDotNet#1370 eager-init constructor. Pass <paramref name="numFeatures"/> at
    /// construction (the channel count for image-like inputs OR the feature count
    /// for MLP inputs) to allocate gamma/beta/running stats immediately and resolve
    /// the layer's input + output shapes. Eliminates the need for a warmup forward
    /// pass before downstream consumers (LoRA wrapping, parameter introspection,
    /// ONNX export) can read shape-dependent state.
    /// </summary>
    /// <param name="numFeatures">
    /// The channel count for image inputs (axis 1 of NCHW) or feature count for MLP
    /// inputs (axis 1 of [B, F]). Must be positive. Per Ioffe &amp; Szegedy 2015 §3,
    /// BatchNorm normalizes per-channel for images and per-feature for MLPs.
    /// </param>
    /// <param name="epsilon">A small value added to the variance for numerical stability.</param>
    /// <param name="momentum">EMA momentum for running mean/variance updates.</param>
    /// <remarks>
    /// <para>
    /// After this constructor returns, <see cref="LayerBase{T}.IsShapeResolved"/> is
    /// <c>true</c> and <see cref="LayerBase{T}.TryDeclareShape"/> returns <c>true</c>
    /// via the default implementation — no override needed on this layer.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">When <paramref name="numFeatures"/> is not positive.</exception>
    public BatchNormalizationLayer(
        int numFeatures,
        double epsilon = NumericalStabilityHelper.LargeEpsilon,
        double momentum = 0.9)
        : base(new[] { numFeatures }, new[] { numFeatures })
    {
        _numFeatures = numFeatures;
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures),
                $"numFeatures must be positive, got {numFeatures}.");

        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _momentum = NumOps.FromDouble(momentum);

        // Eager allocation — same code path as OnFirstForward but driven from ctor.
        // ZeroInitGamma deferral does not apply here (no first-forward to defer to).
        _gamma = new Tensor<T>([numFeatures]);
        _gamma.Fill(NumOps.One);
        _beta = new Tensor<T>([numFeatures]);
        _beta.Fill(NumOps.Zero);
        _runningMean = TensorAllocator.RentPinned<T>([numFeatures]);
        _runningMean.Fill(NumOps.Zero);
        _runningVariance = TensorAllocator.RentPinned<T>([numFeatures]);
        _runningVariance.Fill(NumOps.One);

        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
        RegisterRunningStatisticBuffers();
    }

    /// <summary>
    /// Resolves <c>numFeatures</c> on the first forward call by switching on
    /// the input rank, allocates gamma/beta + running mean/variance tensors,
    /// and registers gamma/beta as trainable parameters. Per the BatchNorm
    /// paper (Ioffe &amp; Szegedy 2015 §3), normalization is per-feature for
    /// rank-1/2 MLP inputs and per-channel for rank-≥4 NCHW image batches.
    /// Rank-3 is treated as channels-first <c>[C, H, W]</c> (unbatched
    /// image, the layout that surfaces during pre-resolve walks of CNN
    /// architectures).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Rank-3 ambiguity:</b> rank-3 input could plausibly mean either
    /// <c>[C, H, W]</c> (channels-first unbatched image) OR <c>[B, S, F]</c>
    /// (features-last batched sequence). We resolve to channels-first per
    /// Ioffe &amp; Szegedy 2015 — paper-faithful BN is per-channel for
    /// image inputs, and sequence/transformer models use LayerNorm
    /// (Ba et al. 2016) not BN.
    /// </para>
    /// <para>
    /// The Forward path at line ~502 handles features-last layouts at
    /// runtime by checking <c>input.Shape[^1] == featureSize</c> and
    /// flattening to <c>[B*..., F]</c> — but that auto-flatten only works
    /// once <c>featureSize</c> is already resolved. The very first forward
    /// call must commit to one interpretation. We pick channels-first to
    /// match the paper; callers using BN with <c>[B, S, F]</c> sequence
    /// inputs should either (a) instantiate the layer with an explicit
    /// known feature count (via <c>ResolveFromShape</c>) before the first
    /// forward, or (b) use LayerNorm instead.
    /// </para>
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        // Per Ioffe & Szegedy 2015 §3 ("Batch Normalization"), BN normalizes per
        // *channel* for image-like inputs. Channel position depends on input
        // rank:
        //   rank 1 [F]                — features in axis 0 (e.g., a flat vector)
        //   rank 2 [B, F]             — features in axis 1 (the standard MLP layout)
        //   rank 3 [C, H, W]          — channels in axis 0 (an unbatched image, no batch dim)
        //   rank ≥ 4 [B, C, H, W, …]  — channels in axis 1 (the canonical NCHW image batch)
        //
        // Rank 3 is the case that surfaces during pre-resolve walks of CNN
        // architectures (ConvolutionalLayer.GetOutputShape returns rank-3
        // [C, H, W] without batch). Without this disambiguation, the prior
        // line `numFeatures = input.Shape[1]` picked the H dim and sized
        // _gamma / _beta / _runningMean / _runningVariance to H instead of C
        // — the Forward path then OOM'd or threw a broadcast error on the
        // first real `[B, C, H, W]` input ("scale [1, H, 1, 1] cannot be
        // broadcast against [B, C, H, W]"). BN is only ever applied per-
        // channel by paper-faithful CNN architectures (Conv → BN → ReLU);
        // sequence models / transformers use LayerNorm per Ba et al. 2016,
        // not BN, so there's no rank-3 ambiguity to mis-route here.
        BindToActualInput(input, preserveExistingValues: true);
    }

    /// <summary>
    /// Rebinds a lazy BatchNorm to the channel width observed by its first real forward when a
    /// prior shape-only network walk used an approximate sequential shape.
    /// </summary>
    /// <remarks>
    /// Only the shape-only provenance path reaches this hook. Explicitly-sized BatchNorm layers
    /// remain strict; an architecture-defined width is never silently changed.
    /// </remarks>
    protected override void ReconcileShapeOnlyResolution(Tensor<T> input)
    {
        BindToActualInput(input, preserveExistingValues: false);
    }

    private void BindToActualInput(Tensor<T> input, bool preserveExistingValues)
    {
        int rank = input.Shape.Length;
        int numFeatures = rank switch
        {
            1 => input.Length,        // [F]
            2 => input.Shape[1],      // [B, F]
            3 => input.Shape[0],      // [C, H, W] — channels-first, unbatched
            _ => input.Shape[1],      // [B, C, H, W, …] — NCHW batched
        };
        if (numFeatures <= 0)
        {
            throw new ArgumentException(
                $"BatchNormalizationLayer cannot resolve numFeatures: derived dim = {numFeatures}.",
                nameof(input));
        }

        // Correctly SIZED is not the same question as correctly REGISTERED, and this guard
        // replaced WeightsAlreadyAllocated, which answered both. Gamma can already be the right
        // length while the trainable registry is still empty -- a shape-only walk resolves the
        // width without allocating, and a deserialize installs values into the fields before any
        // forward has registered them. Comparing lengths alone skipped the block in exactly those
        // cases, so RegisterTrainableParameter never ran and the layer reported zero registered
        // parameters against a layout that carried two ("has 0 registered parameters but
        // received 2" on every clone/deserialize through BatchNorm).
        //
        // Re-entering the block when registration is missing is safe for trained values: the
        // preserveExistingValues copy below carries them into the new tensors whenever the widths
        // already agree, which is the whole reason that copy exists.
        bool affineParametersRegistered = RegisteredTrainableParameterCount >= 2;
        bool needsResize = !affineParametersRegistered
            || _gamma.Length != numFeatures
            || _beta.Length != numFeatures
            || _runningMean.Length != numFeatures
            || _runningVariance.Length != numFeatures;

        // Apply deferred ZeroInitGamma if requested before shape resolution.
        T gammaInit = _zeroInitGammaPending ? NumOps.Zero : NumOps.One;
        _zeroInitGammaPending = false;
        // Norm-layer params are channel-sized (small) — streaming-pool
        // pre-eviction barely moves the needle here, but we route through
        // AllocateLazyWeight for consistency with the rest of the
        // streaming-aware layers and to keep the contract simple. Then
        // fill gamma with the deferred init value (zero-init for
        // post-residual BN, one-init otherwise) and runningVariance with 1.
        // Idempotent: don't re-allocate/re-init gamma/beta/running-stats a clone/deserialize already
        // installed — re-filling gamma and running-variance would drop trained normalization params
        // (#1221 Clone_AfterTraining). See Conv1DLayer.
        if (needsResize)
        {
            var oldGamma = _gamma;
            var oldBeta = _beta;
            var gamma = AllocateLazyWeight([numFeatures]);
            var beta = AllocateLazyWeight([numFeatures]);
            gamma.Fill(gammaInit);

            // A clone/deserialization path may already carry correctly-sized trained values.
            // Shape-only reconciliation, by contrast, must not project values from a guessed
            // width onto a different real channel contract.
            if (preserveExistingValues && oldGamma.Length == numFeatures && oldBeta.Length == numFeatures)
            {
                oldGamma.Data.Span.CopyTo(gamma.AsWritableSpan());
                oldBeta.Data.Span.CopyTo(beta.AsWritableSpan());
            }

            if (!ReplaceTrainableParameter(oldGamma, gamma, PersistentTensorRole.NormalizationParams))
                RegisterTrainableParameter(gamma, PersistentTensorRole.NormalizationParams);
            if (!ReplaceTrainableParameter(oldBeta, beta, PersistentTensorRole.NormalizationParams))
                RegisterTrainableParameter(beta, PersistentTensorRole.NormalizationParams);

            _gamma = gamma;
            _beta = beta;
            // Running statistics are persistent buffers, not scratch. Lazy
            // initialization happens during the first training forward while a
            // TensorArena is active, so use the pinned allocator explicitly;
            // otherwise arena disposal can recycle these buffers before the GAN's
            // subsequent eval-mode discriminator pass.
            _runningMean = TensorAllocator.RentPinned<T>([numFeatures]);
            _runningVariance = TensorAllocator.RentPinned<T>([numFeatures]);
            _runningVariance.Fill(NumOps.One);
            RegisterRunningStatisticBuffers();

            _gammaGradient = null;
            _betaGradient = null;
            _gammaVelocity = null;
            _betaVelocity = null;
            ResetState();
        }

        // BatchNorm is SHAPE-PRESERVING (output shape == input shape). For image
        // inputs (rank ≥ 3) resolve the layer's declared I/O shapes to the
        // per-sample spatial shape [C,H,W], NOT the rank-1 [numFeatures] the
        // learnable params use. Declaring rank-1 for an image broke the pre-forward
        // shape-resolution walk (NeuralNetworkBase.ResolveLazyLayerShapes): a
        // Conv→BN→Conv stack fed the BN's rank-1 GetOutputShape to the next conv,
        // which threw ("expects rank-3/4, got rank 1"), aborting resolution for every
        // downstream conv. Those convs then reported the InputDepth=1 PLACEHOLDER
        // ParameterCount, making whole CNN backbones under-report their size by orders
        // of magnitude (MaskDINO: 228K vs the real ~207M) — which in turn blinded every
        // ParameterCount-gated decision (weight-streaming auto-detect, ShouldUseStreamingTraining,
        // ConfigureInferenceForScale) so the memory-management stack never engaged for
        // exactly the foundation-scale models that need it. Vector inputs (rank ≤ 2)
        // keep the original rank-1 [numFeatures] declaration (the MLP/[B,F] convention),
        // so this only touches the image path. The channel-sized learnable params
        // (_gamma/_beta/_runningMean/_runningVariance) stay [numFeatures] above; only
        // the layer's transformation shape becomes rank-preserving here. Matches
        // ConvolutionalLayer.GetOutputShape's batch-less rank-3 [C,H,W] convention.
        int[] fullShape = input.Shape.ToArray();
        int[] ioShape;
        if (rank <= 2)
        {
            ioShape = new[] { numFeatures };          // [F] / [B,F] → [numFeatures] (unchanged)
        }
        else if (rank == 3)
        {
            ioShape = fullShape;                      // [C,H,W] (unbatched image)
        }
        else
        {
            // [B,C,H,W,…] → [C,H,W,…] (strip the leading batch dim). Manual copy
            // rather than the range operator fullShape[1..], which needs
            // RuntimeHelpers.GetSubArray and does not compile on net471.
            ioShape = new int[fullShape.Length - 1];
            System.Array.Copy(fullShape, 1, ioShape, 0, ioShape.Length);
        }
        ResolveShapes(ioShape, (int[])ioShape.Clone());
    }

    /// <summary>
    /// Performs the forward pass of batch normalization.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, featureSize].</param>
    /// <returns>The normalized, scaled, and shifted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass performs these steps:
    /// 1. If in training mode:
    ///    - Compute mean and variance of the current batch
    ///    - Update running statistics for inference
    ///    - Normalize using batch statistics
    /// 2. If in inference mode:
    ///    - Normalize using running statistics collected during training
    /// 3. Apply scale (gamma) and shift (beta) parameters
    /// </para>
    /// <para>
    /// The normalization formula is: y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    /// </para>
    /// <para><b>For Beginners:</b> This method normalizes the input data and applies learned scaling and shifting.
    ///
    /// During the forward pass, this method:
    ///
    /// 1. Saves the input for later use in backpropagation
    /// 2. If in training mode:
    ///    - Calculates the mean and variance of each feature across the batch
    ///    - Updates the running statistics for use during inference
    ///    - Normalizes the data using the batch statistics
    /// 3. If in inference/testing mode:
    ///    - Uses the running statistics collected during training
    /// 4. Applies the learned scale (gamma) and shift (beta) parameters
    ///
    /// The normalization makes each feature have approximately zero mean and unit variance,
    /// while the scale and shift parameters allow the network to learn the optimal
    /// distribution for each feature.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // Store original shape for backward pass restoration
        _originalInputShape = input._shape;

        // Auto-reshape 1D input to [1, N] for batch normalization compatibility
        _inputWas1D = input.Shape.Length == 1;
        if (_inputWas1D)
        {
            input = Engine.Reshape(input, [1, input.Length]);
        }

        // Features-last flatten: when the layer was constructed with a per-feature
        // gamma/beta contract (sized [featureSize]) and the input arrives rank-3+
        // with features in the LAST axis (the [batch, seq, features] transformer /
        // sequence-model layout), flatten all leading axes into one batch axis
        // before calling Engine.BatchNorm. Otherwise the engine dispatches 3D
        // input to its channels-first BatchNorm3D path ([channels, H, W] image
        // convention) and returns gradients sized [_shape[0]] that don't match
        // our [featureSize] gamma/beta — exactly the smoke-suite regression that
        // surfaced once AiDotNet.Tensors 0.53.1 shipped the correct 3D BN
        // backward.
        //
        // Triggers only when the last axis actually equals featureSize; if a
        // caller is genuinely using channels-first 3D/4D layout with per-channel
        // gamma, we leave input untouched and let the engine's channel-aware
        // paths handle it.
        bool flattenedFeaturesLast = false;
        int[]? preFlattenShape = null;
        int featureSize = _gamma.Length;

        // A caller that has DECLARED channels-first never takes the features-last flatten, even
        // when the trailing axis happens to equal featureSize. That coincidence is not rare and it
        // is not harmless: ContextNet's [1, 32, 32] activations have T == C exactly, so this branch
        // fired and reshaped to [32, 32] with CHANNELS as rows and TIME as columns -- computing
        // mean/variance across channels at each time index, indexing gamma/beta by time, and
        // EMA-updating the running statistics from those transposed moments. Five BN layers of that
        // drive the running variance toward zero; eval-mode then divides by sqrt(~0 + 1e-5), and
        // MSE squares the result. That is the NaN.
        //
        // Rank alone cannot tell [B, C, T] from an unbatched [C, H, W] image, so the layer does not
        // try to guess -- the caller declares it. Default is Infer, which is exactly today's
        // behaviour, so nothing changes for any existing caller.
        bool channelsFirstDeclared = Layout == BatchNormDataLayout.ChannelsFirst;

        if (!channelsFirstDeclared && input.Rank >= 3 && featureSize > 0 && input.Shape[^1] == featureSize)
        {
            preFlattenShape = input._shape;
            int leadingBatch = 1;
            for (int i = 0; i < input.Rank - 1; i++) leadingBatch *= input.Shape[i];
            input = Engine.Reshape(input, [leadingBatch, featureSize]);
            flattenedFeaturesLast = true;
        }

        // _lastInput is layer-side activation retention for a backward path
        // that's never reached when training goes through the tape (tape
        // holds its own intermediate refs already). Skip the assignment
        // when a tape is active so this field doesn't double-root the
        // input activation; null it out so the previous step's tensor is
        // eligible for collection.
        bool tapeActive = AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
            && !AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed;
        _lastInput = tapeActive ? null : input;

        // A single-sample batch (batch size 1) has zero batch variance, so the
        // training-mode normalization (x - mean)/sqrt(var + eps) collapses every
        // feature to 0 → the output is a constant (≈ beta) that is INDEPENDENT of
        // the input and of upstream parameters. Its gradient is therefore zero,
        // which silently detaches the autodiff tape and stops the entire model
        // from learning (surfaced by GradientFlow_ShouldBeNonZeroAndFinite on every
        // BatchNorm model trained one sample at a time). Batch statistics are
        // undefined for a single sample, so fall back to the affine
        // running-statistics path — identical to inference — which is
        // differentiable end-to-end and lets gradients reach the input and the
        // affine parameters. Real training uses batch > 1 and is unaffected.
        int effectiveBatchSize = input.Rank > 0 ? input.Shape[0] : 1;
        if (IsTrainingMode && effectiveBatchSize > 1)
        {
            // Training: Use Engine.BatchNorm to compute batch stats and normalize
            // This is fully GPU accelerated
            var output = Engine.BatchNorm(input, _gamma, _beta, NumOps.ToDouble(_epsilon), out var batchMean, out var batchVariance);

            _lastMean = batchMean;
            _lastVariance = batchVariance;

            // Ensure batch statistics match running statistics shape
            // Engine.BatchNorm may return different shapes based on input configuration
            if (batchMean.Length != _runningMean.Length)
            {
                // Reshape batch statistics to match running statistics
                // This handles cases where input shape doesn't match expected configuration
                var newMeanData = new T[_runningMean.Length];
                var newVarData = new T[_runningVariance.Length];

                // Copy what we can, using first value for padding if needed
                int copyLen = Math.Min(batchMean.Length, _runningMean.Length);
                T meanFillValue = copyLen > 0 ? batchMean.Data.Span[0] : NumOps.Zero;
                T varFillValue = copyLen > 0 ? batchVariance.Data.Span[0] : NumOps.One;

                for (int i = 0; i < _runningMean.Length; i++)
                {
                    newMeanData[i] = i < copyLen ? batchMean.Data.Span[i] : meanFillValue;
                    newVarData[i] = i < copyLen ? batchVariance.Data.Span[i] : varFillValue;
                }

                batchMean = new Tensor<T>(_runningMean._shape, new Vector<T>(newMeanData));
                batchVariance = new Tensor<T>(_runningVariance._shape, new Vector<T>(newVarData));
            }

            // Issue #350 v3: in-place form so the lazy chain captured at
            // CompiledTrainingPlan trace time replays correctly across
            // Step()s. The prior out-of-place form pinned the INITIAL
            // _runningMean reference; every replay computed
            // momentum*init + (1-momentum)*batch instead of
            // momentum*previous + (1-momentum)*batch, so running stats
            // stayed at one EMA step off initial. BatchNormInference
            // (Predict) then divided by sqrt(~0+eps) ≈ 316 per BN layer
            // and blew up the 53-layer pyramid output by ~1e7×. The
            // in-place ops are GraphMode-aware (CpuEngine.cs:2916+ +
            // LazyTensorScope.RecordInPlace) so each replay re-applies
            // the mutation — EMA accumulates correctly under both eager
            // and compiled execution.
            // Running statistics are persistent buffers, not trainable state. Keep
            // their EMA update outside the autodiff graph, matching the semantics of
            // PyTorch BatchNorm buffers. GraphMode still records the in-place
            // mutations for compiled-plan replay; NoGradScope only suppresses tape
            // recording. Without this boundary the batch-variance EMA can be retained
            // as differentiable state and corrupted when the training tape is released.
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                T oneMinusMomentum = NumOps.Subtract(NumOps.One, _momentum);
                Engine.TensorMultiplyScalarInPlace(_runningMean, _momentum);
                var scaledBatchMean = Engine.TensorMultiplyScalar(batchMean, oneMinusMomentum);
                Engine.TensorAddInPlace(_runningMean, scaledBatchMean);
                Engine.TensorMultiplyScalarInPlace(_runningVariance, _momentum);
                var scaledBatchVar = Engine.TensorMultiplyScalar(batchVariance, oneMinusMomentum);
                Engine.TensorAddInPlace(_runningVariance, scaledBatchVar);
            }

            // Restore pre-flatten rank if we collapsed leading axes for the
            // features-last transformer path above. Tape-recorded reshape so
            // backward flows through unchanged.
            if (flattenedFeaturesLast && preFlattenShape is not null)
            {
                output = Engine.Reshape(output, preFlattenShape);
            }

            // Preserve original rank
            if (_inputWas1D)
            {
                output = Engine.Reshape(output, [output.Length]);
            }

            return output;
        }
        else if (IsTrainingMode)
        {
            // #639: batch=1 TRAINING fallback. Batch variance is undefined for a single
            // sample, so we normalize with the running statistics (same VALUE as inference)
            // — but route it through the single differentiable BatchNormAffine engine op
            // instead of the manual sqrt/divide/subtract/broadcast decomposition. Two wins:
            //   1. Op-count: the compiled-plan replay records ONE op per BN layer instead of
            //      the ~4-op cached-scale broadcast chain (the batch=1 op explosion in #639).
            //   2. Correctness: BatchNormAffine carries an exact backward to x, gamma AND beta
            //      every step. The cached-scale path below can DETACH the gamma/beta gradient
            //      whenever inference scale/shift are reused from a prior
            //      step (they were built off-tape), silently zeroing the affine-parameter grads.
            // mean/variance are constant running stats here (batch=1 never updates them), so the
            // captured references stay valid across compiled-plan replays.
            _lastMean = _runningMean;
            _lastVariance = _runningVariance;

            // Compute the affine scale/shift ON-TAPE every step (NOT cached) so gamma and beta keep
            // their gradients through this batch=1 training fallback — the property #639 wanted from the
            // fused Engine.BatchNormAffine. That fused op is not yet in the referenced AiDotNet.Tensors
            // package, so use the manual decomposition (gradient-equivalent: gamma/beta stay live, and
            // ApplyInferenceAnyRank applies the channel-broadcast multiply-add for any rank). Bump the
            // package and restore Engine.BatchNormAffine once it ships (#639).
            var epsilonVec = Tensor<T>.CreateDefault(_runningVariance._shape, _epsilon);
            var stdDev = Engine.TensorSqrt(Engine.TensorAdd(_runningVariance, epsilonVec));
            var affineScale = Engine.TensorDivide(_gamma, stdDev);
            var affineShift = Engine.TensorSubtract(
                _beta, Engine.TensorDivide(Engine.TensorMultiply(_gamma, _runningMean), stdDev));
            var output = ApplyInferenceAnyRank(input, affineScale, affineShift);

            // Restore pre-flatten rank for the features-last path.
            if (flattenedFeaturesLast && preFlattenShape is not null)
            {
                output = Engine.Reshape(output, preFlattenShape);
            }

            // Preserve original rank.
            if (_inputWas1D)
            {
                output = Engine.Reshape(output, [output.Length]);
            }

            return output;
        }
        else
        {
            // Inference: Use running statistics
            // output = gamma * (input - runningMean) / sqrt(runningVar + epsilon) + beta

            // Cache running stats for backward pass support (needed when training with BN in eval mode)
            _lastMean = _runningMean;
            _lastVariance = _runningVariance;

            // Recompute from the live affine parameters. Gamma and beta are optimizer-owned
            // persistent tensors and are commonly updated IN PLACE, so a cache keyed only by a
            // local dirty bit cannot observe their mutation. That stale cache made post-training
            // inference use old parameters and made finite-difference derivatives exactly zero.
            // Engine ops are deterministic; allocation identity is not a numerical contract.
            var epsilonVec = Tensor<T>.CreateDefault(_runningVariance._shape, _epsilon);
            var variancePlusEps = Engine.TensorAdd(_runningVariance, epsilonVec);
            var stdDev = Engine.TensorSqrt(variancePlusEps);
            var (inferenceScale, inferenceShift) =
                CreateInferenceAffine(_gamma, _beta, _runningMean, stdDev);

            // Handle any tensor rank (2D, 3D, 4D, 5D, etc.)
            // Dimension 0 is batch, dimension 1 is features/channels
            // Dimensions 2+ are spatial dimensions
            var result = ApplyInferenceAnyRank(input, inferenceScale, inferenceShift);

            // Restore pre-flatten rank for the features-last path.
            if (flattenedFeaturesLast && preFlattenShape is not null)
            {
                result = Engine.Reshape(result, preFlattenShape);
            }

            // Preserve original rank
            if (_inputWas1D)
            {
                result = Engine.Reshape(result, [result.Length]);
            }

            return result;
        }
    }

    /// <summary>
    /// Builds inference affine coefficients without requesting writable access to persistent operands.
    /// </summary>
    private (Tensor<T> Scale, Tensor<T> Shift) CreateInferenceAffine(
        Tensor<T> gamma,
        Tensor<T> beta,
        Tensor<T> runningMean,
        Tensor<T> standardDeviation)
    {
        if (!gamma._shape.SequenceEqual(beta._shape)
            || !gamma._shape.SequenceEqual(runningMean._shape)
            || !gamma._shape.SequenceEqual(standardDeviation._shape))
        {
            throw new ArgumentException(
                "Batch-normalization affine operands must have identical shapes.");
        }

        // Keep the affine transform on the active tape. Constructing fresh tensors and filling them
        // through scalar indexing reproduces the forward values but severs gamma/beta from the graph,
        // making eval-mode fine tuning and every composite that contains BatchNorm report zero
        // parameter gradients.
        var scale = Engine.TensorDivide(gamma, standardDeviation);
        var shift = Engine.TensorSubtract(beta, Engine.TensorMultiply(scale, runningMean));

        return (scale, shift);
    }

    /// <summary>
    /// Applies batch normalization inference for tensors of any rank.
    /// </summary>
    /// <remarks>
    /// Supports any tensor rank >= 2. Dimension 0 is batch, dimension 1 is features/channels,
    /// and dimensions 2+ are spatial dimensions that are processed element-wise.
    /// </remarks>
    private Tensor<T> ApplyInferenceAnyRank(Tensor<T> input, Tensor<T> scale, Tensor<T> shift)
    {
        // Engine-accelerated batch normalization inference:
        // output = input * scale_broadcast + shift_broadcast
        // The channel axis MUST mirror OnFirstForward's resolution rule (Ioffe & Szegedy 2015 §3):
        //   rank 1 [F] / rank 3 [C, H, W] (unbatched image) — channels in axis 0
        //   rank 2 [B, F] / rank >= 4 [B, C, H, W, ...]     — channels in axis 1
        // Hardcoding axis 1 here broke unbatched rank-3 inputs: scale reshaped to [1, C, 1]
        // against input [C, H, W] — "Tensors with shapes [64, 32, 32] and [1, 64, 1] cannot
        // be broadcast" (surfaced by DenseNetNetwork.Predict on a [3, 32, 32] image).
        int rank = input.Shape.Length;
        // Rank 3 is ambiguous: [C, H, W] (unbatched image, channels axis 0) or [B, C, T] (batched
        // channels-first sequence, channels axis 1). A caller that declared ChannelsFirst gets
        // axis 1; everyone else keeps the historical axis-0 reading. Guessing here is what
        // broadcast the channel scale along ContextNet's BATCH axis.
        int channelAxis = Layout == BatchNormDataLayout.ChannelsFirst && rank >= 3
            ? 1
            : (rank == 1 || rank == 3 ? 0 : 1);

        var broadcastShape = new int[rank];
        for (int d = 0; d < rank; d++)
            broadcastShape[d] = d == channelAxis ? scale.Length : 1;

        var scaleReshaped = Engine.Reshape(scale, broadcastShape);
        var shiftReshaped = Engine.Reshape(shift, broadcastShape);

        // Engine-accelerated broadcast: tape-tracked + SIMD + GPU-capable
        var scaled = Engine.TensorMultiply(input, scaleReshaped);
        return Engine.TensorAdd(scaled, shiftReshaped);
    }

    /// <summary>
    /// Gets whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs GPU-resident batch normalization forward pass.
    /// </summary>
    /// <param name="input">GPU-resident input tensor with shape [batch, features] or [batch, channels, H, W].</param>
    /// <returns>GPU-resident output tensor with same shape as input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when GPU engine is not available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs batch normalization entirely on GPU, avoiding CPU round-trips.
    /// The input and output tensors remain GPU-resident for chained GPU operations.
    /// </para>
    /// <para>
    /// During training mode, running statistics (mean and variance) are updated on GPU
    /// and then downloaded back to CPU for persistence.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];

        // Store input shape for backward pass
        _lastInput = null; // GPU path doesn't store CPU tensor

        double epsilonDouble = NumOps.ToDouble(_epsilon);
        double momentumDouble = NumOps.ToDouble(_momentum);

        // Call GPU-resident batch norm
        var (output, saveMean, saveVar) = gpuEngine.FusedBatchNormGpu(
            input,
            _gamma,
            _beta,
            ref _runningMean,
            ref _runningVariance,
            epsilonDouble,
            momentumDouble,
            IsTrainingMode);

        // GPU backends may replace the ref buffers when materializing device
        // state. Refresh the name-based registration if their identity changed.
        RegisterRunningStatisticBuffers();

        // Store saved values for backward pass (if training)
        if (IsTrainingMode && saveMean is not null && saveVar is not null)
        {
            _lastInputGpu = input;
            _lastMean = saveMean;
            _lastVariance = saveVar;
        }

        return output;
    }

    private static int ComputeTotalElements(int[] shape)
    {
        int total = 1;
        for (int i = 0; i < shape.Length; i++) total *= shape[i];
        return total;
    }

    // --- ILayerSerializationExtras: running mean/variance are non-trainable state ---

    int ILayerSerializationExtras<T>.ExtraParameterCount => _runningMean.Length + _runningVariance.Length;

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_runningMean.Data),
            Vector<T>.FromMemory(_runningVariance.Data));
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        int featureSize = InputShape[0];

        // A layer whose feature count has not resolved yet cannot check anything: InputShape[0] is
        // the -1 free-axis sentinel, and the arithmetic below turned that into the message
        // "extra parameters must have length -2 (mean + variance for -1 features), but got 0" --
        // a demand for a negative number of values, which no caller can satisfy. An empty vector
        // from an equally unresolved source is not a mismatch, it is two sides agreeing that there
        // are no running statistics yet, so accept it and leave the buffers alone. This is what
        // CRNN's clone hit: neither side had run a forward, so neither had statistics.
        if (featureSize <= 0)
        {
            if (extraParameters.Length == 0) return;

            throw new ArgumentException(
                $"BatchNormalization cannot accept {extraParameters.Length} extra parameters until " +
                "its feature count is known; the layer's input shape is still unresolved. Run a " +
                "forward pass, or restore into a layer resolved from the same input shape.",
                nameof(extraParameters));
        }

        if (extraParameters.Length != featureSize * 2)
            throw new ArgumentException(
                $"BatchNormalization extra parameters must have length {featureSize * 2} " +
                $"(mean + variance for {featureSize} features), but got {extraParameters.Length}.",
                nameof(extraParameters));

        var meanVec = extraParameters.Slice(0, featureSize);
        var varVec = extraParameters.Slice(featureSize, featureSize);

        _runningMean = TensorAllocator.RentPinned<T>([featureSize]);
        meanVec.AsSpan().CopyTo(_runningMean.AsWritableSpan());
        _runningVariance = TensorAllocator.RentPinned<T>([featureSize]);
        varVec.AsSpan().CopyTo(_runningVariance.AsWritableSpan());
        RegisterRunningStatisticBuffers();
    }

    /// <summary>
    /// Switches the layer between training and inference behavior. Switching modes invalidates the
    /// cached inference scale/shift so the next inference forward recomputes them from the
    /// <i>current</i> running statistics.
    /// </summary>
    /// <remarks>
    /// Without this invalidation, a cache computed from an intermediate running mean/variance during
    /// training could be reused after switching to eval — producing inference output that lags the
    /// final running statistics. That stale cache also made a freshly deserialized clone (which
    /// always recomputes from the restored running stats) diverge from the original on the very same
    /// weights. Recomputing on every mode switch keeps inference deterministic and round-trip stable.
    /// </remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
    }

    [AiDotNet.Attributes.Buffer]
    private Tensor<T>? _gammaVelocity;
    [AiDotNet.Attributes.Buffer]
    private Tensor<T>? _betaVelocity;

    /// <summary>
    /// Updates the layer's parameters using the computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the gamma (scale) and beta (shift) parameters using gradient descent:
    /// - gamma = gamma - learningRate * gammaGradient
    /// - beta = beta - learningRate * betaGradient
    /// </para>
    /// <para>
    /// The gradients are computed during the backward pass and represent how much
    /// each parameter should change to reduce the loss function.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's learnable parameters during training.
    /// 
    /// After the backward pass calculates how each parameter affects the error,
    /// this method adjusts those parameters to reduce the error:
    /// 
    /// 1. It checks that the backward pass has been called first
    /// 2. It updates the gamma (scale) parameters:
    ///    gamma = gamma - learningRate * gammaGradient
    /// 3. It updates the beta (shift) parameters:
    ///    beta = beta - learningRate * betaGradient
    /// 
    /// The learning rate controls how big the updates are:
    /// - A larger learning rate means bigger changes (faster learning but potentially unstable)
    /// - A smaller learning rate means smaller changes (slower but more stable learning)
    /// 
    /// For example, if a particular gamma value is causing high error, its gradient
    /// will be large, and this method will adjust that parameter more significantly
    /// to reduce the error in the next forward pass.
    /// 
    /// This is the step where actual "learning" happens in the neural network.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("UpdateParameters cannot be called before Backward. No gradients available.");

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_gammaVelocity == null)
            {
                _gammaVelocity = new Tensor<T>(_gamma._shape);
                _gammaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_gammaVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_betaVelocity == null)
            {
                _betaVelocity = new Tensor<T>(_beta._shape);
                _betaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_betaVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_gamma, _gammaGradient, _gammaVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_beta, _betaGradient, _betaVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            // Production-grade: Use Engine operations instead of manual loops
            _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
            _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));

            // Notify GPU that tensor data has changed
            Engine.InvalidatePersistentTensor(_gamma);
            Engine.InvalidatePersistentTensor(_beta);
        }
    }

    /// <summary>
    /// Resets the internal state of the batch normalization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all cached values from the forward and backward passes,
    /// including:
    /// - Last input tensor
    /// - Last normalized values
    /// - Last batch mean and variance
    /// - Gradients for gamma and beta parameters
    /// </para>
    /// <para>
    /// It does NOT reset the learned parameters (gamma and beta) or the running statistics
    /// (running mean and variance) used for inference.
    /// </para>
    /// <para>
    /// This is typically called when starting a new training epoch or when switching
    /// between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of previous calculations.
    /// 
    /// During training, the batch normalization layer keeps track of:
    /// - The last input it processed
    /// - The normalized values it calculated
    /// - The mean and variance of the last batch
    /// - The gradients for its parameters
    /// 
    /// This method clears all of these temporary values, which is useful when:
    /// - Starting a new training epoch
    /// - Switching between training and testing modes
    /// - Ensuring the layer behaves deterministically
    /// 
    /// Important: This does NOT reset the learned parameters (gamma and beta) or
    /// the running statistics (running mean and variance) that are used during inference.
    /// It only clears temporary calculation values.
    /// 
    /// Think of it as clearing the layer's short-term memory while preserving its
    /// long-term learning.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        if (_gammaGradient == null || _betaGradient == null)
            return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate((_gammaGradient is not null ? Vector<T>.FromMemory(_gammaGradient.Data) : new Vector<T>(0)), (_betaGradient is not null ? Vector<T>.FromMemory(_betaGradient.Data) : new Vector<T>(0)));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _gammaGradient = null;
        _betaGradient = null;
    }

    public override void ResetState()
    {
        // Clear CPU cached values
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;

        // Clear GPU cached tensors
        _lastInputGpu = null;
    }

    #region ONNX Export

    /// <summary>
    /// Emits an ONNX <c>BatchNormalization</c> op using the layer's running
    /// statistics (inference mode). 5 graph initializers are added: scale
    /// (gamma), B (beta), mean (running_mean), var (running_variance).
    /// </summary>
    public override AiDotNet.Onnx.OnnxLayerOutputs ConvertToOnnx(
        AiDotNet.Onnx.OnnxGraphBuilder builder,
        AiDotNet.Onnx.OnnxLayerInputs inputs)
    {
        if (builder is null) throw new ArgumentNullException(nameof(builder));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));

        int n = _gamma.Shape[0];

        float[] FlattenRank1(Tensor<T> t)
        {
            var arr = new float[t.Shape[0]];
            for (int i = 0; i < t.Shape[0]; i++) arr[i] = (float)NumOps.ToDouble(t[i]);
            return arr;
        }

        var scaleName = builder.AddFloatInitializer("bn_scale", FlattenRank1(_gamma), new[] { n });
        var biasName  = builder.AddFloatInitializer("bn_B",     FlattenRank1(_beta),  new[] { n });
        var meanName  = builder.AddFloatInitializer("bn_mean",  FlattenRank1(_runningMean), new[] { n });
        var varName   = builder.AddFloatInitializer("bn_var",   FlattenRank1(_runningVariance), new[] { n });

        var outputName = builder.NextTensorName("bn_out");
        var node = builder.AddOp("BatchNormalization",
            inputs: new[] { inputs.Primary, scaleName, biasName, meanName, varName },
            outputs: new[] { outputName });

        // Attach the epsilon attribute matching the layer's configured value.
        node.Attribute.Add(new AiDotNet.Onnx.Protobuf.AttributeProto
        {
            Name = "epsilon",
            Type = AiDotNet.Onnx.Protobuf.AttributeProto.Types.AttributeType.Float,
            F = (float)NumOps.ToDouble(_epsilon),
        });

        return new AiDotNet.Onnx.OnnxLayerOutputs(outputName);
    }

    #endregion
}
