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
public partial class BatchNormalizationLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
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
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]

    private Tensor<T> _gamma;

    /// <summary>
    /// The shift parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as beta. This learnable parameter allows the network to shift
    /// each normalized feature. Initialized to zeros.
    /// </remarks>
    private Tensor<T> _beta;

    /// <summary>
    /// The running mean used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to zeros.
    /// </remarks>
    private Tensor<T> _runningMean;

    /// <summary>
    /// The running variance used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to ones.
    /// </remarks>
    private Tensor<T> _runningVariance;

    // Cached inference scale/shift for deterministic forward pass
    private Tensor<T>? _cachedInferenceScale;
    private Tensor<T>? _cachedInferenceShift;
    private bool _inferenceScaleDirty = true;

    /// <summary>
    /// The input from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
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
    private Tensor<T>? _lastMean;

    /// <summary>
    /// The batch variance from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastVariance;

    /// <summary>
    /// The gradient of the loss with respect to gamma.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update gamma.
    /// </remarks>
    private Tensor<T>? _gammaGradient;

    /// <summary>
    /// The gradient of the loss with respect to beta.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update beta.
    /// </remarks>
    private Tensor<T>? _betaGradient;

    // GPU-resident cached tensors for GPU training pipeline
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
    private void InitializeNormalizationParameters()
    {
        int channels = InputShape[0];
        bool reinit = _gamma is null || _gamma.Length != channels;
        if (reinit)
        {
            _gamma = Tensor<T>.CreateDefault([channels], NumOps.One);
            _beta = Tensor<T>.CreateDefault([channels], NumOps.Zero);
            _runningMean = Tensor<T>.CreateDefault([channels], NumOps.Zero);
            _runningVariance = Tensor<T>.CreateDefault([channels], NumOps.One);
            RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
            RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
        }
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
    public BatchNormalizationLayer(int numFeatures, double epsilon = NumericalStabilityHelper.LargeEpsilon, double momentum = 0.9)
        : base(new[] { numFeatures }, new[] { numFeatures })
    {
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
        _runningMean = new Tensor<T>([numFeatures]);
        _runningVariance = new Tensor<T>([numFeatures]);
        _runningVariance.Fill(NumOps.One);

        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
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

        // Apply deferred ZeroInitGamma if requested before shape resolution.
        T gammaInit = _zeroInitGammaPending ? NumOps.Zero : NumOps.One;
        _zeroInitGammaPending = false;
        // Norm-layer params are channel-sized (small) — streaming-pool
        // pre-eviction barely moves the needle here, but we route through
        // AllocateLazyWeight for consistency with the rest of the
        // streaming-aware layers and to keep the contract simple. Then
        // fill gamma with the deferred init value (zero-init for
        // post-residual BN, one-init otherwise) and runningVariance with 1.
        _gamma = AllocateLazyWeight([numFeatures]);
        _gamma.Fill(gammaInit);
        _beta = AllocateLazyWeight([numFeatures]);
        _runningMean = AllocateLazyWeight([numFeatures]);
        _runningVariance = AllocateLazyWeight([numFeatures]);
        _runningVariance.Fill(NumOps.One);

        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);

        ResolveShapes(new[] { numFeatures }, new[] { numFeatures });
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
    public override Tensor<T> Forward(Tensor<T> input)
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
        if (input.Rank >= 3 && featureSize > 0 && input.Shape[^1] == featureSize)
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
            T oneMinusMomentum = NumOps.Subtract(NumOps.One, _momentum);
            Engine.TensorMultiplyScalarInPlace(_runningMean, _momentum);
            var scaledBatchMean = Engine.TensorMultiplyScalar(batchMean, oneMinusMomentum);
            Engine.TensorAddInPlace(_runningMean, scaledBatchMean);
            Engine.TensorMultiplyScalarInPlace(_runningVariance, _momentum);
            var scaledBatchVar = Engine.TensorMultiplyScalar(batchVariance, oneMinusMomentum);
            Engine.TensorAddInPlace(_runningVariance, scaledBatchVar);

            // Invalidate cached inference scale/shift since running stats changed
            _inferenceScaleDirty = true;

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
        else
        {
            // Inference: Use running statistics
            // output = gamma * (input - runningMean) / sqrt(runningVar + epsilon) + beta

            // Cache running stats for backward pass support (needed when training with BN in eval mode)
            _lastMean = _runningMean;
            _lastVariance = _runningVariance;

            // Cache scale/shift to ensure deterministic forward pass
            // (recomputing creates new tensor allocations that can cause SIMD alignment differences)
            if (_inferenceScaleDirty || _cachedInferenceScale is null || _cachedInferenceShift is null)
            {
                var epsilonVec = Tensor<T>.CreateDefault(_runningVariance._shape, _epsilon);
                var variancePlusEps = Engine.TensorAdd(_runningVariance, epsilonVec);
                var stdDev = Engine.TensorSqrt(variancePlusEps);

                _cachedInferenceScale = Engine.TensorDivide(_gamma, stdDev);
                var term2 = Engine.TensorDivide(Engine.TensorMultiply(_gamma, _runningMean), stdDev);
                _cachedInferenceShift = Engine.TensorSubtract(_beta, term2);
                _inferenceScaleDirty = false;
            }

            // Handle any tensor rank (2D, 3D, 4D, 5D, etc.)
            // Dimension 0 is batch, dimension 1 is features/channels
            // Dimensions 2+ are spatial dimensions
            var result = ApplyInferenceAnyRank(input, _cachedInferenceScale, _cachedInferenceShift);

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
        // Reshape scale/shift to [1, C, 1, 1, ...] for broadcasting across batch and spatial dims.
        int rank = input.Shape.Length;

        // Build broadcast shape: [1, C, 1, 1, ...]
        var broadcastShape = new int[rank];
        broadcastShape[0] = 1;           // batch
        broadcastShape[1] = scale.Length; // channels
        for (int d = 2; d < rank; d++)
            broadcastShape[d] = 1;       // spatial

        var scaleReshaped = Engine.Reshape(scale, broadcastShape);
        var shiftReshaped = Engine.Reshape(shift, broadcastShape);

        // Engine-accelerated broadcast: tape-tracked + SIMD + GPU-capable
        var scaled = Engine.TensorBroadcastMultiply(input, scaleReshaped);
        return Engine.TensorBroadcastAdd(scaled, shiftReshaped);
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

    /// <summary>
    /// Gets all trainable parameters of the batch normalization layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (gamma and beta) concatenated together.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a single vector containing all trainable parameters of the layer:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// </para>
    /// <para>
    /// This is useful for optimization algorithms that need access to all parameters at once,
    /// or for saving/loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable parameters as a single vector.
    ///
    /// Batch normalization has two sets of learnable parameters:
    /// - Gamma (scale): Controls how much to stretch or compress the normalized data
    /// - Beta (shift): Controls how much to move the normalized data up or down
    ///
    /// This method combines both sets into a single vector, with gamma values first,
    /// followed by beta values. For example, with 3 features:
    ///
    /// [gamma1, gamma2, gamma3, beta1, beta2, beta3]
    ///
    /// This format is useful for:
    /// - Saving and loading models
    /// - Advanced optimization algorithms that work with all parameters at once
    /// - Regularization techniques that need to access all parameters
    ///
    /// The total length of the returned vector is twice the number of features,
    /// since there's one gamma and one beta parameter per feature.
    /// </para>
    /// </remarks>
    public override long ParameterCount => _gamma.Length + _beta.Length;

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Production-grade: Use Vector.Concatenate instead of manual loops
        return Vector<T>.Concatenate(Vector<T>.FromMemory(_gamma.Data), Vector<T>.FromMemory(_beta.Data));
    }

    /// <summary>
    /// Sets all trainable parameters of the batch normalization layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (gamma and beta) concatenated together.</param>
    /// <remarks>
    /// <para>
    /// This method expects a single vector containing all trainable parameters:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// </para>
    /// <para>
    /// The length of the parameters vector must be exactly twice the feature size.
    /// This method is useful for loading pre-trained weights or setting parameters
    /// after optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads parameters into the layer from a single vector.
    ///
    /// This is the counterpart to GetParameters() - it takes a vector containing
    /// all parameters and sets them in the layer. The vector must have the format:
    ///
    /// [gamma1, gamma2, ..., gammaN, beta1, beta2, ..., betaN]
    ///
    /// Where N is the number of features. The total length must be exactly 2*N.
    ///
    /// This method is commonly used for:
    /// - Loading pre-trained models
    /// - Setting parameters after external optimization
    /// - Implementing transfer learning
    /// - Testing different parameter configurations
    ///
    /// If the vector doesn't have the expected length, the method will throw an
    /// exception to prevent incorrect parameter assignments.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        // Round-trip from saved parameters when in lazy placeholder state.
        // Layout: [gamma, beta] each featureSize long, so featureSize = length/2.
        if (!IsShapeResolved)
        {
            if (parameters.Length == 0) return;
            if (parameters.Length % 2 != 0 || parameters.Length == 0)
                throw new ArgumentException(
                    $"Cannot infer featureSize for BatchNormalizationLayer from {parameters.Length} parameters.");
            int inferredFeatureSize = parameters.Length / 2;
            ResolveFromShape(new[] { inferredFeatureSize });
        }

        int featureSize = InputShape[0];
        if (parameters.Length != featureSize * 2)
            throw new ArgumentException($"Expected {featureSize * 2} parameters, but got {parameters.Length}", nameof(parameters));

        // Production-grade: Use Tensor.FromVector instead of manual loops
        var gammaVec = parameters.Slice(0, featureSize);
        var betaVec = parameters.Slice(featureSize, featureSize);

        _gamma = Tensor<T>.FromVector(gammaVec, [featureSize]);
        _beta = Tensor<T>.FromVector(betaVec, [featureSize]);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
        _inferenceScaleDirty = true;
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
        if (extraParameters.Length != featureSize * 2)
            throw new ArgumentException(
                $"BatchNormalization extra parameters must have length {featureSize * 2} " +
                $"(mean + variance for {featureSize} features), but got {extraParameters.Length}.",
                nameof(extraParameters));

        var meanVec = extraParameters.Slice(0, featureSize);
        var varVec = extraParameters.Slice(featureSize, featureSize);

        _runningMean = Tensor<T>.FromVector(meanVec, [featureSize]);
        _runningVariance = Tensor<T>.FromVector(varVec, [featureSize]);
        _inferenceScaleDirty = true;
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
        _inferenceScaleDirty = true;
    }

    private Tensor<T>? _gammaVelocity;
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
            _inferenceScaleDirty = true;
        }
        else
        {
            // Production-grade: Use Engine operations instead of manual loops
            _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
            _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));

            // Invalidate cached inference terms since gamma/beta changed
            _inferenceScaleDirty = true;

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
