using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Diffusion Transformer (DiT) noise predictor for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiT (Diffusion Transformer) replaces the traditional U-Net architecture with
/// a pure transformer design. This approach leverages the scalability and
/// effectiveness of transformers, enabling better performance at larger scales.
/// </para>
/// <para>
/// <b>For Beginners:</b> DiT is the "new generation" of noise prediction:
///
/// Traditional U-Net approach:
/// - Uses convolutional neural networks
/// - Has encoder-decoder structure with skip connections
/// - Good, but limited scalability
///
/// DiT approach (this class):
/// - Uses transformer architecture (like GPT, but for images)
/// - Treats image as patches (like words in a sentence)
/// - Scales better with more compute and data
/// - Powers cutting-edge models like DALL-E 3, Sora
///
/// Key advantages:
/// - Better quality at large scales
/// - Simpler architecture (no skip connections needed)
/// - More flexible conditioning mechanisms
/// - Easier to scale training
/// </para>
/// <para>
/// Architecture details:
/// - Patchify: Split image into 2x2 or larger patches
/// - Position embedding: Add spatial information
/// - Transformer blocks: Self-attention + MLP
/// - AdaLN: Adaptive layer normalization for timestep/conditioning
/// - Unpatchify: Reconstruct full resolution output
///
/// Used in: DiT (original), DALL-E 3, Sora, SD3, Pixart-alpha
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create DiT predictor for latent diffusion
/// var dit = new DiTNoisePredictor&lt;float&gt;(
///     inputChannels: 4,      // Latent channels
///     hiddenSize: 1152,      // DiT-XL/2 size
///     numLayers: 28,         // DiT-XL depth
///     numHeads: 16,
///     patchSize: 2);
///
/// // Predict noise
/// var noisePrediction = dit.PredictNoise(noisyLatent, timestep, textEmbedding);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Scalable Diffusion Models with Transformers", "https://arxiv.org/abs/2212.09748")]
public class DiTNoisePredictor<T> : NoisePredictorBase<T>
{
    /// <summary>
    /// Standard DiT model sizes.
    /// </summary>
    public static class ModelSizes
    {
        /// <summary>DiT-S/2: Small model with patch size 2.</summary>
        public static readonly (int hiddenSize, int numLayers, int numHeads) Small = (384, 12, 6);

        /// <summary>DiT-B/2: Base model with patch size 2.</summary>
        public static readonly (int hiddenSize, int numLayers, int numHeads) Base = (768, 12, 12);

        /// <summary>DiT-L/2: Large model with patch size 2.</summary>
        public static readonly (int hiddenSize, int numLayers, int numHeads) Large = (1024, 24, 16);

        /// <summary>DiT-XL/2: Extra-large model with patch size 2.</summary>
        public static readonly (int hiddenSize, int numLayers, int numHeads) XLarge = (1152, 28, 16);
    }

    /// <summary>
    /// Input channels (typically 4 for latent diffusion).
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    private readonly int _hiddenSize;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Parameter-count floor (≈ 1B params ≈ 4 GB fp32 → 2 GB fp16) above which eval keeps the tower's
    /// weights fp16-resident. Below this the model fits comfortably resident at fp32, so it pays nothing.
    /// </summary>
    private const long LowPrecisionResidentThresholdParams = 1_000_000_000L;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Patch size for image tokenization.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// Context dimension for conditioning.
    /// </summary>
    private readonly int _contextDim;

    /// <summary>
    /// MLP hidden dimension ratio.
    /// </summary>
    private readonly double _mlpRatio;

    /// <summary>
    /// Latent spatial size (height = width) for computing patch count.
    /// </summary>
    private readonly int _latentSpatialSize;

    /// <summary>
    /// The neural network architecture configuration, if provided.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <summary>
    /// Patch embedding layer.
    /// </summary>
    private DenseLayer<T>? _patchEmbed;

    /// <summary>
    /// Time embedding layers.
    /// </summary>
    private DenseLayer<T>? _timeEmbed1;
    private DenseLayer<T>? _timeEmbed2;

    /// <summary>
    /// Label/class embedding (optional).
    /// </summary>
    private DenseLayer<T>? _labelEmbed;

    /// <summary>
    /// Transformer blocks.
    /// </summary>
    private readonly List<DiTBlock> _blocks;

    /// <summary>
    /// Final layer norm.
    /// </summary>
    private LayerNormalizationLayer<T>? _finalNorm;

    /// <summary>
    /// Output projection (unpatchify).
    /// </summary>
    private DenseLayer<T>? _outputProj;

    /// <summary>
    /// AdaLN modulation for final layer.
    /// </summary>
    private DenseLayer<T>? _adaln_modulation;

    // Lazy initialization state
    private int _numClasses;
    private List<DiTBlock>? _customBlocks;
    private volatile bool _layersInitialized;
    private readonly object _initLock = new();

    /// <summary>
    /// Whether this predictor's lazy weight tensors have been materialized yet
    /// (i.e. a forward pass, <see cref="GetParameters"/>, or
    /// <see cref="SetParameters"/> has run). While <c>false</c> the predictor
    /// holds no resident weights — its parameters are fully determined by its
    /// construction config — so a clone can be produced by re-running the same
    /// construction rather than copying a flat parameter vector. That matters
    /// for foundation-scale configs (e.g. WanVideo-14B ≈ 15 B params) where the
    /// flat <see cref="Vector{T}"/> copy path is both int.MaxValue-bounded and
    /// far larger than host RAM. Callers that need a true deep copy of resolved
    /// weights use <see cref="CopyParametersFrom"/> instead.
    /// </summary>
    public bool AreLayersInitialized => _layersInitialized;

    /// <summary>
    /// Position embeddings (learnable).
    /// </summary>
    private Tensor<T>? _posEmbed;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    // ──────────────────────────────────────────────────────────────────────────
    // #1672 destination-buffer scratch for the AdaLN / gate broadcasts.
    //
    // Each DiT block runs strictly sequentially, and within a block the AdaLN
    // attn-branch output is fully consumed (by the attention projection) before the
    // mlp-branch AdaLN runs. The AddWithGate `gated` intermediate is consumed by the
    // immediately-following residual add. So a single predictor-level buffer per role,
    // reused across all blocks and the final layer, never aliases a still-live tensor.
    // Reallocated whenever the [B, seq, hidden] shape changes. Used only on the no-tape
    // inference forward (ForwardScratchGate.Enabled). Bit-identical to the allocating path.
    // ──────────────────────────────────────────────────────────────────────────
    private Tensor<T>? _adaLnScaledScratch;   // TensorBroadcastMultiply(x, 1+scale)
    private Tensor<T>? _adaLnOutScratch;       // TensorBroadcastAdd(scaled, shift)
    private Tensor<T>? _gateScratch;           // TensorBroadcastMultiply(residual, gate)

    /// <summary>Element-wise shape-array equality for the #1672 scratch-reuse decision.</summary>
    private static bool ShapeMatches(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// True when the #1672 scratch gate is ON and no gradient tape is recording (inference).
    /// The scratch reuse is safe only on the eager no-tape forward.
    /// </summary>
    private static bool UseForwardScratch()
    {
        if (!AiDotNet.Helpers.ForwardScratchGate.AdaLn) return false;
        bool tapeActive = AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
            && !AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed;
        return !tapeActive;
    }

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <summary>
    /// Gets the hidden size.
    /// </summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Gets the number of layers.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the patch size.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <summary>
    /// Initializes a new instance of the DiTNoisePredictor class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used for the transformer blocks. If null or empty,
    /// industry-standard DiT-XL/2 layers are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input channels (default: 4 for latent diffusion).</param>
    /// <param name="hiddenSize">Hidden dimension size (default: 1152 for DiT-XL).</param>
    /// <param name="numLayers">Number of transformer layers (default: 28 for DiT-XL).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="patchSize">Patch size for tokenization (default: 2).</param>
    /// <param name="contextDim">Conditioning context dimension (default: 1024).</param>
    /// <param name="mlpRatio">MLP hidden dimension ratio (default: 4.0).</param>
    /// <param name="latentSpatialSize">Latent spatial size for computing patch count (default: 32 for 256/8).</param>
    /// <param name="numClasses">Number of classes for class conditioning (0 for text-only).</param>
    /// <param name="customBlocks">
    /// Optional custom transformer blocks. If provided, these blocks are used instead of creating
    /// default blocks. This allows full customization of the transformer architecture.
    /// </param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Random seed for initialization.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults
    /// from the DiT-XL/2 paper. You can create a ready-to-use DiT with no arguments,
    /// or customize any component:
    ///
    /// <code>
    /// // Default DiT-XL/2 configuration (recommended for most users)
    /// var dit = new DiTNoisePredictor&lt;float&gt;();
    ///
    /// // Custom layers via NeuralNetworkArchitecture
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(..., layers: myCustomLayers);
    /// var dit = new DiTNoisePredictor&lt;float&gt;(architecture: arch);
    ///
    /// // DiT-L/2 configuration
    /// var dit = new DiTNoisePredictor&lt;float&gt;(
    ///     hiddenSize: 1024, numLayers: 24, numHeads: 16);
    /// </code>
    /// </para>
    /// </remarks>
    public DiTNoisePredictor(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 4,
        int hiddenSize = 1152,
        int numLayers = 28,
        int numHeads = 16,
        int patchSize = 2,
        int contextDim = 1024,
        double mlpRatio = 4.0,
        int latentSpatialSize = 32,
        int numClasses = 0,
        List<DiTBlock>? customBlocks = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException($"hiddenSize ({hiddenSize}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;
        _mlpRatio = mlpRatio;
        _latentSpatialSize = latentSpatialSize;

        // Class-conditional DiT is fully wired end-to-end per Peebles & Xie 2022
        // §3.2: when numClasses > 0, _labelEmbed projects one-hot class labels
        // [B, numClasses] into the time-embedding space and the Forward() path
        // sums (timeEmbed + classEmbed) into adaLnEmbed, which feeds every
        // block-level AdaLN modulation and the final-layer AdaLN. See Forward()
        // for the routing. When numClasses == 0, the `conditioning` argument
        // instead feeds cross-attention (text-conditional path).
        _blocks = new List<DiTBlock>();
        _numClasses = numClasses;
        // Defensive copy of the caller-owned list — without this, a caller who
        // mutates customBlocks after construction would silently change this
        // model's block graph at the deferred init point. The lazy-init shift
        // turned a constructor-time read into a first-use read, so the list
        // contents must be snapshotted here to preserve construction-time
        // semantics.
        _customBlocks = customBlocks == null ? null : new List<DiTBlock>(customBlocks);
        // Defer heavy layer allocation to first use (lazy initialization)
    }

    /// <summary>
    /// Ensures layers are initialized (lazy init on first use).
    /// </summary>
    /// <remarks>
    /// Retry-safe: if <see cref="InitializeLayers"/> throws after partially
    /// populating <c>_blocks</c>, the next call must rebuild from a clean
    /// slate. Without the pre-clear, a retry would append a second set of
    /// blocks on top of the partial state, doubling the layer graph.
    /// <c>_layersInitialized = true</c> is the LAST step inside the lock so
    /// observers never see a partially-built graph.
    /// </remarks>
    private void EnsureLayersInitialized()
    {
        if (_layersInitialized) return;
        lock (_initLock)
        {
            if (_layersInitialized) return; // double-checked locking
            // Clear any partial state from a prior failed init attempt.
            // _blocks is the only mutable collection touched by InitializeLayers;
            // other fields (_patchEmbed, _timeEmbed1, etc.) are reassigned
            // wholesale by InitializeLayers so partial state there isn't a hazard.
            _blocks.Clear();
            InitializeLayers(_architecture, _numClasses, _customBlocks);
            _layersInitialized = true;
        }
    }

    /// <summary>
    /// Initializes all layers of the DiT, using custom layers from the user
    /// if provided or creating industry-standard layers from the DiT paper.
    /// </summary>
    /// <param name="architecture">Optional architecture with custom layers.</param>
    /// <param name="numClasses">Number of classes for class conditioning.</param>
    /// <param name="customBlocks">Optional custom transformer blocks.</param>
    /// <remarks>
    /// <para>
    /// Layer resolution order:
    /// 1. If custom blocks are provided directly, use those
    /// 2. If a NeuralNetworkArchitecture with layers is provided, wrap those as transformer blocks
    /// 3. Otherwise, create industry-standard DiT-XL/2 layers
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_patchEmbed), nameof(_timeEmbed1), nameof(_timeEmbed2),
                   nameof(_finalNorm), nameof(_outputProj), nameof(_adaln_modulation))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        int numClasses,
        List<DiTBlock>? customBlocks)
    {
        var patchDim = _inputChannels * _patchSize * _patchSize;
        // Faithful DiT (Peebles & Xie 2023): the timestep MLP outputs hidden_size and the AdaLN
        // modulation is Linear(hidden_size, 6*hidden_size). The conditioning vector fed to every block's
        // AdaLN therefore has width = hidden_size. An earlier 4*hidden_size here inflated the dominant
        // AdaLN weight 4x (e.g. [12288,18432] instead of [3072,18432]) — over half a foundation DiT's
        // parameters — and was the primary driver of the foundation-scale OOM (issue #1672).
        var timeEmbedDim = _hiddenSize;

        // Always create patch embedding, time embedding, and final layers. Use
        // LazyDense so weight tensors stay unallocated until the first Forward()
        // pass — DiT-XL's default 4 GB of weights would otherwise OOM the CI
        // runner just from `new DiTNoisePredictor()`.
        _patchEmbed = LazyDense(patchDim, _hiddenSize);

        _timeEmbed1 = LazyDense(_hiddenSize, timeEmbedDim, new SiLUActivation<T>());
        _timeEmbed2 = LazyDense(timeEmbedDim, timeEmbedDim, new SiLUActivation<T>());

        // Class conditioning embedding (Peebles & Xie 2022 §3.2 / Appendix C).
        // LabelEmbedder: one-hot class labels [B, numClasses] → [B, timeEmbedDim],
        // added to the projected time embedding in Forward(). The paper's reference
        // code keeps class and time embeddings in the same space (hidden_size there,
        // timeEmbedDim here since our time MLP projects up to timeEmbedDim).
        // Classifier-free guidance reserves an additional "null" class — the caller
        // is expected to pass a zero one-hot to represent the unconditional token.
        if (numClasses > 0)
        {
            _labelEmbed = LazyDense(numClasses, timeEmbedDim);
        }

        _finalNorm = new LayerNormalizationLayer<T>();
        _adaln_modulation = LazyDense(timeEmbedDim, _hiddenSize * 2);
        _outputProj = LazyDense(_hiddenSize, patchDim);

        // Priority 1: Use custom blocks passed directly
        if (customBlocks != null && customBlocks.Count > 0)
        {
            _blocks.AddRange(customBlocks);
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture as block components
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            foreach (var layer in architecture.Layers)
            {
                // Use a provided DenseLayer<T> for MLP1 if available; otherwise create a
                // standard MLP layer with dimensions (_hiddenSize -> _hiddenSize * _mlpRatio).
                // Note: if the provided DenseLayer has different dimensions, it will auto-resize
                // weights on the first forward pass via EnsureWeightShapeForInput.
                var mlp1 = layer as DenseLayer<T>
                    ?? LazyDense(_hiddenSize, (int)(_hiddenSize * _mlpRatio), new GELUActivation<T>());

                _blocks.Add(new DiTBlock
                {
                    Norm1 = new LayerNormalizationLayer<T>(),
                    Attention = CreateAttentionLayer(),
                    Norm2 = new LayerNormalizationLayer<T>(),
                    MLP1 = mlp1,
                    MLP2 = LazyDense((int)(_hiddenSize * _mlpRatio), _hiddenSize),
                    AdaLNModulation = LazyDense(timeEmbedDim, _hiddenSize * 6),
                    CrossAttnNorm = new LayerNormalizationLayer<T>(),
                    CrossAttnQ = LazyDense(_hiddenSize, _hiddenSize),
                    CrossAttnK = LazyDense(_contextDim, _hiddenSize),
                    CrossAttnV = LazyDense(_contextDim, _hiddenSize),
                    CrossAttnOut = LazyDense(_hiddenSize, _hiddenSize)
                });
            }
            return;
        }

        // Priority 3: Create industry-standard DiT transformer blocks
        CreateDefaultBlocks(timeEmbedDim);
    }

    /// <summary>
    /// Creates industry-standard DiT transformer blocks.
    /// </summary>
    private void CreateDefaultBlocks(int timeEmbedDim)
    {
        var mlpHidden = (int)(_hiddenSize * _mlpRatio);

        for (int i = 0; i < _numLayers; i++)
        {
            _blocks.Add(new DiTBlock
            {
                Norm1 = new LayerNormalizationLayer<T>(),
                Attention = CreateAttentionLayer(),
                Norm2 = new LayerNormalizationLayer<T>(),
                MLP1 = LazyDense(_hiddenSize, mlpHidden, new GELUActivation<T>()),
                MLP2 = LazyDense(mlpHidden, _hiddenSize),
                AdaLNModulation = LazyDense(timeEmbedDim, _hiddenSize * 6),
                CrossAttnNorm = new LayerNormalizationLayer<T>(),
                CrossAttnQ = LazyDense(_hiddenSize, _hiddenSize),
                CrossAttnK = LazyDense(_contextDim, _hiddenSize),
                CrossAttnV = LazyDense(_contextDim, _hiddenSize),
                CrossAttnOut = LazyDense(_hiddenSize, _hiddenSize)
            });
        }
    }

    /// <summary>
    /// Creates an attention layer for the transformer block.
    /// </summary>
    /// <remarks>
    /// Uses SelfAttentionLayer with default sequence length.
    /// Actual sequence length is determined by the number of patches at runtime.
    /// </remarks>
    private SelfAttentionLayer<T> CreateAttentionLayer()
    {
        // Compute sequence length from latent spatial size and patch size
        var numPatches = (_latentSpatialSize / _patchSize) * (_latentSpatialSize / _patchSize);
        // LazySelfAttention keeps Q/K/V weight tensors at size 0 until the first
        // Forward() call — 28 of these per DiT-XL tower × ~32 MB = ~900 MB that
        // would otherwise allocate at model-construction time.
        return LazySelfAttention(
            sequenceLength: numPatches,
            embeddingDimension: _hiddenSize,
            headCount: _numHeads);
    }

    /// <inheritdoc />
    /// <summary>
    /// True when this forward runs the foundation-scale fp16-resident eval path (no active tape,
    /// params over the resident threshold). In that mode the disk-backed streaming scope is bypassed
    /// (its release sweep re-inflates evicted weights → OOM) and the compiled replay is bypassed (it
    /// captures the per-forward-mutated weight tensors → stale references); the eager Forward upcasts
    /// each layer's fp16 weights to fp32 transiently and drops them, keeping only fp16 resident.
    /// </summary>
    private bool UseLowPrecisionResidentEval()
    {
        bool tapeActive = AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
            && !AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed;
        return !tapeActive && ParameterCount > LowPrecisionResidentThresholdParams;
    }

    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        EnsureLayersInitialized();
        _lastInput = noisySample;

        // Get timestep embedding (cheap MLP; computed eagerly each step so the per-step
        // timestep stays a LIVE input the compiled plan re-binds rather than bakes).
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        // Foundation-scale eval: eager fp16-resident forward — NOT the disk-streaming scope (its
        // release sweep re-inflates evicted weights → OOM) and NOT the compiled replay (captures the
        // per-forward-mutated weights). See UseLowPrecisionResidentEval.
        if (UseLowPrecisionResidentEval())
        {
            return Forward(noisySample, timeEmbed, conditioning);
        }

        using var streaming = BeginWeightStreamingForward();
        // Compile the (expensive) DiT forward ONCE and replay it across the denoising loop,
        // re-binding every per-step leaf — noisy sample, timestep embedding, optional
        // conditioning — so a changing timestep is never baked (#1620 / AiDotNet.Tensors#616).
        // Conditioning is declared as an input only when present, so an unconditional model
        // never declares a leaf its forward doesn't read (which would fail closed to eager).
        var inputs = conditioning is not null
            ? new[] { noisySample, timeEmbed, conditioning }
            : new[] { noisySample, timeEmbed };
        return streaming.Complete(
            PredictCompiledMulti(inputs, () => Forward(noisySample, timeEmbed, conditioning)));
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        EnsureLayersInitialized();
        _lastInput = noisySample;

        // See PredictNoise: foundation-scale eval uses the eager fp16-resident forward, bypassing the
        // disk-streaming scope and compiled replay.
        if (UseLowPrecisionResidentEval())
        {
            return Forward(noisySample, timeEmbedding, conditioning);
        }

        using var streaming = BeginWeightStreamingForward();
        // Multi-input compiled replay: every per-step leaf — noisy sample, timestep embedding,
        // optional conditioning — is re-bound each step, so a changing timestep embedding is
        // never baked as a constant. The single-input PredictCompiled marks only the noisy
        // sample as mutable and bakes the rest, which would replay step 0's embedding for the
        // whole denoising loop (silent corruption — #1620). The multi-input compile
        // (AiDotNet.Tensors#616) compiles the expensive forward once while keeping all per-step
        // inputs live; the verify-then-trust gate keeps the output numerically identical to eager.
        var inputs = conditioning is not null
            ? new[] { noisySample, timeEmbedding, conditioning }
            : new[] { noisySample, timeEmbedding };
        return streaming.Complete(
            PredictCompiledMulti(inputs, () => Forward(noisySample, timeEmbedding, conditioning)));
    }

    /// <summary>
    /// Projects timestep embedding through MLP.
    /// </summary>
    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        if (_timeEmbed1 == null || _timeEmbed2 == null)
            throw new InvalidOperationException("Time embedding layers not initialized.");

        var x = _timeEmbed1.Forward(timeEmbed);
        x = _timeEmbed2.Forward(x);
        return x;
    }

    /// <summary>
    /// Forward pass through the DiT.
    /// </summary>
    private Tensor<T> Forward(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        // Trigger lazy allocation if not yet done — callers like
        // PredictNoiseWithEmbedding reach Forward directly on a fresh instance.
        EnsureLayersInitialized();

        if (_patchEmbed == null || _finalNorm == null || _outputProj == null || _adaln_modulation == null)
            throw new InvalidOperationException("Layers not initialized.");

        var shape = x._shape;
        var height = shape[2];
        var width = shape[3];

        // Patchify and embed
        var patches = Patchify(x);
        var numPatches = patches.Shape[1];

        // Linear embed patches
        var hidden = EmbedPatches(patches);

        // Add position embeddings
        hidden = AddPositionEmbedding(hidden, numPatches);

        // Class conditioning (Peebles & Xie 2022 §3.2): when the model was
        // constructed with numClasses > 0, `conditioning` is the class-label
        // tensor (one-hot, shape [B, numClasses]). We project it through the
        // label embedder and ADD it to the time embedding — the sum feeds
        // every AdaLN modulation in every block AND the final-layer AdaLN.
        //
        // When numClasses == 0 (non-class-conditional DiT), `conditioning` is
        // instead used for cross-attention inside each block — that's the
        // text/free-form conditioning path used by Stable-Diffusion-style
        // variants. The two roles are mutually exclusive in the original
        // DiT paper (class-conditional ImageNet DiT had no text path).
        var adaLnEmbed = timeEmbed;
        Tensor<T>? crossAttnCond = conditioning;
        if (_numClasses > 0 && conditioning != null)
        {
            if (_labelEmbed == null)
                throw new InvalidOperationException("Class embedding layer not initialized despite numClasses > 0.");
            if (conditioning.Shape.Length < 2 || conditioning.Shape[^1] != _numClasses)
                throw new ArgumentException(
                    $"Class-conditional DiT expects `conditioning` to be one-hot class labels with last dim {_numClasses}; got shape [{string.Join(",", conditioning.Shape)}].",
                    nameof(conditioning));

            var classEmbed = _labelEmbed.Forward(conditioning);
            adaLnEmbed = Engine.TensorAdd(timeEmbed, classEmbed);
            // Class conditioning consumed as class labels — don't also pass into cross-attention.
            crossAttnCond = null;
        }

        // Process through transformer blocks — every block's AdaLN reads
        // `adaLnEmbed` (= timeEmbed + classEmbed when class-conditional).
        // G4 (#1624): checkpoint each block (recompute activations in backward) — gradient-equivalent.
        // Each closure is a pure function of the residual stream; the AdaLN/cross-attn conditioning is
        // captured as a constant and its gradient is propagated correctly by the checkpoint recompute.
        // Foundation-scale eval: keep each block's large weight matrices fp16-resident (DenseLayer
        // upcasts to fp32 transiently per matmul). Halves the tower's resident weight memory so a
        // multi-GB DiT fits a 16 GB host without disk paging or slow Half arithmetic. Flagged here,
        // before the blocks' first forward, so DenseLayer downcasts its fp32 weights on first use.
        // Gated OFF while a tape is active (training needs the fp32 master for backward).
        bool tapeActive = AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
            && !AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed;
        if (!tapeActive && ParameterCount > LowPrecisionResidentThresholdParams)
        {
            foreach (var b in _blocks) b.EnableLowPrecisionResident();
        }

        var blockForwards = new System.Func<Tensor<T>, Tensor<T>>[_blocks.Count];
        for (int i = 0; i < _blocks.Count; i++)
        {
            var block = _blocks[i];
            blockForwards[i] = h => ForwardBlock(h, adaLnEmbed, crossAttnCond, block);
        }
        hidden = CheckpointBlocks(blockForwards, hidden);

        // Final norm and projection with AdaLN (also uses the combined embedding)
        hidden = FinalLayerWithAdaLN(hidden, adaLnEmbed);

        // Unpatchify back to image
        var output = Unpatchify(hidden, height, width);

        return output;
    }

    /// <summary>
    /// Converts image to patches via a single reshape + permute + reshape.
    /// </summary>
    /// <remarks>
    /// Equivalent to the nested 6-loop scalar copy this used to be:
    /// <c>[B, C, H, W]</c> → reshape <c>[B, C, H/p, p, W/p, p]</c>
    /// → permute to <c>[B, H/p, W/p, C, p, p]</c>
    /// → reshape <c>[B, numPatches, C·p·p]</c>.
    /// All three ops route through <see cref="IEngine"/>, so the permutation
    /// runs through the engine's vectorized memcpy kernel instead of a
    /// scalar C# nested loop. The two reshape steps are zero-copy views; the
    /// permute is the only step that materializes (and only because the
    /// downstream <see cref="DenseLayer{T}.Forward"/> requires contiguous
    /// input).
    /// </remarks>
    private Tensor<T> Patchify(Tensor<T> x)
    {
        var shape = x._shape;
        var batch = shape[0];
        var channels = shape[1];
        var height = shape[2];
        var width = shape[3];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = channels * _patchSize * _patchSize;

        // [B, C, H, W] -> [B, C, H/p, p, W/p, p]
        var split = Engine.Reshape(x,
            new[] { batch, channels, numPatchesH, _patchSize, numPatchesW, _patchSize });
        // -> [B, H/p, W/p, C, p, p]  (axes: 0, 2, 4, 1, 3, 5)
        var permuted = Engine.TensorPermute(split, new[] { 0, 2, 4, 1, 3, 5 });
        // -> [B, numPatches, patchDim]
        return Engine.Reshape(permuted, new[] { batch, numPatches, patchDim });
    }

    /// <summary>
    /// Embeds patches through linear projection using batched forward pass.
    /// </summary>
    /// <remarks>
    /// Uses zero-copy <see cref="IEngine.Reshape"/> views around the dense
    /// projection — the previous <c>TensorAllocator.Rent + CopyTo</c> scratch
    /// buffers were unnecessary because Patchify's output is contiguous.
    /// </remarks>
    private Tensor<T> EmbedPatches(Tensor<T> patches)
    {
        if (_patchEmbed == null)
            throw new InvalidOperationException("Patch embed not initialized.");

        var shape = patches._shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = shape[2];

        // [B, numPatches, patchDim] -> [B·numPatches, patchDim] view, batched dense forward, reshape back as view.
        var flatPatches = Engine.Reshape(patches, new[] { batch * numPatches, patchDim });
        var projected = _patchEmbed.Forward(flatPatches);
        return Engine.Reshape(projected, new[] { batch, numPatches, _hiddenSize });
    }

    /// <summary>
    /// Adds learnable position embeddings using IEngine broadcast add.
    /// </summary>
    private Tensor<T> AddPositionEmbedding(Tensor<T> x, int numPatches)
    {
        // Initialize position embeddings if needed
        if (_posEmbed == null || _posEmbed.Shape[1] != numPatches)
        {
            _posEmbed = CreatePositionEmbedding(numPatches);
        }

        // Position embedding is [1, numPatches, hiddenSize], x is [batch, numPatches, hiddenSize]
        // TensorBroadcastAdd handles the batch dimension broadcasting
        return Engine.TensorBroadcastAdd<T>(x, _posEmbed);
    }

    /// <summary>
    /// Creates sinusoidal position embeddings.
    /// </summary>
    private Tensor<T> CreatePositionEmbedding(int numPatches)
    {
        var posEmbed = TensorAllocator.Rent<T>(new[] { 1, numPatches, _hiddenSize });
        var span = posEmbed.AsWritableSpan();

        for (int pos = 0; pos < numPatches; pos++)
        {
            for (int i = 0; i < _hiddenSize; i += 2)
            {
                var freq = Math.Pow(10000.0, -i / (double)_hiddenSize);
                var angle = pos * freq;

                span[pos * _hiddenSize + i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < _hiddenSize)
                {
                    span[pos * _hiddenSize + i + 1] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }

        return posEmbed;
    }

    /// <summary>
    /// Forward pass through a single DiT block with AdaLN.
    /// </summary>
    /// <remarks>
    /// AdaLN modulation tensor is reshaped to [B, 6, 1, hidden] and the six
    /// shift/scale/gate parameters are obtained as zero-copy
    /// <see cref="IEngine.TensorSliceAxis"/> views — no <c>T[]</c> allocations,
    /// no scalar fill loops. The <c>(1 + scale)</c> precomputation in AdaLN
    /// uses the SIMD <see cref="IEngine.TensorAddScalar"/> primitive instead
    /// of a per-element <c>NumOps.Add(NumOps.One, scale[h])</c> loop.
    /// </remarks>
    private Tensor<T> ForwardBlock(
        Tensor<T> x,
        Tensor<T> timeEmbed,
        Tensor<T>? condEmbed,
        DiTBlock block)
    {
        if (block.Norm1 == null || block.Attention == null ||
            block.Norm2 == null || block.MLP1 == null || block.MLP2 == null ||
            block.AdaLNModulation == null)
        {
            throw new InvalidOperationException("Block layers not initialized.");
        }

        // Get AdaLN modulation parameters (shift1, scale1, gate1, shift2, scale2, gate2).
        // AdaLNModulation forward output is hidden*6 elements per batch — derive
        // batchM from total length so we don't depend on whether the layer
        // returns [B, hidden*6] (rank 2) or [hidden*6] (rank 1, when B=1 and the
        // dense layer collapses leading dims). Reshape to [B, 6, 1, hidden] so
        // TensorSliceAxis(axis=1, index=i) yields a [B, 1, hidden] view directly,
        // broadcastable over [B, seq, hidden] without further reshape.
        var modulation = block.AdaLNModulation.Forward(timeEmbed);
        int stride = 6 * _hiddenSize;
        if (modulation.Length % stride != 0)
        {
            throw new InvalidOperationException(
                $"AdaLNModulation output length {modulation.Length} is not divisible by 6 * hiddenSize " +
                $"({stride}). This indicates the modulation MLP's output size is misconfigured — " +
                $"each DiT block needs exactly 6 * {_hiddenSize} = {stride} modulation parameters per " +
                $"sample (shift/scale/gate × 2 for attention and MLP branches). " +
                $"Check the layer that produced `timeEmbed` and `block.AdaLNModulation`.");
        }
        int batchM = modulation.Length / stride;
        var modReshaped = Engine.Reshape(modulation, new[] { batchM, 6, 1, _hiddenSize });

        var shift1 = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 0);
        var scale1 = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 1);
        var gate1  = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 2);
        var shift2 = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 3);
        var scale2 = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 4);
        var gate2  = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 5);

        // Self-attention with AdaLN
        var normed = block.Norm1.Forward(x);
        normed = ApplyAdaLN(normed, scale1, shift1);

        var attnOut = ApplySelfAttention(normed, block.Attention);
        x = AddWithGate(x, attnOut, gate1);

        // Cross-attention to conditioning (if available)
        if (condEmbed != null)
        {
            x = ApplyCrossAttention(x, condEmbed, block);
        }

        // MLP with AdaLN
        normed = block.Norm2.Forward(x);
        normed = ApplyAdaLN(normed, scale2, shift2);

        var mlpOut = block.MLP1.Forward(normed);
        mlpOut = block.MLP2.Forward(mlpOut);
        x = AddWithGate(x, mlpOut, gate2);

        return x;
    }

    /// <summary>
    /// Applies adaptive layer normalization (Peebles &amp; Xie, 2023):
    /// <c>y = x * (1 + scale) + shift</c>. Both <paramref name="scaleView"/>
    /// and <paramref name="shiftView"/> are <c>[B, 1, hidden]</c> tensor views
    /// sliced from the AdaLN modulation tensor — no scratch allocation, no
    /// scalar fill. The <c>(1 + scale)</c> precomputation runs through the
    /// SIMD <see cref="IEngine.TensorAddScalar"/> kernel.
    /// </summary>
    private Tensor<T> ApplyAdaLN(Tensor<T> x, Tensor<T> scaleView, Tensor<T> shiftView)
    {
        var scalePlusOne = Engine.TensorAddScalar<T>(scaleView, NumOps.One);

        // #1672 destination-buffer path: reuse per-predictor scratch for the two big
        // [B, seq, hidden] broadcasts instead of allocating each step. Same SIMD trailing-
        // repeat kernel → bit-identical. The `scaled` buffer is consumed by the very next
        // broadcast-add; the output buffer is consumed by the caller (attention / MLP /
        // final projection) before the next AdaLN call. See scratch field comments.
        if (UseForwardScratch())
        {
            var needShape = x._shape;
            if (_adaLnScaledScratch == null || !ShapeMatches(_adaLnScaledScratch._shape, needShape))
                _adaLnScaledScratch = new Tensor<T>((int[])needShape.Clone());
            if (_adaLnOutScratch == null || !ShapeMatches(_adaLnOutScratch._shape, needShape))
                _adaLnOutScratch = new Tensor<T>((int[])needShape.Clone());

            Engine.TensorBroadcastMultiplyInto<T>(_adaLnScaledScratch, x, scalePlusOne);
            Engine.TensorBroadcastAddInto<T>(_adaLnOutScratch, _adaLnScaledScratch, shiftView);
            return _adaLnOutScratch;
        }

        var scaled = Engine.TensorBroadcastMultiply<T>(x, scalePlusOne);
        return Engine.TensorBroadcastAdd<T>(scaled, shiftView);
    }

    /// <summary>
    /// Applies self-attention.
    /// </summary>
    private Tensor<T> ApplySelfAttention(
        Tensor<T> x,
        SelfAttentionLayer<T> attention)
    {
        return attention.Forward(x);
    }

    /// <summary>
    /// Applies cross-attention between query (from x) and key/value (from conditioning).
    /// Uses IEngine for hardware-accelerated batched matrix operations.
    /// </summary>
    private Tensor<T> ApplyCrossAttention(
        Tensor<T> x,
        Tensor<T> conditioning,
        DiTBlock block)
    {
        if (block.CrossAttnNorm == null || block.CrossAttnQ == null ||
            block.CrossAttnK == null || block.CrossAttnV == null || block.CrossAttnOut == null)
        {
            return x;
        }

        var shape = x._shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var hidden = shape[2];

        var condShape = conditioning._shape;
        var condSeqLen = condShape.Length > 1 ? condShape[1] : 1;

        // Normalize x
        var normed = block.CrossAttnNorm.Forward(x);

        // Compute Q from x, K and V from conditioning
        var q = block.CrossAttnQ.Forward(normed);
        var k = block.CrossAttnK.Forward(conditioning);
        var v = block.CrossAttnV.Forward(conditioning);

        // Reshape Q, K, V for multi-head attention: [batch, seq, hidden] -> [batch*heads, seq, headDim]
        var headDim = hidden / _numHeads;
        var scaleFactor = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        var qReshaped = ReshapeForHeads(q, batch, seqLen, _numHeads, headDim);
        var kReshaped = ReshapeForHeads(k, batch, condSeqLen, _numHeads, headDim);
        var vReshaped = ReshapeForHeads(v, batch, condSeqLen, _numHeads, headDim);

        // Compute attention scores: Q * K^T using batched matmul.
        // K is batched [batch*heads, condSeqLen, headDim], so transpose only
        // the last two dimensions to [batch*heads, headDim, condSeqLen].
        var kTransposed = Engine.TensorPermute(kReshaped, new[] { 0, 2, 1 });
        var scores = Engine.TensorBatchMatMul<T>(qReshaped, kTransposed);

        // Scale scores
        scores = Engine.TensorMultiplyScalar<T>(scores, scaleFactor);

        // Softmax over last axis (key dimension)
        var attnWeights = Engine.Softmax<T>(scores, axis: -1);

        // Apply attention to values: attn_weights * V [batch*heads, seqLen, headDim]
        var attnOutput = Engine.TensorBatchMatMul<T>(attnWeights, vReshaped);

        // Reshape back: [batch*heads, seqLen, headDim] -> [batch, seqLen, hidden]
        attnOutput = ReshapeFromHeads(attnOutput, batch, seqLen, _numHeads, headDim);

        // Output projection and residual connection
        var projected = block.CrossAttnOut.Forward(attnOutput);
        return Engine.TensorAdd<T>(x, projected);
    }

    /// <summary>
    /// Reshapes tensor from <c>[batch, seq, hidden]</c> to
    /// <c>[batch*heads, seq, headDim]</c> for multi-head attention via a
    /// reshape + permute + reshape pipeline (no scalar nested copy loops).
    /// </summary>
    private Tensor<T> ReshapeForHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        // [B, S, H·D] -> [B, S, H, D]
        var split = Engine.Reshape(tensor, new[] { batch, seq, numHeads, headDim });
        // -> [B, H, S, D]
        var permuted = Engine.TensorPermute(split, new[] { 0, 2, 1, 3 });
        // -> [B·H, S, D]
        return Engine.Reshape(permuted, new[] { batch * numHeads, seq, headDim });
    }

    /// <summary>
    /// Inverse of <see cref="ReshapeForHeads"/> — collapses head and batch
    /// back together via reshape + permute + reshape.
    /// </summary>
    private Tensor<T> ReshapeFromHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        // [B·H, S, D] -> [B, H, S, D]
        var split = Engine.Reshape(tensor, new[] { batch, numHeads, seq, headDim });
        // -> [B, S, H, D]
        var permuted = Engine.TensorPermute(split, new[] { 0, 2, 1, 3 });
        // -> [B, S, H·D]
        return Engine.Reshape(permuted, new[] { batch, seq, numHeads * headDim });
    }

    /// <summary>
    /// Gated residual add: <c>result = x + gateView * residual</c>.
    /// <paramref name="gateView"/> is a <c>[B, 1, hidden]</c> view sliced from
    /// the AdaLN modulation tensor — no scratch allocation, no scalar fill.
    /// Uses <see cref="IEngine.TensorBroadcastMultiply"/> for the per-channel
    /// gate broadcast.
    /// </summary>
    private Tensor<T> AddWithGate(Tensor<T> x, Tensor<T> residual, Tensor<T> gateView)
    {
        // #1672: reuse scratch for the `gated` intermediate (consumed immediately by the
        // following add). The RESULT is the persistent residual stream, so it stays a fresh
        // allocation — never scratch. Same SIMD kernel → bit-identical.
        if (UseForwardScratch())
        {
            var needShape = residual._shape;
            if (_gateScratch == null || !ShapeMatches(_gateScratch._shape, needShape))
                _gateScratch = new Tensor<T>((int[])needShape.Clone());
            Engine.TensorBroadcastMultiplyInto<T>(_gateScratch, residual, gateView);
            return Engine.TensorAdd<T>(x, _gateScratch);
        }

        var gated = Engine.TensorBroadcastMultiply<T>(residual, gateView);
        return Engine.TensorAdd<T>(x, gated);
    }

    /// <summary>
    /// Final layer with AdaLN-zero using batched forward pass.
    /// </summary>
    /// <remarks>
    /// Modulation slicing matches <see cref="ForwardBlock"/>: reshape to
    /// <c>[B, 2, 1, hidden]</c> then <see cref="IEngine.TensorSliceAxis"/> for
    /// shift/scale views. Reshape between the projection input and output uses
    /// <see cref="IEngine.Reshape"/> views instead of
    /// <c>TensorAllocator.Rent</c>+<c>CopyTo</c> scratch buffers, eliminating
    /// two allocations per inference step.
    /// </remarks>
    private Tensor<T> FinalLayerWithAdaLN(Tensor<T> x, Tensor<T> timeEmbed)
    {
        if (_finalNorm == null || _adaln_modulation == null || _outputProj == null)
            throw new InvalidOperationException("Final layers not initialized.");

        var modulation = _adaln_modulation.Forward(timeEmbed);
        int stride = 2 * _hiddenSize;
        if (modulation.Length % stride != 0)
        {
            throw new InvalidOperationException(
                $"Final-layer AdaLN modulation output length {modulation.Length} is not divisible " +
                $"by 2 * hiddenSize ({stride}). The final-layer modulation MLP must emit exactly " +
                $"2 * {_hiddenSize} = {stride} parameters per sample (shift + scale). Check the " +
                $"layer that produced `timeEmbed` and `_adaln_modulation`.");
        }
        int batchM = modulation.Length / stride;
        var modReshaped = Engine.Reshape(modulation, new[] { batchM, 2, 1, _hiddenSize });
        var shiftView = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 0);
        var scaleView = Engine.TensorSliceAxis(modReshaped, axis: 1, index: 1);

        var normed = _finalNorm.Forward(x);
        normed = ApplyAdaLN(normed, scaleView, shiftView);

        var shape = normed._shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = _inputChannels * _patchSize * _patchSize;

        // [batch, numPatches, hiddenSize] -> [batch*numPatches, hiddenSize] via view.
        var flatNormed = Engine.Reshape(normed, new[] { batch * numPatches, _hiddenSize });
        var projected = _outputProj.Forward(flatNormed);
        // Back to [batch, numPatches, patchDim] via view.
        return Engine.Reshape(projected, new[] { batch, numPatches, patchDim });
    }

    /// <summary>
    /// Inverse of <see cref="Patchify"/> — reconstructs the spatial image.
    /// </summary>
    /// <remarks>
    /// <c>[B, numPatches, C·p·p]</c> → reshape <c>[B, H/p, W/p, C, p, p]</c>
    /// → permute to <c>[B, C, H/p, p, W/p, p]</c>
    /// → reshape <c>[B, C, H, W]</c>. Same vectorized-memcpy + view pattern
    /// as <see cref="Patchify"/> — zero scalar loops, one materialization.
    /// </remarks>
    private Tensor<T> Unpatchify(Tensor<T> patches, int height, int width)
    {
        var shape = patches._shape;
        var batch = shape[0];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;

        // [B, numPatches, C·p·p] -> [B, H/p, W/p, C, p, p]
        var unsplit = Engine.Reshape(patches,
            new[] { batch, numPatchesH, numPatchesW, _inputChannels, _patchSize, _patchSize });
        // -> [B, C, H/p, p, W/p, p]  (inverse permutation: axes 0, 3, 1, 4, 2, 5)
        var permuted = Engine.TensorPermute(unsplit, new[] { 0, 3, 1, 4, 2, 5 });
        // -> [B, C, H, W]
        return Engine.Reshape(permuted, new[] { batch, _inputChannels, height, width });
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        EnsureLayersInitialized();

        // Pre-allocate with known size so we avoid the List<T> doubling
        // path AND its ToArray() copy. For a real-scale DiT (Bark uses
        // 24 blocks × 1024 hidden ≈ 250 M parameters; 8-byte doubles ⇒
        // 2 GB), the previous List+ToArray pattern peaked at ~3× that
        // size during the doubling and copy, OOMing CI test hosts.
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        var result = new Vector<T>(totalParams);
        int offset = 0;

        if (_patchEmbed != null) WriteLayerParams(result, ref offset, _patchEmbed);
        if (_timeEmbed1 != null) WriteLayerParams(result, ref offset, _timeEmbed1);
        if (_timeEmbed2 != null) WriteLayerParams(result, ref offset, _timeEmbed2);
        if (_labelEmbed != null) WriteLayerParams(result, ref offset, _labelEmbed);

        foreach (var block in _blocks)
        {
            if (block.Norm1 != null) WriteLayerParams(result, ref offset, block.Norm1);
            if (block.Attention != null) WriteLayerParams(result, ref offset, block.Attention);
            if (block.Norm2 != null) WriteLayerParams(result, ref offset, block.Norm2);
            if (block.MLP1 != null) WriteLayerParams(result, ref offset, block.MLP1);
            if (block.MLP2 != null) WriteLayerParams(result, ref offset, block.MLP2);
            if (block.AdaLNModulation != null) WriteLayerParams(result, ref offset, block.AdaLNModulation);
            if (block.CrossAttnNorm != null) WriteLayerParams(result, ref offset, block.CrossAttnNorm);
            if (block.CrossAttnQ != null) WriteLayerParams(result, ref offset, block.CrossAttnQ);
            if (block.CrossAttnK != null) WriteLayerParams(result, ref offset, block.CrossAttnK);
            if (block.CrossAttnV != null) WriteLayerParams(result, ref offset, block.CrossAttnV);
            if (block.CrossAttnOut != null) WriteLayerParams(result, ref offset, block.CrossAttnOut);
        }

        if (_finalNorm != null) WriteLayerParams(result, ref offset, _finalNorm);
        if (_adaln_modulation != null) WriteLayerParams(result, ref offset, _adaln_modulation);
        if (_outputProj != null) WriteLayerParams(result, ref offset, _outputProj);

        // Validate the final offset matches the pre-allocated buffer.
        // A mismatch means some layer's `ParameterCount` disagreed with
        // its `GetParameters().Length` between the two reads — most
        // likely caused by a lazy-init layer materializing weights
        // mid-walk and changing its reported count. Throwing here turns
        // a silently corrupt parameter dump (random tail garbage or
        // lost trailing layers) into an actionable exception.
        if (offset != totalParams)
        {
            throw new InvalidOperationException(
                $"DiTNoisePredictor.GetParameters wrote {offset} elements but " +
                $"ParameterCount reported {totalParams}. Some layer's " +
                $"GetParameters().Length doesn't match its ParameterCount — " +
                $"check for layers whose weight tensors materialized between " +
                $"the count and the write (lazy init mid-walk).");
        }
        return result;
    }

    private static void WriteLayerParams(Vector<T> dst, ref int offset, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        // Bounds check: catch the failure here instead of letting
        // `dst[offset + i]` throw the harder-to-diagnose
        // ArgumentOutOfRangeException several layers later.
        if (offset + p.Length > dst.Length)
        {
            throw new InvalidOperationException(
                $"WriteLayerParams overflow at layer {layer.GetType().Name}: " +
                $"offset={offset}, p.Length={p.Length}, buffer.Length={dst.Length}. " +
                $"ParameterCount under-counted this layer's actual parameter count.");
        }
        for (int i = 0; i < p.Length; i++)
        {
            dst[offset + i] = p[i];
        }
        offset += p.Length;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        EnsureLayersInitialized();
        int offset = 0;

        // Set patch embed
        if (_patchEmbed != null)
        {
            offset = SetLayerParams(_patchEmbed, parameters, offset);
        }

        // Set time embedding
        if (_timeEmbed1 != null) offset = SetLayerParams(_timeEmbed1, parameters, offset);
        if (_timeEmbed2 != null) offset = SetLayerParams(_timeEmbed2, parameters, offset);

        // Set label embed (optional)
        if (_labelEmbed != null) offset = SetLayerParams(_labelEmbed, parameters, offset);

        // Set transformer blocks
        foreach (var block in _blocks)
        {
            if (block.Norm1 != null) offset = SetLayerParams(block.Norm1, parameters, offset);
            if (block.Attention != null) offset = SetLayerParams(block.Attention, parameters, offset);
            if (block.Norm2 != null) offset = SetLayerParams(block.Norm2, parameters, offset);
            if (block.MLP1 != null) offset = SetLayerParams(block.MLP1, parameters, offset);
            if (block.MLP2 != null) offset = SetLayerParams(block.MLP2, parameters, offset);
            if (block.AdaLNModulation != null) offset = SetLayerParams(block.AdaLNModulation, parameters, offset);
            // Cross-attention layers
            if (block.CrossAttnNorm != null) offset = SetLayerParams(block.CrossAttnNorm, parameters, offset);
            if (block.CrossAttnQ != null) offset = SetLayerParams(block.CrossAttnQ, parameters, offset);
            if (block.CrossAttnK != null) offset = SetLayerParams(block.CrossAttnK, parameters, offset);
            if (block.CrossAttnV != null) offset = SetLayerParams(block.CrossAttnV, parameters, offset);
            if (block.CrossAttnOut != null) offset = SetLayerParams(block.CrossAttnOut, parameters, offset);
        }

        // Set final layers
        if (_finalNorm != null) offset = SetLayerParams(_finalNorm, parameters, offset);
        if (_adaln_modulation != null) offset = SetLayerParams(_adaln_modulation, parameters, offset);
        if (_outputProj != null) offset = SetLayerParams(_outputProj, parameters, offset);
    }

    private int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    /// <summary>
    /// Copies parameters layer-by-layer from <paramref name="source"/> into this
    /// predictor. Avoids the round-trip through a single flat
    /// <see cref="Vector{T}"/> that <see cref="GetParameters"/> +
    /// <see cref="SetParameters"/> would otherwise produce — for real-scale
    /// DiT models (Bark: ~360M parameters; ~3 GB as doubles), the flat
    /// intermediate triples peak memory and OOMs CI test hosts. Per-layer
    /// copy keeps peak at ~2× model weights instead of ~3×.
    /// </summary>
    /// <remarks>
    /// Both <paramref name="source"/> and this instance must have been
    /// initialized with the same architecture (matching layer counts,
    /// hidden sizes, etc.). The walk order must mirror
    /// <see cref="GetParameters"/> exactly so corresponding layers line up.
    /// </remarks>
    public void CopyParametersFrom(DiTNoisePredictor<T> source)
    {
        Guard.NotNull(source);
        source.EnsureLayersInitialized();
        EnsureLayersInitialized();

        if (source._blocks.Count != _blocks.Count)
        {
            throw new ArgumentException(
                $"Source has {source._blocks.Count} transformer blocks but " +
                $"target has {_blocks.Count}; cannot copy parameters across " +
                "different architectures.",
                nameof(source));
        }

        CopyLayerSafely(source._patchEmbed, _patchEmbed);
        CopyLayerSafely(source._timeEmbed1, _timeEmbed1);
        CopyLayerSafely(source._timeEmbed2, _timeEmbed2);
        CopyLayerSafely(source._labelEmbed, _labelEmbed);

        for (int i = 0; i < _blocks.Count; i++)
        {
            var src = source._blocks[i];
            var dst = _blocks[i];
            CopyLayerSafely(src.Norm1, dst.Norm1);
            CopyLayerSafely(src.Attention, dst.Attention);
            CopyLayerSafely(src.Norm2, dst.Norm2);
            CopyLayerSafely(src.MLP1, dst.MLP1);
            CopyLayerSafely(src.MLP2, dst.MLP2);
            CopyLayerSafely(src.AdaLNModulation, dst.AdaLNModulation);
            CopyLayerSafely(src.CrossAttnNorm, dst.CrossAttnNorm);
            CopyLayerSafely(src.CrossAttnQ, dst.CrossAttnQ);
            CopyLayerSafely(src.CrossAttnK, dst.CrossAttnK);
            CopyLayerSafely(src.CrossAttnV, dst.CrossAttnV);
            CopyLayerSafely(src.CrossAttnOut, dst.CrossAttnOut);
        }

        CopyLayerSafely(source._finalNorm, _finalNorm);
        CopyLayerSafely(source._adaln_modulation, _adaln_modulation);
        CopyLayerSafely(source._outputProj, _outputProj);
    }

    private static void CopyLayerSafely(ILayer<T>? source, ILayer<T>? target)
    {
        if (source == null && target == null) return;
        if (source == null || target == null)
        {
            throw new InvalidOperationException(
                "Source and target layer presence mismatch — both must be " +
                "non-null or both null. Architectures may differ.");
        }
        target.SetParameters(source.GetParameters());
    }

    private IEnumerable<ILayer<T>?> EnumerateAllLayers()
    {
        // ORDER IS LOAD-BEARING: GetParameterChunks/SetParameterChunks walk this sequence, so it MUST
        // match GetParameters/SetParameters element-for-element (PredictorParameterStreamingTests'
        // *_Chunks_IndexIdentical contract). The model-level _adaln_modulation is serialized near the
        // END (after _finalNorm, before _outputProj) by GetParameters/SetParameters — emit it there,
        // NOT after _labelEmbed, or the chunk concatenation desyncs from the flat vector.
        yield return _patchEmbed;
        yield return _timeEmbed1;
        yield return _timeEmbed2;
        yield return _labelEmbed;
        foreach (var block in _blocks)
        {
            yield return block.Norm1;
            yield return block.Attention;
            yield return block.Norm2;
            yield return block.MLP1;
            yield return block.MLP2;
            yield return block.AdaLNModulation;
            yield return block.CrossAttnNorm;
            yield return block.CrossAttnQ;
            yield return block.CrossAttnK;
            yield return block.CrossAttnV;
            yield return block.CrossAttnOut;
        }
        yield return _finalNorm;
        yield return _adaln_modulation;
        yield return _outputProj;
    }

    private bool HasMaterializedParameters()
    {
        if (!_layersInitialized) return false;

        foreach (var layer in EnumerateAllLayers())
        {
            if (HasMaterializedParameters(layer))
                return true;
        }

        return false;
    }

    private static bool HasMaterializedParameters(ILayer<T>? layer)
    {
        if (layer is null) return false;

        if (layer is ITrainableLayer<T> trainable)
        {
            foreach (var parameter in trainable.GetTrainableParameters())
            {
                if (parameter.Length > 0)
                    return true;
            }
        }

        foreach (var subLayer in layer.GetSubLayers())
        {
            if (HasMaterializedParameters(subLayer))
                return true;
        }

        return false;
    }

    /// <inheritdoc />
    public override long ParameterCount
    {
        get
        {
            // #1237: long accumulator. DiT-XL/2 with HiddenDim 3072 × 48
            // layers (Sora's paper config) sums to ~5.4 B parameters,
            // overflowing int.MaxValue. Per-layer ParameterCount stays
            // int (single-tensor < 2.1 B); the cross-layer sum is long.
            EnsureLayersInitialized();
            long count = 0;

            if (_patchEmbed != null) count += _patchEmbed.ParameterCount;
            if (_timeEmbed1 != null) count += _timeEmbed1.ParameterCount;
            if (_timeEmbed2 != null) count += _timeEmbed2.ParameterCount;
            if (_labelEmbed != null) count += _labelEmbed.ParameterCount;

            foreach (var block in _blocks)
            {
                if (block.Norm1 != null) count += block.Norm1.ParameterCount;
                if (block.Attention != null) count += block.Attention.ParameterCount;
                if (block.Norm2 != null) count += block.Norm2.ParameterCount;
                if (block.MLP1 != null) count += block.MLP1.ParameterCount;
                if (block.MLP2 != null) count += block.MLP2.ParameterCount;
                if (block.AdaLNModulation != null) count += block.AdaLNModulation.ParameterCount;
                if (block.CrossAttnNorm != null) count += block.CrossAttnNorm.ParameterCount;
                if (block.CrossAttnQ != null) count += block.CrossAttnQ.ParameterCount;
                if (block.CrossAttnK != null) count += block.CrossAttnK.ParameterCount;
                if (block.CrossAttnV != null) count += block.CrossAttnV.ParameterCount;
                if (block.CrossAttnOut != null) count += block.CrossAttnOut.ParameterCount;
            }

            if (_finalNorm != null) count += _finalNorm.ParameterCount;
            if (_adaln_modulation != null) count += _adaln_modulation.ParameterCount;
            if (_outputProj != null) count += _outputProj.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new DiTNoisePredictor<T>(
            inputChannels: _inputChannels,
            hiddenSize: _hiddenSize,
            numLayers: _numLayers,
            numHeads: _numHeads,
            patchSize: _patchSize,
            contextDim: _contextDim,
            mlpRatio: _mlpRatio,
            latentSpatialSize: _latentSpatialSize);

        // Preserve trained/materialized weights without forcing a foundation-scale default
        // constructor to allocate and copy billions of random parameters (HasMaterializedParameters
        // gates the copy to a source that genuinely has allocated weights).
        //
        // The DiT's projection layers are LazyDense — they only ALLOCATE their weight tensors on
        // the first forward, not in EnsureLayersInitialized. A fresh clone therefore has the layer
        // STRUCTURE but unallocated weights, so CopyParametersFrom alone has nothing to copy INTO;
        // the clone would re-initialize those tensors with a fresh RNG on its first real forward
        // and diverge from the source (observed: ~50k fewer materialized params, divergent Predict).
        // Run one throwaway forward at the canonical input shape to materialize EVERY weight tensor
        // on the clone first (weight dims are fixed by config, so the probe's spatial size is
        // irrelevant), then copy the source's trained values. The probe must materialize exactly the
        // paths the source has materialized so CopyParametersFrom finds a target for every source
        // weight. A null-conditioned probe only touches the unconditional path, so if the source was
        // used WITH conditioning its class-embedding (_labelEmbed) and/or cross-attention K/V/Out
        // projections are materialized — leaving the clone's equivalents lazy would let them re-init
        // with fresh RNG on the first conditioned forward and diverge from the source.
        // BuildProbeConditioning returns representative conditioning whenever a conditioned path is
        // materialized on the source (and null otherwise, keeping those layers lazy on both).
        if (HasMaterializedParameters())
        {
            var probe = new Tensor<T>(new[] { 1, _inputChannels, _latentSpatialSize, _latentSpatialSize });
            clone.PredictNoise(probe, timestep: 0, conditioning: BuildProbeConditioning());
            clone.CopyParametersFrom(this);
            // The probe forward traced a compiled plan over the clone's random init; drop it so
            // the next real forward re-traces against the copied weights.
            clone.InvalidateCompiledPlans();
        }
        // else: source has no materialized weights — nothing to copy. The clone shares the same
        // config and initializes lazily on first use; calling GetParameters() here would allocate
        // the full (foundation-scale) parameter vector for nothing.
        return clone;
    }

    /// <summary>
    /// Builds a representative conditioning tensor for the <see cref="Clone"/> probe forward,
    /// matching whichever conditioned path the source has materialized so that path is allocated on
    /// the clone before <see cref="CopyParametersFrom"/> runs. Returns <c>null</c> when no conditioned
    /// path is materialized — the source's conditioned layers are lazy, so the clone's stay lazy too.
    /// </summary>
    private Tensor<T>? BuildProbeConditioning()
    {
        // Class-conditional DiT: a one-hot [1, numClasses] label materializes _labelEmbed.
        if (_numClasses > 0 && _labelEmbed is { IsInitialized: true })
        {
            return new Tensor<T>(new[] { 1, _numClasses });
        }

        // Text/cross-attention DiT: a [1, 1, contextDim] context tensor materializes the per-block
        // cross-attention K/V/Out projections.
        if (_contextDim > 0 && HasMaterializedCrossAttention())
        {
            return new Tensor<T>(new[] { 1, 1, _contextDim });
        }

        return null;
    }

    /// <summary>
    /// True when any transformer block's cross-attention key projection has materialized its weights,
    /// i.e. the source ran at least one conditioned (text/context) forward.
    /// </summary>
    private bool HasMaterializedCrossAttention()
    {
        foreach (var block in _blocks)
        {
            if (block.CrossAttnK is { IsInitialized: true })
            {
                return true;
            }
        }
        return false;
    }

    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize * 4;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;

    /// <summary>
    /// Streams DiT's materialized trainable tensors directly from each layer,
    /// matching the PyTorch <c>nn.Module.parameters()</c> contract: yielded
    /// tensors are the same objects used by forward/training, not flat copies.
    /// Lazy paper-scale defaults may report a structural
    /// <see cref="ParameterCount"/> before their tensors have been allocated;
    /// in that state this iterator yields only already-materialized tensors
    /// instead of forcing a multi-billion-parameter allocation.
    /// </summary>
    public override IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // #1715: engage full-precision weight streaming before initializing/iterating so foundation-scale
        // DiT predictors (e.g. SiT) route their weight allocation through the streaming pool (bounded
        // resident set + lossless write-back) instead of accumulating the full set via RentPinned → OOM.
        // No-op below the param-count/memory threshold; full-precision so the round-trip is exact.
        MaybeEngageWeightStreaming(fullPrecisionStore: true);
        EnsureLayersInitialized();

        foreach (var layer in EnumerateAllLayers())
        {
            foreach (var parameter in EnumerateMaterializedParameters(layer))
                yield return parameter;
        }
    }

    /// <summary>
    /// #1715: per-tensor counterpart to <see cref="GetParameterChunks"/> — copies each incoming chunk
    /// IN PLACE into the corresponding resident weight, in the same EnumerateAllLayers ×
    /// EnumerateMaterializedParameters order, instead of the base implementation that buffers every
    /// chunk into one flat list + Vector (which re-materializes the whole foundation-scale weight set
    /// at once → OOM). Engages full-precision streaming first so the writes round-trip losslessly and
    /// the resident set stays bounded.
    /// </summary>
    public override void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));
        MaybeEngageWeightStreaming(fullPrecisionStore: true);
        EnsureLayersInitialized();

        using var e = chunks.GetEnumerator();
        foreach (var layer in EnumerateAllLayers())
        {
            foreach (var dst in EnumerateMaterializedParameters(layer))
            {
                if (!e.MoveNext())
                    throw new System.ArgumentException(
                        "SetParameterChunks received fewer chunks than the predictor has parameter tensors.",
                        nameof(chunks));
                var src = e.Current;
                if (src is null)
                    throw new System.ArgumentException("SetParameterChunks received a null chunk.", nameof(chunks));
                if (src.Length != dst.Length)
                    throw new System.ArgumentException(
                        $"SetParameterChunks chunk length {src.Length} does not match parameter length {dst.Length}.",
                        nameof(chunks));
                src.Data.Span.CopyTo(dst.Data.Span); // in place — no rebinding, no flat aggregate
            }
        }
    }

    private static IEnumerable<Tensor<T>> EnumerateMaterializedParameters(ILayer<T>? layer)
    {
        if (layer is null) yield break;

        if (layer is ITrainableLayer<T> trainable)
        {
            foreach (var parameter in trainable.GetTrainableParameters())
            {
                if (parameter is null || parameter.Length == 0) continue;
                yield return parameter;
            }
        }

        foreach (var subLayer in layer.GetSubLayers())
        {
            foreach (var parameter in EnumerateMaterializedParameters(subLayer))
                yield return parameter;
        }
    }

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    protected override Vector<T> GetParameterGradients()
    {
        EnsureLayersInitialized();

        // Same pre-allocate-and-fill pattern as GetParameters — see
        // notes there. Avoids List<T> doubling + ToArray() copy on
        // models with hundreds of millions of parameters.
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        var result = new Vector<T>(totalParams);
        int offset = 0;

        WriteLayerGrads(result, ref offset, _patchEmbed);
        WriteLayerGrads(result, ref offset, _timeEmbed1);
        WriteLayerGrads(result, ref offset, _timeEmbed2);
        WriteLayerGrads(result, ref offset, _labelEmbed);

        foreach (var block in _blocks)
        {
            WriteLayerGrads(result, ref offset, block.Norm1);
            WriteLayerGrads(result, ref offset, block.Attention);
            WriteLayerGrads(result, ref offset, block.Norm2);
            WriteLayerGrads(result, ref offset, block.MLP1);
            WriteLayerGrads(result, ref offset, block.MLP2);
            WriteLayerGrads(result, ref offset, block.AdaLNModulation);
            WriteLayerGrads(result, ref offset, block.CrossAttnNorm);
            WriteLayerGrads(result, ref offset, block.CrossAttnQ);
            WriteLayerGrads(result, ref offset, block.CrossAttnK);
            WriteLayerGrads(result, ref offset, block.CrossAttnV);
            WriteLayerGrads(result, ref offset, block.CrossAttnOut);
        }

        WriteLayerGrads(result, ref offset, _finalNorm);
        WriteLayerGrads(result, ref offset, _adaln_modulation);
        WriteLayerGrads(result, ref offset, _outputProj);

        // Mirror the offset validation in GetParameters so a layer whose
        // GetParameterGradients().Length disagrees with ParameterCount —
        // common after a lazy-shape resolve that updated the param tensor
        // shapes but not the cached gradient bookkeeping — surfaces with
        // an actionable error instead of returning a partially filled
        // gradient vector that silently corrupts the optimizer step.
        if (offset != totalParams)
        {
            throw new InvalidOperationException(
                $"DiTNoisePredictor.GetParameterGradients wrote {offset} elements but " +
                $"ParameterCount reported {totalParams}. A child layer's gradient length " +
                $"diverged from its parameter count — check for lazy-shape resolves that " +
                $"updated weights without rebuilding the gradient cache.");
        }

        return result;
    }

    private static void WriteLayerGrads(Vector<T> dst, ref int offset, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        // Bounds-guard the destination write so a layer whose gradient
        // length disagrees with its ParameterCount surfaces here with an
        // actionable error message, instead of throwing an opaque
        // IndexOutOfRangeException at an unrelated later layer.
        if (offset + g.Length > dst.Length)
        {
            throw new InvalidOperationException(
                $"DiTNoisePredictor.WriteLayerGrads: layer of type {layer.GetType().Name} " +
                $"emitted {g.Length} gradients at offset {offset}, but the destination " +
                $"buffer only has {dst.Length} slots. Likely cause: a lazy-shape resolve " +
                $"changed this layer's parameter count after ParameterCount was sampled.");
        }
        for (int i = 0; i < g.Length; i++) dst[offset + i] = g[i];
        offset += g.Length;
    }

    /// <summary>
    /// Block structure for DiT transformer layers containing attention, MLP, and conditioning layers.
    /// </summary>
    public class DiTBlock
    {
        public LayerNormalizationLayer<T>? Norm1 { get; set; }
        public SelfAttentionLayer<T>? Attention { get; set; }
        public LayerNormalizationLayer<T>? Norm2 { get; set; }
        public DenseLayer<T>? MLP1 { get; set; }
        public DenseLayer<T>? MLP2 { get; set; }
        public DenseLayer<T>? AdaLNModulation { get; set; }
        public DenseLayer<T>? CrossAttnQ { get; set; }
        public DenseLayer<T>? CrossAttnK { get; set; }
        public DenseLayer<T>? CrossAttnV { get; set; }
        public DenseLayer<T>? CrossAttnOut { get; set; }
        public LayerNormalizationLayer<T>? CrossAttnNorm { get; set; }

        /// <summary>
        /// Flags this block's large weight matrices (MLP + AdaLN + cross-attention projections) for
        /// fp16-resident inference: each is stored at half precision and upcast to fp32 transiently
        /// per forward (see <see cref="DenseLayer{T}"/>), halving resident weight memory. Norm layers
        /// are tiny and left full precision. Must be called before the block's first forward.
        /// </summary>
        internal void EnableLowPrecisionResident()
        {
            if (MLP1 is not null) MLP1.LowPrecisionResident = true;
            if (MLP2 is not null) MLP2.LowPrecisionResident = true;
            if (AdaLNModulation is not null) AdaLNModulation.LowPrecisionResident = true;
            if (CrossAttnQ is not null) CrossAttnQ.LowPrecisionResident = true;
            if (CrossAttnK is not null) CrossAttnK.LowPrecisionResident = true;
            if (CrossAttnV is not null) CrossAttnV.LowPrecisionResident = true;
            if (CrossAttnOut is not null) CrossAttnOut.LowPrecisionResident = true;
            if (Attention is not null) Attention.LowPrecisionResident = true;
        }
    }
}
