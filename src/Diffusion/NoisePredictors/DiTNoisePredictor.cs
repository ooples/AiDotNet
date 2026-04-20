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
    /// Position embeddings (learnable).
    /// </summary>
    private Tensor<T>? _posEmbed;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

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
        var timeEmbedDim = _hiddenSize * 4;

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

        _finalNorm = new LayerNormalizationLayer<T>(_hiddenSize);
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
                    Norm1 = new LayerNormalizationLayer<T>(_hiddenSize),
                    Attention = CreateAttentionLayer(),
                    Norm2 = new LayerNormalizationLayer<T>(_hiddenSize),
                    MLP1 = mlp1,
                    MLP2 = LazyDense((int)(_hiddenSize * _mlpRatio), _hiddenSize),
                    AdaLNModulation = LazyDense(timeEmbedDim, _hiddenSize * 6),
                    CrossAttnNorm = new LayerNormalizationLayer<T>(_hiddenSize),
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
                Norm1 = new LayerNormalizationLayer<T>(_hiddenSize),
                Attention = CreateAttentionLayer(),
                Norm2 = new LayerNormalizationLayer<T>(_hiddenSize),
                MLP1 = LazyDense(_hiddenSize, mlpHidden, new GELUActivation<T>()),
                MLP2 = LazyDense(mlpHidden, _hiddenSize),
                AdaLNModulation = LazyDense(timeEmbedDim, _hiddenSize * 6),
                CrossAttnNorm = new LayerNormalizationLayer<T>(_hiddenSize),
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
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        EnsureLayersInitialized();
        _lastInput = noisySample;

        // Get timestep embedding
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        return Forward(noisySample, timeEmbed, conditioning);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        EnsureLayersInitialized();
        _lastInput = noisySample;
        return Forward(noisySample, timeEmbedding, conditioning);
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
        foreach (var block in _blocks)
        {
            hidden = ForwardBlock(hidden, adaLnEmbed, crossAttnCond, block);
        }

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

        // Compute attention scores: Q * K^T using batched matmul [batch*heads, seqLen, condSeqLen]
        var kTransposed = Engine.TensorTranspose<T>(kReshaped);
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
        var allParams = new List<T>();

        // Collect from patch embed
        if (_patchEmbed != null)
        {
            AddLayerParams(allParams, _patchEmbed);
        }

        // Collect from time embedding
        if (_timeEmbed1 != null) AddLayerParams(allParams, _timeEmbed1);
        if (_timeEmbed2 != null) AddLayerParams(allParams, _timeEmbed2);

        // Collect from label embed (optional)
        if (_labelEmbed != null) AddLayerParams(allParams, _labelEmbed);

        // Collect from transformer blocks
        foreach (var block in _blocks)
        {
            if (block.Norm1 != null) AddLayerParams(allParams, block.Norm1);
            if (block.Attention != null) AddLayerParams(allParams, block.Attention);
            if (block.Norm2 != null) AddLayerParams(allParams, block.Norm2);
            if (block.MLP1 != null) AddLayerParams(allParams, block.MLP1);
            if (block.MLP2 != null) AddLayerParams(allParams, block.MLP2);
            if (block.AdaLNModulation != null) AddLayerParams(allParams, block.AdaLNModulation);
            // Cross-attention layers
            if (block.CrossAttnNorm != null) AddLayerParams(allParams, block.CrossAttnNorm);
            if (block.CrossAttnQ != null) AddLayerParams(allParams, block.CrossAttnQ);
            if (block.CrossAttnK != null) AddLayerParams(allParams, block.CrossAttnK);
            if (block.CrossAttnV != null) AddLayerParams(allParams, block.CrossAttnV);
            if (block.CrossAttnOut != null) AddLayerParams(allParams, block.CrossAttnOut);
        }

        // Collect from final layers
        if (_finalNorm != null) AddLayerParams(allParams, _finalNorm);
        if (_adaln_modulation != null) AddLayerParams(allParams, _adaln_modulation);
        if (_outputProj != null) AddLayerParams(allParams, _outputProj);

        return new Vector<T>(allParams.ToArray());
    }

    private void AddLayerParams(List<T> allParams, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
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
        var count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            EnsureLayersInitialized();
            int count = 0;

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

        // Preserve trained weights
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize * 4;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    protected override Vector<T> GetParameterGradients()
    {
        EnsureLayersInitialized();
        var allGrads = new List<T>();

        AddLayerGrads(allGrads, _patchEmbed);
        AddLayerGrads(allGrads, _timeEmbed1);
        AddLayerGrads(allGrads, _timeEmbed2);
        AddLayerGrads(allGrads, _labelEmbed);

        foreach (var block in _blocks)
        {
            AddLayerGrads(allGrads, block.Norm1);
            AddLayerGrads(allGrads, block.Attention);
            AddLayerGrads(allGrads, block.Norm2);
            AddLayerGrads(allGrads, block.MLP1);
            AddLayerGrads(allGrads, block.MLP2);
            AddLayerGrads(allGrads, block.AdaLNModulation);
            AddLayerGrads(allGrads, block.CrossAttnNorm);
            AddLayerGrads(allGrads, block.CrossAttnQ);
            AddLayerGrads(allGrads, block.CrossAttnK);
            AddLayerGrads(allGrads, block.CrossAttnV);
            AddLayerGrads(allGrads, block.CrossAttnOut);
        }

        AddLayerGrads(allGrads, _finalNorm);
        AddLayerGrads(allGrads, _adaln_modulation);
        AddLayerGrads(allGrads, _outputProj);

        return new Vector<T>(allGrads.ToArray());
    }

    private static void AddLayerGrads(List<T> list, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) list.Add(g[i]);
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
    }
}
