using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

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
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;
        _mlpRatio = mlpRatio;

        _blocks = new List<DiTBlock>();

        InitializeLayers(architecture, numClasses, customBlocks);
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

        // Always create patch embedding, time embedding, and final layers
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, activationFunction: null);

        _timeEmbed1 = new DenseLayer<T>(
            _hiddenSize,
            timeEmbedDim,
            (IActivationFunction<T>)new SiLUActivation<T>());
        _timeEmbed2 = new DenseLayer<T>(
            timeEmbedDim,
            timeEmbedDim,
            (IActivationFunction<T>)new SiLUActivation<T>());

        // Class embedding (optional)
        if (numClasses > 0)
        {
            _labelEmbed = new DenseLayer<T>(numClasses, _hiddenSize, activationFunction: null);
        }

        _finalNorm = new LayerNormalizationLayer<T>(_hiddenSize);
        _adaln_modulation = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 2, activationFunction: null);
        _outputProj = new DenseLayer<T>(_hiddenSize, patchDim, activationFunction: null);

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
                    ?? new DenseLayer<T>(_hiddenSize, (int)(_hiddenSize * _mlpRatio),
                        (IActivationFunction<T>)new GELUActivation<T>());

                _blocks.Add(new DiTBlock
                {
                    Norm1 = new LayerNormalizationLayer<T>(_hiddenSize),
                    Attention = CreateAttentionLayer(),
                    Norm2 = new LayerNormalizationLayer<T>(_hiddenSize),
                    MLP1 = mlp1,
                    MLP2 = new DenseLayer<T>((int)(_hiddenSize * _mlpRatio), _hiddenSize, activationFunction: null),
                    AdaLNModulation = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 6, activationFunction: null),
                    CrossAttnNorm = new LayerNormalizationLayer<T>(_hiddenSize),
                    CrossAttnQ = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                    CrossAttnK = new DenseLayer<T>(_contextDim, _hiddenSize, activationFunction: null),
                    CrossAttnV = new DenseLayer<T>(_contextDim, _hiddenSize, activationFunction: null),
                    CrossAttnOut = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null)
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
                MLP1 = new DenseLayer<T>(_hiddenSize, mlpHidden, (IActivationFunction<T>)new GELUActivation<T>()),
                MLP2 = new DenseLayer<T>(mlpHidden, _hiddenSize, activationFunction: null),
                AdaLNModulation = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 6, activationFunction: null),
                CrossAttnNorm = new LayerNormalizationLayer<T>(_hiddenSize),
                CrossAttnQ = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                CrossAttnK = new DenseLayer<T>(_contextDim, _hiddenSize, activationFunction: null),
                CrossAttnV = new DenseLayer<T>(_contextDim, _hiddenSize, activationFunction: null),
                CrossAttnOut = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null)
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
        // Default sequence length for 32x32 latent with patch size 2 = 256 patches
        // Actual sequence length is handled dynamically in forward pass
        return new SelfAttentionLayer<T>(
            sequenceLength: 256,
            embeddingDimension: _hiddenSize,
            headCount: _numHeads,
            activationFunction: null);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Get timestep embedding
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        return Forward(noisySample, timeEmbed, conditioning);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
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
        if (_patchEmbed == null || _finalNorm == null || _outputProj == null || _adaln_modulation == null)
            throw new InvalidOperationException("Layers not initialized.");

        var shape = x.Shape;
        var batch = shape[0];
        var height = shape[2];
        var width = shape[3];

        // Patchify and embed
        var patches = Patchify(x);
        var numPatches = patches.Shape[1];

        // Linear embed patches
        var hidden = EmbedPatches(patches);

        // Add position embeddings
        hidden = AddPositionEmbedding(hidden, numPatches);

        // Get conditioning embedding if available
        var condEmbed = conditioning;

        // Process through transformer blocks
        foreach (var block in _blocks)
        {
            hidden = ForwardBlock(hidden, timeEmbed, condEmbed, block);
        }

        // Final norm and projection with AdaLN
        hidden = FinalLayerWithAdaLN(hidden, timeEmbed);

        // Unpatchify back to image
        var output = Unpatchify(hidden, height, width);

        return output;
    }

    /// <summary>
    /// Converts image to patches.
    /// </summary>
    private Tensor<T> Patchify(Tensor<T> x)
    {
        var shape = x.Shape;
        var batch = shape[0];
        var channels = shape[1];
        var height = shape[2];
        var width = shape[3];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = channels * _patchSize * _patchSize;

        var patches = new Tensor<T>(new[] { batch, numPatches, patchDim });
        var patchSpan = patches.AsWritableSpan();
        var xSpan = x.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < numPatchesH; ph++)
            {
                for (int pw = 0; pw < numPatchesW; pw++)
                {
                    // Extract patch
                    int dimIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var srcIdx = b * channels * height * width +
                                             c * height * width +
                                             ih * width + iw;
                                var dstIdx = b * numPatches * patchDim +
                                             patchIdx * patchDim + dimIdx;
                                patchSpan[dstIdx] = xSpan[srcIdx];
                                dimIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return patches;
    }

    /// <summary>
    /// Embeds patches through linear projection using batched forward pass.
    /// </summary>
    private Tensor<T> EmbedPatches(Tensor<T> patches)
    {
        if (_patchEmbed == null)
            throw new InvalidOperationException("Patch embed not initialized.");

        var shape = patches.Shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = shape[2];

        // Reshape [batch, numPatches, patchDim] -> [batch*numPatches, patchDim] for batched forward
        var flatPatches = new Tensor<T>(new[] { batch * numPatches, patchDim });
        patches.AsSpan().CopyTo(flatPatches.AsWritableSpan());

        // Single batched forward pass through the linear layer
        var projected = _patchEmbed.Forward(flatPatches);

        // Reshape back to [batch, numPatches, hiddenSize]
        var embedded = new Tensor<T>(new[] { batch, numPatches, _hiddenSize });
        projected.AsSpan().CopyTo(embedded.AsWritableSpan());

        return embedded;
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
        var posEmbed = new Tensor<T>(new[] { 1, numPatches, _hiddenSize });
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

        // Get AdaLN modulation parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        var modulation = block.AdaLNModulation.Forward(timeEmbed);
        var modSpan = modulation.AsSpan();

        // Split modulation into 6 parts
        var modSize = _hiddenSize;
        var shift1 = ExtractModulation(modSpan, 0, modSize);
        var scale1 = ExtractModulation(modSpan, modSize, modSize);
        var gate1 = ExtractModulation(modSpan, modSize * 2, modSize);
        var shift2 = ExtractModulation(modSpan, modSize * 3, modSize);
        var scale2 = ExtractModulation(modSpan, modSize * 4, modSize);
        var gate2 = ExtractModulation(modSpan, modSize * 5, modSize);

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
    /// Extracts modulation parameters from combined tensor.
    /// </summary>
    private T[] ExtractModulation(ReadOnlySpan<T> modSpan, int offset, int size)
    {
        var result = new T[size];
        for (int i = 0; i < size; i++)
        {
            result[i] = modSpan[offset + i];
        }
        return result;
    }

    /// <summary>
    /// Applies adaptive layer normalization using IEngine for hardware acceleration.
    /// y = x * (1 + scale) + shift
    /// </summary>
    private Tensor<T> ApplyAdaLN(Tensor<T> x, T[] scale, T[] shift)
    {
        var shape = x.Shape;
        var hidden = shape[^1];

        // Create broadcastable tensors for scale and shift: [1, 1, hidden]
        var scaleTensor = new Tensor<T>(new[] { 1, 1, hidden });
        var shiftTensor = new Tensor<T>(new[] { 1, 1, hidden });
        var scaleSpan = scaleTensor.AsWritableSpan();
        var shiftSpan = shiftTensor.AsWritableSpan();

        for (int h = 0; h < hidden; h++)
        {
            scaleSpan[h] = NumOps.Add(NumOps.One, scale[h % scale.Length]);
            shiftSpan[h] = shift[h % shift.Length];
        }

        // y = x * (1 + scale) + shift using engine ops
        var scaled = Engine.TensorMultiply<T>(x, scaleTensor);
        return Engine.TensorAdd<T>(scaled, shiftTensor);
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

        var shape = x.Shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var hidden = shape[2];

        var condShape = conditioning.Shape;
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
    /// Reshapes tensor from [batch, seq, hidden] to [batch*heads, seq, headDim] for multi-head attention.
    /// </summary>
    private static Tensor<T> ReshapeForHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        var result = new Tensor<T>(new[] { batch * numHeads, seq, headDim });
        var srcSpan = tensor.AsSpan();
        var dstSpan = result.AsWritableSpan();
        var hidden = numHeads * headDim;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seq; s++)
                {
                    var srcBase = b * seq * hidden + s * hidden + h * headDim;
                    var dstBase = (b * numHeads + h) * seq * headDim + s * headDim;
                    srcSpan.Slice(srcBase, headDim).CopyTo(dstSpan.Slice(dstBase, headDim));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Reshapes tensor from [batch*heads, seq, headDim] back to [batch, seq, hidden].
    /// </summary>
    private static Tensor<T> ReshapeFromHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        var result = new Tensor<T>(new[] { batch, seq, numHeads * headDim });
        var srcSpan = tensor.AsSpan();
        var dstSpan = result.AsWritableSpan();
        var hidden = numHeads * headDim;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seq; s++)
                {
                    var srcBase = (b * numHeads + h) * seq * headDim + s * headDim;
                    var dstBase = b * seq * hidden + s * hidden + h * headDim;
                    srcSpan.Slice(srcBase, headDim).CopyTo(dstSpan.Slice(dstBase, headDim));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Adds with gating using IEngine for hardware acceleration.
    /// result = x + gate * residual
    /// </summary>
    private Tensor<T> AddWithGate(Tensor<T> x, Tensor<T> residual, T[] gate)
    {
        var hidden = x.Shape[^1];

        // Create broadcastable gate tensor: [1, 1, hidden]
        var gateTensor = new Tensor<T>(new[] { 1, 1, hidden });
        var gateSpan = gateTensor.AsWritableSpan();
        for (int h = 0; h < hidden; h++)
        {
            gateSpan[h] = gate[h % gate.Length];
        }

        // result = x + gate * residual
        var gated = Engine.TensorMultiply<T>(residual, gateTensor);
        return Engine.TensorAdd<T>(x, gated);
    }

    /// <summary>
    /// Final layer with AdaLN zero using batched forward pass.
    /// </summary>
    private Tensor<T> FinalLayerWithAdaLN(Tensor<T> x, Tensor<T> timeEmbed)
    {
        if (_finalNorm == null || _adaln_modulation == null || _outputProj == null)
            throw new InvalidOperationException("Final layers not initialized.");

        // Get final modulation (scale, shift)
        var modulation = _adaln_modulation.Forward(timeEmbed);
        var modSpan = modulation.AsSpan();

        var shift = ExtractModulation(modSpan, 0, _hiddenSize);
        var scale = ExtractModulation(modSpan, _hiddenSize, _hiddenSize);

        // Apply final norm with AdaLN
        var normed = _finalNorm.Forward(x);
        normed = ApplyAdaLN(normed, scale, shift);

        // Project to output dimension using batched forward pass
        var shape = normed.Shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = _inputChannels * _patchSize * _patchSize;

        // Reshape [batch, numPatches, hiddenSize] -> [batch*numPatches, hiddenSize]
        var flatNormed = new Tensor<T>(new[] { batch * numPatches, _hiddenSize });
        normed.AsSpan().CopyTo(flatNormed.AsWritableSpan());

        // Single batched forward pass
        var projected = _outputProj.Forward(flatNormed);

        // Reshape back to [batch, numPatches, patchDim]
        var output = new Tensor<T>(new[] { batch, numPatches, patchDim });
        projected.AsSpan().CopyTo(output.AsWritableSpan());

        return output;
    }

    /// <summary>
    /// Converts patches back to image.
    /// </summary>
    private Tensor<T> Unpatchify(Tensor<T> patches, int height, int width)
    {
        var shape = patches.Shape;
        var batch = shape[0];
        var patchDim = shape[2];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;

        var output = new Tensor<T>(new[] { batch, _inputChannels, height, width });
        var outputSpan = output.AsWritableSpan();
        var patchSpan = patches.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < numPatchesH; ph++)
            {
                for (int pw = 0; pw < numPatchesW; pw++)
                {
                    // Reconstruct patch
                    int dimIdx = 0;
                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var dstIdx = b * _inputChannels * height * width +
                                             c * height * width +
                                             ih * width + iw;
                                var srcIdx = b * numPatchesH * numPatchesW * patchDim +
                                             patchIdx * patchDim + dimIdx;
                                outputSpan[dstIdx] = patchSpan[srcIdx];
                                dimIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
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
            mlpRatio: _mlpRatio);

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
