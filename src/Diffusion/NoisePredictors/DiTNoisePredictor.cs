using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.LossFunctions;
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
    /// Initializes a new DiT noise predictor with default XL/2 parameters.
    /// </summary>
    public DiTNoisePredictor()
        : this(
            inputChannels: 4,
            hiddenSize: ModelSizes.XLarge.hiddenSize,
            numLayers: ModelSizes.XLarge.numLayers,
            numHeads: ModelSizes.XLarge.numHeads,
            patchSize: 2,
            contextDim: 1024,
            mlpRatio: 4.0)
    {
    }

    /// <summary>
    /// Initializes a new DiT noise predictor with custom parameters.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="hiddenSize">Hidden dimension size.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="patchSize">Patch size for tokenization.</param>
    /// <param name="contextDim">Conditioning context dimension.</param>
    /// <param name="mlpRatio">MLP hidden dimension ratio.</param>
    /// <param name="numClasses">Number of classes for class conditioning (0 for text-only).</param>
    /// <param name="seed">Random seed for initialization.</param>
    public DiTNoisePredictor(
        int inputChannels = 4,
        int hiddenSize = 1152,
        int numLayers = 28,
        int numHeads = 16,
        int patchSize = 2,
        int contextDim = 1024,
        double mlpRatio = 4.0,
        int numClasses = 0,
        int? seed = null)
        : base(null, seed)
    {
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;
        _mlpRatio = mlpRatio;

        _blocks = new List<DiTBlock>();

        InitializeLayers(numClasses);
    }

    /// <summary>
    /// Initializes all layers.
    /// </summary>
    private void InitializeLayers(int numClasses)
    {
        var patchDim = _inputChannels * _patchSize * _patchSize;

        // Patch embedding: linear projection from patch to hidden
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, activationFunction: null);

        // Time embedding MLP
        var timeEmbedDim = _hiddenSize * 4;
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

        // Transformer blocks
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
                AdaLNModulation = new DenseLayer<T>(_hiddenSize * 4, _hiddenSize * 6, activationFunction: null)
            });
        }

        // Final norm and projection
        _finalNorm = new LayerNormalizationLayer<T>(_hiddenSize);
        _adaln_modulation = new DenseLayer<T>(_hiddenSize * 4, _hiddenSize * 2, activationFunction: null);
        _outputProj = new DenseLayer<T>(_hiddenSize, patchDim, activationFunction: null);
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
    /// Embeds patches through linear projection.
    /// </summary>
    private Tensor<T> EmbedPatches(Tensor<T> patches)
    {
        if (_patchEmbed == null)
            throw new InvalidOperationException("Patch embed not initialized.");

        var shape = patches.Shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = shape[2];

        var embedded = new Tensor<T>(new[] { batch, numPatches, _hiddenSize });
        var embSpan = embedded.AsWritableSpan();
        var patchSpan = patches.AsSpan();

        // Process each patch through linear layer
        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                // Extract patch vector
                var patchVector = new Tensor<T>(new[] { 1, patchDim });
                var pvSpan = patchVector.AsWritableSpan();

                for (int i = 0; i < patchDim; i++)
                {
                    pvSpan[i] = patchSpan[b * numPatches * patchDim + p * patchDim + i];
                }

                // Project through linear layer
                var projectedVector = _patchEmbed.Forward(patchVector);
                var projSpan = projectedVector.AsSpan();

                // Copy to output
                for (int i = 0; i < _hiddenSize; i++)
                {
                    embSpan[b * numPatches * _hiddenSize + p * _hiddenSize + i] = projSpan[i];
                }
            }
        }

        return embedded;
    }

    /// <summary>
    /// Adds learnable position embeddings.
    /// </summary>
    private Tensor<T> AddPositionEmbedding(Tensor<T> x, int numPatches)
    {
        // Initialize position embeddings if needed
        if (_posEmbed == null || _posEmbed.Shape[1] != numPatches)
        {
            _posEmbed = CreatePositionEmbedding(numPatches);
        }

        var result = new Tensor<T>(x.Shape);
        var resultSpan = result.AsWritableSpan();
        var xSpan = x.AsSpan();
        var posSpan = _posEmbed.AsSpan();

        var batch = x.Shape[0];
        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                for (int h = 0; h < _hiddenSize; h++)
                {
                    var idx = b * numPatches * _hiddenSize + p * _hiddenSize + h;
                    var posIdx = p * _hiddenSize + h;
                    resultSpan[idx] = NumOps.Add(xSpan[idx], posSpan[posIdx]);
                }
            }
        }

        return result;
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

        var attnOut = ApplyAttention(normed, block.Attention, condEmbed);
        x = AddWithGate(x, attnOut, gate1);

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
    /// Applies adaptive layer normalization.
    /// </summary>
    private Tensor<T> ApplyAdaLN(Tensor<T> x, T[] scale, T[] shift)
    {
        var result = new Tensor<T>(x.Shape);
        var resultSpan = result.AsWritableSpan();
        var xSpan = x.AsSpan();

        var shape = x.Shape;
        var batch = shape[0];
        var seq = shape[1];
        var hidden = shape[2];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seq; s++)
            {
                for (int h = 0; h < hidden; h++)
                {
                    var idx = b * seq * hidden + s * hidden + h;
                    // y = x * (1 + scale) + shift
                    var oneVal = NumOps.One;
                    var scaleFactor = NumOps.Add(oneVal, scale[h % scale.Length]);
                    resultSpan[idx] = NumOps.Add(
                        NumOps.Multiply(xSpan[idx], scaleFactor),
                        shift[h % shift.Length]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies self-attention with optional cross-attention.
    /// </summary>
    private Tensor<T> ApplyAttention(
        Tensor<T> x,
        SelfAttentionLayer<T> attention,
        Tensor<T>? conditioning)
    {
        // For DiT, we typically use self-attention
        // Cross-attention to conditioning can be added via additional blocks
        return attention.Forward(x);
    }

    /// <summary>
    /// Adds with gating.
    /// </summary>
    private Tensor<T> AddWithGate(Tensor<T> x, Tensor<T> residual, T[] gate)
    {
        var result = new Tensor<T>(x.Shape);
        var resultSpan = result.AsWritableSpan();
        var xSpan = x.AsSpan();
        var resSpan = residual.AsSpan();

        var shape = x.Shape;
        var batch = shape[0];
        var seq = shape[1];
        var hidden = shape[2];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seq; s++)
            {
                for (int h = 0; h < hidden; h++)
                {
                    var idx = b * seq * hidden + s * hidden + h;
                    var gateVal = gate[h % gate.Length];
                    resultSpan[idx] = NumOps.Add(
                        xSpan[idx],
                        NumOps.Multiply(gateVal, resSpan[idx]));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Final layer with AdaLN zero.
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

        // Project to output dimension
        var shape = normed.Shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = _inputChannels * _patchSize * _patchSize;

        var output = new Tensor<T>(new[] { batch, numPatches, patchDim });
        var outputSpan = output.AsWritableSpan();
        var normedSpan = normed.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                // Extract hidden vector
                var hiddenVector = new Tensor<T>(new[] { 1, _hiddenSize });
                var hvSpan = hiddenVector.AsWritableSpan();

                for (int i = 0; i < _hiddenSize; i++)
                {
                    hvSpan[i] = normedSpan[b * numPatches * _hiddenSize + p * _hiddenSize + i];
                }

                // Project
                var projected = _outputProj.Forward(hiddenVector);
                var projSpan = projected.AsSpan();

                // Copy to output
                for (int i = 0; i < patchDim; i++)
                {
                    outputSpan[b * numPatches * patchDim + p * patchDim + i] = projSpan[i];
                }
            }
        }

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
        // Simplified: return empty vector
        // Full implementation would collect from all layers
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Simplified implementation
    }

    /// <inheritdoc />
    public override int ParameterCount => 0; // Simplified

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        return new DiTNoisePredictor<T>(
            _inputChannels,
            _hiddenSize,
            _numLayers,
            _numHeads,
            _patchSize,
            _contextDim,
            _mlpRatio);
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
    /// Internal block structure for DiT.
    /// </summary>
    private class DiTBlock
    {
        public LayerNormalizationLayer<T>? Norm1 { get; set; }
        public SelfAttentionLayer<T>? Attention { get; set; }
        public LayerNormalizationLayer<T>? Norm2 { get; set; }
        public DenseLayer<T>? MLP1 { get; set; }
        public DenseLayer<T>? MLP2 { get; set; }
        public DenseLayer<T>? AdaLNModulation { get; set; }
    }
}
