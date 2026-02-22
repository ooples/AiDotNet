using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Multi-Modal Diffusion Transformer (MMDiT) noise predictor for SD3 and FLUX architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMDiT extends the standard DiT architecture by processing text and image tokens
/// jointly through shared transformer blocks, rather than using cross-attention.
/// This enables deeper bidirectional interaction between modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> MMDiT is the architecture behind SD3 and FLUX:
///
/// How MMDiT differs from standard DiT:
/// - Standard DiT: Image patches processed by transformer, text injected via cross-attention
/// - MMDiT: Image AND text tokens are concatenated and processed together through JOINT attention
///
/// Key characteristics:
/// - Joint attention: Both text and image tokens attend to each other equally
/// - Separate MLPs: Text and image tokens have independent MLP layers after joint attention
/// - Dual stream: Text and image streams with shared attention but separate feed-forward
/// - AdaLN-Zero: Adaptive layer normalization with zero-init gating
/// - Supports multiple text encoders (CLIP + T5 for SD3, CLIP + T5 for FLUX)
///
/// Used in:
/// - Stable Diffusion 3 / SD 3.5 (Stability AI)
/// - FLUX.1 dev/schnell/pro (Black Forest Labs)
/// - Pixart-Sigma
///
/// Advantages over standard DiT:
/// - Better text-image alignment through joint attention
/// - More expressive conditioning without cross-attention bottleneck
/// - Scales better with model size
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Multi-modal transformer with joint self-attention
/// - SD3 Medium: 2B params, 24 layers, hidden 1536, 24 heads
/// - FLUX.1 dev: ~12B params, 19 double + 38 single layers, hidden 3072, 24 heads
/// - Patch size: 2 (latent space)
/// - Conditioning: Concatenated CLIP + T5 text embeddings
/// - Timestep: Sinusoidal + MLP projection
/// - Positional encoding: RoPE (Rotary Position Embedding)
///
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create MMDiT for SD3
/// var mmdit = new MMDiTNoisePredictor&lt;float&gt;(
///     inputChannels: 16,       // SD3 uses 16 latent channels
///     hiddenSize: 1536,        // SD3 Medium
///     numJointLayers: 24,
///     numHeads: 24,
///     contextDim: 4096);       // T5-XXL
///
/// // Create MMDiT for FLUX
/// var mmdit = new MMDiTNoisePredictor&lt;float&gt;(
///     inputChannels: 16,
///     hiddenSize: 3072,        // FLUX.1
///     numJointLayers: 19,
///     numSingleLayers: 38,
///     numHeads: 24,
///     contextDim: 4096);
/// </code>
/// </example>
public class MMDiTNoisePredictor<T> : NoisePredictorBase<T>
{
    #region Fields

    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numSingleLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _contextDim;
    private readonly double _mlpRatio;
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T> _timeEmbed1;
    private DenseLayer<T> _timeEmbed2;
    private DenseLayer<T> _contextProj;
    private readonly List<MMDiTBlock> _jointBlocks;
    private readonly List<MMDiTSingleBlock> _singleBlocks;
    private LayerNormalizationLayer<T> _finalNorm;
    private DenseLayer<T> _outputProj;
    private DenseLayer<T> _adalnModulation;

    private Tensor<T>? _posEmbed;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize * 4;

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Gets the hidden size of the transformer.
    /// </summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Gets the number of joint (double-stream) transformer layers.
    /// </summary>
    public int NumJointLayers => _numJointLayers;

    /// <summary>
    /// Gets the number of single-stream transformer layers (FLUX-style).
    /// </summary>
    public int NumSingleLayers => _numSingleLayers;

    /// <summary>
    /// Gets the patch size for latent tokenization.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the MMDiTNoisePredictor class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used for the transformer blocks. If null or empty,
    /// industry-standard layers are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input channels (default: 16 for SD3/FLUX).</param>
    /// <param name="hiddenSize">Hidden dimension size (default: 1536 for SD3 Medium).</param>
    /// <param name="numJointLayers">Number of joint/double-stream layers (default: 24).</param>
    /// <param name="numSingleLayers">Number of single-stream layers for FLUX (default: 0).</param>
    /// <param name="numHeads">Number of attention heads (default: 24).</param>
    /// <param name="patchSize">Patch size for latent tokenization (default: 2).</param>
    /// <param name="contextDim">Text conditioning dimension (default: 4096 for T5-XXL).</param>
    /// <param name="mlpRatio">MLP hidden dimension ratio (default: 4.0).</param>
    /// <param name="customJointBlocks">Optional custom joint transformer blocks.</param>
    /// <param name="customSingleBlocks">Optional custom single-stream blocks.</param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults.
    ///
    /// <code>
    /// // SD3 Medium configuration
    /// var sd3 = new MMDiTNoisePredictor&lt;float&gt;();
    ///
    /// // FLUX.1 dev configuration
    /// var flux = new MMDiTNoisePredictor&lt;float&gt;(
    ///     hiddenSize: 3072,
    ///     numJointLayers: 19,
    ///     numSingleLayers: 38);
    /// </code>
    /// </para>
    /// </remarks>
    public MMDiTNoisePredictor(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 16,
        int hiddenSize = 1536,
        int numJointLayers = 24,
        int numSingleLayers = 0,
        int numHeads = 24,
        int patchSize = 2,
        int contextDim = 4096,
        double mlpRatio = 4.0,
        List<MMDiTBlock>? customJointBlocks = null,
        List<MMDiTSingleBlock>? customSingleBlocks = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numJointLayers = numJointLayers;
        _numSingleLayers = numSingleLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;
        _mlpRatio = mlpRatio;

        _jointBlocks = new List<MMDiTBlock>();
        _singleBlocks = new List<MMDiTSingleBlock>();

        InitializeLayers(architecture, customJointBlocks, customSingleBlocks);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes all layers of the MMDiT, using custom layers from the user
    /// if provided or creating industry-standard layers.
    /// </summary>
    [MemberNotNull(nameof(_patchEmbed), nameof(_timeEmbed1), nameof(_timeEmbed2),
                   nameof(_contextProj), nameof(_finalNorm), nameof(_outputProj),
                   nameof(_adalnModulation))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        List<MMDiTBlock>? customJointBlocks,
        List<MMDiTSingleBlock>? customSingleBlocks)
    {
        var patchDim = _inputChannels * _patchSize * _patchSize;
        var timeEmbedDim = _hiddenSize * 4;

        // Patch embedding: linear projection from flattened patch to hidden dim
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, activationFunction: null);

        // Time embedding MLP
        _timeEmbed1 = new DenseLayer<T>(
            _hiddenSize,
            timeEmbedDim,
            (IActivationFunction<T>)new SiLUActivation<T>());
        _timeEmbed2 = new DenseLayer<T>(
            timeEmbedDim,
            timeEmbedDim,
            (IActivationFunction<T>)new SiLUActivation<T>());

        // Context projection: project text embeddings to hidden dim
        _contextProj = new DenseLayer<T>(_contextDim, _hiddenSize, activationFunction: null);

        // Final layers
        _finalNorm = new LayerNormalizationLayer<T>(_hiddenSize);
        _adalnModulation = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 2, activationFunction: null);
        _outputProj = new DenseLayer<T>(_hiddenSize, patchDim, activationFunction: null);

        // Priority 1: Use custom blocks passed directly
        if (customJointBlocks != null && customJointBlocks.Count > 0)
        {
            _jointBlocks.AddRange(customJointBlocks);
            if (customSingleBlocks != null)
            {
                _singleBlocks.AddRange(customSingleBlocks);
            }
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            foreach (var layer in architecture.Layers)
            {
                _jointBlocks.Add(CreateDefaultJointBlock(timeEmbedDim));
            }
            return;
        }

        // Priority 3: Create industry-standard layers
        CreateDefaultJointBlocks(timeEmbedDim);
        CreateDefaultSingleBlocks(timeEmbedDim);
    }

    private void CreateDefaultJointBlocks(int timeEmbedDim)
    {
        for (int i = 0; i < _numJointLayers; i++)
        {
            _jointBlocks.Add(CreateDefaultJointBlock(timeEmbedDim));
        }
    }

    private MMDiTBlock CreateDefaultJointBlock(int timeEmbedDim)
    {
        var mlpHidden = (int)(_hiddenSize * _mlpRatio);

        return new MMDiTBlock
        {
            // Image stream
            ImageNorm1 = new LayerNormalizationLayer<T>(_hiddenSize),
            ImageNorm2 = new LayerNormalizationLayer<T>(_hiddenSize),
            ImageMLP1 = new DenseLayer<T>(_hiddenSize, mlpHidden, (IActivationFunction<T>)new GELUActivation<T>()),
            ImageMLP2 = new DenseLayer<T>(mlpHidden, _hiddenSize, activationFunction: null),
            ImageAdaLN = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 6, activationFunction: null),

            // Text stream
            TextNorm1 = new LayerNormalizationLayer<T>(_hiddenSize),
            TextNorm2 = new LayerNormalizationLayer<T>(_hiddenSize),
            TextMLP1 = new DenseLayer<T>(_hiddenSize, mlpHidden, (IActivationFunction<T>)new GELUActivation<T>()),
            TextMLP2 = new DenseLayer<T>(mlpHidden, _hiddenSize, activationFunction: null),
            TextAdaLN = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 6, activationFunction: null),

            // Joint attention Q/K/V projections
            ImageQProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            ImageKProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            ImageVProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            ImageOutProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),

            TextQProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            TextKProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            TextVProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
            TextOutProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null)
        };
    }

    private void CreateDefaultSingleBlocks(int timeEmbedDim)
    {
        var mlpHidden = (int)(_hiddenSize * _mlpRatio);

        for (int i = 0; i < _numSingleLayers; i++)
        {
            _singleBlocks.Add(new MMDiTSingleBlock
            {
                Norm = new LayerNormalizationLayer<T>(_hiddenSize),
                QProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                KProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                VProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                OutProj = new DenseLayer<T>(_hiddenSize, _hiddenSize, activationFunction: null),
                MLP1 = new DenseLayer<T>(_hiddenSize, mlpHidden, (IActivationFunction<T>)new GELUActivation<T>()),
                MLP2 = new DenseLayer<T>(mlpHidden, _hiddenSize, activationFunction: null),
                AdaLN = new DenseLayer<T>(timeEmbedDim, _hiddenSize * 3, activationFunction: null)
            });
        }
    }

    #endregion

    #region Noise Prediction

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);
        return Forward(noisySample, timeEmbed, conditioning);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        return Forward(noisySample, timeEmbedding, conditioning);
    }

    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        var x = _timeEmbed1.Forward(timeEmbed);
        x = _timeEmbed2.Forward(x);
        return x;
    }

    private Tensor<T> Forward(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        var shape = x.Shape;
        var batch = shape[0];
        var height = shape[2];
        var width = shape[3];

        // Patchify and embed image tokens
        var imageTokens = PatchifyAndEmbed(x);
        var numImageTokens = imageTokens.Shape[1];

        // Add position embeddings to image tokens
        imageTokens = AddPositionEmbedding(imageTokens, numImageTokens);

        // Project conditioning text to hidden dim
        var textTokens = conditioning != null
            ? _contextProj.Forward(conditioning)
            : new Tensor<T>(new[] { batch, 0, _hiddenSize });

        // Process through joint (double-stream) blocks
        foreach (var block in _jointBlocks)
        {
            (imageTokens, textTokens) = ForwardJointBlock(imageTokens, textTokens, timeEmbed, block);
        }

        // Process through single-stream blocks (FLUX-style)
        if (_singleBlocks.Count > 0)
        {
            // Concatenate text and image tokens for single-stream processing
            var combined = ConcatenateSequences(textTokens, imageTokens);
            foreach (var block in _singleBlocks)
            {
                combined = ForwardSingleBlock(combined, timeEmbed, block);
            }
            // Extract image tokens from the combined sequence
            imageTokens = ExtractImageTokens(combined, textTokens.Shape[1], numImageTokens);
        }

        // Final norm and projection
        imageTokens = FinalLayer(imageTokens, timeEmbed);

        // Unpatchify back to image
        return Unpatchify(imageTokens, height, width);
    }

    #endregion

    #region Forward Helpers

    private Tensor<T> PatchifyAndEmbed(Tensor<T> x)
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
                    int dimIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var srcIdx = b * channels * height * width + c * height * width + ih * width + iw;
                                var dstIdx = b * numPatches * patchDim + patchIdx * patchDim + dimIdx;
                                patchSpan[dstIdx] = xSpan[srcIdx];
                                dimIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        // Linear embed through patch projection
        var embedded = new Tensor<T>(new[] { batch, numPatches, _hiddenSize });
        var embSpan = embedded.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                var patchVector = new Tensor<T>(new[] { 1, patchDim });
                var pvSpan = patchVector.AsWritableSpan();
                for (int i = 0; i < patchDim; i++)
                {
                    pvSpan[i] = patchSpan[b * numPatches * patchDim + p * patchDim + i];
                }

                var projected = _patchEmbed.Forward(patchVector);
                var projSpan = projected.AsSpan();
                for (int i = 0; i < _hiddenSize; i++)
                {
                    embSpan[b * numPatches * _hiddenSize + p * _hiddenSize + i] = projSpan[i];
                }
            }
        }

        return embedded;
    }

    private Tensor<T> AddPositionEmbedding(Tensor<T> x, int numPatches)
    {
        if (_posEmbed == null || _posEmbed.Shape[1] != numPatches)
        {
            _posEmbed = CreateSinusoidalPositionEmbedding(numPatches);
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

    private Tensor<T> CreateSinusoidalPositionEmbedding(int numPatches)
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
    /// Processes a joint (double-stream) block where text and image attend to each other.
    /// </summary>
    private (Tensor<T> imageOut, Tensor<T> textOut) ForwardJointBlock(
        Tensor<T> imageTokens,
        Tensor<T> textTokens,
        Tensor<T> timeEmbed,
        MMDiTBlock block)
    {
        var shape = imageTokens.Shape;
        var batch = shape[0];
        var numImageTokens = shape[1];
        var numTextTokens = textTokens.Shape[1];
        var headDim = _hiddenSize / _numHeads;
        var scale = 1.0 / Math.Sqrt(headDim);

        // Get AdaLN modulation for image stream
        var imgMod = block.ImageAdaLN.Forward(timeEmbed);
        var imgModSpan = imgMod.AsSpan();
        var imgShift1 = ExtractMod(imgModSpan, 0, _hiddenSize);
        var imgScale1 = ExtractMod(imgModSpan, _hiddenSize, _hiddenSize);
        var imgGate1 = ExtractMod(imgModSpan, _hiddenSize * 2, _hiddenSize);
        var imgShift2 = ExtractMod(imgModSpan, _hiddenSize * 3, _hiddenSize);
        var imgScale2 = ExtractMod(imgModSpan, _hiddenSize * 4, _hiddenSize);
        var imgGate2 = ExtractMod(imgModSpan, _hiddenSize * 5, _hiddenSize);

        // Get AdaLN modulation for text stream
        var txtMod = block.TextAdaLN.Forward(timeEmbed);
        var txtModSpan = txtMod.AsSpan();
        var txtShift1 = ExtractMod(txtModSpan, 0, _hiddenSize);
        var txtScale1 = ExtractMod(txtModSpan, _hiddenSize, _hiddenSize);
        var txtGate1 = ExtractMod(txtModSpan, _hiddenSize * 2, _hiddenSize);
        var txtShift2 = ExtractMod(txtModSpan, _hiddenSize * 3, _hiddenSize);
        var txtScale2 = ExtractMod(txtModSpan, _hiddenSize * 4, _hiddenSize);
        var txtGate2 = ExtractMod(txtModSpan, _hiddenSize * 5, _hiddenSize);

        // Normalize and modulate
        var imgNormed = block.ImageNorm1.Forward(imageTokens);
        imgNormed = ApplyAdaLN(imgNormed, imgScale1, imgShift1);

        var txtNormed = block.TextNorm1.Forward(textTokens);
        txtNormed = ApplyAdaLN(txtNormed, txtScale1, txtShift1);

        // Compute Q, K, V for both streams
        var imgQ = block.ImageQProj.Forward(imgNormed);
        var imgK = block.ImageKProj.Forward(imgNormed);
        var imgV = block.ImageVProj.Forward(imgNormed);

        var txtQ = block.TextQProj.Forward(txtNormed);
        var txtK = block.TextKProj.Forward(txtNormed);
        var txtV = block.TextVProj.Forward(txtNormed);

        // Joint attention: concatenate K and V from both streams
        var jointK = ConcatenateSequences(txtK, imgK);
        var jointV = ConcatenateSequences(txtV, imgV);

        // Image queries attend to joint K, V
        var imgAttnOut = ScaledDotProductAttention(imgQ, jointK, jointV, scale, batch, numImageTokens, numTextTokens + numImageTokens, headDim);
        imgAttnOut = block.ImageOutProj.Forward(imgAttnOut);
        imageTokens = AddWithGate(imageTokens, imgAttnOut, imgGate1);

        // Text queries attend to joint K, V
        var txtAttnOut = ScaledDotProductAttention(txtQ, jointK, jointV, scale, batch, numTextTokens, numTextTokens + numImageTokens, headDim);
        txtAttnOut = block.TextOutProj.Forward(txtAttnOut);
        textTokens = AddWithGate(textTokens, txtAttnOut, txtGate1);

        // Image MLP
        var imgNormed2 = block.ImageNorm2.Forward(imageTokens);
        imgNormed2 = ApplyAdaLN(imgNormed2, imgScale2, imgShift2);
        var imgMlpOut = block.ImageMLP1.Forward(imgNormed2);
        imgMlpOut = block.ImageMLP2.Forward(imgMlpOut);
        imageTokens = AddWithGate(imageTokens, imgMlpOut, imgGate2);

        // Text MLP
        var txtNormed2 = block.TextNorm2.Forward(textTokens);
        txtNormed2 = ApplyAdaLN(txtNormed2, txtScale2, txtShift2);
        var txtMlpOut = block.TextMLP1.Forward(txtNormed2);
        txtMlpOut = block.TextMLP2.Forward(txtMlpOut);
        textTokens = AddWithGate(textTokens, txtMlpOut, txtGate2);

        return (imageTokens, textTokens);
    }

    /// <summary>
    /// Processes a single-stream block (FLUX-style) on concatenated tokens.
    /// </summary>
    private Tensor<T> ForwardSingleBlock(Tensor<T> combined, Tensor<T> timeEmbed, MMDiTSingleBlock block)
    {
        var shape = combined.Shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var headDim = _hiddenSize / _numHeads;
        var scale = 1.0 / Math.Sqrt(headDim);

        // AdaLN modulation
        var mod = block.AdaLN.Forward(timeEmbed);
        var modSpan = mod.AsSpan();
        var shift = ExtractMod(modSpan, 0, _hiddenSize);
        var scaleArr = ExtractMod(modSpan, _hiddenSize, _hiddenSize);
        var gate = ExtractMod(modSpan, _hiddenSize * 2, _hiddenSize);

        // Self-attention
        var normed = block.Norm.Forward(combined);
        normed = ApplyAdaLN(normed, scaleArr, shift);

        var q = block.QProj.Forward(normed);
        var k = block.KProj.Forward(normed);
        var v = block.VProj.Forward(normed);

        var attnOut = ScaledDotProductAttention(q, k, v, scale, batch, seqLen, seqLen, headDim);
        attnOut = block.OutProj.Forward(attnOut);

        // Parallel MLP (added before gating, FLUX-style)
        var mlpOut = block.MLP1.Forward(normed);
        mlpOut = block.MLP2.Forward(mlpOut);

        // Gate and residual: x + gate * (attn + mlp)
        var residual = AddTensors(attnOut, mlpOut);
        combined = AddWithGate(combined, residual, gate);

        return combined;
    }

    private Tensor<T> FinalLayer(Tensor<T> imageTokens, Tensor<T> timeEmbed)
    {
        var mod = _adalnModulation.Forward(timeEmbed);
        var modSpan = mod.AsSpan();
        var shift = ExtractMod(modSpan, 0, _hiddenSize);
        var scaleArr = ExtractMod(modSpan, _hiddenSize, _hiddenSize);

        var normed = _finalNorm.Forward(imageTokens);
        normed = ApplyAdaLN(normed, scaleArr, shift);

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
                var hiddenVector = new Tensor<T>(new[] { 1, _hiddenSize });
                var hvSpan = hiddenVector.AsWritableSpan();
                for (int i = 0; i < _hiddenSize; i++)
                {
                    hvSpan[i] = normedSpan[b * numPatches * _hiddenSize + p * _hiddenSize + i];
                }

                var projected = _outputProj.Forward(hiddenVector);
                var projSpan = projected.AsSpan();
                for (int i = 0; i < patchDim; i++)
                {
                    outputSpan[b * numPatches * patchDim + p * patchDim + i] = projSpan[i];
                }
            }
        }

        return output;
    }

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
                    int dimIdx = 0;
                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var dstIdx = b * _inputChannels * height * width + c * height * width + ih * width + iw;
                                var srcIdx = b * numPatchesH * numPatchesW * patchDim + patchIdx * patchDim + dimIdx;
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

    #endregion

    #region Tensor Utilities

    private T[] ExtractMod(ReadOnlySpan<T> modSpan, int offset, int size)
    {
        var result = new T[size];
        for (int i = 0; i < size; i++)
        {
            result[i] = modSpan[offset + i];
        }
        return result;
    }

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
                    var scaleFactor = NumOps.Add(NumOps.One, scale[h % scale.Length]);
                    resultSpan[idx] = NumOps.Add(
                        NumOps.Multiply(xSpan[idx], scaleFactor),
                        shift[h % shift.Length]);
                }
            }
        }

        return result;
    }

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
                    resultSpan[idx] = NumOps.Add(
                        xSpan[idx],
                        NumOps.Multiply(gate[h % gate.Length], resSpan[idx]));
                }
            }
        }

        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        var resultSpan = result.AsWritableSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        return result;
    }

    private Tensor<T> ConcatenateSequences(Tensor<T> a, Tensor<T> b)
    {
        var aShape = a.Shape;
        var bShape = b.Shape;
        var batch = aShape[0];
        var aSeq = aShape[1];
        var bSeq = bShape[1];
        var hidden = aShape[2];

        var output = new Tensor<T>(new[] { batch, aSeq + bSeq, hidden });
        var outSpan = output.AsWritableSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        for (int ba = 0; ba < batch; ba++)
        {
            // Copy a tokens
            for (int s = 0; s < aSeq; s++)
            {
                for (int h = 0; h < hidden; h++)
                {
                    outSpan[ba * (aSeq + bSeq) * hidden + s * hidden + h] =
                        aSpan[ba * aSeq * hidden + s * hidden + h];
                }
            }
            // Copy b tokens
            for (int s = 0; s < bSeq; s++)
            {
                for (int h = 0; h < hidden; h++)
                {
                    outSpan[ba * (aSeq + bSeq) * hidden + (aSeq + s) * hidden + h] =
                        bSpan[ba * bSeq * hidden + s * hidden + h];
                }
            }
        }

        return output;
    }

    private Tensor<T> ExtractImageTokens(Tensor<T> combined, int textLen, int imageLen)
    {
        var shape = combined.Shape;
        var batch = shape[0];
        var hidden = shape[2];

        var output = new Tensor<T>(new[] { batch, imageLen, hidden });
        var outSpan = output.AsWritableSpan();
        var combinedSpan = combined.AsSpan();
        var totalSeq = shape[1];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < imageLen; s++)
            {
                for (int h = 0; h < hidden; h++)
                {
                    outSpan[b * imageLen * hidden + s * hidden + h] =
                        combinedSpan[b * totalSeq * hidden + (textLen + s) * hidden + h];
                }
            }
        }

        return output;
    }

    private Tensor<T> ScaledDotProductAttention(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        double scale, int batch, int queryLen, int keyLen, int headDim)
    {
        var hidden = _hiddenSize;
        var output = new Tensor<T>(new[] { batch, queryLen, hidden });
        var outSpan = output.AsWritableSpan();
        var qSpan = q.AsSpan();
        var kSpan = k.AsSpan();
        var vSpan = v.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                var headOffset = h * headDim;

                for (int i = 0; i < queryLen; i++)
                {
                    var scores = new double[keyLen];
                    var maxScore = double.NegativeInfinity;

                    for (int j = 0; j < keyLen; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            var qIdx = b * queryLen * hidden + i * hidden + headOffset + d;
                            var kIdx = b * keyLen * hidden + j * hidden + headOffset + d;
                            score += NumOps.ToDouble(qSpan[qIdx]) * NumOps.ToDouble(kSpan[kIdx]);
                        }
                        score *= scale;
                        scores[j] = score;
                        if (score > maxScore) maxScore = score;
                    }

                    double sumExp = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - maxScore);
                        sumExp += scores[j];
                    }
                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[j] /= sumExp;
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        double weighted = 0;
                        for (int j = 0; j < keyLen; j++)
                        {
                            var vIdx = b * keyLen * hidden + j * hidden + headOffset + d;
                            weighted += scores[j] * NumOps.ToDouble(vSpan[vIdx]);
                        }
                        var outIdx = b * queryLen * hidden + i * hidden + headOffset + d;
                        outSpan[outIdx] = NumOps.FromDouble(weighted);
                    }
                }
            }
        }

        return output;
    }

    #endregion

    #region Parameter Management

    private int CalculateParameterCount()
    {
        int count = 0;

        count += _patchEmbed.ParameterCount;
        count += _timeEmbed1.ParameterCount;
        count += _timeEmbed2.ParameterCount;
        count += _contextProj.ParameterCount;

        foreach (var block in _jointBlocks)
        {
            count += block.ImageNorm1.ParameterCount + block.ImageNorm2.ParameterCount;
            count += block.ImageMLP1.ParameterCount + block.ImageMLP2.ParameterCount;
            count += block.ImageAdaLN.ParameterCount;
            count += block.ImageQProj.ParameterCount + block.ImageKProj.ParameterCount;
            count += block.ImageVProj.ParameterCount + block.ImageOutProj.ParameterCount;
            count += block.TextNorm1.ParameterCount + block.TextNorm2.ParameterCount;
            count += block.TextMLP1.ParameterCount + block.TextMLP2.ParameterCount;
            count += block.TextAdaLN.ParameterCount;
            count += block.TextQProj.ParameterCount + block.TextKProj.ParameterCount;
            count += block.TextVProj.ParameterCount + block.TextOutProj.ParameterCount;
        }

        foreach (var block in _singleBlocks)
        {
            count += block.Norm.ParameterCount;
            count += block.QProj.ParameterCount + block.KProj.ParameterCount;
            count += block.VProj.ParameterCount + block.OutProj.ParameterCount;
            count += block.MLP1.ParameterCount + block.MLP2.ParameterCount;
            count += block.AdaLN.ParameterCount;
        }

        count += _finalNorm.ParameterCount;
        count += _adalnModulation.ParameterCount;
        count += _outputProj.ParameterCount;

        return count;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddLayerParams(allParams, _patchEmbed);
        AddLayerParams(allParams, _timeEmbed1);
        AddLayerParams(allParams, _timeEmbed2);
        AddLayerParams(allParams, _contextProj);

        foreach (var block in _jointBlocks)
        {
            AddLayerParams(allParams, block.ImageNorm1);
            AddLayerParams(allParams, block.ImageQProj);
            AddLayerParams(allParams, block.ImageKProj);
            AddLayerParams(allParams, block.ImageVProj);
            AddLayerParams(allParams, block.ImageOutProj);
            AddLayerParams(allParams, block.ImageNorm2);
            AddLayerParams(allParams, block.ImageMLP1);
            AddLayerParams(allParams, block.ImageMLP2);
            AddLayerParams(allParams, block.ImageAdaLN);
            AddLayerParams(allParams, block.TextNorm1);
            AddLayerParams(allParams, block.TextQProj);
            AddLayerParams(allParams, block.TextKProj);
            AddLayerParams(allParams, block.TextVProj);
            AddLayerParams(allParams, block.TextOutProj);
            AddLayerParams(allParams, block.TextNorm2);
            AddLayerParams(allParams, block.TextMLP1);
            AddLayerParams(allParams, block.TextMLP2);
            AddLayerParams(allParams, block.TextAdaLN);
        }

        foreach (var block in _singleBlocks)
        {
            AddLayerParams(allParams, block.Norm);
            AddLayerParams(allParams, block.QProj);
            AddLayerParams(allParams, block.KProj);
            AddLayerParams(allParams, block.VProj);
            AddLayerParams(allParams, block.OutProj);
            AddLayerParams(allParams, block.MLP1);
            AddLayerParams(allParams, block.MLP2);
            AddLayerParams(allParams, block.AdaLN);
        }

        AddLayerParams(allParams, _finalNorm);
        AddLayerParams(allParams, _adalnModulation);
        AddLayerParams(allParams, _outputProj);

        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        offset = SetLayerParams(_patchEmbed, parameters, offset);
        offset = SetLayerParams(_timeEmbed1, parameters, offset);
        offset = SetLayerParams(_timeEmbed2, parameters, offset);
        offset = SetLayerParams(_contextProj, parameters, offset);

        foreach (var block in _jointBlocks)
        {
            offset = SetLayerParams(block.ImageNorm1, parameters, offset);
            offset = SetLayerParams(block.ImageQProj, parameters, offset);
            offset = SetLayerParams(block.ImageKProj, parameters, offset);
            offset = SetLayerParams(block.ImageVProj, parameters, offset);
            offset = SetLayerParams(block.ImageOutProj, parameters, offset);
            offset = SetLayerParams(block.ImageNorm2, parameters, offset);
            offset = SetLayerParams(block.ImageMLP1, parameters, offset);
            offset = SetLayerParams(block.ImageMLP2, parameters, offset);
            offset = SetLayerParams(block.ImageAdaLN, parameters, offset);
            offset = SetLayerParams(block.TextNorm1, parameters, offset);
            offset = SetLayerParams(block.TextQProj, parameters, offset);
            offset = SetLayerParams(block.TextKProj, parameters, offset);
            offset = SetLayerParams(block.TextVProj, parameters, offset);
            offset = SetLayerParams(block.TextOutProj, parameters, offset);
            offset = SetLayerParams(block.TextNorm2, parameters, offset);
            offset = SetLayerParams(block.TextMLP1, parameters, offset);
            offset = SetLayerParams(block.TextMLP2, parameters, offset);
            offset = SetLayerParams(block.TextAdaLN, parameters, offset);
        }

        foreach (var block in _singleBlocks)
        {
            offset = SetLayerParams(block.Norm, parameters, offset);
            offset = SetLayerParams(block.QProj, parameters, offset);
            offset = SetLayerParams(block.KProj, parameters, offset);
            offset = SetLayerParams(block.VProj, parameters, offset);
            offset = SetLayerParams(block.OutProj, parameters, offset);
            offset = SetLayerParams(block.MLP1, parameters, offset);
            offset = SetLayerParams(block.MLP2, parameters, offset);
            offset = SetLayerParams(block.AdaLN, parameters, offset);
        }

        offset = SetLayerParams(_finalNorm, parameters, offset);
        offset = SetLayerParams(_adalnModulation, parameters, offset);
        SetLayerParams(_outputProj, parameters, offset);
    }

    private void AddLayerParams(List<T> allParams, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
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

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new MMDiTNoisePredictor<T>(
            inputChannels: _inputChannels,
            hiddenSize: _hiddenSize,
            numJointLayers: _numJointLayers,
            numSingleLayers: _numSingleLayers,
            numHeads: _numHeads,
            patchSize: _patchSize,
            contextDim: _contextDim,
            mlpRatio: _mlpRatio);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    #endregion

    #region Block Structures

    /// <summary>
    /// Joint (double-stream) MMDiT block with separate image and text streams
    /// sharing attention but with independent MLPs and AdaLN.
    /// </summary>
    public class MMDiTBlock
    {
        // Image stream
        public LayerNormalizationLayer<T> ImageNorm1 { get; set; } = null!;
        public LayerNormalizationLayer<T> ImageNorm2 { get; set; } = null!;
        public DenseLayer<T> ImageMLP1 { get; set; } = null!;
        public DenseLayer<T> ImageMLP2 { get; set; } = null!;
        public DenseLayer<T> ImageAdaLN { get; set; } = null!;
        public DenseLayer<T> ImageQProj { get; set; } = null!;
        public DenseLayer<T> ImageKProj { get; set; } = null!;
        public DenseLayer<T> ImageVProj { get; set; } = null!;
        public DenseLayer<T> ImageOutProj { get; set; } = null!;

        // Text stream
        public LayerNormalizationLayer<T> TextNorm1 { get; set; } = null!;
        public LayerNormalizationLayer<T> TextNorm2 { get; set; } = null!;
        public DenseLayer<T> TextMLP1 { get; set; } = null!;
        public DenseLayer<T> TextMLP2 { get; set; } = null!;
        public DenseLayer<T> TextAdaLN { get; set; } = null!;
        public DenseLayer<T> TextQProj { get; set; } = null!;
        public DenseLayer<T> TextKProj { get; set; } = null!;
        public DenseLayer<T> TextVProj { get; set; } = null!;
        public DenseLayer<T> TextOutProj { get; set; } = null!;
    }

    /// <summary>
    /// Single-stream MMDiT block (FLUX-style) where text and image tokens
    /// are processed together through shared self-attention and parallel MLP.
    /// </summary>
    public class MMDiTSingleBlock
    {
        public LayerNormalizationLayer<T> Norm { get; set; } = null!;
        public DenseLayer<T> QProj { get; set; } = null!;
        public DenseLayer<T> KProj { get; set; } = null!;
        public DenseLayer<T> VProj { get; set; } = null!;
        public DenseLayer<T> OutProj { get; set; } = null!;
        public DenseLayer<T> MLP1 { get; set; } = null!;
        public DenseLayer<T> MLP2 { get; set; } = null!;
        public DenseLayer<T> AdaLN { get; set; } = null!;
    }

    #endregion
}
