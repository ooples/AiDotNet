using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Standard Variational Autoencoder for latent diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a standard VAE architecture similar to Stable Diffusion's VAE,
/// with an encoder that compresses images to latent space and a decoder that
/// reconstructs images from latents.
/// </para>
/// <para>
/// <b>For Beginners:</b> The StandardVAE is like a very smart image compressor:
///
/// How it works:
/// 1. Encoder: Takes a 512x512x3 image and compresses it to 64x64x4 latent
///    - That's 48x compression! (786,432 values -> 16,384 values)
///    - Uses multiple layers of convolutions and downsampling
///
/// 2. Decoder: Takes the 64x64x4 latent and reconstructs a 512x512x3 image
///    - Uses upsampling and convolutions to expand back to full size
///    - The reconstruction isn't perfect but preserves important visual features
///
/// Why 4 latent channels?
/// - The VAE learns to pack image information into 4 channels
/// - Each channel captures different aspects (colors, edges, textures, etc.)
/// - More channels = better quality but larger latent space
///
/// Why 8x downsampling?
/// - Each side is reduced by 8 (512 -> 64)
/// - This is the sweet spot between compression and quality
/// - Smaller latents = faster diffusion, but potentially lower quality
/// </para>
/// <para>
/// Architecture details:
/// - Input: [batch, 3, H, W] RGB image normalized to [-1, 1]
/// - Encoder: ResBlocks with GroupNorm, downsampling via strided conv
/// - Latent: [batch, 4, H/8, W/8] with mean and variance for sampling
/// - Decoder: ResBlocks with GroupNorm, upsampling via transpose conv
/// - Output: [batch, 3, H, W] reconstructed image
/// </para>
/// </remarks>
public class StandardVAE<T> : VAEModelBase<T>
{
    /// <summary>
    /// Standard Stable Diffusion latent scale factor.
    /// </summary>
    private const double SD_LATENT_SCALE = 0.18215;

    /// <summary>
    /// Number of encoder blocks.
    /// </summary>
    private readonly int _numEncoderBlocks;

    /// <summary>
    /// Number of decoder blocks.
    /// </summary>
    private readonly int _numDecoderBlocks;

    /// <summary>
    /// Base channel count.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Channel multipliers for each level.
    /// </summary>
    private readonly int[] _channelMultipliers;

    /// <summary>
    /// Number of residual blocks per level.
    /// </summary>
    private readonly int _numResBlocksPerLevel;

    /// <summary>
    /// Encoder layers.
    /// </summary>
    private List<ILayer<T>> _encoderLayers;

    /// <summary>
    /// Decoder layers.
    /// </summary>
    private List<ILayer<T>> _decoderLayers;

    /// <summary>
    /// Mean projection layer for latent distribution.
    /// </summary>
    private ConvolutionalLayer<T>? _meanConv;

    /// <summary>
    /// Log variance projection layer for latent distribution.
    /// </summary>
    private ConvolutionalLayer<T>? _logVarConv;

    /// <summary>
    /// Input convolution to initial embedding.
    /// </summary>
    private ConvolutionalLayer<T>? _inputConv;

    /// <summary>
    /// Quant convolution from latent to decoder.
    /// </summary>
    private ConvolutionalLayer<T>? _quantConv;

    /// <summary>
    /// Post-quant convolution in decoder.
    /// </summary>
    private ConvolutionalLayer<T>? _postQuantConv;

    /// <summary>
    /// Output convolution to RGB.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Cached mean from encoding.
    /// </summary>
    private Tensor<T>? _cachedMean;

    /// <summary>
    /// Cached log variance from encoding.
    /// </summary>
    private Tensor<T>? _cachedLogVar;

    /// <summary>
    /// Input channels (3 for RGB).
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Latent channels.
    /// </summary>
    private readonly int _latentChannels;

    /// <summary>
    /// Downsampling factor.
    /// </summary>
    private readonly int _downsampleFactor;

    /// <summary>
    /// Latent scale factor.
    /// </summary>
    private readonly double _latentScaleFactor;

    /// <summary>
    /// The neural network architecture configuration, if provided.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => _downsampleFactor;

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <summary>
    /// Initializes a new instance of the StandardVAE class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used directly for the encoder. If null or empty,
    /// industry-standard layers from the Stable Diffusion paper are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocksPerLevel">Residual blocks per level (default: 2).</param>
    /// <param name="latentScaleFactor">Scale factor for latents (default: 0.18215).</param>
    /// <param name="encoderLayers">
    /// Optional custom encoder layers. If provided, these layers are used instead of creating
    /// default layers. This allows full customization of the encoder architecture.
    /// </param>
    /// <param name="decoderLayers">
    /// Optional custom decoder layers. If provided, these layers are used instead of creating
    /// default layers. This allows full customization of the decoder architecture.
    /// </param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults
    /// from the original Stable Diffusion paper. You can create a ready-to-use VAE
    /// with no arguments, or customize any component:
    ///
    /// <code>
    /// // Default configuration (recommended for most users)
    /// var vae = new StandardVAE&lt;float&gt;();
    ///
    /// // Custom layers via NeuralNetworkArchitecture
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(..., layers: myCustomLayers);
    /// var vae = new StandardVAE&lt;float&gt;(architecture: arch);
    ///
    /// // Custom encoder/decoder layers directly
    /// var vae = new StandardVAE&lt;float&gt;(
    ///     encoderLayers: myEncoderLayers,
    ///     decoderLayers: myDecoderLayers);
    /// </code>
    /// </para>
    /// </remarks>
    public StandardVAE(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMultipliers = null,
        int numResBlocksPerLevel = 2,
        double? latentScaleFactor = null,
        List<ILayer<T>>? encoderLayers = null,
        List<ILayer<T>>? decoderLayers = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? [1, 2, 4, 4];
        _numResBlocksPerLevel = numResBlocksPerLevel;
        _latentScaleFactor = latentScaleFactor ?? SD_LATENT_SCALE;

        // Calculate downsampling factor (2^numLevels where numLevels = multipliers.Length)
        _downsampleFactor = (int)Math.Pow(2, _channelMultipliers.Length - 1);
        _numEncoderBlocks = _channelMultipliers.Length * numResBlocksPerLevel;
        _numDecoderBlocks = _channelMultipliers.Length * numResBlocksPerLevel;

        InitializeLayers(architecture, encoderLayers, decoderLayers);
    }

    /// <summary>
    /// Initializes encoder and decoder layers, using custom layers from the user
    /// if provided or creating industry-standard layers from the Stable Diffusion paper.
    /// </summary>
    /// <param name="architecture">Optional architecture with custom layers.</param>
    /// <param name="customEncoderLayers">Optional custom encoder layers.</param>
    /// <param name="customDecoderLayers">Optional custom decoder layers.</param>
    /// <remarks>
    /// <para>
    /// Layer resolution order:
    /// 1. If custom encoder/decoder layers are provided directly, use those
    /// 2. If a NeuralNetworkArchitecture with layers is provided, use those for the encoder
    /// 3. Otherwise, create industry-standard layers from the Stable Diffusion VAE paper
    ///
    /// When default layers are created, the architecture follows:
    /// - Encoder: InputConv -> [ResBlock + Downsample] per level -> MeanConv + LogVarConv -> QuantConv
    /// - Decoder: PostQuantConv -> [ResBlock + Upsample] per level -> OutputConv
    ///
    /// The default configuration matches the Stable Diffusion 1.5 VAE:
    /// - Base channels: 128, multipliers [1, 2, 4, 4]
    /// - 2 ResBlocks per level with GroupNorm and SiLU activation
    /// - Strided convolution for downsampling, transposed convolution for upsampling
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_encoderLayers), nameof(_decoderLayers))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        List<ILayer<T>>? customEncoderLayers,
        List<ILayer<T>>? customDecoderLayers)
    {
        // Priority 1: Use custom encoder/decoder layers passed directly
        if (customEncoderLayers != null && customEncoderLayers.Count > 0 &&
            customDecoderLayers != null && customDecoderLayers.Count > 0)
        {
            _encoderLayers = new List<ILayer<T>>(customEncoderLayers);
            _decoderLayers = new List<ILayer<T>>(customDecoderLayers);
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            // Architecture layers are used as encoder; decoder is auto-created as mirror
            _encoderLayers = new List<ILayer<T>>(architecture.Layers);
            _decoderLayers = new List<ILayer<T>>();
            CreateDefaultDecoderLayers();
            return;
        }

        // Priority 3: Create industry-standard layers from the Stable Diffusion paper
        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();
        CreateDefaultEncoderLayers();
        CreateDefaultDecoderLayers();
    }

    /// <summary>
    /// Creates industry-standard encoder layers based on the Stable Diffusion VAE paper.
    /// </summary>
    private void CreateDefaultEncoderLayers()
    {
        // Input convolution: [inputChannels] -> [baseChannels]
        _inputConv = new ConvolutionalLayer<T>(
            inputDepth: _inputChannels,
            outputDepth: _baseChannels,
            kernelSize: 3,
            inputHeight: 64,  // Placeholder - actual size handled dynamically
            inputWidth: 64,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Build encoder
        var inChannels = _baseChannels;
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];

            // Residual blocks at this level
            for (int block = 0; block < _numResBlocksPerLevel; block++)
            {
                _encoderLayers.Add(CreateResBlock(inChannels, outChannels));
                inChannels = outChannels;
            }

            // Downsample (except last level)
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderLayers.Add(CreateDownsample(outChannels));
            }
        }

        // Latent projection layers
        var lastEncoderChannels = _baseChannels * _channelMultipliers[^1];
        _meanConv = new ConvolutionalLayer<T>(
            inputDepth: lastEncoderChannels,
            outputDepth: _latentChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _logVarConv = new ConvolutionalLayer<T>(
            inputDepth: lastEncoderChannels,
            outputDepth: _latentChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Quant convolution for latent processing
        _quantConv = new ConvolutionalLayer<T>(
            inputDepth: _latentChannels,
            outputDepth: _latentChannels,
            kernelSize: 1,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Creates industry-standard decoder layers based on the Stable Diffusion VAE paper.
    /// </summary>
    private void CreateDefaultDecoderLayers()
    {
        var lastEncoderChannels = _baseChannels * _channelMultipliers[^1];

        // Post-quant convolution for decoder input
        _postQuantConv = new ConvolutionalLayer<T>(
            inputDepth: _latentChannels,
            outputDepth: lastEncoderChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Build decoder (mirror of encoder)
        var inChannels = lastEncoderChannels;
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];

            // Residual blocks at this level
            for (int block = 0; block < _numResBlocksPerLevel; block++)
            {
                _decoderLayers.Add(CreateResBlock(inChannels, outChannels));
                inChannels = outChannels;
            }

            // Upsample (except first level going backwards)
            if (level > 0)
            {
                _decoderLayers.Add(CreateUpsample(outChannels));
            }
        }

        // Output convolution: [baseChannels] -> [inputChannels]
        _outputConv = new ConvolutionalLayer<T>(
            inputDepth: _baseChannels,
            outputDepth: _inputChannels,
            kernelSize: 3,
            inputHeight: 64,
            inputWidth: 64,
            stride: 1,
            padding: 1,
            activationFunction: new TanhActivation<T>()); // Output in [-1, 1]
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> image, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(image);

        if (sampleMode)
        {
            return Sample(mean, logVar);
        }

        return mean;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image)
    {
        if (_inputConv == null || _meanConv == null || _logVarConv == null || _quantConv == null)
        {
            throw new InvalidOperationException("Encoder layers not initialized.");
        }

        // Initial convolution
        var x = _inputConv.Forward(image);

        // Apply encoder blocks
        foreach (var layer in _encoderLayers)
        {
            x = layer.Forward(x);
        }

        // Project to mean and log variance
        var mean = _meanConv.Forward(x);
        var logVar = _logVarConv.Forward(x);

        // Cache for potential backward pass
        _cachedMean = mean;
        _cachedLogVar = logVar;

        // Apply quant conv and return
        mean = _quantConv.Forward(mean);

        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        if (_postQuantConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Decoder layers not initialized.");
        }

        // Post-quant convolution
        var x = _postQuantConv.Forward(latent);

        // Apply decoder blocks
        foreach (var layer in _decoderLayers)
        {
            x = layer.Forward(x);
        }

        // Output convolution
        x = _outputConv.Forward(x);

        return x;
    }

    /// <summary>
    /// Encodes an image and applies latent scaling for use in diffusion.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="sampleMode">Whether to sample from the distribution.</param>
    /// <returns>Scaled latent representation.</returns>
    public Tensor<T> EncodeForDiffusion(Tensor<T> image, bool sampleMode = true)
    {
        var latent = Encode(image, sampleMode);
        return ScaleLatent(latent);
    }

    /// <summary>
    /// Decodes a diffusion latent back to image space.
    /// </summary>
    /// <param name="latent">The latent from diffusion (already scaled).</param>
    /// <returns>The decoded image.</returns>
    public Tensor<T> DecodeFromDiffusion(Tensor<T> latent)
    {
        var unscaled = UnscaleLatent(latent);
        return Decode(unscaled);
    }

    /// <summary>
    /// Computes the VAE loss (reconstruction + KL divergence).
    /// </summary>
    /// <param name="image">Original input image.</param>
    /// <param name="reconstruction">Reconstructed image.</param>
    /// <param name="klWeight">Weight for KL divergence term (default: 1e-6).</param>
    /// <returns>Combined loss value.</returns>
    public T ComputeVAELoss(Tensor<T> image, Tensor<T> reconstruction, double klWeight = 1e-6)
    {
        // Reconstruction loss
        var reconLoss = LossFunction.CalculateLoss(reconstruction.ToVector(), image.ToVector());

        // KL divergence loss (if we have cached distribution)
        if (_cachedMean != null && _cachedLogVar != null)
        {
            var klLoss = ComputeKLDivergence(_cachedMean, _cachedLogVar);
            var weightedKL = NumOps.Multiply(klLoss, NumOps.FromDouble(klWeight));
            return NumOps.Add(reconLoss, weightedKL);
        }

        return reconLoss;
    }

    #region Layer Factory Methods

    private ILayer<T> CreateResBlock(int inChannels, int outChannels)
    {
        int numGroups = CalculateGroupCount(inChannels, outChannels);
        return new VAEResBlock<T>(inChannels, outChannels, numGroups, spatialSize: 32);
    }

    private static int CalculateGroupCount(int inChannels, int outChannels)
    {
        int[] preferredGroups = [32, 16, 8, 4, 2, 1];

        foreach (int groups in preferredGroups)
        {
            if (inChannels % groups == 0 && outChannels % groups == 0)
            {
                return groups;
            }
        }

        return 1;
    }

    private ILayer<T> CreateDownsample(int channels)
    {
        return new ConvolutionalLayer<T>(
            inputDepth: channels,
            outputDepth: channels,
            kernelSize: 3,
            inputHeight: 32,
            inputWidth: 32,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    private ILayer<T> CreateUpsample(int channels)
    {
        return new DeconvolutionalLayer<T>(
            inputShape: [1, channels, 16, 16],
            outputDepth: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    #endregion

    #region Parameter Management

    private int CalculateParameterCount()
    {
        long count = 0;

        // Input conv
        count += _inputChannels * _baseChannels * 9 + _baseChannels;

        // Encoder blocks
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var channels = _baseChannels * _channelMultipliers[level];
            count += _numResBlocksPerLevel * (channels * channels * 2);
            if (level < _channelMultipliers.Length - 1)
            {
                count += channels * channels * 9;
            }
        }

        // Latent projections
        var lastChannels = _baseChannels * _channelMultipliers[^1];
        count += lastChannels * _latentChannels * 9 * 2; // mean + logvar
        count += _latentChannels * _latentChannels; // quant conv
        count += _latentChannels * lastChannels * 9; // post-quant

        // Decoder blocks (similar to encoder)
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var channels = _baseChannels * _channelMultipliers[level];
            count += _numResBlocksPerLevel * (channels * channels * 2);
            if (level > 0)
            {
                count += channels * channels * 9;
            }
        }

        // Output conv
        count += _baseChannels * _inputChannels * 9 + _inputChannels;

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        AddLayerParameters(parameters, _inputConv);

        foreach (var layer in _encoderLayers)
        {
            AddLayerParameters(parameters, layer);
        }

        AddLayerParameters(parameters, _meanConv);
        AddLayerParameters(parameters, _logVarConv);
        AddLayerParameters(parameters, _quantConv);
        AddLayerParameters(parameters, _postQuantConv);

        foreach (var layer in _decoderLayers)
        {
            AddLayerParameters(parameters, layer);
        }

        AddLayerParameters(parameters, _outputConv);

        return new Vector<T>(parameters.ToArray());
    }

    private void AddLayerParameters(List<T> parameters, ILayer<T>? layer)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        for (int i = 0; i < layerParams.Length; i++)
        {
            parameters.Add(layerParams[i]);
        }
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var index = 0;

        SetLayerParameters(_inputConv, parameters, ref index);

        foreach (var layer in _encoderLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
        }

        SetLayerParameters(_meanConv, parameters, ref index);
        SetLayerParameters(_logVarConv, parameters, ref index);
        SetLayerParameters(_quantConv, parameters, ref index);
        SetLayerParameters(_postQuantConv, parameters, ref index);

        foreach (var layer in _decoderLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
        }

        SetLayerParameters(_outputConv, parameters, ref index);
    }

    private void SetLayerParameters(ILayer<T>? layer, Vector<T> parameters, ref int index)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        var newParams = new Vector<T>(layerParams.Length);
        for (int i = 0; i < layerParams.Length && index < parameters.Length; i++)
        {
            newParams[i] = parameters[index++];
        }
        layer.SetParameters(newParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new StandardVAE<T>(
            inputChannels: _inputChannels,
            latentChannels: _latentChannels,
            baseChannels: _baseChannels,
            channelMultipliers: _channelMultipliers,
            numResBlocksPerLevel: _numResBlocksPerLevel,
            latentScaleFactor: _latentScaleFactor,
            lossFunction: LossFunction);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion
}
