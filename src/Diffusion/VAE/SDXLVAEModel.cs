using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// SDXL-optimized VAE with improved decoder fidelity for 1024x1024 generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SDXL VAE is a fine-tuned version of the Stable Diffusion VAE with improved
/// decoder weights for higher fidelity reconstruction at 1024x1024 resolution. Uses
/// the same architecture as StandardVAE but with SDXL-specific scale factors and
/// optimized decoder weights trained on high-resolution data.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDXL generates 1024x1024 images (4x more pixels than SD 1.5's
/// 512x512). The VAE decoder was retrained specifically for this higher resolution to
/// produce sharper, more detailed reconstructions. The encoder remains compatible with
/// the standard SD VAE, but the decoder is significantly improved.
/// </para>
/// <para>
/// Reference: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution
/// Image Synthesis", ICLR 2024
/// </para>
/// </remarks>
public class SDXLVAEModel<T> : VAEModelBase<T>
{
    private const double SDXL_LATENT_SCALE = 0.13025;
    private const int SDXL_BASE_CHANNELS = 128;

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int _baseChannels;
    private readonly int[] _channelMultipliers;

    private List<ILayer<T>> _encoderLayers;
    private List<ILayer<T>> _decoderLayers;
    private ConvolutionalLayer<T>? _inputConv;
    private ConvolutionalLayer<T>? _meanConv;
    private ConvolutionalLayer<T>? _logVarConv;
    private ConvolutionalLayer<T>? _postQuantConv;
    private ConvolutionalLayer<T>? _outputConv;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 8;

    /// <inheritdoc />
    public override double LatentScaleFactor => SDXL_LATENT_SCALE;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <summary>
    /// Initializes a new SDXL VAE model.
    /// </summary>
    /// <param name="inputChannels">Input channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="seed">Optional random seed.</param>
    public SDXLVAEModel(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = SDXL_BASE_CHANNELS,
        int[]? channelMultipliers = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? [1, 2, 4, 4];

        InitializeLayers();
    }

    [MemberNotNull(nameof(_encoderLayers), nameof(_decoderLayers))]
    private void InitializeLayers()
    {
        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();

        int channels = _baseChannels;

        // Input convolution
        _inputConv = new ConvolutionalLayer<T>(
            inputDepth: _inputChannels, outputDepth: channels,
            kernelSize: 3, inputHeight: 128, inputWidth: 128,
            stride: 1, padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Encoder blocks with progressive downsampling
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            int outChannels = _baseChannels * _channelMultipliers[level];

            // Two ResBlocks per level
            for (int block = 0; block < 2; block++)
            {
                int numGroups = CalculateGroupCount(channels, outChannels);
                _encoderLayers.Add(new VAEResBlock<T>(channels, outChannels, numGroups, spatialSize: 32));
                channels = outChannels;
            }

            // Downsample (except last level)
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderLayers.Add(new ConvolutionalLayer<T>(
                    inputDepth: channels, outputDepth: channels,
                    kernelSize: 3, inputHeight: 32, inputWidth: 32,
                    stride: 2, padding: 1,
                    activationFunction: new IdentityActivation<T>()));
            }
        }

        // Mean and log-var projections
        int lastChannels = _baseChannels * _channelMultipliers[^1];
        _meanConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels, outputDepth: _latentChannels,
            kernelSize: 3, inputHeight: 16, inputWidth: 16,
            stride: 1, padding: 1,
            activationFunction: new IdentityActivation<T>());

        _logVarConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels, outputDepth: _latentChannels,
            kernelSize: 3, inputHeight: 16, inputWidth: 16,
            stride: 1, padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Decoder: improved SDXL decoder with higher fidelity
        _postQuantConv = new ConvolutionalLayer<T>(
            inputDepth: _latentChannels, outputDepth: lastChannels,
            kernelSize: 3, inputHeight: 16, inputWidth: 16,
            stride: 1, padding: 1,
            activationFunction: new IdentityActivation<T>());

        channels = lastChannels;
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            int outChannels = _baseChannels * _channelMultipliers[level];

            for (int block = 0; block < 3; block++) // SDXL uses 3 blocks in decoder (vs 2 in encoder)
            {
                int numGroups = CalculateGroupCount(channels, outChannels);
                _decoderLayers.Add(new VAEResBlock<T>(channels, outChannels, numGroups, spatialSize: 32));
                channels = outChannels;
            }

            if (level > 0)
            {
                _decoderLayers.Add(new DeconvolutionalLayer<T>(
                    inputShape: [1, channels, 16, 16],
                    outputDepth: channels,
                    kernelSize: 4, stride: 2, padding: 1,
                    activationFunction: new IdentityActivation<T>()));
            }
        }

        _outputConv = new ConvolutionalLayer<T>(
            inputDepth: _baseChannels, outputDepth: _inputChannels,
            kernelSize: 3, inputHeight: 128, inputWidth: 128,
            stride: 1, padding: 1,
            activationFunction: new TanhActivation<T>());
    }

    private static int CalculateGroupCount(int inChannels, int outChannels)
    {
        int[] preferredGroups = [32, 16, 8, 4, 2, 1];
        foreach (int groups in preferredGroups)
        {
            if (inChannels % groups == 0 && outChannels % groups == 0)
                return groups;
        }
        return 1;
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> image, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(image);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image)
    {
        if (_inputConv == null || _meanConv == null || _logVarConv == null)
            throw new InvalidOperationException("Encoder layers not initialized.");

        var x = _inputConv.Forward(image);

        foreach (var layer in _encoderLayers)
            x = layer.Forward(x);

        var mean = _meanConv.Forward(x);
        var logVar = _logVarConv.Forward(x);

        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        if (_postQuantConv == null || _outputConv == null)
            throw new InvalidOperationException("Decoder layers not initialized.");

        var x = _postQuantConv.Forward(latent);

        foreach (var layer in _decoderLayers)
            x = layer.Forward(x);

        x = _outputConv.Forward(x);
        return x;
    }

    private int CalculateParameterCount()
    {
        long count = 0;

        if (_inputConv != null) count += _inputConv.GetParameters().Length;
        foreach (var layer in _encoderLayers) count += layer.GetParameters().Length;
        if (_meanConv != null) count += _meanConv.GetParameters().Length;
        if (_logVarConv != null) count += _logVarConv.GetParameters().Length;
        if (_postQuantConv != null) count += _postQuantConv.GetParameters().Length;
        foreach (var layer in _decoderLayers) count += layer.GetParameters().Length;
        if (_outputConv != null) count += _outputConv.GetParameters().Length;

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        AddLayerParams(parameters, _inputConv);
        foreach (var layer in _encoderLayers) AddLayerParams(parameters, layer);
        AddLayerParams(parameters, _meanConv);
        AddLayerParams(parameters, _logVarConv);
        AddLayerParams(parameters, _postQuantConv);
        foreach (var layer in _decoderLayers) AddLayerParams(parameters, layer);
        AddLayerParams(parameters, _outputConv);

        return new Vector<T>(parameters.ToArray());
    }

    private static void AddLayerParams(List<T> parameters, ILayer<T>? layer)
    {
        if (layer == null) return;
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) parameters.Add(p[i]);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        SetLayerParams(_inputConv, parameters, ref index);
        foreach (var layer in _encoderLayers) SetLayerParams(layer, parameters, ref index);
        SetLayerParams(_meanConv, parameters, ref index);
        SetLayerParams(_logVarConv, parameters, ref index);
        SetLayerParams(_postQuantConv, parameters, ref index);
        foreach (var layer in _decoderLayers) SetLayerParams(layer, parameters, ref index);
        SetLayerParams(_outputConv, parameters, ref index);
    }

    private static void SetLayerParams(ILayer<T>? layer, Vector<T> parameters, ref int index)
    {
        if (layer == null) return;
        var p = layer.GetParameters();
        var np = new Vector<T>(p.Length);
        for (int i = 0; i < p.Length && index < parameters.Length; i++)
            np[i] = parameters[index++];
        layer.SetParameters(np);
    }

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new SDXLVAEModel<T>(
            _inputChannels, _latentChannels, _baseChannels,
            _channelMultipliers, LossFunction);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
