using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Deep Compression Autoencoder (DC-AE) for extremely high spatial compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DC-AE achieves 32x-128x spatial compression (vs standard 8x) by combining residual
/// autoencoding with a decoupled two-stage training: first train a standard AE, then
/// add a lightweight latent adapter to achieve extreme compression while preserving quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard VAEs compress images 8x spatially (512x512 → 64x64).
/// DC-AE compresses 32x-128x (512x512 → 16x16 or even 4x4), making diffusion dramatically
/// faster because the model works on much smaller latent tensors. The two-stage training
/// ensures image quality doesn't suffer despite the extreme compression.
/// </para>
/// <para>
/// Reference: Chen et al., "Deep Compression Autoencoder for Efficient High-Resolution
/// Diffusion Models", NeurIPS 2024
/// </para>
/// </remarks>
public class DeepCompressionVAE<T> : VAEModelBase<T>
{
    private const double DCAE_LATENT_SCALE = 0.3611;
    private const int DEFAULT_DOWNSAMPLE = 32;

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int _downsampleFactor;
    private readonly double _latentScaleFactor;
    private readonly int _baseChannels;

    private List<ILayer<T>> _encoderLayers;
    private List<ILayer<T>> _decoderLayers;

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

    /// <summary>
    /// Initializes a new Deep Compression Autoencoder.
    /// </summary>
    /// <param name="inputChannels">Input image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Latent channels (default: 32 for high-info retention).</param>
    /// <param name="downsampleFactor">Spatial compression factor (default: 32).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="seed">Optional random seed.</param>
    public DeepCompressionVAE(
        int inputChannels = 3,
        int latentChannels = 32,
        int downsampleFactor = DEFAULT_DOWNSAMPLE,
        int baseChannels = 128,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _downsampleFactor = downsampleFactor;
        _baseChannels = baseChannels;
        _latentScaleFactor = DCAE_LATENT_SCALE;

        InitializeLayers();
    }

    [MemberNotNull(nameof(_encoderLayers), nameof(_decoderLayers))]
    private void InitializeLayers()
    {
        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();

        // Number of downsample stages = log2(downsampleFactor)
        int numStages = (int)Math.Log(Math.Max(2, _downsampleFactor), 2);
        int channels = _baseChannels;

        // Encoder: progressive downsampling with residual blocks
        for (int i = 0; i < numStages; i++)
        {
            int outChannels = Math.Min(channels * 2, 512);
            _encoderLayers.Add(new ConvolutionalLayer<T>(
                inputDepth: i == 0 ? _inputChannels : channels,
                outputDepth: outChannels,
                kernelSize: 3, inputHeight: 32, inputWidth: 32,
                stride: 2, padding: 1,
                activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));
            channels = outChannels;
        }

        // Latent projection
        _encoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: channels, outputDepth: _latentChannels,
            kernelSize: 1, inputHeight: 8, inputWidth: 8,
            stride: 1, padding: 0,
            activationFunction: new IdentityActivation<T>()));

        // Decoder: progressive upsampling
        _decoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: _latentChannels, outputDepth: channels,
            kernelSize: 1, inputHeight: 8, inputWidth: 8,
            stride: 1, padding: 0,
            activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));

        for (int i = numStages - 1; i >= 0; i--)
        {
            int outChannels = i == 0 ? _inputChannels : Math.Min(_baseChannels * (int)Math.Pow(2, i - 1), 512);
            var activation = i == 0
                ? (IActivationFunction<T>)new TanhActivation<T>()
                : (IActivationFunction<T>)new GELUActivation<T>();
            _decoderLayers.Add(new DeconvolutionalLayer<T>(
                inputShape: [1, channels, 8, 8],
                outputDepth: outChannels,
                kernelSize: 4, stride: 2, padding: 1,
                activationFunction: activation));
            channels = outChannels;
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> image, bool sampleMode = true)
    {
        var (mean, _) = EncodeWithDistribution(image);
        return mean;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image)
    {
        var x = image;
        foreach (var layer in _encoderLayers)
            x = layer.Forward(x);

        // DC-AE is deterministic (not variational) — logVar is zeros
        var logVar = new Tensor<T>(x.Shape);
        return (x, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        var x = latent;
        foreach (var layer in _decoderLayers)
            x = layer.Forward(x);
        return x;
    }

    private int CalculateParameterCount()
    {
        long count = 0;
        foreach (var layer in _encoderLayers)
            count += layer.GetParameters().Length;
        foreach (var layer in _decoderLayers)
            count += layer.GetParameters().Length;
        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        foreach (var layer in _encoderLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) parameters.Add(p[i]);
        }
        foreach (var layer in _decoderLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) parameters.Add(p[i]);
        }
        return new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in _encoderLayers)
        {
            var p = layer.GetParameters();
            var np = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length && index < parameters.Length; i++)
                np[i] = parameters[index++];
            layer.SetParameters(np);
        }
        foreach (var layer in _decoderLayers)
        {
            var p = layer.GetParameters();
            var np = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length && index < parameters.Length; i++)
                np[i] = parameters[index++];
            layer.SetParameters(np);
        }
    }

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new DeepCompressionVAE<T>(
            _inputChannels, _latentChannels, _downsampleFactor, _baseChannels,
            LossFunction);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
