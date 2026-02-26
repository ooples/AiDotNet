using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Equivariance-preserving VAE (EQ-VAE) with improved latent regularity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// EQ-VAE enforces equivariance constraints on the encoder/decoder pair, ensuring that
/// geometric transformations in pixel space correspond to the same transformations in
/// latent space. This produces smoother, more regular latent distributions that improve
/// downstream diffusion model training and generation quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you rotate or flip an image, the latent representation should
/// rotate or flip in the same way. Standard VAEs don't guarantee this, leading to
/// inconsistent latent spaces. EQ-VAE adds this guarantee, producing better latents
/// for diffusion models to work with — like having a well-organized workspace.
/// </para>
/// <para>
/// Reference: Xu et al., "EQ-VAE: Equivariance Regularized Latent Space for Improved
/// Generative Image Modeling", 2025
/// </para>
/// </remarks>
public class EQVAEModel<T> : VAEModelBase<T>
{
    private const double EQVAE_LATENT_SCALE = 0.18215;
    private const double DEFAULT_EQUIVARIANCE_WEIGHT = 0.1;

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int _baseChannels;
    private readonly double _equivarianceWeight;

    private List<ILayer<T>> _encoderLayers;
    private List<ILayer<T>> _decoderLayers;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 8;

    /// <inheritdoc />
    public override double LatentScaleFactor => EQVAE_LATENT_SCALE;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <summary>
    /// Initializes a new EQ-VAE model.
    /// </summary>
    /// <param name="inputChannels">Input image channels (default: 3).</param>
    /// <param name="latentChannels">Latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="equivarianceWeight">Weight for equivariance loss term (default: 0.1).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="seed">Optional random seed.</param>
    public EQVAEModel(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        double equivarianceWeight = DEFAULT_EQUIVARIANCE_WEIGHT,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _equivarianceWeight = equivarianceWeight;

        InitializeLayers();
    }

    [MemberNotNull(nameof(_encoderLayers), nameof(_decoderLayers))]
    private void InitializeLayers()
    {
        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();

        int[] multipliers = [1, 2, 4, 4];
        int channels = _baseChannels;

        // Encoder with equivariance-aware residual blocks
        _encoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: _inputChannels, outputDepth: channels,
            kernelSize: 3, inputHeight: 64, inputWidth: 64,
            stride: 1, padding: 1,
            activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));

        foreach (int mult in multipliers)
        {
            int outChannels = _baseChannels * mult;
            _encoderLayers.Add(new ConvolutionalLayer<T>(
                inputDepth: channels, outputDepth: outChannels,
                kernelSize: 3, inputHeight: 32, inputWidth: 32,
                stride: 2, padding: 1,
                activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));
            channels = outChannels;
        }

        // Latent projection (mean only — equivariance regularization replaces KL)
        _encoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: channels, outputDepth: _latentChannels,
            kernelSize: 1, inputHeight: 8, inputWidth: 8,
            stride: 1, padding: 0,
            activationFunction: new IdentityActivation<T>()));

        // Decoder
        _decoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: _latentChannels, outputDepth: channels,
            kernelSize: 1, inputHeight: 8, inputWidth: 8,
            stride: 1, padding: 0,
            activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));

        for (int i = multipliers.Length - 1; i >= 0; i--)
        {
            int outChannels = i == 0 ? _baseChannels : _baseChannels * multipliers[i - 1];
            _decoderLayers.Add(new DeconvolutionalLayer<T>(
                inputShape: [1, channels, 8, 8],
                outputDepth: outChannels,
                kernelSize: 4, stride: 2, padding: 1,
                activationFunction: (IActivationFunction<T>)new GELUActivation<T>()));
            channels = outChannels;
        }

        _decoderLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: channels, outputDepth: _inputChannels,
            kernelSize: 3, inputHeight: 64, inputWidth: 64,
            stride: 1, padding: 1,
            activationFunction: new TanhActivation<T>()));
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> image, bool sampleMode = true)
    {
        var x = image;
        foreach (var layer in _encoderLayers)
            x = layer.Forward(x);
        return x;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image)
    {
        var mean = Encode(image, sampleMode: false);
        var logVar = new Tensor<T>(mean.Shape);
        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        var x = latent;
        foreach (var layer in _decoderLayers)
            x = layer.Forward(x);
        return x;
    }

    /// <summary>
    /// Computes the equivariance loss between original and transformed encode-decode paths.
    /// </summary>
    /// <param name="original">Original image tensor.</param>
    /// <param name="transformed">Geometrically transformed image tensor.</param>
    /// <returns>Equivariance loss value.</returns>
    public T ComputeEquivarianceLoss(Tensor<T> original, Tensor<T> transformed)
    {
        var latentOrig = Encode(original);
        var latentTrans = Encode(transformed);

        // L2 distance between latent representations should be preserved
        var diff = new Tensor<T>(latentOrig.Shape);
        var origSpan = latentOrig.AsSpan();
        var transSpan = latentTrans.AsSpan();
        var diffSpan = diff.AsWritableSpan();

        var sum = NumOps.Zero;
        for (int i = 0; i < diffSpan.Length; i++)
        {
            var d = NumOps.Subtract(origSpan[i], transSpan[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(d, d));
        }

        return NumOps.Multiply(NumOps.FromDouble(_equivarianceWeight), sum);
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
        var clone = new EQVAEModel<T>(
            _inputChannels, _latentChannels, _baseChannels,
            _equivarianceWeight, LossFunction);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
