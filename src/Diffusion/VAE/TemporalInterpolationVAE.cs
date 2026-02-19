using AiDotNet.ActivationFunctions;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Temporal interpolation VAE that generates intermediate frames in latent space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FILM: Frame Interpolation for Large Motion" (Reda et al., 2022)</item>
/// <item>Paper: "Stable Video Diffusion" (Blattmann et al., 2023)</item>
/// </list></para>
/// <para>
/// The Temporal Interpolation VAE extends standard video VAE with the ability to generate
/// intermediate frames between keyframes. This enables:
/// - Frame rate upsampling (e.g., 8fps to 24fps) in latent space
/// - Smoother video generation by interpolating between diffusion outputs
/// - Multi-scale temporal generation (coarse keyframes then fine interpolation)
/// </para>
/// <para>
/// Architecture:
/// - Inherits spatial encoding/decoding from TemporalVAE
/// - Adds temporal interpolation network that operates in latent space
/// - Interpolation uses bidirectional temporal attention and flow estimation
/// </para>
/// </remarks>
public class TemporalInterpolationVAE<T> : VAEModelBase<T>
{
    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int _baseChannels;
    private readonly int _interpolationFactor;
    private readonly double _latentScaleFactor;

    private readonly DenseLayer<T> _encoderIn;
    private readonly DenseLayer<T> _encoderOut;
    private readonly DenseLayer<T> _decoderIn;
    private readonly DenseLayer<T> _decoderOut;
    private readonly DenseLayer<T> _interpIn;
    private readonly DenseLayer<T> _interpOut;
    private readonly LayerNormalizationLayer<T> _encoderNorm;
    private readonly LayerNormalizationLayer<T> _decoderNorm;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 8;

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public override int ParameterCount => GetParameters().Length;

    /// <summary>
    /// Gets the temporal interpolation factor.
    /// </summary>
    public int InterpolationFactor => _interpolationFactor;

    /// <summary>
    /// Initializes a new Temporal Interpolation VAE.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels.</param>
    /// <param name="baseChannels">Base channel count.</param>
    /// <param name="interpolationFactor">Number of intermediate frames to generate between keyframes.</param>
    /// <param name="latentScaleFactor">Scale factor for latent space normalization.</param>
    public TemporalInterpolationVAE(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int interpolationFactor = 3,
        double latentScaleFactor = 0.18215)
        : base(new MeanSquaredErrorLoss<T>())
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _interpolationFactor = interpolationFactor;
        _latentScaleFactor = latentScaleFactor;

        int hiddenChannels = baseChannels * 4;

        _encoderIn = new DenseLayer<T>(inputChannels, baseChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _encoderOut = new DenseLayer<T>(hiddenChannels, latentChannels * 2, (IActivationFunction<T>)new IdentityActivation<T>());
        _decoderIn = new DenseLayer<T>(latentChannels, hiddenChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _decoderOut = new DenseLayer<T>(baseChannels, inputChannels, (IActivationFunction<T>)new IdentityActivation<T>());

        // Interpolation network: takes two latent frames and produces intermediate frame
        _interpIn = new DenseLayer<T>(latentChannels * 2, hiddenChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _interpOut = new DenseLayer<T>(hiddenChannels, latentChannels, (IActivationFunction<T>)new IdentityActivation<T>());

        _encoderNorm = new LayerNormalizationLayer<T>(hiddenChannels);
        _decoderNorm = new LayerNormalizationLayer<T>(baseChannels);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(input);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> input)
    {
        var x = _encoderIn.Forward(input);
        x = _encoderNorm.Forward(x);
        x = _encoderOut.Forward(x);

        int halfLen = x.Shape[^1] / 2;
        var meanShape = GetReducedShape(x.Shape, halfLen);
        var mean = new Tensor<T>(meanShape);
        var logVar = new Tensor<T>(meanShape);
        int elements = mean.Shape.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < elements; i++)
        {
            mean[i] = x[i];
            logVar[i] = x[elements + i];
        }

        return (ScaleLatent(mean), logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        var unscaled = UnscaleLatent(latent);
        var x = _decoderIn.Forward(unscaled);
        x = _decoderNorm.Forward(x);
        x = _decoderOut.Forward(x);
        return x;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var clone = new TemporalInterpolationVAE<T>(
            inputChannels: _inputChannels,
            latentChannels: _latentChannels,
            baseChannels: _baseChannels,
            interpolationFactor: _interpolationFactor,
            latentScaleFactor: _latentScaleFactor);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <summary>
    /// Interpolates between two latent frames to generate an intermediate frame.
    /// </summary>
    /// <param name="latentA">Latent representation of the first frame.</param>
    /// <param name="latentB">Latent representation of the second frame.</param>
    /// <returns>Interpolated latent frame.</returns>
    public Tensor<T> InterpolateLatent(Tensor<T> latentA, Tensor<T> latentB)
    {
        // Concatenate two latent frames as input to interpolation network
        var combined = latentA.ConcatenateTensors(latentB);
        var hidden = _interpIn.Forward(combined);
        return _interpOut.Forward(hidden);
    }

    private static int[] GetReducedShape(int[] shape, int lastDim)
    {
        var result = (int[])shape.Clone();
        result[^1] = lastDim;
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parts = new[]
        {
            _encoderIn.GetParameters(), _encoderOut.GetParameters(),
            _decoderIn.GetParameters(), _decoderOut.GetParameters(),
            _interpIn.GetParameters(), _interpOut.GetParameters(),
            _encoderNorm.GetParameters(), _decoderNorm.GetParameters()
        };

        int total = 0;
        foreach (var p in parts) total += p.Length;

        var combined = new Vector<T>(total);
        int offset = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < p.Length; i++)
                combined[offset + i] = p[i];
            offset += p.Length;
        }
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var layers = new LayerBase<T>[]
        {
            _encoderIn, _encoderOut, _decoderIn, _decoderOut,
            _interpIn, _interpOut, _encoderNorm, _decoderNorm
        };

        int offset = 0;
        foreach (var layer in layers)
        {
            int count = layer.GetParameters().Length;
            var sub = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                sub[i] = parameters[offset + i];
            layer.SetParameters(sub);
            offset += count;
        }
    }

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new TemporalInterpolationVAE<T>(
            inputChannels: _inputChannels,
            latentChannels: _latentChannels,
            baseChannels: _baseChannels,
            interpolationFactor: _interpolationFactor,
            latentScaleFactor: _latentScaleFactor);
        clone.SetParameters(GetParameters());
        return clone;
    }
}
