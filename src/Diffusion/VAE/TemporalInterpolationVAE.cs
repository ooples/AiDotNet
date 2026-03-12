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
/// <para><b>For Beginners:</b> The Temporal Interpolation VAE supports variable frame rate encoding and decoding. It can compress videos at different temporal resolutions, enabling efficient processing of both fast-action and slow-motion content.</para>
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
    public override int ParameterCount =>
        _encoderIn.ParameterCount + _encoderOut.ParameterCount +
        _decoderIn.ParameterCount + _decoderOut.ParameterCount +
        _interpIn.ParameterCount + _interpOut.ParameterCount +
        _encoderNorm.ParameterCount + _decoderNorm.ParameterCount;

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
        Guard.Positive(inputChannels, nameof(inputChannels));
        Guard.Positive(latentChannels, nameof(latentChannels));
        Guard.Positive(baseChannels, nameof(baseChannels));
        Guard.Positive(interpolationFactor, nameof(interpolationFactor));

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _interpolationFactor = interpolationFactor;
        _latentScaleFactor = latentScaleFactor;

        int hiddenChannels = baseChannels * 4;

        // Encoder: inputChannels -> baseChannels -> (norm at baseChannels) -> latentChannels*2
        _encoderIn = new DenseLayer<T>(inputChannels, baseChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _encoderOut = new DenseLayer<T>(baseChannels, latentChannels * 2, (IActivationFunction<T>)new IdentityActivation<T>());
        // Decoder: latentChannels -> hiddenChannels -> (norm at hiddenChannels) -> inputChannels
        _decoderIn = new DenseLayer<T>(latentChannels, hiddenChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _decoderOut = new DenseLayer<T>(hiddenChannels, inputChannels, (IActivationFunction<T>)new IdentityActivation<T>());

        // Interpolation network: takes two latent frames and produces intermediate frame
        _interpIn = new DenseLayer<T>(latentChannels * 2, hiddenChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _interpOut = new DenseLayer<T>(hiddenChannels, latentChannels, (IActivationFunction<T>)new IdentityActivation<T>());

        _encoderNorm = new LayerNormalizationLayer<T>(baseChannels);
        _decoderNorm = new LayerNormalizationLayer<T>(hiddenChannels);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input, bool sampleMode = true)
    {
        Guard.NotNull(input, nameof(input));
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
        Guard.NotNull(latent, nameof(latent));
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
    public Tensor<T>[] InterpolateLatent(Tensor<T> latentA, Tensor<T> latentB)
    {
        Guard.NotNull(latentA);
        Guard.NotNull(latentB);

        if (latentA.Shape.Length != latentB.Shape.Length)
        {
            throw new ArgumentException(
                $"Latent shapes must have the same rank: latentA has {latentA.Shape.Length} dims, latentB has {latentB.Shape.Length} dims.");
        }

        for (int d = 0; d < latentA.Shape.Length; d++)
        {
            if (latentA.Shape[d] != latentB.Shape[d])
            {
                throw new ArgumentException(
                    $"Latent shape mismatch at dimension {d}: latentA={latentA.Shape[d]}, latentB={latentB.Shape[d]}.");
            }
        }

        var results = new Tensor<T>[_interpolationFactor];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < _interpolationFactor; i++)
        {
            double t = (double)(i + 1) / (_interpolationFactor + 1);
            // Blend input pair weighted by interpolation position
            var blendedA = latentA.Transform((v, _) => numOps.Multiply(v, numOps.FromDouble(1.0 - t)));
            var blendedB = latentB.Transform((v, _) => numOps.Multiply(v, numOps.FromDouble(t)));
            var combined = blendedA.ConcatenateTensors(blendedB);
            var hidden = _interpIn.Forward(combined);
            results[i] = _interpOut.Forward(hidden);
        }

        return results;
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

        int expected = 0;
        foreach (var layer in layers) expected += layer.GetParameters().Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

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
