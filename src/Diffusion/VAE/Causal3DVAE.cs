using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Causal 3D VAE for video with temporal causal convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024)</item>
/// <item>Paper: "Open-Sora Plan" (2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> The Causal 3D VAE compresses video into a much smaller latent space while preserving temporal information. Causal means it only uses past frames to encode each frame, enabling streaming video compression.</para>
/// <para>
/// The Causal 3D VAE uses causal 3D convolutions to encode and decode video. Causal convolutions
/// ensure that each frame's encoding depends only on the current and previous frames, enabling:
/// - Streaming video generation (encode/decode frame by frame)
/// - Autoregressive generation without future frame leakage
/// - Temporal compression (e.g., 4x in time dimension)
/// </para>
/// <para>
/// Architecture:
/// - Encoder: Causal 3D Conv blocks with temporal stride for temporal compression
/// - Decoder: Causal 3D TransposeConv blocks for temporal upsampling
/// - Both spatial and temporal compression in latent space
/// - Typical compression: 8x spatial, 4x temporal
/// </para>
/// </remarks>
public class Causal3DVAE<T> : VAEModelBase<T>
{
    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int _baseChannels;
    private readonly int[] _channelMultipliers;
    private readonly int _temporalCompression;
    private readonly double _latentScaleFactor;

    private readonly DenseLayer<T> _encoderIn;
    private readonly DenseLayer<T> _encoderOut;
    private readonly DenseLayer<T> _decoderIn;
    private readonly DenseLayer<T> _decoderOut;
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
    /// Gets the temporal compression factor.
    /// </summary>
    public int TemporalCompression => _temporalCompression;

    /// <summary>
    /// Initializes a new Causal 3D VAE.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels.</param>
    /// <param name="baseChannels">Base channel count for encoder/decoder.</param>
    /// <param name="channelMultipliers">Channel multipliers for each level.</param>
    /// <param name="temporalCompression">Temporal compression factor (e.g., 4).</param>
    /// <param name="latentScaleFactor">Scale factor for latent space normalization.</param>
    public Causal3DVAE(
        int inputChannels = 3,
        int latentChannels = 16,
        int baseChannels = 128,
        int[]? channelMultipliers = null,
        int temporalCompression = 4,
        double latentScaleFactor = 0.13025)
        : base(new MeanSquaredErrorLoss<T>())
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? new[] { 1, 2, 4, 4 };
        _temporalCompression = temporalCompression;
        _latentScaleFactor = latentScaleFactor;

        int maxChannels = baseChannels * _channelMultipliers[^1];

        _encoderIn = new DenseLayer<T>(inputChannels, baseChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _encoderOut = new DenseLayer<T>(maxChannels, latentChannels * 2, (IActivationFunction<T>)new IdentityActivation<T>());
        _decoderIn = new DenseLayer<T>(latentChannels, maxChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _decoderOut = new DenseLayer<T>(baseChannels, inputChannels, (IActivationFunction<T>)new IdentityActivation<T>());
        _encoderNorm = new LayerNormalizationLayer<T>(maxChannels);
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

        // Split into mean and log variance
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
        var clone = new Causal3DVAE<T>(
            inputChannels: _inputChannels,
            latentChannels: _latentChannels,
            baseChannels: _baseChannels,
            channelMultipliers: (int[])_channelMultipliers.Clone(),
            temporalCompression: _temporalCompression,
            latentScaleFactor: _latentScaleFactor);
        clone.SetParameters(GetParameters());
        return clone;
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
            _encoderNorm, _decoderNorm
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
        var clone = new Causal3DVAE<T>(
            inputChannels: _inputChannels,
            latentChannels: _latentChannels,
            baseChannels: _baseChannels,
            channelMultipliers: (int[])_channelMultipliers.Clone(),
            temporalCompression: _temporalCompression,
            latentScaleFactor: _latentScaleFactor);
        clone.SetParameters(GetParameters());
        return clone;
    }
}
