using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// KL-regularized Variational Autoencoder for latent diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AutoencoderKL is the standard VAE architecture used in Stable Diffusion and other
/// latent diffusion models. It compresses high-resolution images to a compact latent
/// representation while maintaining perceptual quality through KL-regularization.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoencoderKL is the "image compressor" used by Stable Diffusion.
///
/// Why use KL-regularized VAE?
/// 1. Compression: 512x512x3 image -> 64x64x4 latent (48x smaller!)
/// 2. KL-regularization: Keeps the latent space well-organized (Gaussian distribution)
/// 3. This organization makes diffusion work better in latent space
///
/// The "KL" in AutoencoderKL refers to Kullback-Leibler divergence, which measures
/// how different the encoder's output distribution is from a standard normal.
/// By minimizing KL divergence, we ensure the latent space is smooth and continuous.
///
/// Architecture:
/// ```
///     Image (512x512x3)
///           │
///           ├─→ VAEEncoder ─→ [mean, logvar] (64x64x8)
///           │                        │
///           │               Sample using reparameterization
///           │                        │
///           │                        ↓
///           │              Latent z (64x64x4)
///           │                        │
///           │                 [Scale by 0.18215]
///           │                        │
///           │                        ↓
///           │              Scaled latent (for diffusion)
///           │                        │
///           │                 [Unscale by 1/0.18215]
///           │                        │
///           │                        ↓
///           │              Latent z (64x64x4)
///           │                        │
///           └────────────────→ VAEDecoder
///                                    │
///                                    ↓
///                          Reconstructed Image (512x512x3)
/// ```
/// </para>
/// </remarks>
public class AutoencoderKL<T> : VAEModelBase<T>
{
    /// <summary>
    /// Standard Stable Diffusion latent scale factor.
    /// This normalizes the latent distribution for better diffusion performance.
    /// </summary>
    private const double SD_LATENT_SCALE = 0.18215;

    /// <summary>
    /// The encoder component.
    /// </summary>
    private readonly VAEEncoder<T> _encoder;

    /// <summary>
    /// The decoder component.
    /// </summary>
    private readonly VAEDecoder<T> _decoder;

    /// <summary>
    /// Number of input/output image channels.
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Number of latent channels.
    /// </summary>
    private readonly int _latentChannels;

    /// <summary>
    /// Base channel count for encoder/decoder.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Channel multipliers for each level.
    /// </summary>
    private readonly int[] _channelMults;

    /// <summary>
    /// Latent scale factor.
    /// </summary>
    private readonly double _latentScaleFactor;

    /// <summary>
    /// Cached mean from last encoding.
    /// </summary>
    private Tensor<T>? _cachedMean;

    /// <summary>
    /// Cached log variance from last encoding.
    /// </summary>
    private Tensor<T>? _cachedLogVar;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => _encoder.DownsampleFactor;

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public override int ParameterCount => _encoder.GetParameters().Length + _decoder.GetParameters().Length;

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <summary>
    /// Initializes a new instance of the AutoencoderKL class with default Stable Diffusion configuration.
    /// </summary>
    /// <param name="inputChannels">Number of input image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMults">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per level (default: 2).</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="latentScaleFactor">Scale factor for latents (default: 0.18215).</param>
    /// <param name="inputSpatialSize">Spatial size of input images (default: 512).</param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create an AutoencoderKL with sensible defaults for most use cases.
    ///
    /// Default configuration matches Stable Diffusion v1.5/v2.1 VAE:
    /// - 3 RGB channels in/out
    /// - 4 latent channels
    /// - 8x spatial downsampling (512x512 -> 64x64)
    /// - Channel progression: 128 -> 256 -> 512 -> 512
    ///
    /// For custom configurations:
    /// - Smaller latentChannels = more compression, potentially lower quality
    /// - Larger baseChannels = more capacity, but slower and more memory
    /// - More channelMults levels = more downsampling, smaller latents
    /// </para>
    /// </remarks>
    public AutoencoderKL(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMults = null,
        int numResBlocks = 2,
        int numGroups = 32,
        double? latentScaleFactor = null,
        int inputSpatialSize = 512,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMults = channelMults ?? new[] { 1, 2, 4, 4 };
        _latentScaleFactor = latentScaleFactor ?? SD_LATENT_SCALE;

        // Create encoder
        _encoder = new VAEEncoder<T>(
            inputChannels: inputChannels,
            latentChannels: latentChannels,
            baseChannels: baseChannels,
            channelMults: _channelMults,
            numResBlocks: numResBlocks,
            numGroups: numGroups,
            inputSpatialSize: inputSpatialSize);

        // Create decoder
        _decoder = new VAEDecoder<T>(
            outputChannels: inputChannels,
            latentChannels: latentChannels,
            baseChannels: baseChannels,
            channelMults: _channelMults,
            numResBlocks: numResBlocks,
            numGroups: numGroups,
            outputSpatialSize: inputSpatialSize);
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
        var (mean, logVar) = _encoder.EncodeWithDistribution(image);
        _cachedMean = mean;
        _cachedLogVar = logVar;
        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        return _decoder.Forward(latent);
    }

    /// <summary>
    /// Encodes an image and applies latent scaling for use in diffusion.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="sampleMode">Whether to sample from the distribution (default: true).</param>
    /// <returns>Scaled latent representation ready for diffusion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this method when preparing images for diffusion.
    ///
    /// The latent scaling is important because:
    /// 1. It normalizes the latent distribution to unit variance
    /// 2. This helps the diffusion model work with consistent noise levels
    /// 3. The scale factor (0.18215) was empirically determined for SD VAE
    /// </para>
    /// </remarks>
    public Tensor<T> EncodeForDiffusion(Tensor<T> image, bool sampleMode = true)
    {
        var latent = Encode(image, sampleMode);
        return ScaleLatent(latent);
    }

    /// <summary>
    /// Decodes a diffusion latent back to image space.
    /// </summary>
    /// <param name="latent">The scaled latent from diffusion.</param>
    /// <returns>The decoded image.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this method to convert diffusion output to images.
    ///
    /// Steps:
    /// 1. Unscale the latent (divide by scale factor)
    /// 2. Decode through the VAE decoder
    /// 3. Result is an image in [-1, 1] range
    ///
    /// To display/save, convert from [-1, 1] to [0, 255]:
    /// pixel = (value + 1) * 127.5
    /// </para>
    /// </remarks>
    public Tensor<T> DecodeFromDiffusion(Tensor<T> latent)
    {
        var unscaled = UnscaleLatent(latent);
        return Decode(unscaled);
    }

    /// <summary>
    /// Performs a full forward pass: encode -> sample -> decode.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <returns>Reconstructed image.</returns>
    public Tensor<T> Forward(Tensor<T> image)
    {
        var latent = Encode(image, sampleMode: true);
        return Decode(latent);
    }

    /// <summary>
    /// Computes the VAE loss (reconstruction + KL divergence).
    /// </summary>
    /// <param name="image">Original input image.</param>
    /// <param name="reconstruction">Reconstructed image from Forward().</param>
    /// <param name="klWeight">Weight for KL divergence term (default: 1e-6).</param>
    /// <returns>Combined loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The VAE loss has two parts:
    ///
    /// 1. Reconstruction loss: How different is the output from the input?
    ///    - Uses MSE (mean squared error) by default
    ///    - Lower = better reconstruction
    ///
    /// 2. KL divergence loss: How different is the latent distribution from N(0,1)?
    ///    - Regularizes the latent space to be smooth
    ///    - Lower = more organized latent space
    ///
    /// The klWeight controls the trade-off:
    /// - Higher klWeight = more regularized latent space, potentially blurrier reconstructions
    /// - Lower klWeight = sharper reconstructions, but less organized latent space
    ///
    /// Default 1e-6 is very small because we prioritize reconstruction quality
    /// for diffusion applications.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Trains the VAE on a single image.
    /// </summary>
    /// <param name="input">Input image to reconstruct.</param>
    /// <param name="expectedOutput">Target output (usually same as input for VAE).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var reconstruction = Forward(input);

        // Compute combined loss
        var loss = ComputeVAELoss(expectedOutput, reconstruction, klWeight: 1e-6);

        // Backward pass (simplified - actual implementation would propagate gradients)
        var reconstructionGrad = LossFunction.CalculateDerivative(reconstruction.ToVector(), expectedOutput.ToVector());
        var gradTensor = new Tensor<T>(reconstruction.Shape);
        var gradSpan = gradTensor.AsWritableSpan();
        for (int i = 0; i < gradSpan.Length && i < reconstructionGrad.Length; i++)
        {
            gradSpan[i] = reconstructionGrad[i];
        }

        // Backward through decoder
        var decoderGrad = _decoder.Backward(gradTensor);

        // Backward through encoder
        _encoder.Backward(decoderGrad);

        // Update parameters
        var lr = NumOps.FromDouble(1e-4);
        _encoder.UpdateParameters(lr);
        _decoder.UpdateParameters(lr);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var encoderParams = _encoder.GetParameters();
        var decoderParams = _decoder.GetParameters();

        var combined = new T[encoderParams.Length + decoderParams.Length];
        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[i] = encoderParams[i];
        }
        for (int i = 0; i < decoderParams.Length; i++)
        {
            combined[encoderParams.Length + i] = decoderParams[i];
        }

        return new Vector<T>(combined);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var encoderParams = _encoder.GetParameters();
        var decoderParams = _decoder.GetParameters();

        var newEncoderParams = new Vector<T>(encoderParams.Length);
        for (int i = 0; i < encoderParams.Length && i < parameters.Length; i++)
        {
            newEncoderParams[i] = parameters[i];
        }
        _encoder.SetParameters(newEncoderParams);

        var newDecoderParams = new Vector<T>(decoderParams.Length);
        for (int i = 0; i < decoderParams.Length && encoderParams.Length + i < parameters.Length; i++)
        {
            newDecoderParams[i] = parameters[encoderParams.Length + i];
        }
        _decoder.SetParameters(newDecoderParams);
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets the internal state of encoder and decoder.
    /// </summary>
    public void ResetState()
    {
        _cachedMean = null;
        _cachedLogVar = null;
        _encoder.ResetState();
        _decoder.ResetState();
    }

    /// <inheritdoc />
    public override void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Save version
        writer.Write(2); // Version 2 for AutoencoderKL

        // Save configuration
        writer.Write(_inputChannels);
        writer.Write(_latentChannels);
        writer.Write(_baseChannels);
        writer.Write(_channelMults.Length);
        foreach (var mult in _channelMults)
        {
            writer.Write(mult);
        }
        writer.Write(_latentScaleFactor);

        // Save encoder and decoder
        _encoder.Serialize(writer);
        _decoder.Serialize(writer);

        stream.Flush();
    }

    /// <inheritdoc />
    public override void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read version
        var version = reader.ReadInt32();
        if (version != 2)
            throw new InvalidOperationException($"Unsupported model version: {version}");

        // Read and validate configuration
        var inputChannels = reader.ReadInt32();
        var latentChannels = reader.ReadInt32();
        var baseChannels = reader.ReadInt32();
        var numMults = reader.ReadInt32();
        var channelMults = new int[numMults];
        for (int i = 0; i < numMults; i++)
        {
            channelMults[i] = reader.ReadInt32();
        }
        _ = reader.ReadDouble(); // latentScaleFactor

        if (inputChannels != _inputChannels || latentChannels != _latentChannels ||
            baseChannels != _baseChannels || !channelMults.SequenceEqual(_channelMults))
        {
            throw new InvalidOperationException("Architecture mismatch in AutoencoderKL deserialization.");
        }

        // Load encoder and decoder
        _encoder.Deserialize(reader);
        _decoder.Deserialize(reader);
    }

    #endregion

    #region Cloning

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new AutoencoderKL<T>(
            _inputChannels,
            _latentChannels,
            _baseChannels,
            _channelMults,
            numResBlocks: 2,
            numGroups: 32,
            _latentScaleFactor,
            inputSpatialSize: 512,
            LossFunction);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion

    /// <summary>
    /// Gets the encoder component for direct access.
    /// </summary>
    public VAEEncoder<T> Encoder => _encoder;

    /// <summary>
    /// Gets the decoder component for direct access.
    /// </summary>
    public VAEDecoder<T> Decoder => _decoder;

    /// <summary>
    /// Creates a default AutoencoderKL matching Stable Diffusion v1.5 configuration.
    /// </summary>
    public static AutoencoderKL<T> StableDiffusionV1() => new(
        inputChannels: 3,
        latentChannels: 4,
        baseChannels: 128,
        channelMults: new[] { 1, 2, 4, 4 },
        numResBlocks: 2,
        numGroups: 32,
        latentScaleFactor: 0.18215,
        inputSpatialSize: 512);

    /// <summary>
    /// Creates an AutoencoderKL matching SDXL configuration.
    /// </summary>
    public static AutoencoderKL<T> SDXL() => new(
        inputChannels: 3,
        latentChannels: 4,
        baseChannels: 128,
        channelMults: new[] { 1, 2, 4, 4 },
        numResBlocks: 2,
        numGroups: 32,
        latentScaleFactor: 0.13025, // SDXL uses different scale
        inputSpatialSize: 1024);

    /// <summary>
    /// Creates a lightweight AutoencoderKL for testing/experimentation.
    /// </summary>
    public static AutoencoderKL<T> Lightweight() => new(
        inputChannels: 3,
        latentChannels: 4,
        baseChannels: 64,
        channelMults: new[] { 1, 2, 4 },
        numResBlocks: 1,
        numGroups: 16,
        latentScaleFactor: 0.18215,
        inputSpatialSize: 256);
}
