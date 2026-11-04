using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// The KL-regularized autoencoder of latent diffusion (Rombach et al., 2022), the VAE of Stable Diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022), Sec. 3.1
/// and Appendix G</item>
/// <item>Implementation: diffusers <c>AutoencoderKL</c> (the <c>runwayml/stable-diffusion-v1-5</c> <c>vae</c> folder),
/// which this reproduces (<c>tools/reference-data/autoencoder_kl_reference.py</c>); <see cref="LoadDiffusersWeights"/>
/// loads its state dict.</item>
/// </list></para>
/// <para><b>For Beginners:</b> AutoencoderKL is the "image compressor" Stable Diffusion works through. It squeezes a
/// 512x512 RGB image into a 64x64 grid of 4 latent channels; the diffusion model works there, and the decoder turns
/// its result back into an image. The "KL" is the Kullback-Leibler penalty that keeps the latent close to a standard
/// normal, so the latent space stays smooth.</para>
/// <para>
/// Architecture: a 3x3 conv in; one stage of residual blocks per width, each stage but the last halving both axes with a
/// stride-2 conv; a middle block of residual block, single-head self-attention and residual block; GroupNorm, SiLU and a
/// 3x3 conv to the posterior's mean and log-variance; a 1x1 quant conv. The decoder mirrors it (a 1x1 post-quant conv
/// first, one more residual block per stage, nearest 2x upsampling then a conv). The widths are
/// <c>baseChannels</c> times each of <c>channelMults</c>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var vae = AutoencoderKL&lt;float&gt;.StableDiffusionV1();
/// var image = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 3, 512, 512 });
/// var latent = vae.Encode(image, sampleMode: false); // [1, 4, 64, 64] for a [1, 3, 512, 512] image
/// var decoded = vae.Decode(latent);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("High-Resolution Image Synthesis with Latent Diffusion Models", "https://arxiv.org/abs/2112.10752", Year = 2022, Authors = "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer")]
[ResearchPaper("Auto-Encoding Variational Bayes", "https://arxiv.org/abs/1312.6114", Year = 2013, Authors = "Kingma & Welling")]
public partial class AutoencoderKL<T> : VAEModelBase<T>
{
    /// <summary>The scaling factor of Stable Diffusion 1.x and 2.x's VAE.</summary>
    public const double StableDiffusionLatentScale = 0.18215;

    /// <summary>The scaling factor of SDXL's VAE.</summary>
    public const double SdxlLatentScale = 0.13025;

    // diffusers' DiagonalGaussianDistribution clamps the log-variance to this range.
    private const double MinLogVariance = -30.0;
    private const double MaxLogVariance = 20.0;

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_encoder);
        RegisterParameterComponent(_quantConv);
        RegisterParameterComponent(_postQuantConv);
        RegisterParameterComponent(_decoder);
    }

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int[] _blockOutChannels;
    private readonly double _latentScaleFactor;

    private readonly KlEncoder<T> _encoder;
    private readonly KlConv2d<T> _quantConv;
    private readonly KlConv2d<T> _postQuantConv;
    private readonly KlDecoder<T> _decoder;

    [Scratch]
    private Tensor<T>? _cachedMean;

    [Scratch]
    private Tensor<T>? _cachedLogVar;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 1 << (_blockOutChannels.Length - 1);

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <inheritdoc />
    /// <remarks>Revision 2: latent diffusion's autoencoder replaced a network without the middle block's attention,
    /// with separate mean and log-variance convolutions and transposed-convolution upsampling.</remarks>
    protected override int ArchitectureRevision => 2;

    /// <inheritdoc />
    protected override string ArchitectureRevisionNote =>
        "Revision 2 is latent diffusion's KL autoencoder (diffusers AutoencoderKL); revision 1 had no middle-block attention, separate mean and log-variance convolutions and transposed-convolution upsampling.";

    /// <summary>
    /// Initializes an AutoencoderKL; the defaults are Stable Diffusion v1.5's (diffusers <c>AutoencoderKL</c>,
    /// <c>runwayml/stable-diffusion-v1-5</c>).
    /// </summary>
    /// <param name="inputChannels">Image channels (3).</param>
    /// <param name="latentChannels">Latent channels (4).</param>
    /// <param name="baseChannels">Width of the first stage (128).</param>
    /// <param name="channelMults">Each stage's width as a multiple of <paramref name="baseChannels"/> (1, 2, 4, 4); each
    /// stage but the last halves both axes, so four stages compress 8x.</param>
    /// <param name="numResBlocks">Residual blocks per encoder stage (2); the decoder's stages have one more.</param>
    /// <param name="numGroups">GroupNorm groups (32); every width must divide by it.</param>
    /// <param name="latentScaleFactor">The latent's scale for diffusion (<see cref="StableDiffusionLatentScale"/>).</param>
    /// <param name="lossFunction">Reconstruction loss for training (mean squared error by default).</param>
    /// <param name="seed">Seed for the weight initialization and the posterior sampling.</param>
    public AutoencoderKL(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMults = null,
        int numResBlocks = 2,
        int numGroups = 32,
        double? latentScaleFactor = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction ?? new MeanSquaredErrorLoss<T>(), seed)
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (baseChannels <= 0) throw new ArgumentOutOfRangeException(nameof(baseChannels));
        if (numResBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numResBlocks));
        double scale = latentScaleFactor ?? StableDiffusionLatentScale;
        if (scale <= 0) throw new ArgumentOutOfRangeException(nameof(latentScaleFactor));
        var mults = channelMults ?? new[] { 1, 2, 4, 4 };
        if (mults.Length == 0) throw new ArgumentException("At least one stage is required.", nameof(channelMults));
        var widths = new int[mults.Length];
        for (int i = 0; i < mults.Length; i++)
        {
            if (mults[i] <= 0) throw new ArgumentOutOfRangeException(nameof(channelMults), "Every multiplier must be positive.");
            widths[i] = baseChannels * mults[i];
            if (numGroups <= 0 || widths[i] % numGroups != 0)
                throw new ArgumentException($"Stage width {widths[i]} does not divide into {numGroups} groups.", nameof(numGroups));
        }

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _blockOutChannels = widths;
        _latentScaleFactor = scale;

        _encoder = new KlEncoder<T>(inputChannels, latentChannels, widths, numResBlocks, numGroups);
        _quantConv = new KlConv2d<T>(2 * latentChannels, 2 * latentChannels, 1);
        _postQuantConv = new KlConv2d<T>(latentChannels, latentChannels, 1);
        _decoder = new KlDecoder<T>(latentChannels, inputChannels, widths, numResBlocks, numGroups);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> image, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(image);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    /// <param name="image">An image, [batch, channels, height, width].</param>
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image)
    {
        RequireImage(image, _inputChannels, nameof(image));
        var moments = _quantConv.Forward(_encoder.Forward(image));
        var s = moments.Shape;
        var mean = Engine.TensorSlice(moments, new[] { 0, 0, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] });
        var logVar = Engine.TensorClamp(
            Engine.TensorSlice(moments, new[] { 0, _latentChannels, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] }),
            NumOps.FromDouble(MinLogVariance), NumOps.FromDouble(MaxLogVariance));
        _cachedMean = mean;
        _cachedLogVar = logVar;
        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        RequireImage(latent, _latentChannels, nameof(latent));
        // Every denoising step decodes at the same latent shape, so later calls replay a compiled plan (#1272).
        return DecodeCompiled(latent, () => _decoder.Forward(_postQuantConv.Forward(latent)));
    }

    /// <summary>
    /// Loads a diffusers <c>AutoencoderKL</c> state dict of this configuration (a Stable Diffusion <c>vae</c> folder).
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and this model's shape.</param>
    /// <returns>The names read: every weight of the model.</returns>
    public IReadOnlyCollection<string> LoadDiffusersWeights(Func<string, int[], double[]> read)
    {
        var names = KlModule<T>.Load(
            _encoder.NamedTensors("encoder")
                .Concat(_quantConv.NamedTensors("quant_conv"))
                .Concat(_postQuantConv.NamedTensors("post_quant_conv"))
                .Concat(_decoder.NamedTensors("decoder")),
            read);
        InvalidateCompiledHosts();
        return names;
    }

    private static void RequireImage(Tensor<T> x, int channels, string name)
    {
        if (x is null) throw new ArgumentNullException(name);
        if (x.Rank != 4)
            throw new ArgumentException($"Expected [batch, channels, height, width]; got rank {x.Rank}.", name);
        if (x.Shape[1] != channels)
            throw new ArgumentException($"Expected {channels} channel(s); got {x.Shape[1]}.", name);
    }

    /// <summary>
    /// Encodes an image and scales the latent for diffusion.
    /// </summary>
    /// <param name="image">The image, [batch, channels, height, width], in [-1, 1].</param>
    /// <param name="sampleMode">Whether to sample the posterior (true) or take its mean.</param>
    /// <returns>The latent times <see cref="LatentScaleFactor"/>.</returns>
    public Tensor<T> EncodeForDiffusion(Tensor<T> image, bool sampleMode = true) => ScaleLatent(Encode(image, sampleMode));

    /// <summary>
    /// Decodes a diffusion latent (divided by <see cref="LatentScaleFactor"/> first) back to an image in [-1, 1].
    /// </summary>
    /// <param name="latent">The scaled latent.</param>
    /// <returns>The decoded image.</returns>
    public Tensor<T> DecodeFromDiffusion(Tensor<T> latent) => Decode(UnscaleLatent(latent));

    /// <summary>
    /// Encodes, samples the posterior and decodes.
    /// </summary>
    /// <param name="image">The image.</param>
    /// <returns>The reconstruction.</returns>
    public Tensor<T> Forward(Tensor<T> image) => Decode(Encode(image, sampleMode: true));

    /// <summary>
    /// The VAE loss: reconstruction plus the KL divergence of the last encoding's posterior, weighted.
    /// </summary>
    /// <param name="image">The original image.</param>
    /// <param name="reconstruction">The reconstruction from <see cref="Forward"/>.</param>
    /// <param name="klWeight">Weight of the KL term (1e-6, latent diffusion's).</param>
    /// <returns>The combined loss.</returns>
    public T ComputeVAELoss(Tensor<T> image, Tensor<T> reconstruction, double klWeight = 1e-6)
    {
        var reconLoss = LossFunction.CalculateLoss(reconstruction.ToVector(), image.ToVector());
        if (_cachedMean is null || _cachedLogVar is null) return reconLoss;
        var klLoss = ComputeKLDivergence(_cachedMean, _cachedLogVar);
        return NumOps.Add(reconLoss, NumOps.Multiply(klLoss, NumOps.FromDouble(klWeight)));
    }

    /// <summary>Forgets the last encoding's posterior.</summary>
    public void ResetState()
    {
        _cachedMean = null;
        _cachedLogVar = null;
    }

    /// <summary>
    /// Drops the decoder's cached compiled plans, so the next <see cref="Decode"/> re-traces. Call after changing the
    /// weights outside <see cref="LoadDiffusersWeights"/> and the parameter registry.
    /// </summary>
    public void InvalidateCompiledHosts() => InvalidateVAECompiledPlans();

    /// <summary>Stable Diffusion v1.5's VAE.</summary>
    public static AutoencoderKL<T> StableDiffusionV1() => new(latentScaleFactor: StableDiffusionLatentScale);

    /// <summary>SDXL's VAE: the same architecture with its own latent scale.</summary>
    public static AutoencoderKL<T> SDXL() => new(latentScaleFactor: SdxlLatentScale);

    /// <summary>A smaller VAE of the same architecture, for experiments.</summary>
    public static AutoencoderKL<T> Lightweight() => new(
        baseChannels: 64, channelMults: new[] { 1, 2, 4 }, numResBlocks: 1, numGroups: 16);

    /// <inheritdoc />
    /// <remarks>The gradient tape differentiates this VAE; it keeps no per-layer gradient buffers.</remarks>
    protected override Vector<T> GetParameterGradients() =>
        throw new NotSupportedException($"{GetType().Name} computes its gradients on the tape; it keeps no per-layer gradients.");
}