using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Stable Video Diffusion's VAE: the latent-diffusion KL encoder applied to each frame, and a decoder that mixes
/// information across frames so the decoded clip is temporally consistent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (Blattmann et al.,
/// 2023), Sec. 3 and Appendix D (the temporal decoder)</item>
/// <item>Implementation: diffusers <c>AutoencoderKLTemporalDecoder</c> (the
/// <c>stabilityai/stable-video-diffusion-img2vid</c> <c>vae</c> folder), which this reproduces
/// (<c>tools/reference-data/svd_vae_reference.py</c>); <see cref="LoadDiffusersWeights"/> loads its state dict.</item>
/// </list></para>
/// <para><b>For Beginners:</b> A video VAE compresses each frame to a small latent and back. Decoding every frame on
/// its own makes the result flicker, so this VAE's decoder also looks across neighbouring frames: each block blends a
/// per-frame result with one that mixes the frames, and a final convolution over time smooths the clip.</para>
/// <para>
/// Architecture (diffusers' <c>AutoencoderKLTemporalDecoder</c>): the encoder and its 1x1 quant conv are the KL
/// autoencoder's, run on every frame. The decoder is a 3x3 conv in; a middle block of spatio-temporal residual blocks
/// around single-head self-attention; one stage per width from the deepest, of <c>layersPerBlock + 1</c>
/// spatio-temporal residual blocks and, but the last, a nearest 2x upsample; GroupNorm, SiLU and a 3x3 conv out; and a
/// (3, 1, 1) convolution over time. Each spatio-temporal block is a 2D residual block on every frame, a residual block
/// of (3, 1, 1) convolutions across the frames, and a learned blend of the two. There is no post-quant conv. A clip is
/// [batch, channels, frames, height, width]; an image, [batch, channels, height, width], is a one-frame clip.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var vae = new TemporalInterpolationVAE&lt;float&gt;();
/// var latent = vae.Encode(clip, sampleMode: false); // [1, 4, 14, 72, 128] for a [1, 3, 14, 576, 1024] clip
/// var decoded = vae.Decode(latent);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets", "https://arxiv.org/abs/2311.15127", Year = 2023, Authors = "Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach")]
public partial class TemporalInterpolationVAE<T> : VAEModelBase<T>, IVideoVAEModel<T>
{
    /// <summary>The scaling factor of Stable Video Diffusion's VAE.</summary>
    public const double SvdLatentScale = 0.18215;

    // diffusers' DiagonalGaussianDistribution clamps the log-variance to this range.
    private const double MinLogVariance = -30.0;
    private const double MaxLogVariance = 20.0;

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_encoder);
        RegisterParameterComponent(_quantConv);
        RegisterParameterComponent(_decoder);
    }

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int[] _blockOutChannels;
    private readonly double _latentScaleFactor;

    private readonly KlEncoder<T> _encoder;
    private readonly KlConv2d<T> _quantConv;
    private readonly KlTemporalDecoder<T> _decoder;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 1 << (_blockOutChannels.Length - 1);

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    /// <remarks>The encoder works on each frame, so the latent has as many frames as the clip.</remarks>
    public int TemporalCompression => 1;

    /// <inheritdoc />
    public int GetLatentFrameCount(int frames)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        return frames;
    }

    /// <inheritdoc />
    /// <remarks>Revision 2: Stable Video Diffusion's temporal-decoder VAE replaced the dense placeholder.</remarks>
    protected override int ArchitectureRevision => 2;

    /// <inheritdoc />
    protected override string ArchitectureRevisionNote =>
        "Revision 2 is Stable Video Diffusion's VAE (diffusers AutoencoderKLTemporalDecoder); revision 1 was dense layers over the channel axis.";

    /// <summary>
    /// Initializes the VAE; the defaults are Stable Video Diffusion's (diffusers <c>AutoencoderKLTemporalDecoder</c>,
    /// <c>stabilityai/stable-video-diffusion-img2vid</c>).
    /// </summary>
    /// <param name="inputChannels">Frame channels (3).</param>
    /// <param name="latentChannels">Latent channels (4).</param>
    /// <param name="blockOutChannels">Width of each encoder stage, shallowest first (128, 256, 512, 512); each but the
    /// last halves height and width.</param>
    /// <param name="layersPerBlock">Residual blocks per encoder stage (2); the decoder's stages have one more.</param>
    /// <param name="normGroups">GroupNorm groups (32, which diffusers fixes); every width must divide by it.</param>
    /// <param name="latentScaleFactor">The latent's scale for diffusion (<see cref="SvdLatentScale"/>).</param>
    /// <param name="lossFunction">Reconstruction loss for training (mean squared error by default).</param>
    /// <param name="seed">Seed for the weight initialization and the posterior sampling.</param>
    public TemporalInterpolationVAE(
        int inputChannels = 3,
        int latentChannels = 4,
        int[]? blockOutChannels = null,
        int layersPerBlock = 2,
        int normGroups = 32,
        double latentScaleFactor = SvdLatentScale,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction ?? new MeanSquaredErrorLoss<T>(), seed)
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (layersPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(layersPerBlock));
        if (latentScaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(latentScaleFactor));
        var widths = blockOutChannels ?? new[] { 128, 256, 512, 512 };
        if (widths.Length == 0) throw new ArgumentException("At least one stage is required.", nameof(blockOutChannels));
        foreach (int width in widths)
        {
            if (width <= 0 || normGroups <= 0 || width % normGroups != 0)
                throw new ArgumentException($"Stage width {width} does not divide into {normGroups} groups.", nameof(blockOutChannels));
        }

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _blockOutChannels = (int[])widths.Clone();
        _latentScaleFactor = latentScaleFactor;

        _encoder = new KlEncoder<T>(inputChannels, latentChannels, _blockOutChannels, layersPerBlock, normGroups);
        _quantConv = new KlConv2d<T>(2 * latentChannels, 2 * latentChannels, 1);
        _decoder = new KlTemporalDecoder<T>(latentChannels, inputChannels, _blockOutChannels, layersPerBlock, normGroups);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(input);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    /// <param name="input">A clip, [batch, channels, frames, height, width], or an image, [batch, channels, height,
    /// width].</param>
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> input)
    {
        bool image = RequireVideoOrImage(input, _inputChannels, nameof(input));
        int frames = image ? 1 : input.Shape[2];
        var moments = _quantConv.Forward(_encoder.Forward(image ? input : FramesToBatch(input)));
        var s = moments.Shape;
        var mean = Engine.TensorSlice(moments, new[] { 0, 0, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] });
        var logVar = Engine.TensorClamp(
            Engine.TensorSlice(moments, new[] { 0, _latentChannels, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] }),
            NumOps.FromDouble(MinLogVariance), NumOps.FromDouble(MaxLogVariance));
        return image ? (mean, logVar) : (BatchToFrames(mean, frames), BatchToFrames(logVar, frames));
    }

    /// <inheritdoc />
    /// <param name="latent">A clip latent, [batch, latent channels, frames, height, width], or an image latent, [batch,
    /// latent channels, height, width].</param>
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        bool image = RequireVideoOrImage(latent, _latentChannels, nameof(latent));
        int frames = image ? 1 : latent.Shape[2];
        var decoded = _decoder.Forward(image ? latent : FramesToBatch(latent), frames);
        return image ? decoded : BatchToFrames(decoded, frames);
    }

    /// <summary>
    /// Loads a diffusers <c>AutoencoderKLTemporalDecoder</c> state dict of this configuration (a Stable Video Diffusion
    /// <c>vae</c> folder).
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and this model's shape.</param>
    /// <returns>The names read: every weight of the model.</returns>
    public IReadOnlyCollection<string> LoadDiffusersWeights(Func<string, int[], double[]> read) =>
        KlModule<T>.Load(
            _encoder.NamedTensors("encoder")
                .Concat(_quantConv.NamedTensors("quant_conv"))
                .Concat(_decoder.NamedTensors("decoder")),
            read);

    private static bool RequireVideoOrImage(Tensor<T> x, int channels, string name)
    {
        if (x is null) throw new ArgumentNullException(name);
        if (x.Rank != 4 && x.Rank != 5)
            throw new ArgumentException($"Expected [batch, channels, frames, height, width] or [batch, channels, height, width]; got rank {x.Rank}.", name);
        if (x.Shape[1] != channels)
            throw new ArgumentException($"Expected {channels} channels; got {x.Shape[1]}.", name);
        return x.Rank == 4;
    }

    // [B, C, F, H, W] -> [B * F, C, H, W], frames of one clip adjacent, as diffusers lays a clip out.
    private Tensor<T> FramesToBatch(Tensor<T> x)
    {
        var s = x.Shape;
        return Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 2, 1, 3, 4 }), new[] { s[0] * s[2], s[1], s[3], s[4] });
    }

    // [B * F, C, H, W] -> [B, C, F, H, W].
    private Tensor<T> BatchToFrames(Tensor<T> x, int frames)
    {
        var s = x.Shape;
        return Engine.TensorPermute(Engine.Reshape(x, new[] { s[0] / frames, frames, s[1], s[2], s[3] }), new[] { 0, 2, 1, 3, 4 });
    }

    /// <inheritdoc />
    /// <remarks>The gradient tape differentiates this VAE; it keeps no per-layer gradient buffers.</remarks>
    protected override Vector<T> GetParameterGradients() =>
        throw new NotSupportedException($"{GetType().Name} computes its gradients on the tape; it keeps no per-layer gradients.");
}