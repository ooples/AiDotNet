using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// CogVideoX's causal 3D VAE: video compressed 8x in height and width and 4x in time, every convolution causal in time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (Yang et al., 2024), section 2.1</item>
/// <item>Implementation: diffusers <c>AutoencoderKLCogVideoX</c>, which this reproduces (see
/// <c>tools/reference-data/cogvideox_vae_reference.py</c>); <see cref="LoadCogVideoXWeights"/> loads its state dict.</item>
/// </list></para>
/// <para><b>For Beginners:</b> The Causal 3D VAE compresses video into a much smaller latent space while preserving
/// temporal information. Causal means every convolution reads only the current and earlier frames, so the first frame
/// keeps its own latent frame and a still image is simply a one-frame clip. (The group normalizations take their
/// statistics over the whole clip, as the reference does, so a later frame still shifts earlier latents slightly.)</para>
/// <para>
/// Architecture (the paper's figure 3, diffusers' layout):
/// - Encoder: a causal 3x3x3 conv, one stage of residual blocks per width in <c>blockOutChannels</c>, each stage but
///   the last halving height and width (a stride-2 conv) and the first log2(temporalCompression) also averaging frame
///   pairs; two residual blocks; GroupNorm, SiLU, and a causal conv to the posterior's mean and log-variance.
/// - Decoder: the mirror, with one more residual block per stage, nearest-neighbour upsampling, and every
///   normalization a spatial norm modulated by the latent (MoVQ's SpatialNorm).
/// - Causal padding repeats the first frame in front, so a clip of 1 + 4k frames encodes to 1 + k latent frames.
/// </para>
/// <para>
/// An image, [batch, channels, height, width], is encoded as a one-frame clip and its latent is returned without the
/// frame axis. diffusers encodes long clips in 8-frame batches carrying a convolution cache to bound memory; this
/// model encodes the clip whole, which is the network both describe.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var vae = new Causal3DVAE&lt;float&gt;();
/// var video = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 3, 9, 64, 64 });
/// var latent = vae.Encode(video, sampleMode: false); // [1, 16, 3, 8, 8]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer", "https://arxiv.org/abs/2408.06072", Year = 2024, Authors = "Yang et al.")]
public partial class Causal3DVAE<T> : VAEModelBase<T>, IVideoVAEModel<T>
{
    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_encoder);
        RegisterParameterComponent(_decoder);
    }

    // diffusers' DiagonalGaussianDistribution clamps the log-variance to this range.
    private const double MinLogVariance = -30.0;
    private const double MaxLogVariance = 20.0;

    private readonly int _inputChannels;
    private readonly int _latentChannels;
    private readonly int[] _blockOutChannels;
    private readonly int _temporalCompression;
    private readonly int _temporalCompressLevel;
    private readonly double _latentScaleFactor;

    private readonly CausalVideoEncoder<T> _encoder;
    private readonly CausalVideoDecoder<T> _decoder;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 1 << (_blockOutChannels.Length - 1);

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public int TemporalCompression => _temporalCompression;

    /// <inheritdoc />
    /// <remarks>Revision 2: CogVideoX's causal 3D convolutional VAE replaced the dense placeholder.</remarks>
    protected override int ArchitectureRevision => 2;

    /// <inheritdoc />
    protected override string ArchitectureRevisionNote =>
        "Revision 2 is CogVideoX's causal 3D convolutional VAE; revision 1 was a stack of dense layers.";

    /// <summary>
    /// Initializes a new Causal 3D VAE. The defaults are CogVideoX's (diffusers' <c>AutoencoderKLCogVideoX</c> config).
    /// </summary>
    /// <param name="inputChannels">Channels per frame (3 for RGB).</param>
    /// <param name="latentChannels">Latent channels (16 in CogVideoX).</param>
    /// <param name="blockOutChannels">Width of each encoder stage, shallowest first (128, 256, 256, 512); each stage
    /// but the last halves height and width.</param>
    /// <param name="layersPerBlock">Residual blocks per encoder stage (3); the decoder's stages have one more.</param>
    /// <param name="normGroups">GroupNorm groups (32); every width must divide by it.</param>
    /// <param name="temporalCompression">Time compression, a power of two up to 2^(stages - 1) (4).</param>
    /// <param name="latentScaleFactor">The latent's scale for diffusion (1.15258426, CogVideoX-2b's).</param>
    /// <param name="lossFunction">Reconstruction loss for training (mean squared error by default).</param>
    /// <param name="seed">Seed for the weight initialization and the posterior sampling.</param>
    public Causal3DVAE(
        int inputChannels = 3,
        int latentChannels = 16,
        int[]? blockOutChannels = null,
        int layersPerBlock = 3,
        int normGroups = 32,
        int temporalCompression = 4,
        double latentScaleFactor = 1.15258426,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction ?? new MeanSquaredErrorLoss<T>(), seed)
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (layersPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(layersPerBlock));
        if (latentScaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(latentScaleFactor));
        var widths = blockOutChannels ?? new[] { 128, 256, 256, 512 };
        if (widths.Length == 0) throw new ArgumentException("At least one stage is required.", nameof(blockOutChannels));
        foreach (int width in widths)
        {
            if (width <= 0 || normGroups <= 0 || width % normGroups != 0)
                throw new ArgumentException($"Stage width {width} does not divide into {normGroups} groups.", nameof(blockOutChannels));
        }
        if (temporalCompression <= 0 || (temporalCompression & (temporalCompression - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(temporalCompression), "The temporal compression must be a power of two.");
        int level = 0;
        while ((1 << level) < temporalCompression) level++;
        if (level > widths.Length - 1)
            throw new ArgumentOutOfRangeException(nameof(temporalCompression),
                $"{widths.Length} stages compress time at most {1 << (widths.Length - 1)}x.");

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _blockOutChannels = (int[])widths.Clone();
        _temporalCompression = temporalCompression;
        _temporalCompressLevel = level;
        _latentScaleFactor = latentScaleFactor;

        _encoder = new CausalVideoEncoder<T>(inputChannels, latentChannels, _blockOutChannels, layersPerBlock, normGroups, 1e-6, level);
        _decoder = new CausalVideoDecoder<T>(latentChannels, inputChannels, _blockOutChannels, layersPerBlock, normGroups, 1e-6, level);
    }

    /// <inheritdoc />
    public int GetLatentFrameCount(int frames)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        // Each time-compressing stage keeps the first frame of an odd count and averages the rest in pairs.
        for (int i = 0; i < _temporalCompressLevel; i++) frames = frames % 2 == 1 ? 1 + (frames - 1) / 2 : frames / 2;
        return frames;
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
        bool image = RequireVideoOrImage(input, _inputChannels, nameof(input));
        var moments = _encoder.Forward(image ? AddFrameAxis(input) : input);
        var s = moments.Shape;
        var mean = Engine.TensorSlice(moments, new[] { 0, 0, 0, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3], s[4] });
        var logVar = Engine.TensorSlice(moments, new[] { 0, _latentChannels, 0, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3], s[4] });
        logVar = Engine.TensorClamp(logVar, NumOps.FromDouble(MinLogVariance), NumOps.FromDouble(MaxLogVariance));
        return image ? (DropFrameAxis(mean), DropFrameAxis(logVar)) : (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        bool image = RequireVideoOrImage(latent, _latentChannels, nameof(latent));
        var output = _decoder.Forward(image ? AddFrameAxis(latent) : latent);
        return image ? DropFrameAxis(output) : output;
    }

    /// <summary>
    /// Loads a diffusers <c>AutoencoderKLCogVideoX</c> state dict of this configuration.
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and this model's shape for
    /// it.</param>
    /// <returns>The names read: every weight of the model, so a state dict with names left over did not fit.</returns>
    public IReadOnlyCollection<string> LoadCogVideoXWeights(Func<string, int[], double[]> read)
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        var names = new List<string>();
        foreach (var (name, tensor) in _encoder.NamedTensors("encoder").Concat(_decoder.NamedTensors("decoder")))
        {
            var values = read(name, tensor.Shape.ToArray());
            if (values is null || values.Length != tensor.Length)
                throw new ArgumentException($"'{name}' needs {tensor.Length} values; {values?.Length ?? 0} were given.", nameof(read));
            for (int i = 0; i < values.Length; i++) tensor[i] = NumOps.FromDouble(values[i]);
            names.Add(name);
        }
        return names;
    }

    private static bool RequireVideoOrImage(Tensor<T> x, int channels, string name)
    {
        if (x is null) throw new ArgumentNullException(name);
        if (x.Rank != 4 && x.Rank != 5)
            throw new ArgumentException($"Expected [batch, channels, frames, height, width] or [batch, channels, height, width]; got rank {x.Rank}.", name);
        if (x.Shape[1] != channels)
            throw new ArgumentException($"Expected {channels} channels; got {x.Shape[1]}.", name);
        return x.Rank == 4;
    }

    private Tensor<T> AddFrameAxis(Tensor<T> x)
    {
        var s = x.Shape;
        return Engine.Reshape(x, new[] { s[0], s[1], 1, s[2], s[3] });
    }

    private Tensor<T> DropFrameAxis(Tensor<T> x)
    {
        var s = x.Shape;
        return Engine.Reshape(x, new[] { s[0], s[1], s[3], s[4] });
    }

    /// <inheritdoc />
    /// <remarks>The gradient tape differentiates this VAE (<see cref="VAEModelBase{T}.ComputeGradients"/>); it keeps no
    /// per-layer gradient buffers.</remarks>
    protected override Vector<T> GetParameterGradients() =>
        throw new NotSupportedException($"{GetType().Name} computes its gradients on the tape; it keeps no per-layer gradients.");
}
