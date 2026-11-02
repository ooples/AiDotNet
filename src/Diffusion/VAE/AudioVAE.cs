using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// AudioLDM's variational autoencoder: the latent-diffusion KL autoencoder applied to a log-mel spectrogram as a
/// one-channel image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (Liu et al., ICML 2023), Sec. 3.2</item>
/// <item>Implementation: diffusers <c>AutoencoderKL</c> as the <c>cvssp/audioldm-s-full-v2</c> and <c>cvssp/audioldm2</c>
/// pipelines configure it, which this reproduces (<c>tools/reference-data/audioldm_vae_reference.py</c>);
/// <see cref="LoadDiffusersWeights"/> loads its state dict.</item>
/// </list></para>
/// <para><b>For Beginners:</b> Raw audio is far too long for a diffusion model, so AudioLDM works on a compressed
/// picture of it. The audio becomes a mel spectrogram (time on one axis, pitch bands on the other), and this VAE squeezes
/// that picture 4x along both axes into 8 latent channels. The diffusion model works there, and the decoder turns its
/// result back into a spectrogram.</para>
/// <para>
/// Architecture (diffusers' <c>AutoencoderKL</c>): a 3x3 conv in; one stage of residual blocks per width in
/// <c>blockOutChannels</c>, each stage but the last halving both axes; a middle block of residual block, single-head
/// self-attention and residual block; GroupNorm, SiLU and a 3x3 conv to the posterior's mean and log-variance; a 1x1
/// quant conv. The decoder mirrors it (a 1x1 post-quant conv first, one more residual block per stage, nearest 2x
/// upsampling). A spectrogram is [batch, 1, time frames, mel bins], as AudioLDM lays it out.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var vae = new AudioVAE&lt;float&gt;();
/// var waveform = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 16000 }); // one second at 16 kHz
/// var mel = vae.AudioToMelSpectrogram(waveform);   // [1, 1, frames, 64]
/// var latent = vae.Encode(mel, sampleMode: false); // [1, 8, frames / 4, 16]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AudioLDM: Text-to-Audio Generation with Latent Diffusion Models", "https://arxiv.org/abs/2301.12503", Year = 2023, Authors = "Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, Mark D. Plumbley")]
public partial class AudioVAE<T> : VAEModelBase<T>
{
    /// <summary>The scaling factor of AudioLDM's VAE (<c>cvssp/audioldm-s-full-v2</c>).</summary>
    public const double AudioLdmLatentScale = 0.9392935633659363;

    /// <summary>The scaling factor of AudioLDM 2's VAE (<c>cvssp/audioldm2</c>).</summary>
    public const double AudioLdm2LatentScale = 0.4110932946205139;

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
    private readonly int _melBins;
    private readonly double _latentScaleFactor;

    private readonly KlEncoder<T> _encoder;
    private readonly KlConv2d<T> _quantConv;
    private readonly KlConv2d<T> _postQuantConv;
    private readonly KlDecoder<T> _decoder;

    private MelSpectrogram<T>? _melSpectrogramProcessor;
    private GriffinLim<T>? _griffinLimProcessor;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => 1 << (_blockOutChannels.Length - 1);

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <summary>Gets the number of mel bins the spectrogram helpers produce and read.</summary>
    public int MelBins => _melBins;

    /// <inheritdoc />
    /// <remarks>Revision 2: AudioLDM's 2D KL autoencoder replaced the 1D placeholder.</remarks>
    protected override int ArchitectureRevision => 2;

    /// <inheritdoc />
    protected override string ArchitectureRevisionNote =>
        "Revision 2 is AudioLDM's 2D KL autoencoder over [batch, 1, frames, mel bins]; revision 1 was a 1D network over mel bins as channels.";

    /// <summary>
    /// Initializes an AudioVAE in AudioLDM's configuration (diffusers <c>AutoencoderKL</c>, <c>cvssp/audioldm-s-full-v2</c>).
    /// </summary>
    /// <param name="inputChannels">Spectrogram channels (1: a log-mel spectrogram is one image channel).</param>
    /// <param name="latentChannels">Latent channels (8).</param>
    /// <param name="blockOutChannels">Width of each encoder stage, shallowest first (128, 256, 512); each but the last
    /// halves both axes.</param>
    /// <param name="layersPerBlock">Residual blocks per encoder stage (2); the decoder's stages have one more.</param>
    /// <param name="normGroups">GroupNorm groups (32); every width must divide by it.</param>
    /// <param name="melBins">Mel bins of the spectrogram helpers (64, AudioLDM's).</param>
    /// <param name="latentScaleFactor">The latent's scale for diffusion (<see cref="AudioLdmLatentScale"/>; AudioLDM 2
    /// uses <see cref="AudioLdm2LatentScale"/>).</param>
    /// <param name="lossFunction">Reconstruction loss for training (mean squared error by default).</param>
    /// <param name="seed">Seed for the weight initialization and the posterior sampling.</param>
    public AudioVAE(
        int inputChannels = 1,
        int latentChannels = 8,
        int[]? blockOutChannels = null,
        int layersPerBlock = 2,
        int normGroups = 32,
        int melBins = 64,
        double latentScaleFactor = AudioLdmLatentScale,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction ?? new MeanSquaredErrorLoss<T>(), seed)
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (layersPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(layersPerBlock));
        if (melBins <= 0) throw new ArgumentOutOfRangeException(nameof(melBins));
        if (latentScaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(latentScaleFactor));
        var widths = blockOutChannels ?? new[] { 128, 256, 512 };
        if (widths.Length == 0) throw new ArgumentException("At least one stage is required.", nameof(blockOutChannels));
        foreach (int width in widths)
        {
            if (width <= 0 || normGroups <= 0 || width % normGroups != 0)
                throw new ArgumentException($"Stage width {width} does not divide into {normGroups} groups.", nameof(blockOutChannels));
        }

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _blockOutChannels = (int[])widths.Clone();
        _melBins = melBins;
        _latentScaleFactor = latentScaleFactor;

        _encoder = new KlEncoder<T>(inputChannels, latentChannels, _blockOutChannels, layersPerBlock, normGroups);
        _quantConv = new KlConv2d<T>(2 * latentChannels, 2 * latentChannels, 1);
        _postQuantConv = new KlConv2d<T>(latentChannels, latentChannels, 1);
        _decoder = new KlDecoder<T>(latentChannels, inputChannels, _blockOutChannels, layersPerBlock, normGroups);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(input);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    /// <param name="input">A spectrogram, [batch, channels, time frames, mel bins].</param>
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> input)
    {
        RequireImage(input, _inputChannels, nameof(input));
        var moments = _quantConv.Forward(_encoder.Forward(input));
        var s = moments.Shape;
        var mean = Engine.TensorSlice(moments, new[] { 0, 0, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] });
        var logVar = Engine.TensorSlice(moments, new[] { 0, _latentChannels, 0, 0 }, new[] { s[0], _latentChannels, s[2], s[3] });
        return (mean, Engine.TensorClamp(logVar, NumOps.FromDouble(MinLogVariance), NumOps.FromDouble(MaxLogVariance)));
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        RequireImage(latent, _latentChannels, nameof(latent));
        return _decoder.Forward(_postQuantConv.Forward(latent));
    }

    /// <summary>
    /// Loads a diffusers <c>AutoencoderKL</c> state dict of this configuration (an AudioLDM <c>vae</c> folder).
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and this model's shape.</param>
    /// <returns>The names read: every weight of the model.</returns>
    public IReadOnlyCollection<string> LoadDiffusersWeights(Func<string, int[], double[]> read)
    {
        return KlModule<T>.Load(
            _encoder.NamedTensors("encoder")
                .Concat(_quantConv.NamedTensors("quant_conv"))
                .Concat(_postQuantConv.NamedTensors("post_quant_conv"))
                .Concat(_decoder.NamedTensors("decoder")),
            read);
    }

    private static void RequireImage(Tensor<T> x, int channels, string name)
    {
        if (x is null) throw new ArgumentNullException(name);
        if (x.Rank != 4)
            throw new ArgumentException($"Expected [batch, channels, time frames, mel bins]; got rank {x.Rank}.", name);
        if (x.Shape[1] != channels)
            throw new ArgumentException($"Expected {channels} channel(s); got {x.Shape[1]}.", name);
    }

    /// <summary>
    /// Converts a waveform to the log-mel spectrogram this VAE reads, [batch, 1, time frames, mel bins].
    /// </summary>
    /// <param name="waveform">[samples] or [batch, samples].</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="hopLength">STFT hop length.</param>
    /// <param name="fftSize">STFT size.</param>
    /// <returns>The log-mel spectrogram, [batch, 1, time frames, <see cref="MelBins"/>].</returns>
    public virtual Tensor<T> AudioToMelSpectrogram(
        Tensor<T> waveform,
        int sampleRate = 16000,
        int hopLength = 512,
        int fftSize = 2048)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        _melSpectrogramProcessor ??= new MelSpectrogram<T>(
            sampleRate: sampleRate, nFft: fftSize, hopLength: hopLength, nMels: _melBins,
            fMin: 0, fMax: sampleRate / 2.0, logMel: true);

        var shape = waveform._shape;
        int batch = shape.Length == 1 ? 1 : shape[0];
        int samples = shape.Length == 1 ? shape[0] : shape[1];
        Tensor<T>? result = null;
        for (int b = 0; b < batch; b++)
        {
            var single = shape.Length == 1 ? waveform : Engine.Reshape(
                Engine.TensorSlice(waveform, new[] { b, 0 }, new[] { 1, samples }), new[] { samples });
            // The processor returns [time frames, mel bins], AudioLDM's spectrogram layout.
            var mel = _melSpectrogramProcessor.Forward(single);
            int frames = mel.Shape[0];
            result ??= new Tensor<T>(new[] { batch, 1, frames, _melBins });
            var source = mel.AsSpan();
            var target = result.AsWritableSpan();
            int offset = b * frames * _melBins;
            for (int i = 0; i < frames * _melBins && i < source.Length; i++) target[offset + i] = source[i];
        }
        return result ?? new Tensor<T>(new[] { 0, 1, 0, _melBins });
    }

    /// <summary>
    /// Converts a log-mel spectrogram, [batch, 1, time frames, mel bins], back to a waveform by inverting the mel
    /// filterbank and reconstructing the phase with Griffin-Lim (AudioLDM itself uses a HiFi-GAN vocoder).
    /// </summary>
    /// <param name="melSpectrogram">[batch, 1, time frames, mel bins].</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="hopLength">STFT hop length.</param>
    /// <returns>The waveform, [batch, samples].</returns>
    public virtual Tensor<T> MelSpectrogramToAudio(
        Tensor<T> melSpectrogram,
        int sampleRate = 16000,
        int hopLength = 512)
    {
        RequireImage(melSpectrogram, 1, nameof(melSpectrogram));
        const int fftSize = 2048;
        _melSpectrogramProcessor ??= new MelSpectrogram<T>(
            sampleRate: sampleRate, nFft: fftSize, hopLength: hopLength, nMels: _melBins,
            fMin: 0, fMax: sampleRate / 2.0, logMel: true);
        _griffinLimProcessor ??= new GriffinLim<T>(nFft: fftSize, hopLength: hopLength, iterations: 32, momentum: 0.99);

        var shape = melSpectrogram._shape;
        int batch = shape[0], frames = shape[2], bins = shape[3];
        int samples = frames * hopLength;
        var result = new Tensor<T>(new[] { batch, samples });
        var target = result.AsWritableSpan();
        for (int b = 0; b < batch; b++)
        {
            // [time frames, mel bins] for one item.
            var single = Engine.Reshape(
                Engine.TensorSlice(melSpectrogram, new[] { b, 0, 0, 0 }, new[] { 1, 1, frames, bins }), new[] { frames, bins });
            var magnitude = _melSpectrogramProcessor.InvertMelToMagnitude(single);
            var audio = _griffinLimProcessor.Reconstruct(magnitude, samples).AsSpan();
            for (int s = 0; s < samples && s < audio.Length; s++) target[b * samples + s] = audio[s];
        }
        return result;
    }

    /// <inheritdoc />
    /// <remarks>The gradient tape differentiates this VAE; it keeps no per-layer gradient buffers.</remarks>
    protected override Vector<T> GetParameterGradients() =>
        throw new NotSupportedException($"{GetType().Name} computes its gradients on the tape; it keeps no per-layer gradients.");
}
