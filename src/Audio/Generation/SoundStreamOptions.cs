using AiDotNet.Audio.Codecs;

namespace AiDotNet.Audio.Generation;

/// <summary>Where SoundStream applies its denoising FiLM conditioning (§III-F).</summary>
public enum SoundStreamFilmPosition
{
    /// <summary>No conditioning: a plain codec.</summary>
    None,

    /// <summary>On the encoder's output, before quantization (the paper's encoder-side denoising).</summary>
    Encoder,

    /// <summary>On the quantized embedding, before the decoder (the paper's decoder-side denoising).</summary>
    Decoder,
}

/// <summary>
/// Options for SoundStream ("SoundStream: An End-to-End Neural Audio Codec", Zeghidour et al. 2021). The defaults are the
/// paper's bitrate-scalable 24 kHz model: C = 32, strides (2, 4, 5, 8), 24 quantizers of 1024 codes (18 kbps) trained with
/// quantizer dropout and used at 6 kbps.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SoundStream compresses audio to a few numbers per 13 ms and rebuilds it. A single trained
/// model works at any bitrate up to 18 kbps by using more or fewer codebooks per frame, and can optionally remove
/// background noise while it compresses.</para>
/// </remarks>
public class SoundStreamOptions : NeuralAudioCodecOptions
{
    /// <summary>Creates the paper's bitrate-scalable configuration.</summary>
    public SoundStreamOptions()
    {
        SampleRate = 24000;
        Channels = 1;
        NumQuantizers = 24;
        CodebookSize = 1024;
        TargetBandwidthKbps = 6.0;
        SegmentSize = 24000;
    }

    // ---------------------------------------------------------------- encoder / decoder (§III-B, Fig. 3)

    /// <summary>Gets or sets the base channel count C (C_enc = C_dec = 32).</summary>
    public int Filters { get; set; } = 32;

    /// <summary>Gets or sets the strides in decoder order (8, 5, 4, 2); the encoder applies them reversed (2, 4, 5, 8).</summary>
    public int[] Ratios { get; set; } = [8, 5, 4, 2];

    /// <summary>Gets or sets the embedding dimension D (the paper leaves its value unstated; 128, as EnCodec's
    /// reimplementation).</summary>
    public int Dimension { get; set; } = 128;

    /// <summary>Gets or sets the residual units per block (3: dilations 1, 3, 9).</summary>
    public int ResidualLayers { get; set; } = 3;

    /// <summary>Gets or sets the dilation growth over a block's residual units (3: 1, 3, 9).</summary>
    public int DilationBase { get; set; } = 3;

    /// <summary>Gets or sets the residual unit's kernels (7, then 1).</summary>
    public int[] ResidualKernelSizes { get; set; } = [7, 1];

    /// <summary>Gets or sets whether convolutions zero-pad the past rather than reflecting the signal (false: zeros, as
    /// causal padding of the past).</summary>
    public bool ReflectPadding { get; set; }

    // ---------------------------------------------------------------- quantizer (§III-C)

    /// <summary>Gets or sets whether each training example uses n_q ~ U[1, N_q] codebooks (quantizer dropout; true).</summary>
    public bool QuantizerDropout { get; set; } = true;

    /// <summary>Gets or sets the codebooks' EMA decay (0.99).</summary>
    public double CodebookDecay { get; set; } = 0.99;

    /// <summary>Gets or sets the k-means iterations of the first-batch initialization (50).</summary>
    public int KMeansIterations { get; set; } = 50;

    /// <summary>Gets or sets the EMA assignment count below which a code is replaced by a frame of the batch (2).</summary>
    public int DeadCodeThreshold { get; set; } = 2;

    /// <summary>Gets or sets the commitment loss weight (0: Eq. 6 has no commitment term; the codebooks learn by EMA).</summary>
    public double CommitmentLossWeight { get; set; }

    // ---------------------------------------------------------------- denoising (§III-F)

    /// <summary>Gets or sets where the FiLM denoising conditioning sits (Encoder).</summary>
    public SoundStreamFilmPosition FilmPosition { get; set; } = SoundStreamFilmPosition.Encoder;

    /// <summary>Gets or sets whether inference asks for denoising (false).</summary>
    public bool Denoise { get; set; }

    // ---------------------------------------------------------------- discriminators (§III-D)

    /// <summary>Gets or sets the wave-based discriminator's resolutions (3: original, 2× and 4× down-sampled).</summary>
    public int WaveDiscriminatorScales { get; set; } = 3;

    /// <summary>Gets or sets the wave-based discriminators' grouped-convolution widths (64, 256, 1024, 1024).</summary>
    public int[] WaveDiscriminatorChannels { get; set; } = [64, 256, 1024, 1024];

    /// <summary>Gets or sets a divisor on the wave-based discriminators' widths (1: the paper's).</summary>
    public int WaveDiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the STFT discriminator's window W (1024).</summary>
    public int StftWindow { get; set; } = 1024;

    /// <summary>Gets or sets the STFT discriminator's hop H (256).</summary>
    public int StftHop { get; set; } = 256;

    /// <summary>Gets or sets the STFT discriminator's base channels C (32).</summary>
    public int StftChannels { get; set; } = 32;

    // ---------------------------------------------------------------- losses (§III-E, Eq. 6)

    /// <summary>Gets or sets λ_adv (1).</summary>
    public double AdversarialLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets λ_feat (100).</summary>
    public double FeatureLossWeight { get; set; } = 100.0;

    /// <summary>Gets or sets λ_rec (1).</summary>
    public double ReconstructionLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the reconstruction loss's scales i (window and FFT 2^i, hop 2^i / 4): 6 … 11.</summary>
    public int[] MelScales { get; set; } = [6, 7, 8, 9, 10, 11];

    /// <summary>Gets or sets the mel bins (64).</summary>
    public int MelBins { get; set; } = 64;

    // ---------------------------------------------------------------- optimizer (gap: EnCodec's reimplementation)

    /// <summary>Gets or sets Adam's learning rate (3e-4; the paper states none, EnCodec's reimplementation's).</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets Adam's β1 (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Gets or sets Adam's β2 (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;
}
