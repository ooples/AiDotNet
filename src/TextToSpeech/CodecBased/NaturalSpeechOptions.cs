namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>NaturalSpeech's training stages (Tan et al. 2022, Algorithm 1 and App. F).</summary>
public enum NaturalSpeechStage
{
    /// <summary>Mixed-phoneme pre-training of the phoneme encoder (masked phoneme and sup-phoneme prediction on text).</summary>
    PhonemePretraining,

    /// <summary>The warm-up stage (the paper's first 1k epochs): durations from monotonic alignment search train the
    /// duration predictor and drive the upsampling, no memory bank, the backward loss only.</summary>
    Warmup,

    /// <summary>The main stage: predicted durations, the memory bank (initialize it first with
    /// <see cref="NaturalSpeech{T}.InitializeMemoryBank"/>), the backward and forward losses.</summary>
    Joint,

    /// <summary>The tuning stage (the paper's last 2k epochs): only the end-to-end loss, only the durator trained.</summary>
    Tuning,
}

/// <summary>Options for <see cref="NaturalSpeech{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's (Tan et al. 2022, Table 8 and §4.1). It leaves some details unstated; those come from
/// heatz123/naturalspeech, and where that port departs from the paper its choice is an option
/// (<see cref="Heatz123"/> selects all of them).</para>
/// <para><b>For Beginners:</b> NaturalSpeech reads text aloud in one voice (it is trained on one speaker, as on LJSpeech).
/// Train it in stages: optionally pre-train the phoneme encoder on text, then warm up, initialize the memory bank, train
/// jointly and finish with tuning.</para>
/// </remarks>
[Newtonsoft.Json.JsonConverter(typeof(NaturalSpeechOptionsConverter))]
public class NaturalSpeechOptions : TtsModelOptions
{
    /// <summary>The native HiFi-GAN decoder's upsampling product (256 in the released configuration).</summary>
    public override int HopSize
    {
        get => UpsampleRates.Aggregate(1, (product, rate) => checked(product * rate));
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "HopSize must be positive.");
            if (value != HopSize) RecomposeDecoder(value, UpsampleRates.Length);
        }
    }

    /// <summary>The native waveform decoder's upsampling stages (four in the released configuration).</summary>
    public override int NumDecoderLayers
    {
        get => UpsampleRates.Length;
        set
        {
            if (value != NumDecoderLayers) RecomposeDecoder(HopSize, value);
        }
    }

    private void RecomposeDecoder(int hop, int depth)
    {
        if (depth < 1 || depth > 30 || UpsampleInitialChannels <= 0 || UpsampleInitialChannels % (1L << depth) != 0)
            throw new ArgumentOutOfRangeException(nameof(depth), "The decoder's channels must halve to a positive integer at every stage.");
        if (UpsampleRates.Length == 0 || UpsampleRates.Any(rate => rate <= 0))
            throw new ArgumentException("UpsampleRates must contain positive strides.", nameof(UpsampleRates));
        int Gcd(int a, int b)
        {
            while (b != 0) { int remainder = a % b; a = b; b = remainder; }
            return a;
        }
        var rates = new int[depth];
        var kernels = new int[depth];
        int remaining = hop;
        for (int i = 0; i < depth; i++)
        {
            rates[i] = i == depth - 1 ? remaining
                : Gcd(i < UpsampleRates.Length ? UpsampleRates[i] : 1, remaining);
            remaining /= rates[i];
            // Matching stride/kernel parity makes ConvTranspose emit exactly inputFrames * stride, including odd hops.
            kernels[i] = checked((int)(2L * rates[i] - (rates[i] & 1)));
        }
        UpsampleRates = rates;
        UpsampleKernels = kernels;
    }


    /// <summary>Creates the paper's configuration.</summary>
    public NaturalSpeechOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 192;
        NumHeads = 2;
        NumEncoderLayers = 6;
        VocabSize = 178;
        MaxTextLength = 1000;
        DropoutRate = 0.1;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>heatz123/naturalspeech's choices where they depart from the paper: blank-separated phonemes, the VITS phoneme encoder, a 2-layer,
    /// 256-filter duration predictor, its learnable upsampling, the plain KL instead of soft-DTW (with the warp scaled to
    /// 0.07·192 should soft-DTW be on), the duration loss kept after warm-up, and its loss weights (forward KL 0.001,
    /// end-to-end 0.1, duration 5).</summary>
    public static NaturalSpeechOptions Heatz123() => new()
    {
        AddBlank = true,
        PhonemeEncoder = NaturalSpeechPhonemeEncoder.Vits,
        DurationLayers = 2,
        DurationFilter = 256,
        Upsampling = NaturalSpeechUpsampling.Heatz123,
        UseSoftDtw = false,
        SoftDtwWarp = 0.07 * 192,
        DurationLossAfterWarmup = true,
        ForwardKlWeight = 0.001,
        EndToEndWeight = 0.1,
        DurationLossWeight = 5,
    };

    /// <summary>The stage training runs.</summary>
    public NaturalSpeechStage CurrentStage { get; set; } = NaturalSpeechStage.Warmup;

    // ---------------------------------------------------------------- text

    /// <summary>eSpeak NG's voice for the phonemizer (the paper's phonemizer with the espeak backend; "en-us").</summary>
    public string EspeakLanguage { get; set; } = "en-us";

    /// <summary>Whether a blank symbol separates the phonemes (VITS's add_blank, as heatz123 trains; off by default since
    /// the paper reads phonemes only).</summary>
    public bool AddBlank { get; set; }

    /// <summary>The sup-phoneme merges (pairs of ids in merge order; see <see cref="NaturalSpeech{T}.LearnSupPhonemes"/>);
    /// null reads phonemes alone and pre-trains with masked phoneme prediction only.</summary>
    public IReadOnlyList<(int Left, int Right)>? SupPhonemeMerges { get; set; }

    /// <summary>The fraction of sup-phonemes (or phonemes) masked in pre-training (15%).</summary>
    public double MaskRatio { get; set; } = 0.15;

    // ---------------------------------------------------------------- networks

    /// <summary>The phoneme encoder (the paper's FFT blocks).</summary>
    public NaturalSpeechPhonemeEncoder PhonemeEncoder { get; set; } = NaturalSpeechPhonemeEncoder.FeedForwardTransformer;

    /// <summary>The phoneme encoder's feed-forward filter (768).</summary>
    public int FilterChannels { get; set; } = 768;

    /// <summary>The phoneme encoder's convolution kernel (3).</summary>
    public int EncoderKernel { get; set; } = 3;

    /// <summary>The duration predictor's convolution layers (3; heatz123 2).</summary>
    public int DurationLayers { get; set; } = 3;

    /// <summary>The duration predictor's filters (192; heatz123 256).</summary>
    public int DurationFilter { get; set; } = 192;

    /// <summary>The duration predictor's kernel (3).</summary>
    public int DurationKernel { get; set; } = 3;

    /// <summary>The duration predictor's dropout (0.5).</summary>
    public double DurationDropout { get; set; } = 0.5;

    /// <summary>The learnable upsampling layer (the paper's).</summary>
    public NaturalSpeechUpsampling Upsampling { get; set; } = NaturalSpeechUpsampling.Paper;

    /// <summary>The upsampling convolution's kernel (3).</summary>
    public int UpsamplingKernel { get; set; } = 3;

    /// <summary>The upsampling convolution's output channels (8).</summary>
    public int UpsamplingFilter { get; set; } = 8;

    /// <summary>The attention heads q of the upsampling (4).</summary>
    public int UpsamplingQ { get; set; } = 4;

    /// <summary>The auxiliary context width p of the upsampling (2).</summary>
    public int UpsamplingP { get; set; } = 2;

    /// <summary>The flow's affine coupling layers (4).</summary>
    public int Couplings { get; set; } = 4;

    /// <summary>The WaveNet layers in each coupling (4).</summary>
    public int CouplingLayers { get; set; } = 4;

    /// <summary>The waveform decoder's upsampling rates (8, 8, 2, 2).</summary>
    public int[] UpsampleRates { get; set; } = { 8, 8, 2, 2 };

    /// <summary>The waveform decoder's transposed-convolution kernels (16, 16, 4, 4).</summary>
    public int[] UpsampleKernels { get; set; } = { 16, 16, 4, 4 };

    /// <summary>The waveform decoder's first width (256).</summary>
    public int UpsampleInitialChannels { get; set; } = 256;

    /// <summary>The waveform decoder's residual-block kernels (3, 7, 11).</summary>
    public int[] ResblockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>The waveform decoder's residual-block dilations (1, 3, 5 each).</summary>
    public int[][] ResblockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The memory bank's size (1000).</summary>
    public int MemorySize { get; set; } = 1000;

    /// <summary>The memory attention's heads (2).</summary>
    public int MemoryHeads { get; set; } = 2;

    /// <summary>The multi-period discriminator's periods (2, 3, 5, 7, 11, with VITS's scale discriminator for the paper's
    /// period 1).</summary>
    public int[] DiscriminatorPeriods { get; set; } = { 2, 3, 5, 7, 11 };

    /// <summary>A divisor on every discriminator width (1: the paper's widths).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    // ---------------------------------------------------------------- training

    /// <summary>The decoder's training segment in frames (32: heatz123's 8192 samples).</summary>
    public int SegmentFrames { get; set; } = 32;

    /// <summary>Whether the backward and forward KLs use soft dynamic time warping (App. D; heatz123's default is the plain KL).</summary>
    public bool UseSoftDtw { get; set; } = true;

    /// <summary>The soft-DTW smoothing γ (0.01).</summary>
    public double SoftDtwGamma { get; set; } = 0.01;

    /// <summary>The soft-DTW warp penalty (0.07).</summary>
    public double SoftDtwWarp { get; set; } = 0.07;

    /// <summary>The mel loss's weight (45, HiFi-GAN's, which the paper's waveform loss follows).</summary>
    public double MelLossWeight { get; set; } = 45;

    /// <summary>The backward KL's weight (1).</summary>
    public double BackwardKlWeight { get; set; } = 1;

    /// <summary>The forward KL's weight (1: Eq. 5's unweighted sum; heatz123 0.001).</summary>
    public double ForwardKlWeight { get; set; } = 1;

    /// <summary>The end-to-end GAN loss's weight (1; heatz123 0.1).</summary>
    public double EndToEndWeight { get; set; } = 1;

    /// <summary>The duration loss's weight (1; heatz123 5).</summary>
    public double DurationLossWeight { get; set; } = 1;

    /// <summary>Whether the duration loss stays after warm-up (the paper trains the predictor on alignment labels only in
    /// warm-up; heatz123 keeps it).</summary>
    public bool DurationLossAfterWarmup { get; set; }

    /// <summary>AdamW's β₁ (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>AdamW's β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW's ε (1e-9, VITS's).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>The per-epoch learning-rate decay (0.999875).</summary>
    public double LearningRateDecay { get; set; } = 0.999875;

    /// <summary>Training steps per epoch (null: no decay).</summary>
    public int? UpdatesPerEpoch { get; set; }

    /// <summary>K-means iterations of the memory bank's initialization (20).</summary>
    public int MemoryKMeansIterations { get; set; } = 20;

    // ---------------------------------------------------------------- inference

    /// <summary>The prior's noise scale at inference (1, heatz123's default).</summary>
    public double NoiseScale { get; set; } = 1.0;

    /// <summary>The durations' scale at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

}
