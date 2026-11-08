namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>What <see cref="NaturalSpeech3{T}"/>'s training step trains.</summary>
public enum NaturalSpeech3Stage
{
    /// <summary>FACodec (App. B.1): reconstruction, adversarial, codebook, supervision and gradient-reversal losses.</summary>
    Codec,

    /// <summary>The voice-conversion redecoder, on a frozen FACodec's codes and timbre (not in the paper; the release ships it).</summary>
    Redecoder,

    /// <summary>The factorized diffusion model (§3.3) on a frozen FACodec.</summary>
    Diffusion,
}

/// <summary>Options for <see cref="NaturalSpeech3{T}"/>.</summary>
/// <remarks>
/// <para>FACodec follows the released code and checkpoints (amphion/naturalspeech3_facodec) with the paper's training
/// (App. B.1); the factorized diffusion model follows the paper (§3.3, App. A), with MaskGCT's conventions where the paper
/// is silent. Where the release and the paper differ, the other is an option.</para>
/// <para><b>For Beginners:</b> NaturalSpeech 3 speaks text in the voice and style of a short prompt. Its codec splits speech
/// into prosody, content, acoustic detail and timbre; diffusion models generate the first three for the text, each
/// following the prompt's own, and the codec puts them back together in the prompt's timbre. Train the codec first, then
/// the diffusion model.</para>
/// </remarks>
public class NaturalSpeech3Options : TtsModelOptions
{
    /// <summary>The codec hop, which must equal both stride products.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("HopSize must equal both CodecEncoderRates and CodecDecoderRates products; paired native codec round trips and unpaired rejection tests cover this contract.")]
    public new int HopSize { get => base.HopSize; set => base.HopSize = value; }

    /// <summary>Inherited mel-output metadata; NaturalSpeech 3 decodes waveforms through FACodec.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("NaturalSpeech 3 emits codec-decoded waveforms, not mel outputs. MelBins controls the codec's spectral training losses; native round trips verify waveform geometry independently of inherited MelChannels.")]
    public new int MelChannels { get => base.MelChannels; set => base.MelChannels = value; }

    /// <summary>Creates the configuration.</summary>
    public NaturalSpeech3Options()
    {
        SampleRate = 16000;
        HopSize = 200;
        HiddenDim = 512;
        NumHeads = 8;
        NumEncoderLayers = 6;
        VocabSize = 100;
        DropoutRate = 0.1;
        LearningRate = 1e-4;
        WeightDecay = 0.01;
    }

    /// <summary>What the training step trains (Diffusion).</summary>
    public NaturalSpeech3Stage CurrentStage { get; set; } = NaturalSpeech3Stage.Diffusion;

    /// <summary>A pronunciation lexicon consulted before g2p_en (see <see cref="NaturalSpeech2Options.LexiconPath"/>).</summary>
    public string? LexiconPath { get; set; }

    // ---------------------------------------------------------------- FACodec (§3.2, App. B.1; the release)

    /// <summary>The encoder's first width (32).</summary>
    public int CodecEncoderDim { get; set; } = 32;

    /// <summary>The encoder's strides (2, 4, 5, 5: a hop of 200 at 16 kHz).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The encoder stride product must equal HopSize and the decoder stride product; native tests exercise paired hops and reject unpaired strides.")]
    public int[] CodecEncoderRates { get; set; } = { 2, 4, 5, 5 };

    /// <summary>The latent width (256).</summary>
    public int CodecDim { get; set; } = 256;

    /// <summary>The decoder's first width (1024).</summary>
    public int CodecDecoderChannels { get; set; } = 1024;

    /// <summary>The decoder's strides (5, 5, 4, 2).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The decoder stride product must equal HopSize and the encoder stride product; native tests exercise paired hops and reject unpaired strides.")]
    public int[] CodecDecoderRates { get; set; } = { 5, 5, 4, 2 };

    /// <summary>The prosody quantizers N_qp (1).</summary>
    public int ProsodyQuantizers { get; set; } = 1;

    /// <summary>The content quantizers N_qc (2).</summary>
    public int ContentQuantizers { get; set; } = 2;

    /// <summary>The acoustic-detail quantizers N_qd (3).</summary>
    public int DetailQuantizers { get; set; } = 3;

    /// <summary>The factorized codes' dimension, the information bottleneck (8).</summary>
    public int CodebookDim { get; set; } = 8;

    /// <summary>The codes per codebook (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>The phone classes the content supervision predicts (5003, the release's).</summary>
    public int CodecPhones { get; set; } = 5003;

    /// <summary>The speakers the speaker-reversal head classifies (245200, the release's).</summary>
    public int CodecSpeakers { get; set; } = 245200;

    /// <summary>The F0-reversal head on z_d (true).</summary>
    public bool DetailF0Reversal { get; set; } = true;

    /// <summary>The phone-reversal head on z_d (true).</summary>
    public bool DetailPhoneReversal { get; set; } = true;

    /// <summary>The speaker-reversal head on z_p + z_c + z_d (true).</summary>
    public bool TimbreReversal { get; set; } = true;

    /// <summary>The F0-reversal head on z_c, which the paper describes and the release leaves off (false).</summary>
    public bool ContentF0Reversal { get; set; }

    /// <summary>The phone-reversal head on z_p, which the paper describes and the release leaves off (false).</summary>
    public bool ProsodyPhoneReversal { get; set; }

    /// <summary>FACodec v2: prosody quantized from the first 20 mel bins rather than the encoder latent (false).</summary>
    public bool MelProsody { get; set; }

    /// <summary>Whether the voice-conversion redecoder is built (false).</summary>
    public bool IncludeRedecoder { get; set; }

    /// <summary>Whether the decoder reads stop(z_p) + stop(z_c) during training, as the release's forward does (true); false
    /// trains the quantizers through the reconstruction too.</summary>
    public bool DetachProsodyContent { get; set; } = true;

    /// <summary>The detail dropout's probability of masking z_d during training (0.75, the release's; the paper's p).</summary>
    public double DetailDropout { get; set; } = 0.75;

    /// <summary>The codec's training clip in samples (16000).</summary>
    public int CodecSegment { get; set; } = 16000;

    /// <summary>λ_rec (10).</summary>
    public double ReconstructionWeight { get; set; } = 10.0;

    /// <summary>λ_adv (2).</summary>
    public double AdversarialWeight { get; set; } = 2.0;

    /// <summary>λ_feat (2).</summary>
    public double FeatureWeight { get; set; } = 2.0;

    /// <summary>λ_codebook (1).</summary>
    public double CodebookWeight { get; set; } = 1.0;

    /// <summary>λ_commit (0.25).</summary>
    public double CommitmentWeight { get; set; } = 0.25;

    /// <summary>λ_f0 (5).</summary>
    public double F0Weight { get; set; } = 5.0;

    /// <summary>λ_ph (5).</summary>
    public double PhoneWeight { get; set; } = 5.0;

    /// <summary>λ_gr-f0 (5).</summary>
    public double F0ReversalWeight { get; set; } = 5.0;

    /// <summary>λ_gr-ph (5).</summary>
    public double PhoneReversalWeight { get; set; } = 5.0;

    /// <summary>λ_gr-spk (1).</summary>
    public double SpeakerReversalWeight { get; set; } = 1.0;

    /// <summary>The multi-scale mel loss's windows (DAC's 32 ... 2048).</summary>
    public int[] MelWindows { get; set; } = { 32, 64, 128, 256, 512, 1024, 2048 };

    /// <summary>The multi-scale mel loss's mel bins (DAC's 5 ... 320).</summary>
    public int[] MelBins { get; set; } = { 5, 10, 20, 40, 80, 160, 320 };

    /// <summary>The multi-period discriminator's periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = { 2, 3, 5, 7, 11 };

    /// <summary>The multi-band multi-scale STFT discriminator's windows (2048, 1024, 512).</summary>
    public int[] DiscriminatorWindows { get; set; } = { 2048, 1024, 512 };

    /// <summary>A divisor on every discriminator width (1).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>The codec's Adam learning rate (2e-4).</summary>
    public double CodecLearningRate { get; set; } = 2e-4;

    /// <summary>The codec's Adam β1 (0.5).</summary>
    public double CodecBeta1 { get; set; } = 0.5;

    /// <summary>The codec's Adam β2 (0.9).</summary>
    public double CodecBeta2 { get; set; } = 0.9;

    // ---------------------------------------------------------------- factorized diffusion (§3.3, App. A)

    /// <summary>The phoneme encoder's feed-forward filter (2048).</summary>
    public int FilterChannels { get; set; } = 2048;

    /// <summary>The phoneme encoder's feed-forward kernel (9).</summary>
    public int EncoderKernel { get; set; } = 9;

    /// <summary>The diffusion Transformers' width (1024).</summary>
    public int DiffusionDim { get; set; } = 1024;

    /// <summary>Their attention heads (8).</summary>
    public int DiffusionHeads { get; set; } = 8;

    /// <summary>Their feed-forward filter (2048).</summary>
    public int DiffusionFilter { get; set; } = 2048;

    /// <summary>Their feed-forward kernel (3).</summary>
    public int DiffusionKernel { get; set; } = 3;

    /// <summary>The shared prosody/content/detail Transformer's layers (12).</summary>
    public int FrameDiffusionLayers { get; set; } = 12;

    /// <summary>The duration (and phoneme-level prosody) Transformer's layers (6).</summary>
    public int PhonemeDiffusionLayers { get; set; } = 6;

    /// <summary>The longest duration token in frames (63: durations are 64 classes, longer ones clamped). The paper does not
    /// say how durations are tokenized.</summary>
    public int MaxDuration { get; set; } = 63;

    /// <summary>
    /// The phoneme-level prosody diffusion the paper's inference lists but never defines (false). When on, it generates each
    /// phoneme's most frequent prosody code first, which then conditions the duration and prosody diffusions.
    /// </summary>
    public bool PhonemeProsody { get; set; }

    /// <summary>p_cfg, the prompt's drop probability in training (0.15).</summary>
    public double PromptDropProbability { get; set; } = 0.15;

    /// <summary>The iterations of each diffusion (4).</summary>
    public int SamplingSteps { get; set; } = 4;

    /// <summary>The top-k filter (20).</summary>
    public int TopK { get; set; } = 20;

    /// <summary>The sampling temperature at the first iteration, annealed to 0 (1.5).</summary>
    public double Temperature { get; set; } = 1.5;

    /// <summary>The classifier-free guidance scale α for prosody, content and detail (1.0; durations use none).</summary>
    public double GuidanceScale { get; set; } = 1.0;

    /// <summary>The diffusion's AdamW β2 (0.98; β1 0.9).</summary>
    public double DiffusionBeta2 { get; set; } = 0.98;

    /// <summary>The diffusion's warm-up steps before the inverse square root (5000).</summary>
    public int WarmupSteps { get; set; } = 5000;
}
