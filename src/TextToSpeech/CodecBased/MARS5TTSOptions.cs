using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of MARS5's networks <see cref="MARS5TTS{T}"/> trains.</summary>
public enum Mars5Stage
{
    /// <summary>The autoregressive model: next-token cross-entropy over the text and the first codebook's speech tokens.</summary>
    Autoregressive,

    /// <summary>The non-autoregressive model: the multinomial diffusion loss over all eight codebooks, the first held to
    /// the ground truth.</summary>
    NonAutoregressive,
}

/// <summary>Options for <see cref="MARS5TTS{T}"/>.</summary>
/// <remarks>
/// <para>MARS5 (Camb.ai, 2024) has no paper; the release (Camb-ai/MARS5-TTS) defines the model and its inference. The
/// defaults are the release's (<c>inference.py</c>'s model sizes and <c>InferenceConfig</c>). The release ships no training
/// code: the losses follow its architecture notes and its diffusion code, with every unstated choice an option here.</para>
/// <para>The text and speech tokenizers are stored in the released AR checkpoint's metadata; build the options with
/// <see cref="FromCheckpoint"/> to load them. Without it, byte-level tokenizers with no merges stand in, so an untrained
/// model works end to end.</para>
/// <para><b>For Beginners:</b> MARS5 copies a voice from about six seconds of recording. Give it the recording's transcript
/// too ("deep clone") for a closer copy.</para>
/// </remarks>
public class MARS5TTSOptions : TtsModelOptions
{
    /// <summary>Creates the release's configuration.</summary>
    public MARS5TTSOptions()
    {
        SampleRate = 24000;
        HopSize = 320;
        HiddenDim = 1536;
        NumHeads = 24;
        MaxTextLength = 2000;
        DropoutRate = 0.0;
        LearningRate = 4e-4;
        WeightDecay = 0.1;
        Codec = CreateReleaseCodec();
    }

    /// <summary>EnCodec's 24 kHz model at 6 kbps (eight codebooks), as the release encodes with it.</summary>
    public static EnCodecOptions CreateReleaseCodec()
    {
        var codec = EnCodecOptions.OfficialCheckpoint24kHz();
        codec.TargetBandwidthKbps = 6.0;
        return codec;
    }

    /// <summary>The release's configuration with the tokenizers stored in an AR checkpoint's metadata
    /// (<c>mars5_ar.safetensors</c>).</summary>
    public static MARS5TTSOptions FromCheckpoint(string arCheckpointPath)
    {
        var metadata = Mars5Checkpoint.Metadata(arCheckpointPath);
        return new MARS5TTSOptions
        {
            TextTokenizerModel = metadata.TryGetValue("texttok.model", out var text) ? text : throw new InvalidDataException("The checkpoint stores no texttok.model."),
            SpeechTokenizerModel = metadata.TryGetValue("speechtok.model", out var speech) ? speech : throw new InvalidDataException("The checkpoint stores no speechtok.model."),
        };
    }

    /// <summary>The codec (EnCodec 24 kHz at 6 kbps).</summary>
    public EnCodecOptions Codec { get; set; }

    /// <summary>The codebooks the models read (8).</summary>
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Codes per codebook (1024; the NAR adds one pad class).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>The text tokenizer's minbpe model (the AR checkpoint's <c>texttok.model</c>); null for a byte-level
    /// tokenizer with no merges and the two text specials.</summary>
    public string? TextTokenizerModel { get; set; }

    /// <summary>The speech tokenizer's minbpe model (<c>speechtok.model</c>); null for codes with no merges and the
    /// end-of-speech special.</summary>
    public string? SpeechTokenizerModel { get; set; }

    // ---------------------------------------------------------------- AR

    /// <summary>The AR decoder's layers (26); its width is <see cref="TtsModelOptions.HiddenDim"/> (1536) and its heads
    /// <see cref="TtsModelOptions.NumHeads"/> (24).</summary>
    public int ArLayers { get; set; } = 26;

    /// <summary>The AR decoder's SwiGLU width (3584 = 1536 · 7/3).</summary>
    public int ArFeedForward { get; set; } = 3584;

    /// <summary>The AR decoder's head width (64: Mistral's default, which the release never overrides).</summary>
    public int ArHeadDim { get; set; } = 64;

    /// <summary>The AR speaker encoder's layers (2).</summary>
    public int ArSpeakerLayers { get; set; } = 2;

    /// <summary>The AR decoder's attention window (3000).</summary>
    public int SlidingWindow { get; set; } = 3000;

    // ---------------------------------------------------------------- NAR

    /// <summary>The NAR's width (1024).</summary>
    public int NarDim { get; set; } = 1024;

    /// <summary>The NAR's heads (16).</summary>
    public int NarHeads { get; set; } = 16;

    /// <summary>The NAR's encoder layers (8).</summary>
    public int NarEncoderLayers { get; set; } = 8;

    /// <summary>The NAR's decoder layers (16).</summary>
    public int NarDecoderLayers { get; set; } = 16;

    /// <summary>The NAR speaker encoder's layers (3).</summary>
    public int NarSpeakerLayers { get; set; } = 3;

    /// <summary>The timestep embedding's width (1024).</summary>
    public int TimeDim { get; set; } = 1024;

    /// <summary>The probability of dropping the NAR's conditioning in training (<c>p_cond_drop</c>, 0.1).</summary>
    public double ConditionDropProbability { get; set; } = 0.1;

    /// <summary>Diffusion steps (200).</summary>
    public int DiffusionSteps { get; set; } = 200;

    // ---------------------------------------------------------------- Vocos

    /// <summary>Vocos's width (384, vocos-encodec-24khz).</summary>
    public int DecoderDim { get; set; } = 384;

    /// <summary>Vocos's feed-forward width (1152).</summary>
    public int DecoderIntermediateDim { get; set; } = 1152;

    /// <summary>Vocos's layers (8).</summary>
    public int DecoderLayers { get; set; } = 8;

    /// <summary>The Vocos bandwidth class decoding uses (1, the 3 kbps class: the release's smoothing).</summary>
    public int VocosBandwidthId { get; set; } = 1;

    // ---------------------------------------------------------------- inference

    /// <summary>Whether to deep clone (continue the reference's transcript and codes) when the voice has a transcript.</summary>
    public bool DeepClone { get; set; } = true;

    /// <summary>The AR temperature (0.7).</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>The AR top-k (200).</summary>
    public int TopK { get; set; } = 200;

    /// <summary>The AR top-p (0.2).</summary>
    public double TopP { get; set; } = 0.2;

    /// <summary>The AR typical-p mass (1: off).</summary>
    public double TypicalP { get; set; } = 1.0;

    /// <summary>The frequency penalty (3).</summary>
    public double FrequencyPenalty { get; set; } = 3;

    /// <summary>The presence penalty (0.4).</summary>
    public double PresencePenalty { get; set; } = 0.4;

    /// <summary>Tokens the penalties look back over (80, about 5 s).</summary>
    public int PenaltyWindow { get; set; } = 80;

    /// <summary>The end-of-speech penalty's decay (0.5).</summary>
    public double EosPenaltyDecay { get; set; } = 0.5;

    /// <summary>The end-of-speech penalty's factor (1).</summary>
    public double EosPenaltyFactor { get; set; } = 1;

    /// <summary>The expected generation length as a multiple of the text's characters (1).</summary>
    public double EosEstimatedLengthFactor { get; set; } = 1.0;

    /// <summary>The longest AR sequence, prompt included (2000).</summary>
    public int MaxGenerationLength { get; set; } = 2000;

    /// <summary>The NAR's x₀ temperature (0.7).</summary>
    public double X0Temperature { get; set; } = 0.7;

    /// <summary>The NAR's classifier-free guidance weight (3).</summary>
    public double NarGuidance { get; set; } = 3;

    /// <summary>The final diffusion steps at which the NAR may revise the first codebook (20).</summary>
    public int Q0OverrideSteps { get; set; } = 20;

    /// <summary>RePaint's jump length (1: no resampling).</summary>
    public int JumpLength { get; set; } = 1;

    /// <summary>RePaint's resamples per jump (1).</summary>
    public int JumpSamples { get; set; } = 1;

    /// <summary>Silence trimmed from the output's ends, in dB below its loudest frame (27).</summary>
    public double TrimDb { get; set; } = 27;

    /// <summary>Silence prepended to the reference, in seconds (0).</summary>
    public double ReferencePadSeconds { get; set; }

    /// <summary>The seed of sampling and diffusion.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- training

    /// <summary>The network <see cref="MARS5TTS{T}"/>'s training updates.</summary>
    public Mars5Stage CurrentStage { get; set; } = Mars5Stage.Autoregressive;

    /// <summary>The AR loss's weight on the text tokens' cross-entropy (0.01, Tortoise's and XTTS's text weight; the
    /// release says only "a smaller weight").</summary>
    public double TextLossWeight { get; set; } = 0.01;

    /// <summary>Whether the NAR loss adds the KL prior in place of the t = 0 decoder term (<c>compute_Lt</c>'s default).</summary>
    public bool IncludeKlPrior { get; set; } = true;

    /// <summary>The AR's peak learning rate (the checkpoint's 4e-4).</summary>
    public double ArLearningRate { get; set; } = 4e-4;

    /// <summary>The AR's final learning rate (2e-5).</summary>
    public double ArEndLearningRate { get; set; } = 2e-5;

    /// <summary>The step the AR's rate reaches its end (700,000).</summary>
    public int ArEndSteps { get; set; } = 700_000;

    /// <summary>The AR's weight decay (0.1).</summary>
    public double ArWeightDecay { get; set; } = 0.1;

    /// <summary>The AR's AdamW β₁ (PyTorch's 0.9; the AR config names none).</summary>
    public double ArBeta1 { get; set; } = 0.9;

    /// <summary>The AR's AdamW β₂ (PyTorch's 0.999).</summary>
    public double ArBeta2 { get; set; } = 0.999;

    /// <summary>The NAR's peak learning rate (1e-4).</summary>
    public double NarLearningRate { get; set; } = 1e-4;

    /// <summary>The NAR's final learning rate (5e-6).</summary>
    public double NarEndLearningRate { get; set; } = 5e-6;

    /// <summary>The step the NAR's rate reaches its end (800,000).</summary>
    public int NarEndSteps { get; set; } = 800_000;

    /// <summary>The NAR's weight decay (0.002).</summary>
    public double NarWeightDecay { get; set; } = 0.002;

    /// <summary>The NAR's AdamW β₁ (0.8).</summary>
    public double NarBeta1 { get; set; } = 0.8;

    /// <summary>The NAR's AdamW β₂ (0.99).</summary>
    public double NarBeta2 { get; set; } = 0.99;

    /// <summary>AdamW's ε (PyTorch's 1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Warmup steps of both rates (10,000), linear from zero; the rate then falls linearly to its end (the
    /// schedule's shape is not recorded).</summary>
    public int WarmupSteps { get; set; } = 10_000;

    /// <summary>The gradient norm both stages clip to (1).</summary>
    public double GradientClipNorm { get; set; } = 1.0;

    /// <summary>The AR's speaker reference length range in seconds, cropped from the recording when the sample has no
    /// separate reference (0.3 to 8, the checkpoint's <c>spk_ref_min</c>/<c>spk_ref_max</c>).</summary>
    public double ArReferenceMinSeconds { get; set; } = 0.3;

    /// <summary>The AR's longest speaker reference in seconds (8).</summary>
    public double ArReferenceMaxSeconds { get; set; } = 8.0;

    /// <summary>The NAR's shortest prompt in seconds (0.5, <c>prompt_length_min</c>).</summary>
    public double NarPromptMinSeconds { get; set; } = 0.5;

    /// <summary>The NAR's longest prompt in seconds (14, <c>prompt_length_max</c>).</summary>
    public double NarPromptMaxSeconds { get; set; } = 14.0;

    /// <summary>The training draws' seed.</summary>
    public int TrainingSeed { get; set; } = 1234;
}
