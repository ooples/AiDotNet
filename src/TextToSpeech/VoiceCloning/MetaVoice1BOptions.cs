namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for the MetaVoice-1B multi-stage voice-cloning TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MetaVoice-1B model. The defaults reproduce
/// the released first-stage causal GPT, second-stage noncausal GPT, speaker encoder,
/// eight-codebook EnCodec and four-band diffusion decoder.</para>
/// </remarks>
public class MetaVoice1BOptions : VoiceCloningOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MetaVoice1BOptions(MetaVoice1BOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
        FirstStageDim = other.FirstStageDim;
        NumFirstStageLayers = other.NumFirstStageLayers;
        SecondStageDim = other.SecondStageDim;
        NumSecondStageLayers = other.NumSecondStageLayers;
        FirstStageCodebooks = other.FirstStageCodebooks;
        CodecLatentDim = other.CodecLatentDim;
        VocoderChannels = other.VocoderChannels;
        VocoderUpsampleFactor = other.VocoderUpsampleFactor;
        SwiGLUMultipleOf = other.SwiGLUMultipleOf;
        RoPETheta = other.RoPETheta;
        NumHeads = other.NumHeads;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        SecondStageHeads = other.SecondStageHeads;
        FirstStageContext = other.FirstStageContext;
        SecondStageContext = other.SecondStageContext;
        MaximumNewTokens = other.MaximumNewTokens;
        FirstStageTopK = other.FirstStageTopK;
        FirstStageTopP = other.FirstStageTopP;
        FirstStageTemperature = other.FirstStageTemperature;
        SecondStageTopK = other.SecondStageTopK;
        SecondStageTemperature = other.SecondStageTemperature;
        SpeakerGuidance = other.SpeakerGuidance;
        PromptGuidance = other.PromptGuidance;
        ReferenceSampleRate = other.ReferenceSampleRate;
        SpeakerHiddenDim = other.SpeakerHiddenDim;
        SpeakerMelChannels = other.SpeakerMelChannels;
        SpeakerEncoderLayers = other.SpeakerEncoderLayers;
        FineTuneLastBlocks = other.FineTuneLastBlocks;
        TrainingDropout = other.TrainingDropout;
        RawDecoderOutput = other.RawDecoderOutput;
        EnhanceOutput = other.EnhanceOutput;
    }

    public MetaVoice1BOptions()
    {
        MinReferenceDurationSec = 30.0;
        NumHeads = 16;
        NumCodebooks = 8;
        CodebookSize = 1024;
        // MetaVoice's GPTConfig sets dropout = 0.0 (no dropout in the released 1B model).
        DropoutRate = 0.0;
        SampleRate = 24000;
        HopSize = 320;
        CodecFrameRate = 75;
        HiddenDim = FirstStageDim;
        LLMDim = FirstStageDim;
        NumLLMLayers = NumFirstStageLayers;
        MaxTextLength = FirstStageContext;
        MaxCodecFrames = SecondStageContext;
        MelChannels = SpeakerMelChannels;
    }

    /// <summary>Legacy encoder width (retained for serialization compatibility).</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Legacy decoder width (retained for serialization compatibility).</summary>
    public int DecoderDim { get; set; } = 1024;

    /// <summary>First-stage causal GPT width, divisible by the first-stage head count.</summary>
    public int FirstStageDim { get; set; } = 2048;

    /// <summary>Number of first-stage causal transformer blocks.</summary>
    public int NumFirstStageLayers { get; set; } = 24;

    /// <summary>Released noncausal GPT width.</summary>
    public int SecondStageDim { get; set; } = 384;

    /// <summary>Number of second-stage non-causal transformer blocks.</summary>
    public int NumSecondStageLayers { get; set; } = 6;

    /// <summary>EnCodec RVQ hierarchies predicted by the first stage (paper: the first two).</summary>
    public int FirstStageCodebooks { get; set; } = 2;

    /// <summary>EnCodec continuous-latent width (summed RVQ codebook embedding dim); the vocoder's
    /// input channel count.</summary>
    public int CodecLatentDim { get; set; } = 128;

    /// <summary>Legacy factory setting retained for serialized options; the native decoder is multiband diffusion.</summary>
    public int VocoderChannels { get; set; } = 512;

    /// <summary>Legacy factory setting retained for serialized options.</summary>
    public int VocoderUpsampleFactor { get; set; } = 2;

    /// <summary>Rounds SwiGLU FFN hidden dimensions to this multiple.</summary>
    public int SwiGLUMultipleOf { get; set; } = 256;

    /// <summary>Legacy factory setting; the released native transformers use learned absolute positions.</summary>
    public double RoPETheta { get; set; } = 10000.0;

    /// <summary>Released noncausal GPT head count.</summary>
    public int SecondStageHeads { get; set; } = 6;
    /// <summary>First-stage learned-position context.</summary>
    public int FirstStageContext { get; set; } = 2048;
    /// <summary>Second-stage fixed padded context.</summary>
    public int SecondStageContext { get; set; } = 1024;
    /// <summary>Original canonical generation budget.</summary>
    public int MaximumNewTokens { get; set; } = 864;
    /// <summary>Optional first-stage top-k; choose either this or top-p.</summary>
    public int? FirstStageTopK { get; set; }
    /// <summary>First-stage nucleus probability.</summary>
    public double? FirstStageTopP { get; set; } = .95;
    /// <summary>First-stage logit temperature.</summary>
    public double FirstStageTemperature { get; set; } = 1;
    /// <summary>Second-stage top-k filtering.</summary>
    public int? SecondStageTopK { get; set; } = 200;
    /// <summary>Second-stage logit temperature.</summary>
    public double SecondStageTemperature { get; set; } = 1;
    /// <summary>Speaker classifier-free guidance scale.</summary>
    public double SpeakerGuidance { get; set; } = 3;
    /// <summary>Text-prompt classifier-free guidance scale.</summary>
    public double PromptGuidance { get; set; } = 1;
    /// <summary>Sample rate of tensor-only reference recordings, before resampling to 16 kHz.</summary>
    public int ReferenceSampleRate { get; set; } = 16000;
    /// <summary>Released speaker LSTM width.</summary>
    public int SpeakerHiddenDim { get; set; } = 256;
    /// <summary>Released speaker power-mel channels.</summary>
    public int SpeakerMelChannels { get; set; } = 40;
    /// <summary>Released speaker LSTM depth.</summary>
    public int SpeakerEncoderLayers { get; set; } = 3;
    /// <summary>Original fine-tuning freezes everything except the final norm and these last blocks; zero selects all blocks.</summary>
    public int FineTuneLastBlocks { get; set; } = 1;
    /// <summary>Original fine-tuning embedding, attention and residual dropout.</summary>
    public double TrainingDropout { get; set; } = .1;
    /// <summary>Returns raw 24 kHz neural output when true; the default reproduces the release's loudness and PCM policy.</summary>
    public bool RawDecoderOutput { get; set; }
    /// <summary>Runs the release's DeepFilterNet3 stage and returns 48 kHz audio. Ignored for raw decoder output.</summary>
    public bool EnhanceOutput { get; set; } = true;
}
