namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for the MetaVoice-1B multi-stage voice-cloning TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MetaVoice-1B model. The defaults reproduce
/// the paper's 1.2B-parameter scale (metavoiceio/metavoice-src): a large first-stage causal
/// GPT/LLaMA transformer, a second-stage non-causal transformer, and a HiFi-GAN-style vocoder.
/// Every dimension is fully customizable for smaller / larger builds.</para>
/// </remarks>
public class MetaVoice1BOptions : VoiceCloningOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MetaVoice1BOptions(MetaVoice1BOptions other)
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
    }

    public MetaVoice1BOptions()
    {
        MinReferenceDurationSec = 3.0;
        // Paper scale (1.2B first-stage LLaMA transformer). RoPE needs an even per-head dim:
        // 2048/16 = 128 and 1024/16 = 64 are both even.
        NumHeads = 16;
        NumCodebooks = 8;
        CodebookSize = 1024;
        // MetaVoice's GPTConfig sets dropout = 0.0 (no dropout in the released 1B model).
        DropoutRate = 0.0;
    }

    /// <summary>Legacy encoder width (retained for serialization compatibility).</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Legacy decoder width (retained for serialization compatibility).</summary>
    public int DecoderDim { get; set; } = 1024;

    /// <summary>First-stage (causal GPT/LLaMA) transformer hidden width. Must be divisible by
    /// <see cref="AiDotNet.TextToSpeech.TtsModelOptions.NumHeads"/> with an even per-head dimension.</summary>
    public int FirstStageDim { get; set; } = 2048;

    /// <summary>Number of first-stage causal transformer blocks.</summary>
    public int NumFirstStageLayers { get; set; } = 24;

    /// <summary>Second-stage (non-causal) transformer hidden width. Must be divisible by
    /// <see cref="AiDotNet.TextToSpeech.TtsModelOptions.NumHeads"/> with an even per-head dimension.</summary>
    public int SecondStageDim { get; set; } = 1024;

    /// <summary>Number of second-stage non-causal transformer blocks.</summary>
    public int NumSecondStageLayers { get; set; } = 12;

    /// <summary>EnCodec RVQ hierarchies predicted by the first stage (paper: the first two).</summary>
    public int FirstStageCodebooks { get; set; } = 2;

    /// <summary>EnCodec continuous-latent width (summed RVQ codebook embedding dim); the vocoder's
    /// input channel count.</summary>
    public int CodecLatentDim { get; set; } = 128;

    /// <summary>HiFi-GAN vocoder upsample-stem channel width.</summary>
    public int VocoderChannels { get; set; } = 512;

    /// <summary>Vocoder time-upsampling factor for the transposed convolution.</summary>
    public int VocoderUpsampleFactor { get; set; } = 2;

    /// <summary>Rounds SwiGLU FFN hidden dimensions to this multiple.</summary>
    public int SwiGLUMultipleOf { get; set; } = 256;

    /// <summary>RoPE base frequency (theta) for both transformer stages.</summary>
    public double RoPETheta { get; set; } = 10000.0;
}
