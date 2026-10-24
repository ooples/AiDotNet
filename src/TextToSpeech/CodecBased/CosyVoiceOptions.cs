namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for CosyVoice.</summary>
/// <remarks>
/// <para>The defaults are CosyVoice-300M's (FunAudioLLM/CosyVoice <c>examples/libritts/cosyvoice/conf/cosyvoice.yaml</c>):
/// Whisper's 100-language text tokens (51,866) and the 50 Hz S3 speech tokens (4,096); the LM's 6-block Conformer text
/// encoder (512 in, 1,024 wide, 16 heads, 4,096 units) and 14-block Transformer (1,024 wide); the flow's 6-block
/// Conformer (512 wide, 8 heads, 2,048 units) and U-Net estimator (channels 256, 256; 4 blocks; 12 mid blocks; 8 heads
/// of 64), 10 Euler steps with guidance 0.7; HiFT at 22.05 kHz (base 512, 8 harmonics, upsampling 8·8, iSTFT 16/4); a
/// 192-wide CAM++ x-vector.</para>
/// <para><b>For Beginners:</b> These options configure the CosyVoice model. Default values follow the released model.</para>
/// </remarks>
public class CosyVoiceOptions : CosyVoiceFamilyOptions
{
    public CosyVoiceOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        NumCodebooks = 1;
        CodebookSize = 4096;
        CodecFrameRate = 50;
        VocabSize = 51866;
        TextEncoderDim = 512;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 14;
        NumHeads = 16;
        SpeakerEmbeddingDim = 192;
        DropoutRate = 0.1;
        DecoderChannels = new[] { 256, 256 };
        HiftUpsampleRates = new[] { 8, 8 };
        HiftUpsampleKernels = new[] { 16, 16 };
        HiftSourceKernels = new[] { 7, 11 };
        WarmupSteps = 2500;
        VocoderSegmentLength = 24576;
    }

    /// <summary>Initializes a new instance by copying all CosyVoice settings.</summary>
    public CosyVoiceOptions(CosyVoiceOptions other)
        : base(other)
    {
        TextLanguages = other.TextLanguages;
        LmFeedForwardDim = other.LmFeedForwardDim;
    }

    /// <summary>Language tokens in the text tokenizer: 100 (Whisper large-v3's) for CosyVoice-300M.</summary>
    public int TextLanguages { get; set; } = 100;

    /// <summary>The LM's feed-forward width (<c>linear_units</c>), in the text encoder and the Transformer alike.</summary>
    public int LmFeedForwardDim { get; set; } = 4096;
}
