namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for Voicebox TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Voicebox model. Default values follow the original paper settings.</para>
/// </remarks>
public class VoiceboxOptions : CodecTtsOptions
{
    /// <summary>Initializes a copy of a Voicebox configuration.</summary>
    /// <param name="other">The options instance to copy.</param>
    public VoiceboxOptions(VoiceboxOptions other)
        : base(other)
    {
    }

    public VoiceboxOptions()
    {
        // Le et al. 2023, Appendix D, Table D13: the audio vector-field
        // estimator operates on continuous 80-bin log-mel features with a
        // 24-layer, 1024-wide Transformer, 16 heads and a 4096-wide FFN.
        MelChannels = 80;
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 24;
        NumHeads = 16;
        DropoutRate = 0.1;
        LearningRate = 1e-4;
    }
}
