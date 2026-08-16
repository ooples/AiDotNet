using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for E2TTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the E2TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class E2TTSOptions : CodecTtsOptions
{
    public E2TTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        VocabSize = 399;
        TextEncoderDim = 1024;
        NumEncoderLayers = 0;
        NumHeads = 16;
        NumCodebooks = 1;
        CodebookSize = MelChannels;
        CodecFrameRate = SampleRate / HopSize;
        LLMDim = 1024;
        NumLLMLayers = 24;
        MaxMelLength = 4000;
        LearningRate = 7.5e-5;
    }

    /// <summary>Initializes a new instance by copying all user-customizable options.</summary>
    /// <param name="other">The options instance to copy.</param>
    public E2TTSOptions(E2TTSOptions other)
        : base(other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
    }
}
