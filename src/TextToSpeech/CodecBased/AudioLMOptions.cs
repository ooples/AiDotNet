namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for AudioLM TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AudioLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class AudioLMOptions : CodecTtsOptions
{
    public AudioLMOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 12;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        TextEncoderDim = 1024;
        LLMDim = 1024;
        NumEncoderLayers = 0;
        NumLLMLayers = 12;
        NumHeads = 16;
        DropoutRate = 0.1;
    }

    /// <summary>Initializes a new instance by copying all user-customizable options.</summary>
    /// <param name="other">The options instance to copy.</param>
    public AudioLMOptions(AudioLMOptions other)
        : base(other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
    }
}
