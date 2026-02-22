using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>Options for WhisperSpeech.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WhisperSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class WhisperSpeechOptions : CodecTtsOptions
{
    public WhisperSpeechOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 2;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 768;
        NumLLMLayers = 6;
    }
}
