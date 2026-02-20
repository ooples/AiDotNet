using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>Options for WhisperSpeech.</summary>
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
