using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>Options for PromptTTS description-based TTS model.</summary>
public class PromptTTSOptions : EndToEndTtsOptions
{
    public PromptTTSOptions()
    {
        NumFlowSteps = 0;
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int PromptEncoderDim { get; set; } = 128;
    public int NumPromptLayers { get; set; } = 3;
}
