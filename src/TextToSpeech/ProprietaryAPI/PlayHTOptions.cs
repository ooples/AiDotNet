using AiDotNet.TextToSpeech.EndToEnd;
namespace AiDotNet.TextToSpeech.ProprietaryAPI;
/// <summary>Options for PlayHT TTS API wrapper.</summary>
public class PlayHTOptions : EndToEndTtsOptions
{
    public PlayHTOptions() { NumFlowSteps = 0; }
    public string ApiKey { get; set; } = string.Empty;
    public string ApiEndpoint { get; set; } = string.Empty;
    public string VoiceId { get; set; } = "default";
    public new int NumEncoderLayers { get; set; } = 2;
    public new int NumDecoderLayers { get; set; } = 2;
    public new int NumHeads { get; set; } = 4;
    public new double DropoutRate { get; set; } = 0.0;
}
