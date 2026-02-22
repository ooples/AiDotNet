namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for SeedTTSClone voice cloning model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SeedTTSClone model. Default values follow the original paper settings.</para>
/// </remarks>
public class SeedTTSCloneOptions : VoiceCloningOptions
{
    public SeedTTSCloneOptions()
    {
        MinReferenceDurationSec = 5.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
