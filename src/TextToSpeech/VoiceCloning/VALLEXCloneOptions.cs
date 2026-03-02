namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for VALLEXClone voice cloning model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VALLEXClone model. Default values follow the original paper settings.</para>
/// </remarks>
public class VALLEXCloneOptions : VoiceCloningOptions
{
    public VALLEXCloneOptions()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
