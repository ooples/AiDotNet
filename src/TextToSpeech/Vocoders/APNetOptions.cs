namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet (amplitude-phase network with dual-stream spectrum prediction and anti-wrapping phase loss).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the APNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class APNetOptions : VocoderOptions
{
    public APNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
    }
}
