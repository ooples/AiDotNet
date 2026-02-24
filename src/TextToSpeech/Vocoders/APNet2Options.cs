namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet2 (improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the APNet2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class APNet2Options : VocoderOptions
{
    public APNet2Options()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
    }
}
