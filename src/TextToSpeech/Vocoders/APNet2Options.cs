namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet2 (improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss).</summary>
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
