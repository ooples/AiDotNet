namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet (amplitude-phase network with dual-stream spectrum prediction and anti-wrapping phase loss).</summary>
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
