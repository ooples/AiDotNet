namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for iSTFTNet (inverse STFT-based vocoder that outputs STFT coefficients then iSTFT).</summary>
public class ISTFTNetOptions : VocoderOptions
{
    public ISTFTNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
    }

    public int StftWindow { get; set; } = 1024;
    public int NumUpsampleLayers { get; set; } = 4;
}
