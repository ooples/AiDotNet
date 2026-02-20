namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Vocos (ConvNeXt-based Fourier vocoder predicting STFT magnitude and instantaneous frequency).</summary>
public class VocosOptions : VocoderOptions
{
    public VocosOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
    }

    public int ConvNeXtDim { get; set; } = 512;
}
