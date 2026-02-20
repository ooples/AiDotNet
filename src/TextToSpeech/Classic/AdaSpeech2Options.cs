namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech 2 (adaptive TTS with untranscribed speech data).</summary>
public class AdaSpeech2Options : AcousticModelOptions
{
    public AdaSpeech2Options() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the mel-to-phoneme encoder dimension for untranscribed data.</summary>
    public int Mel2PhDim { get; set; } = 256;
}
