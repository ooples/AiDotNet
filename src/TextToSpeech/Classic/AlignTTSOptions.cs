namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AlignTTS (alignment-free non-autoregressive TTS with mix density network).</summary>
public class AlignTTSOptions : AcousticModelOptions
{
    public AlignTTSOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the number of mixture components for the alignment loss.</summary>
    public int NumMixtures { get; set; } = 1;
}
