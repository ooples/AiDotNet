namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Base configuration options for classic acoustic TTS models.
/// </summary>
public class AcousticModelOptions : TtsModelOptions
{
    /// <summary>Gets or sets the encoder embedding dimension.</summary>
    public int EncoderDim { get; set; } = 256;

    /// <summary>Gets or sets the decoder output dimension (mel channels).</summary>
    public int DecoderDim { get; set; } = 80;

    /// <summary>Gets or sets the number of mel frames generated per decoder step (reduction factor).</summary>
    public int OutputsPerStep { get; set; } = 1;

    /// <summary>Gets or sets whether to use a postnet for mel refinement.</summary>
    public bool UsePostnet { get; set; } = true;

    /// <summary>Gets or sets the postnet embedding dimension.</summary>
    public int PostnetDim { get; set; } = 512;

    /// <summary>Gets or sets the number of postnet convolution layers.</summary>
    public int PostnetLayers { get; set; } = 5;
}
