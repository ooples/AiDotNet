namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron 2 (location-sensitive attention with WaveNet vocoder).</summary>
public class Tacotron2Options : AcousticModelOptions
{
    public Tacotron2Options() { EncoderDim = 512; DecoderDim = 80; HiddenDim = 512; NumEncoderLayers = 3; NumDecoderLayers = 2; NumHeads = 1; OutputsPerStep = 1; UsePostnet = true; PostnetDim = 512; PostnetLayers = 5; }

    /// <summary>Gets or sets the prenet dimension.</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the attention RNN dimension.</summary>
    public int AttentionRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the decoder RNN dimension.</summary>
    public int DecoderRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the attention location feature channels.</summary>
    public int AttentionLocationChannels { get; set; } = 32;
}
