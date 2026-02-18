namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Transformer TTS (multi-head self-attention acoustic model).</summary>
public class TransformerTTSOptions : AcousticModelOptions
{
    public TransformerTTSOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 4; }

    /// <summary>Gets or sets the feedforward dimension in transformer blocks.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the positional encoding dimension.</summary>
    public int PositionalEncodingDim { get; set; } = 256;
}
