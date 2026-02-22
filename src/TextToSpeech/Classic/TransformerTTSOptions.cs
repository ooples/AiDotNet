namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Transformer TTS (multi-head self-attention acoustic model).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the TransformerTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class TransformerTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransformerTTSOptions(TransformerTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FeedForwardDim = other.FeedForwardDim;
        PositionalEncodingDim = other.PositionalEncodingDim;
    }

    public TransformerTTSOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 4; }

    /// <summary>Gets or sets the feedforward dimension in transformer blocks.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the positional encoding dimension.</summary>
    public int PositionalEncodingDim { get; set; } = 256;
}
