namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Deep Voice 3 (fully convolutional attention-based TTS).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DeepVoice3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class DeepVoice3Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeepVoice3Options(DeepVoice3Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ConvKernelSize = other.ConvKernelSize;
        NumSpeakers = other.NumSpeakers;
        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        ConverterDim = other.ConverterDim;
    }

    public DeepVoice3Options() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 7; NumDecoderLayers = 4; NumHeads = 1; OutputsPerStep = 4; }

    /// <summary>Gets or sets the convolution kernel size for encoder blocks.</summary>
    public int ConvKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the number of speaker embeddings for multi-speaker.</summary>
    public int NumSpeakers { get; set; } = 1;

    /// <summary>Gets or sets the speaker embedding dimension.</summary>
    public int SpeakerEmbeddingDim { get; set; } = 64;

    /// <summary>Gets or sets the converter (post-net) hidden dim.</summary>
    public int ConverterDim { get; set; } = 256;
}
