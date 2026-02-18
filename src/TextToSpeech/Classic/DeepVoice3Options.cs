namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Deep Voice 3 (fully convolutional attention-based TTS).</summary>
public class DeepVoice3Options : AcousticModelOptions
{
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
