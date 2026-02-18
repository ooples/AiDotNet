using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base configuration options for text-to-speech models.
/// </summary>
public class TtsModelOptions : ModelOptions
{
    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel-spectrogram frequency channels.</summary>
    public int MelChannels { get; set; } = 80;

    /// <summary>Gets or sets the hop size for mel-spectrogram computation.</summary>
    public int HopSize { get; set; } = 256;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the model hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the phoneme/text vocabulary size.</summary>
    public int VocabSize { get; set; } = 256;

    /// <summary>Gets or sets the maximum input text length.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Gets or sets the maximum output audio length in mel frames.</summary>
    public int MaxMelLength { get; set; } = 2048;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
