using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for NeMo Citrinet (NVIDIA, 2021): 1D time-channel separable convolution CTC model.</summary>
public class NeMoCitrinetOptions : ModelOptions
{
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 23;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 1024;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    /// <summary>Squeeze-excitation reduction ratio for channel attention.</summary>
    public int SqueezeExcitationRatio { get; set; } = 8;
}
