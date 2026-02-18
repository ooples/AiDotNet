namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for FastSpeech (non-autoregressive TTS with duration predictor).</summary>
public class FastSpeechOptions : AcousticModelOptions
{
    public FastSpeechOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the duration predictor filter size.</summary>
    public int DurationPredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the duration predictor kernel size.</summary>
    public int DurationPredictorKernelSize { get; set; } = 3;
}
