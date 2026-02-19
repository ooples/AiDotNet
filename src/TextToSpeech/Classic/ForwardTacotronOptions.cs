namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Forward Tacotron (non-autoregressive Tacotron with duration predictor).</summary>
public class ForwardTacotronOptions : AcousticModelOptions
{
    public ForwardTacotronOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 3; NumDecoderLayers = 1; NumHeads = 1; }

    /// <summary>Gets or sets the prenet dimension for the LSTM encoder.</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the highway network dimension.</summary>
    public int HighwayDim { get; set; } = 128;

    /// <summary>Gets or sets the duration scale factor for phoneme duration prediction.</summary>
    public double DurationScale { get; set; } = 4.0;
}
