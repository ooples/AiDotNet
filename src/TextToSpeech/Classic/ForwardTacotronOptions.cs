namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Forward Tacotron (non-autoregressive Tacotron with duration predictor).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ForwardTacotron model. Default values follow the original paper settings.</para>
/// </remarks>
public class ForwardTacotronOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ForwardTacotronOptions(ForwardTacotronOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        PrenetDim = other.PrenetDim;
        HighwayDim = other.HighwayDim;
        DurationScale = other.DurationScale;
    }

    public ForwardTacotronOptions()
    {
        EncoderDim = 256;
        DecoderDim = 80;
        HiddenDim = 256;
        NumEncoderLayers = 3;
        NumDecoderLayers = 1;
        NumHeads = 1;
    }

    /// <summary>Gets or sets the prenet dimension for the LSTM encoder.</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the highway network dimension.</summary>
    public int HighwayDim { get; set; } = 128;

    /// <summary>Gets or sets the duration scale factor for phoneme duration prediction.</summary>
    public double DurationScale { get; set; } = 4.0;
}
