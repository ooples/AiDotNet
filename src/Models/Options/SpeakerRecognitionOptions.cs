namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the speaker recognition models — the embedding extractor and the
/// verifier built on top of it.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models turn a recording of someone speaking into a short list of
/// numbers — a "voiceprint" — that is close to other recordings of the same person and far from
/// everyone else's. The settings here describe the audio they expect and the network that
/// produces the voiceprint.
/// </para>
/// <para>
/// Both models follow the x-vector design of Snyder et al. (ICASSP 2018): a stack of time-delay
/// layers over frame-level features, statistics pooling across time, then dense layers producing
/// the embedding. There is no attention in that design, which is why there is no head count here.
/// </para>
/// </remarks>
public abstract class SpeakerRecognitionOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the sample rate, in hertz, that input audio is expected to have. Default: 16000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 16 kHz is the standard for speech: it keeps everything the human
    /// voice produces while being far cheaper to process than music-grade audio.</para>
    /// </remarks>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the length of the voiceprint the model produces. Default: 256.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of values used to describe a voice. Longer captures
    /// more detail but needs more enrolment audio to be reliable.</para>
    /// </remarks>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// Gets or sets the width of the encoder layers. Default: 256.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets how many encoder layers the model builds. Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the number of layers the model constructs for itself. It is not
    /// <see cref="NeuralNetworkOptions.EncoderLayerCount"/>, which says where the encoder ends in
    /// a layer list the caller supplied.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Throws if a dimension every speaker model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its published defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(SampleRate, nameof(SampleRate));
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        Require(HiddenDim, nameof(HiddenDim));
        Require(NumEncoderLayers, nameof(NumEncoderLayers));
    }
}
