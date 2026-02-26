namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for FastSpeech (non-autoregressive TTS with duration predictor).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FastSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class FastSpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FastSpeechOptions(FastSpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        DurationPredictorFilterSize = other.DurationPredictorFilterSize;
        DurationPredictorKernelSize = other.DurationPredictorKernelSize;
        DurationScale = other.DurationScale;
        MaxDuration = other.MaxDuration;
    }

    public FastSpeechOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the duration predictor filter size.</summary>
    public int DurationPredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the duration predictor kernel size.</summary>
    public int DurationPredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the duration scale factor for phoneme duration prediction.</summary>
    public double DurationScale { get; set; } = 2.5;

    /// <summary>Gets or sets the maximum frames per phoneme.</summary>
    public int MaxDuration { get; set; } = 15;
}
