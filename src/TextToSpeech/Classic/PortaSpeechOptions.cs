namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for PortaSpeech (portable TTS with word-level prosody modeling and normalizing flow post-net).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PortaSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class PortaSpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PortaSpeechOptions(PortaSpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumFlowLayers = other.NumFlowLayers;
        ProsodyDim = other.ProsodyDim;
        DurationScale = other.DurationScale;
        MaxDuration = other.MaxDuration;
    }

    public PortaSpeechOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the number of normalizing flow layers for the post-net.</summary>
    public int NumFlowLayers { get; set; } = 16;

    /// <summary>Gets or sets the word-level prosody embedding dimension.</summary>
    public int ProsodyDim { get; set; } = 64;

    /// <summary>Gets or sets the duration scale factor for phoneme duration prediction.</summary>
    public double DurationScale { get; set; } = 2.5;

    /// <summary>Gets or sets the maximum frames per phoneme.</summary>
    public int MaxDuration { get; set; } = 15;
}
