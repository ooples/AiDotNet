using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for MatchaTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MatchaTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class MatchaTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MatchaTTSOptions(MatchaTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FlowDim = other.FlowDim;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public MatchaTTSOptions()
    {
        NumFlowSteps = 4;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the hidden width of the flow decoder that maps noise to the mel spectrogram.</summary>
    public int FlowDim { get; set; } = 256;
}
