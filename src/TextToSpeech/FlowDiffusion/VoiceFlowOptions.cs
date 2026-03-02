using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for VoiceFlow TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VoiceFlow model. Default values follow the original paper settings.</para>
/// </remarks>
public class VoiceFlowOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VoiceFlowOptions(VoiceFlowOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FlowDim = other.FlowDim;
    }

    public VoiceFlowOptions()
    {
        NumFlowSteps = 2;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
}
