using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for DiTToTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DiTToTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class DiTToTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiTToTTSOptions(DiTToTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FlowDim = other.FlowDim;
        NumFlowLayers = other.NumFlowLayers;
    }

    public DiTToTTSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
    public int NumFlowLayers { get; set; } = 4;
}
