using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for CoMoSpeech TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CoMoSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class CoMoSpeechOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CoMoSpeechOptions(CoMoSpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FlowDim = other.FlowDim;
        NumFlowLayers = other.NumFlowLayers;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public CoMoSpeechOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the hidden width of the flow decoder that maps noise to the mel spectrogram.</summary>
    public int FlowDim { get; set; } = 256;
    /// <summary>Gets or sets the number of layers in the flow decoder.</summary>
    public int NumFlowLayers { get; set; } = 4;
}
