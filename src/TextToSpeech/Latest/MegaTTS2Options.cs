using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for MegaTTS2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MegaTTS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MegaTTS2Options : EndToEndTtsOptions
{
    public MegaTTS2Options(MegaTTS2Options other) : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
    }

    public MegaTTS2Options()
    {
        EncoderDim = 192;
        DecoderDim = MelChannels;
        HiddenDim = 192;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
