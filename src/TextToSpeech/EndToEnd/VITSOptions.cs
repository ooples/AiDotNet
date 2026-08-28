namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for VITS (end-to-end TTS with conditional VAE, normalizing flows, and adversarial training).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VITS model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITSOptions : EndToEndTtsOptions
{
    public VITSOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        HiddenDim = 192;
        NumFlowSteps = 4;
        // Kim et al. 2021 section 4.1: AdamW, beta = (0.8, 0.99), weight decay 0.01,
        // initial learning rate 2e-4. The generic TTS defaults are Adam's usual
        // (0.9, 0.999) at 1e-4, which trains this stack unstably.
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>First AdamW moment coefficient used by the released VITS recipe.</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Second AdamW moment coefficient used by the released VITS recipe.</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW numerical-stability epsilon used by the released VITS recipe.</summary>
    public double Epsilon { get; set; } = 1e-9;
}
