namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for VITS2 (improved VITS with duration discriminator, Gaussian mixture prior, and speaker-conditional flow).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VITS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITS2Options : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VITS2Options(VITS2Options other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumMixtureComponents = other.NumMixtureComponents;
    }

    public VITS2Options()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        HiddenDim = 192;
        NumFlowSteps = 4;
        // VITS2 (Kong et al. 2023) and YourTTS (Casanova et al. 2022) both keep VITS's
        // optimizer recipe: AdamW, beta = (0.8, 0.99), weight decay 0.01, lr 2e-4.
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    public int NumMixtureComponents { get; set; } = 4;

    /// <summary>First AdamW moment coefficient used by the released VITS2 recipe.</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Second AdamW moment coefficient used by the released VITS2 recipe.</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW numerical-stability epsilon used by the released VITS2 recipe.</summary>
    public double Epsilon { get; set; } = 1e-9;
}
