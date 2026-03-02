namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for VITS2 (improved VITS with duration discriminator, Gaussian mixture prior, and speaker-conditional flow).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VITS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITS2Options : EndToEndTtsOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VITS2Options(VITS2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumMixtureComponents = other.NumMixtureComponents;
    }
 public VITS2Options() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } public int NumMixtureComponents { get; set; } = 4; }
