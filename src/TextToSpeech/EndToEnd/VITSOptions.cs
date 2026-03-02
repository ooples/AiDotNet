namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for VITS (end-to-end TTS with conditional VAE, normalizing flows, and adversarial training).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VITS model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITSOptions : EndToEndTtsOptions { public VITSOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } }
