namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for VITS2 (improved VITS with duration discriminator, Gaussian mixture prior, and speaker-conditional flow).</summary>
public class VITS2Options : EndToEndTtsOptions { public VITS2Options() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } public int NumMixtureComponents { get; set; } = 4; }
