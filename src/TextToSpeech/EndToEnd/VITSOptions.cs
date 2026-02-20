namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for VITS (end-to-end TTS with conditional VAE, normalizing flows, and adversarial training).</summary>
public class VITSOptions : EndToEndTtsOptions { public VITSOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } }
