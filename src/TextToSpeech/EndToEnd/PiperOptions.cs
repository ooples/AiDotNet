namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for Piper (lightweight VITS-based TTS optimized for edge/embedded deployment).</summary>
public class PiperOptions : EndToEndTtsOptions { public PiperOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; NumEncoderLayers = 4; } public double LengthScale { get; set; } = 1.0; public double NoiseScale { get; set; } = 0.667; public double NoiseScaleW { get; set; } = 0.8; }
