namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveGlow (flow-based vocoder combining Glow and WaveNet).</summary>
public class WaveGlowOptions : VocoderOptions { public WaveGlowOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; } public int NumFlows { get; set; } = 12; public int NumWaveNetLayers { get; set; } = 8; public int EarlyOutputChannels { get; set; } = 2; }
