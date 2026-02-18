namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveGrad (gradient-based conditional waveform diffusion).</summary>
public class WaveGradOptions : VocoderOptions { public WaveGradOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 300; NumDiffusionSteps = 50; } public int NumDownsampleBlocks { get; set; } = 4; }
