namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for BigVGAN (universal vocoder with anti-aliased multi-periodicity composition and Snake activation).</summary>
public class BigVGANOptions : VocoderOptions { public BigVGANOptions() { SampleRate = 24000; MelChannels = 100; HopSize = 256; } public int HiddenChannels { get; set; } = 512; public int NumUpsampleLayers { get; set; } = 4; public int NumPeriods { get; set; } = 5; public double SnakeAlpha { get; set; } = 1.0; }
