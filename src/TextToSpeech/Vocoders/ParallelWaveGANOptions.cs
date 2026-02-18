namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for Parallel WaveGAN (non-autoregressive GAN vocoder with multi-resolution STFT loss).</summary>
public class ParallelWaveGANOptions : VocoderOptions { public ParallelWaveGANOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 300; } public int NumResBlocks { get; set; } = 30; public int ResChannels { get; set; } = 64; }
