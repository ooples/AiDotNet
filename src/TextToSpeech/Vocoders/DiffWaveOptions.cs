namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for DiffWave (diffusion-based vocoder using denoising score matching).</summary>
public class DiffWaveOptions : VocoderOptions { public DiffWaveOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; NumDiffusionSteps = 50; } public int NumResLayers { get; set; } = 30; public int ResChannels { get; set; } = 64; }
