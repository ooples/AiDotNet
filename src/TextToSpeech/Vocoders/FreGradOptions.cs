namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for FreGrad (frequency-domain diffusion vocoder with DWT sub-band processing).</summary>
public class FreGradOptions : VocoderOptions { public FreGradOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; NumDiffusionSteps = 4; } public int NumResBlocks { get; set; } = 15; public int NumWaveletLevels { get; set; } = 3; }
