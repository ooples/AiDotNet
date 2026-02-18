namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for UnivNet (universal neural vocoder with multi-resolution spectrogram discriminator).</summary>
public class UnivNetOptions : VocoderOptions { public UnivNetOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; } public int NumKernels { get; set; } = 3; public int NumLMBlocks { get; set; } = 5; }
