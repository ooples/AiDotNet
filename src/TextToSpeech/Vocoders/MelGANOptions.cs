namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for MelGAN (lightweight GAN vocoder with no need for paired training data).</summary>
public class MelGANOptions : VocoderOptions { public MelGANOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; } public int NumResStacks { get; set; } = 3; public int NgfBase { get; set; } = 512; }
