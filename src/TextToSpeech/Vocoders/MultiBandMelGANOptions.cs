namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for Multi-band MelGAN (multi-band signal decomposition for faster vocoding).</summary>
public class MultiBandMelGANOptions : VocoderOptions { public MultiBandMelGANOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 300; } public int NumBands { get; set; } = 4; }
