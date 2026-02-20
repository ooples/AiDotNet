namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for PriorGrad (diffusion vocoder with data-dependent prior for adaptive noise).</summary>
public class PriorGradOptions : VocoderOptions { public PriorGradOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; NumDiffusionSteps = 6; } public int NumResBlocks { get; set; } = 15; }
