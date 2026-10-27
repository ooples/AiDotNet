using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The front end <see cref="MegaTTS{T}"/> reads text with.</summary>
public enum MegaTtsLanguage
{
    /// <summary>NATSpeech's English processor (g2p_en, word separators, &lt;BOS&gt;/&lt;EOS&gt;).</summary>
    English,

    /// <summary>The NATSpeech-family Chinese processor (chn_text_norm, pypinyin initials and toned finals).</summary>
    Chinese,
}

/// <summary>Which of Mega-TTS's two models <see cref="MegaTTS{T}"/> trains.</summary>
public enum MegaTtsStage
{
    /// <summary>The VQGAN-based acoustic model: the encoders, the duration predictor and the mel decoder against the
    /// multi-window discriminator (Jiang et al. 2023, Eq. 1–2).</summary>
    Acoustic,

    /// <summary>The prosody large language model, on the frozen acoustic model's prosody codes (§3.2).</summary>
    ProsodyLanguageModel,
}

/// <summary>Options for <see cref="MegaTTS{T}"/>. The sizes are the paper's (Jiang et al. 2023, Table 7); what it leaves
/// unstated comes from the authors' NATSpeech and SyntaSpeech code.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Mega-TTS speaks in a voice from a few seconds of it, copying both how the person sounds and
/// how they tend to speak. These options size its parts; train it on recordings with phoneme durations.</para>
/// </remarks>
public class MegaTTSOptions : TtsModelOptions
{
    public MegaTTSOptions()
    {
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        HiddenDim = 320;
        DropoutRate = 0;
        WeightDecay = 0;
    }

    /// <summary>The text front end (English).</summary>
    public MegaTtsLanguage Language { get; set; } = MegaTtsLanguage.English;

    /// <summary>The phone set (NATSpeech's <c>ph_set.json</c>): its sorted phones follow the three reserved tokens. Null
    /// takes every phone the English front end writes.</summary>
    public string[]? PhoneSet { get; set; }

    /// <summary>The analysis window, in samples (NATSpeech's 1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>The mel filterbank's lowest frequency (NATSpeech's 80 Hz).</summary>
    public double MelMinFrequency { get; set; } = 80;

    /// <summary>The mel filterbank's highest frequency (NATSpeech's 7600 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 7600;

    /// <summary>Content encoder layers (4) and heads (2), feed-forward kernel 5.</summary>
    public int ContentLayers { get; set; } = 4;

    /// <summary>See <see cref="ContentLayers"/>.</summary>
    public int ContentHeads { get; set; } = 2;

    /// <summary>Convolution blocks of the prosody encoder's two stacks, the timbre encoder and the mel decoder (5).</summary>
    public int ConvBlocks { get; set; } = 5;

    /// <summary>The timbre encoder's kernel (31).</summary>
    public int TimbreKernel { get; set; } = 31;

    /// <summary>The low mel bins the prosody encoder reads (the first 20).</summary>
    public int ProsodyBins { get; set; } = 20;

    /// <summary>VQ codebook entries (2048) and their width (256).</summary>
    public int Codes { get; set; } = 2048;

    /// <summary>See <see cref="Codes"/>.</summary>
    public int CodeDimension { get; set; } = 256;

    /// <summary>The prosody language model's layers (8), width (512) and heads (8), feed-forward kernel 5.</summary>
    public int LmLayers { get; set; } = 8;

    /// <summary>See <see cref="LmLayers"/>.</summary>
    public int LmHidden { get; set; } = 512;

    /// <summary>See <see cref="LmLayers"/>.</summary>
    public int LmHeads { get; set; } = 8;

    /// <summary>The discriminator's windows (32, 64, 128 frames) and width (192).</summary>
    public int[] DiscriminatorWindows { get; set; } = { 32, 64, 128 };

    /// <summary>See <see cref="DiscriminatorWindows"/>.</summary>
    public int DiscriminatorHidden { get; set; } = 192;

    /// <summary>Top-k sampling of the prosody codes (5).</summary>
    public int TopK { get; set; } = 5;

    /// <summary>The seed of the prosody sampling.</summary>
    public int SamplingSeed { get; set; } = 0;

    /// <summary>The duration predictor's convolutions (3) and their kernel (NATSpeech's 3).</summary>
    public int DurationLayers { get; set; } = 3;

    /// <summary>See <see cref="DurationLayers"/>.</summary>
    public int DurationKernel { get; set; } = 3;

    /// <summary>HiFi-GAN V1's first width (512), upsampling rates and kernels, and residual-block kernels and dilations
    /// (Kong et al. 2020, Table 5); the rates multiply to <see cref="TtsModelOptions.HopSize"/>.</summary>
    public int VocoderInitialChannels { get; set; } = 512;

    /// <summary>See <see cref="VocoderInitialChannels"/>.</summary>
    public int[] VocoderRates { get; set; } = { 8, 8, 2, 2 };

    /// <summary>See <see cref="VocoderInitialChannels"/>.</summary>
    public int[] VocoderKernels { get; set; } = { 16, 16, 4, 4 };

    /// <summary>See <see cref="VocoderInitialChannels"/>.</summary>
    public int[] VocoderResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>See <see cref="VocoderInitialChannels"/>.</summary>
    public int[][] VocoderResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    // ---------------------------------------------------------------- training

    /// <summary>The model <see cref="MegaTTS{T}"/> trains.</summary>
    public MegaTtsStage Stage { get; set; } = MegaTtsStage.Acoustic;

    /// <summary>Adam's β1 (0.9), β2 (0.98) and ε (1e-9), on the Transformer schedule.</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>See <see cref="Beta1"/>.</summary>
    public double Beta2 { get; set; } = 0.98;

    /// <summary>See <see cref="Beta1"/>.</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>The Transformer schedule's warm-up steps (4000).</summary>
    public int WarmupSteps { get; set; } = 4000;

    /// <summary>The Transformer schedule's scale (1: Vaswani et al.'s rate, d^−½ · min(t^−½, t · warmup^−1.5)).</summary>
    public double ScheduleFactor { get; set; } = 1.0;

    /// <summary>The gradient norm every step clips to (NATSpeech's <c>clip_grad_norm</c> of 1); 0 turns clipping off.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>The adversarial loss weight (SyntaSpeech's 0.05).</summary>
    public double AdversarialWeight { get; set; } = 0.05;

    /// <summary>The phoneme-duration loss weight (NATSpeech's 0.1); word and sentence durations weigh 1.</summary>
    public double DurationWeight { get; set; } = 0.1;

    /// <summary>The word- and sentence-duration loss weights (NATSpeech's 1 and 1).</summary>
    public double WordDurationWeight { get; set; } = 1.0;

    /// <summary>See <see cref="WordDurationWeight"/>.</summary>
    public double SentenceDurationWeight { get; set; } = 1.0;

    /// <summary>The discriminator's learning rate (1e-4), ε (1e-6), and its step decay (halved every 40000 steps), as
    /// SyntaSpeech trains it.</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public double DiscriminatorEpsilon { get; set; } = 1e-6;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public int DiscriminatorDecaySteps { get; set; } = 40000;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public double DiscriminatorDecay { get; set; } = 0.5;

    /// <summary>The sentences of one speaker the P-LLM reads as one sequence in training (Table 7's 7 contextual
    /// sentences); see <see cref="MegaTTS{T}.TrainProsodyModel"/>.</summary>
    public int ContextSentences { get; set; } = 7;

    /// <summary>The seed of training's draws (the discriminator windows).</summary>
    public int TrainingSeed { get; set; } = 0;

    internal string[] Vocabulary => NatSpeechFrontEnd.Vocabulary(PhoneSet ?? NatSpeechFrontEnd.EnglishPhones());
}
