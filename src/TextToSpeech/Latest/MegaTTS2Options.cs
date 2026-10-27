using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Which of Mega-TTS 2's training stages <see cref="MegaTTS2{T}"/> runs.</summary>
public enum MegaTts2Stage
{
    /// <summary>The compressive acoustic autoencoder against the multi-length discriminator (Jiang et al. 2024, §3.2).</summary>
    Acoustic,

    /// <summary>The prosody latent language model and the autoregressive duration model, on the frozen autoencoder's
    /// prosody codes and content (§3.3).</summary>
    LanguageModels,
}

/// <summary>Options for <see cref="MegaTTS2{T}"/>. The sizes are the paper's (Jiang et al. 2024, Appendix A.4 and Table 5);
/// what it leaves unstated comes from the authors' code (NATSpeech, SyntaSpeech, MegaTTS 3's release) and from CVQ-VAE.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Mega-TTS 2 speaks in a voice it has never heard from a few seconds, or a few minutes, of
/// it: the more prompt sentences you give it, the closer it gets to the speaker. These options size its parts; train it
/// on recordings with phoneme durations.</para>
/// </remarks>
public class MegaTTS2Options : TtsModelOptions
{
    public MegaTTS2Options()
    {
        SampleRate = 16000;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        HiddenDim = 512;
        DropoutRate = 0;
        WeightDecay = 0;
    }

    /// <summary>The text front end (English; the paper trains on LibriLight).</summary>
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

    /// <summary>The content encoder's layers (8), heads (NATSpeech's 2) and feed-forward filter (1024); kernel 5.</summary>
    public int ContentLayers { get; set; } = 8;

    /// <summary>See <see cref="ContentLayers"/>.</summary>
    public int ContentHeads { get; set; } = 2;

    /// <summary>See <see cref="ContentLayers"/>.</summary>
    public int ContentFilter { get; set; } = 1024;

    /// <summary>The VQ prosody encoder's width (384) and blocks per stack (3 blocks of 2 layers, kernel 5).</summary>
    public int ProsodyHidden { get; set; } = 384;

    /// <summary>See <see cref="ProsodyHidden"/>.</summary>
    public int ProsodyBlocks { get; set; } = 3;

    /// <summary>The VQ encoder's compression rate r: the max pooling of the mel spectrogram (8).</summary>
    public int Compression { get; set; } = 8;

    /// <summary>The codebook's entries (1024) and their width (256).</summary>
    public int Codes { get; set; } = 1024;

    /// <summary>See <see cref="Codes"/>.</summary>
    public int CodeDimension { get; set; } = 256;

    /// <summary>How a vector finds its code (CVQ-VAE's cosine similarity).</summary>
    public MegaTts2CodeDistance CodeDistance { get; set; } = MegaTts2CodeDistance.Cosine;

    /// <summary>Where the dynamic initialization draws its anchors (CVQ-VAE's closest features).</summary>
    public MegaTts2CodeAnchor CodeAnchor { get; set; } = MegaTts2CodeAnchor.Closest;

    /// <summary>The VQ loss's commitment weight β (0.25, van den Oord et al. and CVQ-VAE).</summary>
    public double CommitmentWeight { get; set; } = 0.25;

    /// <summary>The decay of the codes' running usage (CVQ-VAE's 0.99).</summary>
    public double CodeUsageDecay { get; set; } = 0.99;

    /// <summary>The MRTE's acoustic width (the key encoder's 256), blocks per stack (5 blocks of 2 layers, kernel 3) and
    /// downsampling rate d (16).</summary>
    public int TimbreHidden { get; set; } = 256;

    /// <summary>See <see cref="TimbreHidden"/>.</summary>
    public int TimbreBlocks { get; set; } = 5;

    /// <summary>See <see cref="TimbreHidden"/>.</summary>
    public int Downsampling { get; set; } = 16;

    /// <summary>The timbre-to-content attention's heads (NATSpeech's 2).</summary>
    public int TimbreHeads { get; set; } = 2;

    /// <summary>The frames of the other recordings the MRTE reads in training, drawn at random (2000).</summary>
    public int TimbreReferenceFrames { get; set; } = 2000;

    /// <summary>The mel decoder's blocks (4 blocks of 2 layers, kernel 5).</summary>
    public int DecoderBlocks { get; set; } = 4;

    /// <summary>The P-LLM's layers (12), width (1024) and heads (16).</summary>
    public int LmLayers { get; set; } = 12;

    /// <summary>See <see cref="LmLayers"/>.</summary>
    public int LmHidden { get; set; } = 1024;

    /// <summary>See <see cref="LmLayers"/>.</summary>
    public int LmHeads { get; set; } = 16;

    /// <summary>The autoregressive duration model's layers (8), width (512) and heads (8, MegaTTS 3's).</summary>
    public int DurationLayers { get; set; } = 8;

    /// <summary>See <see cref="DurationLayers"/>.</summary>
    public int DurationHidden { get; set; } = 512;

    /// <summary>See <see cref="DurationLayers"/>.</summary>
    public int DurationHeads { get; set; } = 8;

    /// <summary>The largest duration the duration model reads as a token; longer ones read as it (MegaTTS 3's 128).</summary>
    public int DurationCodes { get; set; } = 128;

    /// <summary>The multi-length discriminator's windows (32, 64, 128 frames) and width (192).</summary>
    public int[] DiscriminatorWindows { get; set; } = { 32, 64, 128 };

    /// <summary>See <see cref="DiscriminatorWindows"/>.</summary>
    public int DiscriminatorHidden { get; set; } = 192;

    /// <summary>Top-k sampling of the prosody codes (10, Appendix A.6).</summary>
    public int TopK { get; set; } = 10;

    /// <summary>The seed of the prosody sampling.</summary>
    public int SamplingSeed { get; set; } = 0;

    /// <summary>Prosody interpolation's weight γ on the auxiliary prosody prompt (0.8 in the paper's transfer
    /// experiments, Eq. 3); used when <see cref="MegaTTS2{T}.ProsodyPrompts"/> is set.</summary>
    public double ProsodyInterpolation { get; set; } = 0.8;

    /// <summary>HiFi-GAN V1's first width (512), upsampling rates and kernels, and residual-block kernels and dilations
    /// (Kong et al. 2020); the rates multiply to <see cref="TtsModelOptions.HopSize"/>.</summary>
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

    /// <summary>The stage <see cref="MegaTTS2{T}"/> trains.</summary>
    public MegaTts2Stage Stage { get; set; } = MegaTts2Stage.Acoustic;

    /// <summary>Adam's β1 (0.9), β2 (0.999) and ε (1e-9), on the Transformer schedule (§4.1).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>See <see cref="Beta1"/>.</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>See <see cref="Beta1"/>.</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>The Transformer schedule's warm-up steps (Vaswani et al.'s 4000).</summary>
    public int WarmupSteps { get; set; } = 4000;

    /// <summary>The Transformer schedule's scale (1: d^−½ · min(t^−½, t · warmup^−1.5)).</summary>
    public double ScheduleFactor { get; set; } = 1.0;

    /// <summary>The gradient norm every step clips to (NATSpeech's <c>clip_grad_norm</c> of 1); 0 turns clipping off.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>The adversarial loss weight (SyntaSpeech's 0.05).</summary>
    public double AdversarialWeight { get; set; } = 0.05;

    /// <summary>The discriminator's learning rate (1e-4), ε (1e-6) and step decay (halved every 40000 steps), as
    /// SyntaSpeech trains it.</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public double DiscriminatorEpsilon { get; set; } = 1e-6;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public int DiscriminatorDecaySteps { get; set; } = 40000;

    /// <summary>See <see cref="DiscriminatorLearningRate"/>.</summary>
    public double DiscriminatorDecay { get; set; } = 0.5;

    /// <summary>The most prosody tokens one language-model step reads (the paper's 4000 per GPU).</summary>
    public int ProsodyTokenBudget { get; set; } = 4000;

    /// <summary>The seed of training's draws (the discriminator windows, the MRTE's frames, CVQ-VAE's anchors).</summary>
    public int TrainingSeed { get; set; } = 0;

    internal string[] Vocabulary => NatSpeechFrontEnd.Vocabulary(PhoneSet ?? NatSpeechFrontEnd.EnglishPhones());
}
