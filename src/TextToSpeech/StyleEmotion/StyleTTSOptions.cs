namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Which loss weighting StyleTTS trains with.</summary>
public enum StyleTtsObjective
{
    /// <summary>The paper's (Li et al. 2022, §II-B, §III-B): every loss an L1 norm, weighted λs2s = 0.2, λmono = 5,
    /// λadv = 1, λfm = 0.2, λdur = 1, λf0 = 0.1, λn = 1, and the reconstruction losses at 1.</summary>
    Paper,

    /// <summary>The released configuration (yl4579/StyleTTS <c>Configs/config.yml</c>, <c>train_*.py</c>): mel loss ×5,
    /// λs2s = 1, the monotonic loss ×10 at λmono = 1, λfm = 0.1, and smooth-L1 pitch (÷10) and energy losses.</summary>
    Reference,
}

/// <summary>The modules a StyleTTS training step updates (Li et al. 2022, §II-B).</summary>
public enum StyleTtsTrainingStage
{
    /// <summary>Stage 1: mel reconstruction — the text encoder, style encoder, decoder and, once the transferable
    /// monotonic aligner's objectives start, the text aligner and pitch extractor.</summary>
    Reconstruction,

    /// <summary>Stage 2: the duration and prosody predictors, with every other module fixed.</summary>
    Prosody,
}

/// <summary>The options StyleTTS (Li et al. 2022) and StyleTTS 2 (Li et al. 2023) share: audio features, the text
/// encoder, style encoders, decoder residual widths, prosody predictor, text aligner and pitch extractor, and the
/// AdamW settings and first-stage schedule of both.</summary>
/// <remarks>
/// <para>
/// The defaults are StyleTTS's paper's (§III, Appendix D) with the released code's configuration where the paper is silent
/// (yl4579/StyleTTS <c>Configs/config.yml</c>): 24 kHz audio, 80-bin mel spectrograms from a 2048-point FFT with a
/// 1200-sample window and 300-sample hop, normalized as <c>(log(1e-5 + mel) + 4) / 4</c>; 178 phoneme symbols; a
/// 512-wide text encoder of three convolutions and a BiLSTM; 128-dimensional styles from a style encoder whose first
/// width is 64; a decoder of seven residual blocks with 64-wide residual features; a three-layer prosody predictor with
/// dropout 0.2; a text aligner pre-trained for ASR (256 wide, 512-dimensional token embeddings) and a JDC pitch extractor.
/// Both stages train with AdamW (β = 0, 0.99, weight decay 1e-4, learning rate 1e-4).
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of StyleTTS's networks, its audio features and how it is
/// trained. The defaults reproduce the paper.</para>
/// </remarks>
public abstract class StyleTtsBaseOptions : TtsModelOptions
{
    /// <summary>Creates the papers' shared options.</summary>
    protected StyleTtsBaseOptions()
    {
        SampleRate = 24000;
        HopSize = 300;
        FftSize = 2048;
        MelChannels = 80;
        HiddenDim = 512;
        VocabSize = 178;
        DropoutRate = 0.2;
        LearningRate = 1e-4;
        WeightDecay = 1e-4;
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    protected StyleTtsBaseOptions(StyleTtsBaseOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        WindowLength = other.WindowLength;
        MelMean = other.MelMean;
        MelStd = other.MelStd;
        StyleDim = other.StyleDim;
        TextLayers = other.TextLayers;
        TextKernel = other.TextKernel;
        StyleEncoderDim = other.StyleEncoderDim;
        MaxConvDim = other.MaxConvDim;
        ResidualDim = other.ResidualDim;
        AlignerHidden = other.AlignerHidden;
        AlignerTokenEmbedding = other.AlignerTokenEmbedding;
        AlignerConvLayers = other.AlignerConvLayers;
        PitchExtractorChannels = (int[])other.PitchExtractorChannels.Clone();
        PitchExtractorLstm = other.PitchExtractorLstm;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        HardAlignmentProbability = other.HardAlignmentProbability;
        TmaStartStep = other.TmaStartStep;
        MinTrainingFrames = other.MinTrainingFrames;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The analysis window, in samples (1200).</summary>
    public int WindowLength { get; set; } = 1200;

    /// <summary>The log mel spectrogram's normalization mean (−4).</summary>
    public double MelMean { get; set; } = -4.0;

    /// <summary>The log mel spectrogram's normalization standard deviation (4).</summary>
    public double MelStd { get; set; } = 4.0;

    /// <summary>Width of the style vector (128).</summary>
    public int StyleDim { get; set; } = 128;

    /// <summary>Convolutions of the text encoder, and layers of the duration encoder (3).</summary>
    public int TextLayers { get; set; } = 3;

    /// <summary>The text encoder's convolution kernel (5).</summary>
    public int TextKernel { get; set; } = 5;

    /// <summary>The style encoder's and discriminator's first width (64).</summary>
    public int StyleEncoderDim { get; set; } = 64;

    /// <summary>The style encoder's and discriminator's widest layer (512).</summary>
    public int MaxConvDim { get; set; } = 512;

    /// <summary>The decoder's residual phoneme, pitch and energy feature width (64).</summary>
    public int ResidualDim { get; set; } = 64;

    /// <summary>The text aligner's width (256).</summary>
    public int AlignerHidden { get; set; } = 256;

    /// <summary>The text aligner's token embedding width (512).</summary>
    public int AlignerTokenEmbedding { get; set; } = 512;

    /// <summary>The text aligner's residual convolution blocks (6).</summary>
    public int AlignerConvLayers { get; set; } = 6;

    /// <summary>The pitch extractor's conv block and residual block widths (64, 128, 192, 256: the JDC network's).</summary>
    public int[] PitchExtractorChannels { get; set; } = { 64, 128, 192, 256 };

    /// <summary>Each direction of the pitch extractor's BiLSTM (256).</summary>
    public int PitchExtractorLstm { get; set; } = 256;

    /// <summary>AdamW β₁ (0, §III-B).</summary>
    public double Beta1 { get; set; }

    /// <summary>AdamW β₂ (0.99, §III-B).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW ε (1e-9, the released code's).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>The share of reconstruction steps that read the hard monotonic alignment instead of the soft attention
    /// (0.5: the paper's 50%-50% strategy, §II-B and Appendix A-A).</summary>
    public double HardAlignmentProbability { get; set; } = 0.5;

    /// <summary>Reconstruction updates before the aligner's objectives start and the aligner and pitch extractor learn
    /// (0; the released code waits 20 epochs, whose length depends on your data — set it for yours).</summary>
    public int TmaStartStep { get; set; }

    /// <summary>Training segments shorter than this many frames are skipped, as the released code skips them (80).</summary>
    public int MinTrainingFrames { get; set; } = 80;

    /// <summary>Seed of the training draws and of dropout.</summary>
    public int SamplingSeed { get; set; }
}

/// <summary>Options for StyleTTS (Li et al. 2022, "StyleTTS: A Style-Based Generative Model for Natural and Diverse
/// Text-to-Speech Synthesis").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's (§III, Appendix D) with the released code's configuration where the paper is silent
/// (yl4579/StyleTTS <c>Configs/config.yml</c>): 24 kHz audio, 80-bin mel spectrograms from a 2048-point FFT with a
/// 1200-sample window and 300-sample hop, normalized as <c>(log(1e-5 + mel) + 4) / 4</c>; 178 phoneme symbols; a
/// 512-wide text encoder of three convolutions and a BiLSTM; 128-dimensional styles from a style encoder whose first
/// width is 64; a decoder of seven residual blocks with 64-wide residual features; a three-layer prosody predictor with
/// dropout 0.2; a text aligner pre-trained for ASR (256 wide, 512-dimensional token embeddings) and a JDC pitch extractor.
/// Both stages train with AdamW (β = 0, 0.99, weight decay 1e-4, learning rate 1e-4).
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of StyleTTS's networks, its audio features and how it is
/// trained. The defaults reproduce the paper.</para>
/// </remarks>
public class StyleTTSOptions : StyleTtsBaseOptions
{
    /// <summary>Creates the paper's options.</summary>
    public StyleTTSOptions()
    {
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public StyleTTSOptions(StyleTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Objective = other.Objective;
        R1Weight = other.R1Weight;
        MaxStretch = other.MaxStretch;
    }

    /// <summary>The loss weighting (the paper's).</summary>
    public StyleTtsObjective Objective { get; set; } = StyleTtsObjective.Paper;

    /// <summary>Weight of the discriminator's R1 penalty on real spectrograms (1, the released code's; the paper does not
    /// mention it).</summary>
    public double R1Weight { get; set; } = 1.0;

    /// <summary>The largest relative time stretch of the prosody stage's augmentation (0.25: scales from 0.75 to 1.25,
    /// §III-B).</summary>
    public double MaxStretch { get; set; } = 0.25;
}
