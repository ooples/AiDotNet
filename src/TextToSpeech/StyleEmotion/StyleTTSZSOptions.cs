namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for <see cref="StyleTTSZS{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's (Li et al. 2024, §III–IV and Appendix C–D): 512-wide conformer modules (8 heads of 64,
/// feed-forward 1024; Tables VII–XI), a 50-row time-varying style quantized by 9 codebooks of 1024 codes in 8 dimensions,
/// the style denoiser's 100-step DDIM sampling with guidance 5, the distilled student, AdamW (β = 0, 0.99, decay 1e-4,
/// rate 1e-4). StyleTTS-ZS released no code: where the paper is silent the values follow StyleTTS 2's code (the decoder,
/// aligner, pitch extractor, PL-BERT and wave discriminators it says it reuses) or the cited components (wenet's
/// conformer, DAC's quantizer, WeSpeaker's ResNet34), each noted below.</para>
/// <para><b>For Beginners:</b> These settings size StyleTTS-ZS's networks and control how it samples the speaking style of
/// a short prompt recording.</para>
/// </remarks>
public class StyleTTSZSOptions : StyleTtsWaveformOptions
{
    /// <summary>Creates the paper's options.</summary>
    public StyleTTSZSOptions()
    {
        StyleDim = 512;
        MaxTrainingFrames = 240;
    }

    /// <summary>Copies another instance.</summary>
    public StyleTTSZSOptions(StyleTTSZSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ModelDim = other.ModelDim;
        Heads = other.Heads;
        FeedForwardDim = other.FeedForwardDim;
        ConformerDropout = other.ConformerDropout;
        PromptTextKernels = (int[])other.PromptTextKernels.Clone();
        ProsodyEncoderKernels = (int[])other.ProsodyEncoderKernels.Clone();
        ProsodyDecoderKernels = (int[])other.ProsodyDecoderKernels.Clone();
        DenoiserKernels = (int[])other.DenoiserKernels.Clone();
        DiscriminatorKernels = (int[])other.DiscriminatorKernels.Clone();
        Separators = other.Separators;
        MaxPhonemes = other.MaxPhonemes;
        StyleLength = other.StyleLength;
        Quantizers = other.Quantizers;
        CodebookSize = other.CodebookSize;
        CodeDim = other.CodeDim;
        CommitmentWeight = other.CommitmentWeight;
        CodebookWeight = other.CodebookWeight;
        ProsodyMaskProbability = other.ProsodyMaskProbability;
        StyleDropout = other.StyleDropout;
        PromptFrames = other.PromptFrames;
        SamplingSteps = other.SamplingSteps;
        Guidance = other.Guidance;
        ConditionDropout = other.ConditionDropout;
        UseDistilledSampler = other.UseDistilledSampler;
        DistillationGuidanceMin = other.DistillationGuidanceMin;
        DistillationGuidanceMax = other.DistillationGuidanceMax;
        S2sLossWeight = other.S2sLossWeight;
        MonotonicLossWeight = other.MonotonicLossWeight;
        F0LossWeight = other.F0LossWeight;
        NormLossWeight = other.NormLossWeight;
        SpeakerChannels = other.SpeakerChannels;
        SpeakerBlocks = (int[])other.SpeakerBlocks.Clone();
        SpeakerEmbedding = other.SpeakerEmbedding;
        BertLearningRate = other.BertLearningRate;
        BertBeta1 = other.BertBeta1;
        BertWeightDecay = other.BertWeightDecay;
    }

    /// <summary>Width of every new module (512, Tables VII–XI).</summary>
    public int ModelDim { get; set; } = 512;

    /// <summary>Attention heads of the conformer blocks (8, of 64 features each).</summary>
    public int Heads { get; set; } = 8;

    /// <summary>Feed-forward width of the conformer blocks (1024).</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Dropout inside the conformer blocks (0.1, wenet's default; the paper is silent).</summary>
    public double ConformerDropout { get; set; } = 0.1;

    /// <summary>Kernels of the prompt-text encoder's conformer blocks (one of 31, five of 15; Table VII).</summary>
    public int[] PromptTextKernels { get; set; } = { 31, 15, 15, 15, 15, 15 };

    /// <summary>Kernels of the prosody encoder's conformer blocks (one of 31, five of 15; Table VIII).</summary>
    public int[] ProsodyEncoderKernels { get; set; } = { 31, 15, 15, 15, 15, 15 };

    /// <summary>Kernels of each prosody decoder's conformer blocks (six of 31; Table IX).</summary>
    public int[] ProsodyDecoderKernels { get; set; } = { 31, 31, 31, 31, 31, 31 };

    /// <summary>Kernels of the style denoiser's and the student's conformer blocks (two of 31, ten of 15; Table X).</summary>
    public int[] DenoiserKernels { get; set; } = { 31, 31, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 };

    /// <summary>Kernels of each multimodal discriminator's conformer blocks (six of 15; Table XI).</summary>
    public int[] DiscriminatorKernels { get; set; } = { 15, 15, 15, 15, 15, 15 };

    /// <summary>Whether a learned separator row sits between joined sequences (the &lt;SEP&gt; tokens of Fig. 2; true).</summary>
    public bool Separators { get; set; } = true;

    /// <summary>Phonemes the prosody encoder's position embedding covers (512, PL-BERT's positions).</summary>
    public int MaxPhonemes { get; set; } = 512;

    /// <summary>Rows K of the time-varying style (50).</summary>
    public int StyleLength { get; set; } = 50;

    /// <summary>Codebooks of the residual quantizer (9).</summary>
    public int Quantizers { get; set; } = 9;

    /// <summary>Codes per codebook (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Dimension of the factorized codes (8, following DAC).</summary>
    public int CodeDim { get; set; } = 8;

    /// <summary>Weight of the quantizer's commitment loss (0.25, DAC's; the paper lists no quantizer loss).</summary>
    public double CommitmentWeight { get; set; } = 0.25;

    /// <summary>Weight of the quantizer's codebook loss (1, DAC's).</summary>
    public double CodebookWeight { get; set; } = 1.0;

    /// <summary>Probability that a training step masks a random span of the prosody encoder's input (0.5; the paper
    /// states the masking without its rate). The span covers a uniformly random fraction of the frames.</summary>
    public double ProsodyMaskProbability { get; set; } = 0.5;

    /// <summary>Dropout on the global style in training (0.2, §III-A).</summary>
    public double StyleDropout { get; set; } = 0.2;

    /// <summary>Mel frames of the longest training prompt, a random segment of the utterance (240: 3 s, as the evaluation
    /// prompts; the paper truncates prompts to the batch's shortest utterance).</summary>
    public int PromptFrames { get; set; } = 240;

    /// <summary>DDIM steps L of the teacher's sampling (100).</summary>
    public int SamplingSteps { get; set; } = 100;

    /// <summary>Classifier-free guidance scale ω at inference (5).</summary>
    public double Guidance { get; set; } = 5.0;

    /// <summary>Rate at which training drops the denoiser's conditions (0.1).</summary>
    public double ConditionDropout { get; set; } = 0.1;

    /// <summary>Whether inference samples with the distilled one-step student (true) or the teacher's DDIM.</summary>
    public bool UseDistilledSampler { get; set; } = true;

    /// <summary>Smallest guidance scale distillation draws (1, Eq. 6).</summary>
    public double DistillationGuidanceMin { get; set; } = 1.0;

    /// <summary>Largest guidance scale distillation draws (15, Eq. 6).</summary>
    public double DistillationGuidanceMax { get; set; } = 15.0;

    /// <summary>λ_s2s (0.2, App. C-A).</summary>
    public double S2sLossWeight { get; set; } = 0.2;

    /// <summary>λ_mono (5, App. C-A).</summary>
    public double MonotonicLossWeight { get; set; } = 5.0;

    /// <summary>λ_f0 (0.1, App. C-B).</summary>
    public double F0LossWeight { get; set; } = 0.1;

    /// <summary>λ_n (1, App. C-B).</summary>
    public double NormLossWeight { get; set; } = 1.0;

    /// <summary>First width of the WeSpeaker ResNet of the speaker-embedding feature matching (32: ResNet34).</summary>
    public int SpeakerChannels { get; set; } = 32;

    /// <summary>Residual blocks per stage of that ResNet ({3, 4, 6, 3}: ResNet34).</summary>
    public int[] SpeakerBlocks { get; set; } = { 3, 4, 6, 3 };

    /// <summary>Its embedding width (256).</summary>
    public int SpeakerEmbedding { get; set; } = 256;

    /// <summary>PL-BERT's learning rate (1e-5, StyleTTS 2's; the paper is silent).</summary>
    public double BertLearningRate { get; set; } = 1e-5;

    /// <summary>PL-BERT's AdamW β₁ (0.9, StyleTTS 2's).</summary>
    public double BertBeta1 { get; set; } = 0.9;

    /// <summary>PL-BERT's AdamW weight decay (0.01, StyleTTS 2's).</summary>
    public double BertWeightDecay { get; set; } = 0.01;
}
