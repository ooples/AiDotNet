namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for <see cref="StyleTTS2{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's multispeaker model as the released LibriTTS configuration builds it
/// (yl4579/StyleTTS2 <c>Configs/config_libritts.yml</c>): HiFi-GAN decoder, 128-dimensional acoustic and prosodic styles,
/// a 3-layer style-diffusion transformer conditioned on PL-BERT, 50 duration bins. The LJSpeech configuration is
/// <see cref="Multispeaker"/> false with the iSTFTNet decoder (<see cref="DecoderType"/>, <see cref="UpsampleRates"/>
/// {10, 6}, <see cref="UpsampleKernels"/> {20, 12}).</para>
/// <para><b>For Beginners:</b> These settings size the networks and control how speech is generated; the defaults
/// reproduce the published multispeaker model.</para>
/// </remarks>
public class StyleTTS2Options : StyleTtsWaveformOptions
{
    /// <summary>Creates the released LibriTTS configuration.</summary>
    public StyleTTS2Options()
    {
    }

    /// <summary>Copies another instance.</summary>
    public StyleTTS2Options(StyleTTS2Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Multispeaker = other.Multispeaker;
        DurationBins = other.DurationBins;
        ResBlockKernels = (int[])other.ResBlockKernels.Clone();
        UpsampleRates = (int[])other.UpsampleRates.Clone();
        UpsampleKernels = (int[])other.UpsampleKernels.Clone();
        DiffusionLayers = other.DiffusionLayers;
        DiffusionHeads = other.DiffusionHeads;
        DiffusionHeadFeatures = other.DiffusionHeadFeatures;
        DiffusionMultiplier = other.DiffusionMultiplier;
        DiffusionMaxLength = other.DiffusionMaxLength;
        EmbeddingMaskProbability = other.EmbeddingMaskProbability;
        SigmaData = other.SigmaData;
        EstimateSigmaData = other.EstimateSigmaData;
        LogSigmaMean = other.LogSigmaMean;
        LogSigmaStd = other.LogSigmaStd;
        DiffusionSteps = other.DiffusionSteps;
        EmbeddingScale = other.EmbeddingScale;
        Alpha = other.Alpha;
        Beta = other.Beta;
        TrimEndSamples = other.TrimEndSamples;
        FinalPhonemeExtraFrames = other.FinalPhonemeExtraFrames;
        SlmDiscriminatorChannels = other.SlmDiscriminatorChannels;
        DiffusionStartStep = other.DiffusionStartStep;
        JointStartStep = other.JointStartStep;
        MelLossWeight = other.MelLossWeight;
        GeneratorLossWeight = other.GeneratorLossWeight;
        SlmLossWeight = other.SlmLossWeight;
        MonotonicLossWeight = other.MonotonicLossWeight;
        S2sLossWeight = other.S2sLossWeight;
        F0LossWeight = other.F0LossWeight;
        NormLossWeight = other.NormLossWeight;
        DurationLossWeight = other.DurationLossWeight;
        DurationCrossEntropyWeight = other.DurationCrossEntropyWeight;
        StyleLossWeight = other.StyleLossWeight;
        DiffusionLossWeight = other.DiffusionLossWeight;
        BertLearningRate = other.BertLearningRate;
        BertBeta1 = other.BertBeta1;
        BertWeightDecay = other.BertWeightDecay;
        FineTuneLearningRate = other.FineTuneLearningRate;
        SlmAdversarialMinFrames = other.SlmAdversarialMinFrames;
        SlmAdversarialMaxFrames = other.SlmAdversarialMaxFrames;
        SlmDiscriminatorInterval = other.SlmDiscriminatorInterval;
        SlmGradientThreshold = other.SlmGradientThreshold;
        SlmGradientScale = other.SlmGradientScale;
        SlmDurationSigma = other.SlmDurationSigma;
        OutOfDistributionTexts = other.OutOfDistributionTexts?.Select(t => (int[])t.Clone()).ToList();
    }

    /// <summary>Whether the style denoiser reads a reference style (the LibriTTS model, true) or not (LJSpeech).</summary>
    public bool Multispeaker { get; set; } = true;

    /// <summary>Duration bins of the duration head (<c>max_dur</c>, 50): the duration is the sum of their sigmoids.</summary>
    public int DurationBins { get; set; } = 50;

    /// <summary>Blocks of the style-diffusion transformer (3).</summary>
    public int DiffusionLayers { get; set; } = 3;

    /// <summary>Its attention heads (8).</summary>
    public int DiffusionHeads { get; set; } = 8;

    /// <summary>Its per-head width (64).</summary>
    public int DiffusionHeadFeatures { get; set; } = 64;

    /// <summary>Its feed-forward multiplier (2).</summary>
    public int DiffusionMultiplier { get; set; } = 2;

    /// <summary>Length of the denoiser's unconditional embedding table (512, <c>StyleTransformer1d</c>'s default).</summary>
    public int DiffusionMaxLength { get; set; } = 512;

    /// <summary>Probability of training the denoiser unconditionally (0.1), for classifier-free guidance.</summary>
    public double EmbeddingMaskProbability { get; set; } = 0.1;

    /// <summary>σ_data of the EDM preconditioning (0.2, replaced by each batch's estimate when
    /// <see cref="EstimateSigmaData"/>).</summary>
    public double SigmaData { get; set; } = 0.2;

    /// <summary>Whether training re-estimates σ_data as the mean standard deviation of the target styles (true).</summary>
    public bool EstimateSigmaData { get; set; } = true;

    /// <summary>Mean of ln σ for the training noise levels (−3).</summary>
    public double LogSigmaMean { get; set; } = -3.0;

    /// <summary>Standard deviation of ln σ (1).</summary>
    public double LogSigmaStd { get; set; } = 1.0;

    /// <summary>Sampling steps of the style diffusion at inference (5).</summary>
    public int DiffusionSteps { get; set; } = 5;

    /// <summary>Classifier-free guidance scale at inference (1: none).</summary>
    public double EmbeddingScale { get; set; } = 1.0;

    /// <summary>Weight of the sampled acoustic style against the reference's (0.3; multispeaker only).</summary>
    public double Alpha { get; set; } = 0.3;

    /// <summary>Weight of the sampled prosodic style against the reference's (0.7; multispeaker only).</summary>
    public double Beta { get; set; } = 0.7;

    /// <summary>Samples cut from the end of the waveform (50, the LibriTTS notebook's fix for a pulse at the end).</summary>
    public int TrimEndSamples { get; set; } = 50;

    /// <summary>Frames added to the last phoneme's duration (0; the LJSpeech notebook adds 5).</summary>
    public int FinalPhonemeExtraFrames { get; set; }

    /// <summary>First width of the SLM discriminator head (64).</summary>
    public int SlmDiscriminatorChannels { get; set; } = 64;

    /// <summary>Joint-training updates before the style diffusion and the wave discriminators start (the reference's <c>diff_epoch</c>: 10 epochs for LibriTTS, 20 for LJSpeech, whose length depends on your data — set it for yours).</summary>
    public int DiffusionStartStep { get; set; }

    /// <summary>Joint-training updates before the decoder and style encoder are fine-tuned and SLM adversarial training starts (the reference's <c>joint_epoch</c>: 15 epochs for LibriTTS, 50 for LJSpeech).</summary>
    public int JointStartStep { get; set; }

    /// <summary>λ_mel, the multi-resolution mel loss (5).</summary>
    public double MelLossWeight { get; set; } = 5.0;

    /// <summary>λ_gen, the wave discriminators' adversarial loss (1).</summary>
    public double GeneratorLossWeight { get; set; } = 1.0;

    /// <summary>λ_slm, WavLM feature matching (1).</summary>
    public double SlmLossWeight { get; set; } = 1.0;

    /// <summary>λ_mono, the aligner's monotonic loss (1; the loss itself is ×10).</summary>
    public double MonotonicLossWeight { get; set; } = 1.0;

    /// <summary>λ_s2s, the aligner's token cross-entropy (1).</summary>
    public double S2sLossWeight { get; set; } = 1.0;

    /// <summary>λ_F0, the pitch reconstruction loss (1; the loss itself is ÷10).</summary>
    public double F0LossWeight { get; set; } = 1.0;

    /// <summary>λ_norm, the energy reconstruction loss (1).</summary>
    public double NormLossWeight { get; set; } = 1.0;

    /// <summary>λ_dur, the duration L1 loss (1).</summary>
    public double DurationLossWeight { get; set; } = 1.0;

    /// <summary>λ_ce, the duration bins' binary cross-entropy (20).</summary>
    public double DurationCrossEntropyWeight { get; set; } = 20.0;

    /// <summary>λ_sty, the sampled style's L1 reconstruction loss (1).</summary>
    public double StyleLossWeight { get; set; } = 1.0;

    /// <summary>λ_diff, the EDM score-matching loss (1).</summary>
    public double DiffusionLossWeight { get; set; } = 1.0;

    /// <summary>PL-BERT's learning rate in joint training (1e-5).</summary>
    public double BertLearningRate { get; set; } = 1e-5;

    /// <summary>PL-BERT's AdamW β₁ (0.9; the other modules use 0).</summary>
    public double BertBeta1 { get; set; } = 0.9;

    /// <summary>PL-BERT's AdamW weight decay (0.01).</summary>
    public double BertWeightDecay { get; set; } = 0.01;

    /// <summary>The decoder's and acoustic style encoder's learning rate in joint training (1e-5).</summary>
    public double FineTuneLearningRate { get; set; } = 1e-5;

    /// <summary>Mel frames of the shortest SLM adversarial segment (<c>slmadv_params.min_len</c>, 400); utterances shorter than this skip the step.</summary>
    public int SlmAdversarialMinFrames { get; set; } = 400;

    /// <summary>Mel frames of the longest SLM adversarial segment (500).</summary>
    public int SlmAdversarialMaxFrames { get; set; } = 500;

    /// <summary>Joint updates per SLM discriminator update (<c>iter</c>: 20 for LibriTTS, 10 for LJSpeech).</summary>
    public int SlmDiscriminatorInterval { get; set; } = 20;

    /// <summary>The predictor's gradient norm above which the SLM step's gradients are divided by it (5).</summary>
    public double SlmGradientThreshold { get; set; } = 5.0;

    /// <summary>Scale of the SLM step's gradients on the duration head, its LSTM and the diffusion model (0.01).</summary>
    public double SlmGradientScale { get; set; } = 0.01;

    /// <summary>σ of the Gaussian upsampling of differentiable duration modeling (1.5).</summary>
    public double SlmDurationSigma { get; set; } = 1.5;

    /// <summary>Phoneme id sequences of other texts (framed by the pad symbol) for SLM adversarial training's out-of-
    /// distribution half (the reference's <c>OOD_texts.txt</c>); without them every SLM step reads the training text.</summary>
    public IReadOnlyList<int[]>? OutOfDistributionTexts { get; set; }
}
