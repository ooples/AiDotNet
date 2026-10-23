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
public class StyleTTS2Options : StyleTtsBaseOptions
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
        DecoderType = other.DecoderType;
        DecoderBottleneck = other.DecoderBottleneck;
        ResBlockKernels = (int[])other.ResBlockKernels.Clone();
        ResBlockDilations = other.ResBlockDilations.Select(d => (int[])d.Clone()).ToArray();
        UpsampleRates = (int[])other.UpsampleRates.Clone();
        UpsampleKernels = (int[])other.UpsampleKernels.Clone();
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        IstftFftSize = other.IstftFftSize;
        IstftHopSize = other.IstftHopSize;
        Bert = other.Bert with { };
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
        SlmHidden = other.SlmHidden;
        SlmLayers = other.SlmLayers;
        SlmDiscriminatorChannels = other.SlmDiscriminatorChannels;
        SlmSampleRate = other.SlmSampleRate;
        SlmHeads = other.SlmHeads;
        SlmIntermediate = other.SlmIntermediate;
        SlmConvChannels = other.SlmConvChannels;
        SlmPositionalConvKernel = other.SlmPositionalConvKernel;
        SlmPositionalConvGroups = other.SlmPositionalConvGroups;
        SlmRelativePositionBuckets = other.SlmRelativePositionBuckets;
        DiffusionStartStep = other.DiffusionStartStep;
        JointStartStep = other.JointStartStep;
        MaxTrainingFrames = other.MaxTrainingFrames;
        WavePadding = other.WavePadding;
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

    /// <summary>The waveform generator (HiFi-GAN for LibriTTS, iSTFTNet for LJSpeech).</summary>
    public StyleTts2DecoderType DecoderType { get; set; } = StyleTts2DecoderType.HiFiGan;

    /// <summary>Width of the decoder's AdaIN residual blocks (1024 in the released code).</summary>
    public int DecoderBottleneck { get; set; } = 1024;

    /// <summary>Kernel sizes of the generator's parallel residual blocks.</summary>
    public int[] ResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>Dilations of each of those residual blocks.</summary>
    public int[][] ResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The generator's upsampling factors; their product (times <see cref="IstftHopSize"/> for iSTFTNet) is the
    /// mel hop (<see cref="TtsModelOptions.HopSize"/>, 300).</summary>
    public int[] UpsampleRates { get; set; } = { 10, 5, 3, 2 };

    /// <summary>Kernel sizes of the upsampling transposed convolutions.</summary>
    public int[] UpsampleKernels { get; set; } = { 20, 10, 6, 4 };

    /// <summary>Channels entering the generator; each upsampling halves them.</summary>
    public int UpsampleInitialChannels { get; set; } = 512;

    /// <summary>iSTFTNet's FFT size (20).</summary>
    public int IstftFftSize { get; set; } = 20;

    /// <summary>iSTFTNet's hop (5).</summary>
    public int IstftHopSize { get; set; } = 5;

    /// <summary>The PL-BERT encoder (ALBERT: 178 phoneme tokens, width 768, 12 heads, feed-forward 2048, 12 shared-layer
    /// passes; <c>Utils/PLBERT/config.yml</c>).</summary>
    internal AlbertConfiguration Bert { get; set; } = new()
    {
        VocabSize = 178,
        Hidden = 768,
        Heads = 12,
        Intermediate = 2048,
        Layers = 12,
        MaxPositions = 512,
    };

    /// <summary>PL-BERT's width (768).</summary>
    public int BertHidden
    {
        get => Bert.Hidden;
        set => Bert = Bert with { Hidden = value };
    }

    /// <summary>PL-BERT's layer passes (12).</summary>
    public int BertLayers
    {
        get => Bert.Layers;
        set => Bert = Bert with { Layers = value };
    }

    /// <summary>PL-BERT's attention heads (12).</summary>
    public int BertHeads
    {
        get => Bert.Heads;
        set => Bert = Bert with { Heads = value };
    }

    /// <summary>PL-BERT's feed-forward width (2048).</summary>
    public int BertIntermediate
    {
        get => Bert.Intermediate;
        set => Bert = Bert with { Intermediate = value };
    }

    /// <summary>PL-BERT's factorized embedding width (128).</summary>
    public int BertEmbedding
    {
        get => Bert.EmbeddingSize;
        set => Bert = Bert with { EmbeddingSize = value };
    }

    /// <summary>PL-BERT's position table length (512).</summary>
    public int BertMaxPositions
    {
        get => Bert.MaxPositions;
        set => Bert = Bert with { MaxPositions = value };
    }

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

    /// <summary>Width of the speech language model's hidden states (WavLM-base-plus: 768).</summary>
    public int SlmHidden { get; set; } = 768;

    /// <summary>Hidden states the SLM discriminator stacks (13: the convolutional features' projection and 12 layers).</summary>
    public int SlmLayers { get; set; } = 13;

    /// <summary>First width of the SLM discriminator head (64).</summary>
    public int SlmDiscriminatorChannels { get; set; } = 64;

    /// <summary>The SLM's sample rate (16 kHz); waveforms are resampled to it.</summary>
    public int SlmSampleRate { get; set; } = 16000;

    /// <summary>WavLM's attention heads (12).</summary>
    public int SlmHeads { get; set; } = 12;

    /// <summary>WavLM's feed-forward width (3072).</summary>
    public int SlmIntermediate { get; set; } = 3072;

    /// <summary>Channels of WavLM's seven-layer convolutional feature encoder (512).</summary>
    public int SlmConvChannels { get; set; } = 512;

    /// <summary>WavLM's convolutional position embedding kernel (128).</summary>
    public int SlmPositionalConvKernel { get; set; } = 128;

    /// <summary>WavLM's convolutional position embedding groups (16).</summary>
    public int SlmPositionalConvGroups { get; set; } = 16;

    /// <summary>Buckets of WavLM's gated relative position bias (320).</summary>
    public int SlmRelativePositionBuckets { get; set; } = 320;

    /// <summary>The frozen speech language model: WavLM-base-plus (Chen et al. 2022; <c>microsoft/wavlm-base-plus</c>) —
    /// wav2vec 2.0's architecture with gated relative position bias, <see cref="SlmLayers"/> − 1 Transformer layers.</summary>
    internal AiDotNet.Audio.Foundations.WavLmConfiguration SlmConfiguration => new()
    {
        ConvDims = Enumerable.Repeat(SlmConvChannels, 7).ToArray(),
        HiddenSize = SlmHidden,
        Layers = SlmLayers - 1,
        Heads = SlmHeads,
        IntermediateSize = SlmIntermediate,
        PositionalConvKernel = SlmPositionalConvKernel,
        PositionalConvGroups = SlmPositionalConvGroups,
        RelativePositionBuckets = SlmRelativePositionBuckets,
    };

    /// <summary>Joint-training updates before the style diffusion and the wave discriminators start (the reference's <c>diff_epoch</c>: 10 epochs for LibriTTS, 20 for LJSpeech, whose length depends on your data — set it for yours).</summary>
    public int DiffusionStartStep { get; set; }

    /// <summary>Joint-training updates before the decoder and style encoder are fine-tuned and SLM adversarial training starts (the reference's <c>joint_epoch</c>: 15 epochs for LibriTTS, 50 for LJSpeech).</summary>
    public int JointStartStep { get; set; }

    /// <summary>Mel frames of the longest training segment (<c>max_len</c>: 300 for LibriTTS, 400 for LJSpeech).</summary>
    public int MaxTrainingFrames { get; set; } = 300;

    /// <summary>Zero samples the reference's data loader adds on each side of every recording (5000).</summary>
    public int WavePadding { get; set; } = 5000;

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
