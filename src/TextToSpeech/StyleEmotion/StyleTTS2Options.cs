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
public class StyleTTS2Options : StyleTTSOptions
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
    /// hop of the decoder's frames, which are half the mel frames' (<see cref="TtsModelOptions.HopSize"/> 300).</summary>
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
}
