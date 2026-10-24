namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// The options StyleTTS 2 (Li et al. 2023) and StyleTTS-ZS (Li et al. 2024) share on top of StyleTTS's: the waveform
/// decoder's generator, PL-BERT, the frozen WavLM, and the data loader's segmenting. The defaults are StyleTTS 2's released
/// LibriTTS configuration (yl4579/StyleTTS2 <c>Configs/config_libritts.yml</c>), which StyleTTS-ZS keeps for the modules it
/// reuses.
/// </summary>
public abstract class StyleTtsWaveformOptions : StyleTtsBaseOptions
{
    /// <summary>Creates the released configuration.</summary>
    protected StyleTtsWaveformOptions()
    {
    }

    /// <summary>Copies another instance.</summary>
    protected StyleTtsWaveformOptions(StyleTtsWaveformOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        DecoderType = other.DecoderType;
        DecoderBottleneck = other.DecoderBottleneck;
        ResBlockDilations = other.ResBlockDilations.Select(d => (int[])d.Clone()).ToArray();
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        IstftFftSize = other.IstftFftSize;
        IstftHopSize = other.IstftHopSize;
        Bert = other.Bert with { };
        SlmHidden = other.SlmHidden;
        SlmLayers = other.SlmLayers;
        SlmSampleRate = other.SlmSampleRate;
        SlmHeads = other.SlmHeads;
        SlmIntermediate = other.SlmIntermediate;
        SlmConvChannels = other.SlmConvChannels;
        SlmPositionalConvKernel = other.SlmPositionalConvKernel;
        SlmPositionalConvGroups = other.SlmPositionalConvGroups;
        SlmRelativePositionBuckets = other.SlmRelativePositionBuckets;
        MaxTrainingFrames = other.MaxTrainingFrames;
        WavePadding = other.WavePadding;
    }

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

    /// <summary>Width of the speech language model's hidden states (WavLM-base-plus: 768).</summary>
    public int SlmHidden { get; set; } = 768;

    /// <summary>Hidden states the SLM discriminator stacks (13: the convolutional features' projection and 12 layers).</summary>
    public int SlmLayers { get; set; } = 13;

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

    /// <summary>Mel frames of the longest training segment (<c>max_len</c>: 300 for LibriTTS, 400 for LJSpeech).</summary>
    public int MaxTrainingFrames { get; set; } = 300;

    /// <summary>Zero samples the reference's data loader adds on each side of every recording (5000).</summary>
    public int WavePadding { get; set; } = 5000;
}
