using System.Linq;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// Options for <see cref="MeloTTS{T}"/>. The defaults are MeloTTS's (myshell-ai/MeloTTS <c>configs/config.json</c>,
/// <c>train.py</c> and <c>api.py</c>): a 44.1 kHz Bert-VITS2 synthesizer for one language.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MeloTTS speaks one language per model; <see cref="Language"/> picks which (it decides
/// the text front end and the BERT model the synthesizer listens to). A released checkpoint loads with
/// <see cref="MeloTTS{T}.LoadCheckpoint"/> and its BERT with <see cref="MeloTTS{T}.LoadBertWeights"/>.</para>
/// </remarks>
public class MeloTTSOptions : EndToEndTtsOptions
{
    /// <summary>Copies another instance.</summary>
    public MeloTTSOptions(MeloTTSOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        Language = other.Language;
        WindowSize = other.WindowSize;
        SegmentSize = other.SegmentSize;
        NumSpeakers = other.NumSpeakers;
        SpeakerChannels = other.SpeakerChannels;
        FlowTransformerLayers = other.FlowTransformerLayers;
        KernelSize = other.KernelSize;
        ResblockKernelSizes = (int[])other.ResblockKernelSizes.Clone();
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        UpsampleRates = (int[])other.UpsampleRates.Clone();
        UpsampleKernelSizes = (int[])other.UpsampleKernelSizes.Clone();
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        DurationDropout = other.DurationDropout;
        UseDurationDiscriminator = other.UseDurationDiscriminator;
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        DurationDiscriminatorDropout = other.DurationDiscriminatorDropout;
        MasNoiseInitial = other.MasNoiseInitial;
        MasNoiseDecay = other.MasNoiseDecay;
        BertChannels = other.BertChannels;
        JaBertChannels = other.JaBertChannels;
        BertLayers = other.BertLayers;
        BertHeads = other.BertHeads;
        BertIntermediate = other.BertIntermediate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        MelLossWeight = other.MelLossWeight;
        KlLossWeight = other.KlLossWeight;
        SdpRatio = other.SdpRatio;
        NoiseScale = other.NoiseScale;
        NoiseScaleW = other.NoiseScaleW;
        Speed = other.Speed;
        SpeakerId = other.SpeakerId;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates MeloTTS's defaults.</summary>
    public MeloTTSOptions()
    {
        SampleRate = 44100;
        FftSize = 2048;
        HopSize = 512;
        MelChannels = 128;
        HiddenDim = 192;
        InterChannels = 192;
        FilterChannels = 768;
        NumHeads = 2;
        NumEncoderLayers = 6;
        DropoutRate = 0.1;
        NumFlowSteps = 4;
        VocabSize = MeloSymbols.Symbols.Count;
        LearningRate = 3e-4;
        WeightDecay = 0.01;
    }

    /// <summary>The language the model speaks (default English).</summary>
    public MeloLanguage Language { get; set; } = MeloLanguage.English;

    /// <summary>The STFT window length (2048).</summary>
    public int WindowSize { get; set; } = 2048;

    /// <summary>The training segment the decoder synthesizes, in samples (16384).</summary>
    public int SegmentSize { get; set; } = 16384;

    /// <summary>Rows of the speaker table (256).</summary>
    public int NumSpeakers { get; set; } = 256;

    /// <summary>The speaker embedding width, gin_channels (256).</summary>
    public int SpeakerChannels { get; set; } = 256;

    /// <summary>Transformer layers in each flow coupling, n_layers_trans_flow (3).</summary>
    public int FlowTransformerLayers { get; set; } = 3;

    /// <summary>The text encoder's feed-forward kernel (3).</summary>
    public int KernelSize { get; set; } = 3;

    /// <summary>The decoder's residual-block kernels (3, 7, 11).</summary>
    public int[] ResblockKernelSizes { get; set; } = { 3, 7, 11 };

    /// <summary>The decoder's residual-block dilations ((1, 3, 5) for each kernel).</summary>
    public int[][] ResblockDilationSizes { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The decoder's upsampling rates (8, 8, 2, 2, 2); their product is the hop size.</summary>
    public int[] UpsampleRates { get; set; } = { 8, 8, 2, 2, 2 };

    /// <summary>The decoder's upsampling kernels (16, 16, 8, 2, 2).</summary>
    public int[] UpsampleKernelSizes { get; set; } = { 16, 16, 8, 2, 2 };

    /// <summary>The decoder's first width (512).</summary>
    public int UpsampleInitialChannels { get; set; } = 512;

    /// <summary>Dropout in the stochastic and deterministic duration predictors (0.5).</summary>
    public double DurationDropout { get; set; } = 0.5;

    /// <summary>Whether VITS2's duration discriminator trains alongside (true).</summary>
    public bool UseDurationDiscriminator { get; set; } = true;

    /// <summary>A divisor on every multi-period discriminator width (1: MeloTTS's; larger values shrink it for tests).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>The duration discriminator's dropout (0.1).</summary>
    public double DurationDiscriminatorDropout { get; set; } = 0.1;

    /// <summary>The alignment search's starting noise scale (0.01).</summary>
    public double MasNoiseInitial { get; set; } = 0.01;

    /// <summary>The alignment noise scale's decrease per training step (2e-6).</summary>
    public double MasNoiseDecay { get; set; } = 2e-6;

    /// <summary>The width of the Chinese BERT features (chinese-roberta-wwm-ext-large: 1024).</summary>
    public int BertChannels { get; set; } = 1024;

    /// <summary>The width of every other language's BERT features (768).</summary>
    public int JaBertChannels { get; set; } = 768;

    /// <summary>The BERT's layer count; null for the language's released model (24 Chinese, 12 others).</summary>
    public int? BertLayers { get; set; }

    /// <summary>The BERT's attention heads; null for the language's released model (16 Chinese, 12 others).</summary>
    public int? BertHeads { get; set; }

    /// <summary>The BERT's feed-forward width; null for the language's released model (4096 Chinese, 3072 others).</summary>
    public int? BertIntermediate { get; set; }

    /// <summary>AdamW's β₁ (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>AdamW's β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW's ε (1e-9).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>The learning rate's decay per epoch (0.999875).</summary>
    public double LearningRateDecay { get; set; } = 0.999875;

    /// <summary>
    /// The updates in one epoch, after which the learning rate decays once (2084: LJSpeech's 12,500 clips at MeloTTS's
    /// batch of 6; set it for your dataset).
    /// </summary>
    public int UpdatesPerEpoch { get; set; } = 2084;

    /// <summary>The mel reconstruction loss weight c_mel (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>The KL loss weight c_kl (1).</summary>
    public double KlLossWeight { get; set; } = 1.0;

    /// <summary>The stochastic predictor's share of the durations at synthesis (0.2).</summary>
    public double SdpRatio { get; set; } = 0.2;

    /// <summary>The prior's noise scale at synthesis (0.6).</summary>
    public double NoiseScale { get; set; } = 0.6;

    /// <summary>The stochastic duration predictor's noise scale at synthesis (0.8).</summary>
    public double NoiseScaleW { get; set; } = 0.8;

    /// <summary>The speaking speed (1.0); durations scale by its inverse.</summary>
    public double Speed { get; set; } = 1.0;

    /// <summary>The speaker synthesis uses (0).</summary>
    public int SpeakerId { get; set; }

    /// <summary>The seed of training's and synthesis's random draws (0).</summary>
    public int SamplingSeed { get; set; }
}
