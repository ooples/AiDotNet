namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for <see cref="EmotiVoice{T}"/>.</summary>
/// <remarks>
/// <para>EmotiVoice has no paper: the defaults are its released configuration (netease-youdao/EmotiVoice
/// <c>config/joint/config.py</c> and <c>config.yaml</c>): 16 kHz audio, 80 mel bins up to 8 kHz from a 1024-point FFT with
/// a 256-sample hop; 384-wide four-block Transformer encoder and decoder (8 heads, kernel-3 feed-forward of 1536,
/// dropout 0.2); 2014 speakers and 502 phoneme tokens; SimBERT (BERT-base Chinese, 13,685 tokens) embeddings of the
/// emotion prompt and of the text; HiFi-GAN V1 (upsampling 8·8·2·2) trained jointly on 32-frame segments; Adam
/// (rate 1.25e-5, β = 0.5, 0.9) with an exponential decay of 0.999875 per epoch.</para>
/// <para><b>For Beginners:</b> These settings size EmotiVoice's networks; the defaults reproduce the released model.</para>
/// </remarks>
public class EmotiVoiceOptions : TtsModelOptions
{
    /// <summary>Creates the released configuration.</summary>
    public EmotiVoiceOptions()
    {
        SampleRate = 16000;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        HiddenDim = 384;
        VocabSize = 502;
        DropoutRate = 0.2;
        LearningRate = 1.25e-5;
        WeightDecay = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public EmotiVoiceOptions(EmotiVoiceOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        WindowLength = other.WindowLength;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        SpeakerCount = other.SpeakerCount;
        Heads = other.Heads;
        EncoderLayers = other.EncoderLayers;
        DecoderLayers = other.DecoderLayers;
        FeedForwardKernel = other.FeedForwardKernel;
        VarianceChannels = other.VarianceChannels;
        PitchLayers = other.PitchLayers;
        EnergyLayers = other.EnergyLayers;
        VarianceKernel = other.VarianceKernel;
        VarianceDropout = other.VarianceDropout;
        DurationLayers = other.DurationLayers;
        DurationKernel = other.DurationKernel;
        DurationDropout = other.DurationDropout;
        VarianceEmbedKernel = other.VarianceEmbedKernel;
        StyleBert = other.StyleBert with { };
        EmotionLabels = other.EmotionLabels;
        PitchLabels = other.PitchLabels;
        SpeedLabels = other.SpeedLabels;
        EnergyLabels = other.EnergyLabels;
        UpsampleRates = (int[])other.UpsampleRates.Clone();
        UpsampleKernels = (int[])other.UpsampleKernels.Clone();
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        ResBlockKernels = (int[])other.ResBlockKernels.Clone();
        ResBlockDilations = other.ResBlockDilations.Select(d => (int[])d.Clone()).ToArray();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        SegmentFrames = other.SegmentFrames;
        PitchMean = other.PitchMean;
        PitchStd = other.PitchStd;
        EnergyMean = other.EnergyMean;
        EnergyStd = other.EnergyStd;
        TrimSilence = other.TrimSilence;
        MelLossWeight = other.MelLossWeight;
        ForwardSumWeight = other.ForwardSumWeight;
        BinarizationWeight = other.BinarizationWeight;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The analysis window, in samples (1024).</summary>
    public int WindowLength { get; set; } = 1024;

    /// <summary>Lowest mel frequency (0 Hz).</summary>
    public double MelMinFrequency { get; set; }

    /// <summary>Highest mel frequency (8000 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;

    /// <summary>Speakers in the speaker table (2014, the released list).</summary>
    public int SpeakerCount { get; set; } = 2014;

    /// <summary>Attention heads of the encoder and decoder (8).</summary>
    public int Heads { get; set; } = 8;

    /// <summary>Encoder blocks (4).</summary>
    public int EncoderLayers { get; set; } = 4;

    /// <summary>Decoder blocks (4).</summary>
    public int DecoderLayers { get; set; } = 4;

    /// <summary>Kernel of the blocks' convolutional feed-forward (3); its width is 4 × <see cref="TtsModelOptions.HiddenDim"/>.</summary>
    public int FeedForwardKernel { get; set; } = 3;

    /// <summary>Channels of the variance predictors (384).</summary>
    public int VarianceChannels { get; set; } = 384;

    /// <summary>Convolutions of the pitch predictor (3).</summary>
    public int PitchLayers { get; set; } = 3;

    /// <summary>Convolutions of the energy predictor (2).</summary>
    public int EnergyLayers { get; set; } = 2;

    /// <summary>Kernel of the pitch and energy predictors (3).</summary>
    public int VarianceKernel { get; set; } = 3;

    /// <summary>Dropout of the pitch and energy predictors (0.1).</summary>
    public double VarianceDropout { get; set; } = 0.1;

    /// <summary>Convolutions of the duration predictor (2).</summary>
    public int DurationLayers { get; set; } = 2;

    /// <summary>Kernel of the duration predictor (3).</summary>
    public int DurationKernel { get; set; } = 3;

    /// <summary>Dropout of the duration predictor (0.5).</summary>
    public double DurationDropout { get; set; } = 0.5;

    /// <summary>Kernel of the pitch and energy embedding convolutions (9).</summary>
    public int VarianceEmbedKernel { get; set; } = 9;

    /// <summary>SimBERT (WangZeJun/simbert-base-chinese): BERT-base with a 13,685-token Chinese vocabulary.</summary>
    internal AiDotNet.NeuralNetworks.BertConfiguration StyleBert { get; set; } = new() { VocabSize = 13685 };

    /// <summary>SimBERT's width (768).</summary>
    public int StyleBertHidden
    {
        get => StyleBert.Hidden;
        set => StyleBert = StyleBert with { Hidden = value };
    }

    /// <summary>SimBERT's layers (12).</summary>
    public int StyleBertLayers
    {
        get => StyleBert.Layers;
        set => StyleBert = StyleBert with { Layers = value };
    }

    /// <summary>SimBERT's heads (12).</summary>
    public int StyleBertHeads
    {
        get => StyleBert.Heads;
        set => StyleBert = StyleBert with { Heads = value };
    }

    /// <summary>SimBERT's feed-forward width (3072).</summary>
    public int StyleBertIntermediate
    {
        get => StyleBert.Intermediate;
        set => StyleBert = StyleBert with { Intermediate = value };
    }

    /// <summary>Emotion classes of the style encoder's pre-training (7: 普通 生气 开心 惊讶 悲伤 厌恶 恐惧).</summary>
    public int EmotionLabels { get; set; } = 7;

    /// <summary>Pitch classes (3: normal, high, low).</summary>
    public int PitchLabels { get; set; } = 3;

    /// <summary>Speed classes (3).</summary>
    public int SpeedLabels { get; set; } = 3;

    /// <summary>Energy classes (3).</summary>
    public int EnergyLabels { get; set; } = 3;

    /// <summary>HiFi-GAN's upsampling factors (8, 8, 2, 2); their product is the hop.</summary>
    public int[] UpsampleRates { get; set; } = { 8, 8, 2, 2 };

    /// <summary>HiFi-GAN's upsampling kernels (16, 16, 4, 4).</summary>
    public int[] UpsampleKernels { get; set; } = { 16, 16, 4, 4 };

    /// <summary>HiFi-GAN's first width (512).</summary>
    public int UpsampleInitialChannels { get; set; } = 512;

    /// <summary>HiFi-GAN's residual-block kernels (3, 7, 11).</summary>
    public int[] ResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>Their dilations ({1, 3, 5} each).</summary>
    public int[][] ResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>A divisor on every width of HiFi-GAN's multi-period and multi-scale discriminators (1: the released 32–1024
    /// channels).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Frames of the random segment the vocoder trains on (32).</summary>
    public int SegmentFrames { get; set; } = 32;

    /// <summary>Pitch normalization mean (225.089 Hz).</summary>
    public double PitchMean { get; set; } = 225.089;

    /// <summary>Pitch normalization standard deviation (53.78).</summary>
    public double PitchStd { get; set; } = 53.78;

    /// <summary>Energy normalization mean (30.610).</summary>
    public double EnergyMean { get; set; } = 30.610;

    /// <summary>Energy normalization standard deviation (21.78).</summary>
    public double EnergyStd { get; set; } = 21.78;

    /// <summary>Whether training trims the recording's silence (below 0.5 % of its peak) and pads 50 ms on each side, as the
    /// released data loader does (true).</summary>
    public bool TrimSilence { get; set; } = true;

    /// <summary>Weight of the mel L1 loss (45).</summary>
    public double MelLossWeight { get; set; } = 45;

    /// <summary>Weight of the forward-sum loss (2).</summary>
    public double ForwardSumWeight { get; set; } = 2;

    /// <summary>Weight of the binarization loss (2).</summary>
    public double BinarizationWeight { get; set; } = 2;

    /// <summary>Adam β₁ (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Adam β₂ (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>Adam ε (1e-9).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>Learning-rate decay per epoch (0.999875).</summary>
    public double LearningRateDecay { get; set; } = 0.999875;

    /// <summary>Updates per epoch, after which the learning rate decays (0: never; the released code decays once per epoch
    /// of your data — set it for yours).</summary>
    public int UpdatesPerEpoch { get; set; }

    /// <summary>Seed of the training draws and of dropout.</summary>
    public int SamplingSeed { get; set; }
}
