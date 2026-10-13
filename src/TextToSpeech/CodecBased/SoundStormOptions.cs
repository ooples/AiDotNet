using System;
using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SoundStorm (Borsos et al. 2023, "SoundStorm: Efficient Parallel Audio Generation").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's acoustic generator (§4.1): a 350M-parameter bidirectional Conformer (12 layers,
/// 16 heads, width 1024, feed-forward 4096, convolution kernel 5, rotary positions) over a SoundStream codec at
/// 50 frames per second with Q = 12 RVQ levels of 1024 codes, conditioned on semantic tokens (1024 k-means clusters at
/// 25 Hz, each duplicated to the codec's 50 Hz). Decoding runs (16, 1, …, 1) iterations: 16 confidence-based
/// iterations on the first level, whose last iteration is greedy (§3.3), then one greedy pass per finer level.
/// </para>
/// <para>
/// Synthesis from text uses the paper's text-to-semantic model (§5): ByT5-large (36 encoder and 12 decoder layers,
/// width 1536, feed-forward 3840, Xue et al. 2022), sampling semantic tokens at temperature 0.9 from the top 125. Only
/// its decoder trains; the encoder comes from a text-pretrained ByT5 checkpoint (<see cref="SoundStorm{T}.LoadTextEncoderWeights"/>).
/// </para>
/// <para>
/// The paper gives no optimizer, dropout or convolution expansion, and no released code exists. Those gaps follow
/// lucidrains/soundstorm-pytorch: no dropout, convolution expansion 2, a learned mask embedding, and its trainer's
/// Adam (β = 0.9, 0.99, ε = 1e-8, no weight decay) at 3e-4 after a linear warmup from 1e-5, decayed by a cosine over
/// the training steps, with gradients clipped at norm 0.5. Its trainer requires the warmup and total steps and gives
/// neither a default, so <see cref="WarmupSteps"/> and <see cref="TrainingSteps"/> are left for you to set; while they
/// are 0 the learning rate stays at <see cref="TtsModelOptions.LearningRate"/>.
/// </para>
/// <para><b>For Beginners:</b> SoundStorm turns "what is said" tokens into the audio codec's tokens all at once
/// instead of one by one, which makes it about a hundred times faster than generating them in order.</para>
/// </remarks>
public class SoundStormOptions : TtsModelOptions
{
    /// <summary>Creates the paper's options (§4.1, §5).</summary>
    public SoundStormOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        HiddenDim = 1024;
        NumHeads = 16;
        NumEncoderLayers = 36;
        NumDecoderLayers = 12;
        DropoutRate = 0.0;
        LearningRate = 3e-4;
        WeightDecay = 0.0;
        MaxTextLength = 1024;
        // AudioLM's SoundStream (Borsos et al. 2022): 16 kHz, strides (2, 4, 5, 8) for 50 frames per second, 12 RVQ
        // levels of 1024 codes.
        SoundStream = new SoundStreamOptions { SampleRate = 16000, NumQuantizers = 12, CodebookSize = 1024, TargetBandwidthKbps = 6.0 };
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public SoundStormOptions(SoundStormOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ConformerLayers = other.ConformerLayers;
        ConformerHeadDim = other.ConformerHeadDim;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        ConvExpansion = other.ConvExpansion;
        ConvKernel = other.ConvKernel;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        SemanticCodes = other.SemanticCodes;
        SemanticFrameRate = other.SemanticFrameRate;
        FirstLevelIterations = other.FirstLevelIterations;
        TextModelDim = other.TextModelDim;
        TextFeedForwardDim = other.TextFeedForwardDim;
        TextKeyValueDim = other.TextKeyValueDim;
        TextDropoutRate = other.TextDropoutRate;
        Temperature = other.Temperature;
        TopK = other.TopK;
        MaxNewSemanticTokens = other.MaxNewSemanticTokens;
        WarmupStartLearningRate = other.WarmupStartLearningRate;
        WarmupSteps = other.WarmupSteps;
        TrainingSteps = other.TrainingSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        SoundStream = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.SoundStream);
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>
    /// The paper's dialogue system (§5): semantic tokens from 4096 k-means clusters of a 0.6B BEST-RQ at 25 Hz, over a
    /// 24 kHz SoundStream at 50 frames per second with 12 RVQ levels.
    /// </summary>
    /// <param name="codecStrides">The codec's strides in decoder order. The paper gives the frame rate but not the
    /// strides, so they are left for you to set; their product must be 480 (24 000 / 50).</param>
    public static SoundStormOptions DialogueSynthesis(int[] codecStrides)
    {
        if (codecStrides is null) throw new ArgumentNullException(nameof(codecStrides));
        int hop = 1;
        foreach (int stride in codecStrides) hop *= stride;
        if (hop != 480)
            throw new ArgumentException($"The strides give a {hop}-sample hop; 24 kHz at 50 frames per second needs 480.", nameof(codecStrides));
        return new SoundStormOptions
        {
            SampleRate = 24000,
            HopSize = 480,
            SemanticCodes = 4096,
            SoundStream = new SoundStreamOptions
            {
                SampleRate = 24000, NumQuantizers = 12, CodebookSize = 1024, TargetBandwidthKbps = 6.0,
                Ratios = (int[])codecStrides.Clone(),
            },
        };
    }

    // ---------------------------------------------------------------- acoustic Conformer (§4.1)

    /// <summary>Conformer layers (12).</summary>
    public int ConformerLayers { get; set; } = 12;

    /// <summary>Width of each attention head (64: 16 heads over width 1024).</summary>
    public int ConformerHeadDim { get; set; } = 64;

    /// <summary>Feed-forward width as a multiple of the model width (4: 4096 over 1024).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Expansion of the convolution module (2, from lucidrains/soundstorm-pytorch; the paper does not say).</summary>
    public int ConvExpansion { get; set; } = 2;

    /// <summary>Convolution kernel size (5).</summary>
    public int ConvKernel { get; set; } = 5;

    // ---------------------------------------------------------------- tokens (§4.1)

    /// <summary>Q, the RVQ levels modeled (12).</summary>
    public int NumCodebooks { get; set; } = 12;

    /// <summary>C, the codes per RVQ level (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Semantic token vocabulary (1024 w2v-BERT k-means clusters).</summary>
    public int SemanticCodes { get; set; } = 1024;

    /// <summary>Semantic tokens per second (25); each is duplicated to the codec's frame rate.</summary>
    public int SemanticFrameRate { get; set; } = 25;

    /// <summary>Confidence-based iterations on the first RVQ level (16); every finer level takes one greedy pass.</summary>
    public int FirstLevelIterations { get; set; } = 16;

    // ---------------------------------------------------------------- text-to-semantic (§5, ByT5-large)

    /// <summary>ByT5's width (1536).</summary>
    public int TextModelDim { get; set; } = 1536;

    /// <summary>ByT5's feed-forward width (3840).</summary>
    public int TextFeedForwardDim { get; set; } = 3840;

    /// <summary>ByT5's attention head width (64).</summary>
    public int TextKeyValueDim { get; set; } = 64;

    /// <summary>ByT5's dropout (0.1, the released checkpoints' <c>dropout_rate</c>).</summary>
    public double TextDropoutRate { get; set; } = 0.1;

    /// <summary>Sampling temperature for semantic tokens (0.9).</summary>
    public double Temperature { get; set; } = 0.9;

    /// <summary>Semantic tokens are sampled from the top K (125).</summary>
    public int TopK { get; set; } = 125;

    /// <summary>The most semantic tokens generated (750: 30 seconds at 25 Hz, §5).</summary>
    public int MaxNewSemanticTokens { get; set; } = 750;

    // ---------------------------------------------------------------- optimization (lucidrains/soundstorm-pytorch)

    /// <summary>Learning rate at the start of the warmup (1e-5).</summary>
    public double WarmupStartLearningRate { get; set; } = 1e-5;

    /// <summary>Linear warmup steps. Left for you to set (0: no warmup).</summary>
    public int WarmupSteps { get; set; }

    /// <summary>Total training steps the cosine decay spans. Left for you to set (0: no decay).</summary>
    public int TrainingSteps { get; set; }

    /// <summary>Gradient norm clip (0.5).</summary>
    public double MaxGradientNorm { get; set; } = 0.5;

    /// <summary>The SoundStream codec's options.</summary>
    public SoundStreamOptions SoundStream { get; set; }

    /// <summary>Seed of synthesis's and training's random draws.</summary>
    public int SamplingSeed { get; set; } = 1234;
}
