namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Which of SpeechT5's objectives <see cref="SpeechT5{T}"/> trains.</summary>
public enum SpeechT5Stage
{
    /// <summary>
    /// The joint pre-training (Ao et al. 2022 §2.3): a sample with a recording and its HuBERT labels trains the speech
    /// objectives (masked prediction, the feature penalty, the speech decoder's reconstruction, the codebook); a sample
    /// with text alone trains BART's text infilling. Both mix the encoder states with the cross-modal codebook.
    /// </summary>
    Pretraining,

    /// <summary>Text-to-speech fine-tuning (§3.4): the speech decoder's L1, the stop token's BCE and the guided
    /// attention loss on recordings and their text.</summary>
    TextToSpeech,
}

/// <summary>When SpeechT5's generation stops.</summary>
public enum SpeechT5StopCriterion
{
    /// <summary>The official <c>generate_speech</c>: when any of a step's r stop probabilities reaches the
    /// threshold.</summary>
    AnyFrame,

    /// <summary>The Hugging Face port's <c>_generate_speech</c>: when a step's r stop probabilities summed reach it.</summary>
    SummedProbability,
}

/// <summary>Options for <see cref="SpeechT5{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's base model (Ao et al. 2022: 12 encoder and 6 decoder layers of width 768, 12
/// heads, feed-forward 3072, relative position embeddings; 100 × 2 codebook entries mixed into 10% of the states) with
/// the official recipe filling what the paper leaves out (<c>t5_transformer_base</c>, the pre-training and TTS
/// commands), and the paper's training rates: pre-training warms up over 8% of 500k updates to 2e-4 then decays
/// linearly; text-to-speech fine-tuning warms up over 10k of 120k steps to 4e-4 then decays by the inverse square root.
/// The released weights (<c>microsoft/speecht5_tts</c>) use these sizes.</para>
/// <para><b>For Beginners:</b> SpeechT5 is one encoder-decoder pre-trained on both speech and text and fine-tuned to read
/// text aloud in the voice of a speaker embedding (an x-vector computed from a recording). These settings size the
/// networks and control training and generation.</para>
/// </remarks>
public class SpeechT5Options : TtsModelOptions
{
    /// <summary>Creates the paper's base configuration.</summary>
    public SpeechT5Options()
    {
        SampleRate = 16000;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 768;
        NumEncoderLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        VocabSize = 81;
        MaxTextLength = 600;
        MaxMelLength = 1876;
        DropoutRate = 0.15;
        LearningRate = 4e-4;
        WeightDecay = 0.0;
    }

    /// <summary>The joint pre-training's configuration (the recipe's <c>t5_transformer_base</c>: dropout 0.1, LayerDrop
    /// 0.05 in both stacks), training <see cref="SpeechT5Stage.Pretraining"/>.</summary>
    public static SpeechT5Options Pretraining() => new()
    {
        DropoutRate = 0.1,
        AttentionDropout = 0.1,
        ActivationDropout = 0.1,
        EncoderLayerDrop = 0.05,
        DecoderLayerDrop = 0.05,
        CurrentStage = SpeechT5Stage.Pretraining,
    };

    // ---------------------------------------------------------------- the shared encoder-decoder

    /// <summary>Feed-forward width (3072).</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Clipping distance of the encoder's relative position embedding (160).</summary>
    public int MaxRelativePosition { get; set; } = 160;

    /// <summary>Attention-weight dropout (0.15, the TTS recipe's).</summary>
    public double AttentionDropout { get; set; } = 0.15;

    /// <summary>Dropout after the feed-forward activation (0.15, the TTS recipe's).</summary>
    public double ActivationDropout { get; set; } = 0.15;

    /// <summary>Dropout after the scaled positional encodings (0.1).</summary>
    public double PositionalDropout { get; set; } = 0.1;

    /// <summary>LayerDrop of the encoder layers (0 in the TTS recipe; 0.05 in pre-training).</summary>
    public double EncoderLayerDrop { get; set; }

    /// <summary>LayerDrop of the decoder layers (0 in the TTS recipe; 0.05 in pre-training).</summary>
    public double DecoderLayerDrop { get; set; }

    // ---------------------------------------------------------------- text

    /// <summary>The character SentencePiece model (<c>spm_char.model</c>), which turning text into tokens needs.</summary>
    public string? SentencePieceModelPath { get; set; }

    /// <summary>The padding token (1).</summary>
    public int PadId { get; set; } = 1;

    /// <summary>The end-of-sentence token, appended to every text (2).</summary>
    public int EosId { get; set; } = 2;

    /// <summary>The beginning-of-sentence token, which the text pre-training wraps around its text (0).</summary>
    public int BosId { get; set; }

    /// <summary>The mask token of the text infilling (79, <c>&lt;mask&gt;</c>).</summary>
    public int MaskId { get; set; } = 79;

    // ---------------------------------------------------------------- the speech pre- and post-nets

    /// <summary>Mel frames per decoder step (2).</summary>
    public int ReductionFactor { get; set; } = 2;

    /// <summary>Width of the speaker x-vector (512).</summary>
    public int SpeakerDim { get; set; } = 512;

    /// <summary>The speech decoder pre-net's units (256).</summary>
    public int PrenetUnits { get; set; } = 256;

    /// <summary>The speech decoder pre-net's layers (2).</summary>
    public int PrenetLayers { get; set; } = 2;

    /// <summary>The speech decoder pre-net's always-on dropout (0.5, Tacotron 2's).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>The post-net's channels (256).</summary>
    public int PostnetChannels { get; set; } = 256;

    /// <summary>The post-net's convolutions (5).</summary>
    public int PostnetLayers { get; set; } = 5;

    /// <summary>The post-net's kernel (5).</summary>
    public int PostnetKernel { get; set; } = 5;

    /// <summary>The post-net's dropout (0.5).</summary>
    public double PostnetDropout { get; set; } = 0.5;

    /// <summary>The speech encoder pre-net's convolutions as (channels, kernel, stride): wav2vec 2.0's seven.</summary>
    public (int Channels, int Kernel, int Stride)[] FeatureLayers { get; set; } =
        { (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2) };

    /// <summary>The convolutional position embedding's kernel (128).</summary>
    public int ConvPositionKernel { get; set; } = 128;

    /// <summary>The convolutional position embedding's groups (16).</summary>
    public int ConvPositionGroups { get; set; } = 16;

    // ---------------------------------------------------------------- pre-training

    /// <summary>Width of the HuBERT post-net's projection (256).</summary>
    public int FinalDim { get; set; } = 256;

    /// <summary>HuBERT label classes per label set (one set of 500 k-means units plus fairseq's 4 special symbols,
    /// 504).</summary>
    public int[] HubertClasses { get; set; } = { 504 };

    /// <summary>HuBERT labels per second (50).</summary>
    public double LabelRate { get; set; } = 50;

    /// <summary>The span masking's start probability (0.8 → ≈8% of frames start a span, §2.3).</summary>
    public double MaskProbability { get; set; } = 0.8;

    /// <summary>The masked spans' length (10).</summary>
    public int MaskLength { get; set; } = 10;

    /// <summary>The longest pre-training recording in samples, randomly cropped to (250000).</summary>
    public int MaxSpeechSamples { get; set; } = 250000;

    /// <summary>Weight of the feature penalty (10).</summary>
    public double FeaturePenaltyWeight { get; set; } = 10;

    /// <summary>Weight of the codebook diversity loss (0.1).</summary>
    public double DiversityWeight { get; set; } = 0.1;

    /// <summary>Codebook entries per group (100).</summary>
    public int LatentVars { get; set; } = 100;

    /// <summary>Codebook groups (2).</summary>
    public int LatentGroups { get; set; } = 2;

    /// <summary>Fraction of encoder states replaced by their quantized vectors (0.1).</summary>
    public double CodebookProbability { get; set; } = 0.1;

    /// <summary>Fraction of text tokens masked by the infilling (0.3).</summary>
    public double TextMaskRatio { get; set; } = 0.3;

    /// <summary>Fraction of masks replaced by random tokens instead (0.1).</summary>
    public double TextRandomRatio { get; set; } = 0.1;

    /// <summary>The Poisson λ of the infilling's span lengths (3.5).</summary>
    public double PoissonLambda { get; set; } = 3.5;

    /// <summary>Whether a fresh model draws its weights as the official pre-training's <c>--bert-init</c> does
    /// (N(0, 0.02) linear layers and embeddings, zero biases; true).</summary>
    public bool BertInitialization { get; set; } = true;

    // ---------------------------------------------------------------- training

    /// <summary>The objective <see cref="SpeechT5{T}"/> trains (text-to-speech fine-tuning).</summary>
    public SpeechT5Stage CurrentStage { get; set; } = SpeechT5Stage.TextToSpeech;

    /// <summary>Pre-training's peak learning rate (2e-4, §3.1).</summary>
    public double PretrainLearningRate { get; set; } = 2e-4;

    /// <summary>Pre-training updates (500k, §3.1); the rate warms up over the first 8% and decays linearly to 0.</summary>
    public int PretrainUpdates { get; set; } = 500000;

    /// <summary>Fraction of pre-training updates spent warming up (0.08).</summary>
    public double PretrainWarmupFraction { get; set; } = 0.08;

    /// <summary>Pre-training's decoupled weight decay (0.01, the recipe's).</summary>
    public double PretrainWeightDecay { get; set; } = 0.01;

    /// <summary>Pre-training's Adam ε (1e-6, the recipe's).</summary>
    public double PretrainEpsilon { get; set; } = 1e-6;

    /// <summary>Pre-training's gradient-norm clip (5, the recipe's).</summary>
    public double PretrainMaxGradientNorm { get; set; } = 5.0;

    /// <summary>Text-to-speech warm-up steps before the inverse-square-root decay (10k, Appendix B.2).</summary>
    public int WarmupUpdates { get; set; } = 10000;

    /// <summary>Adam β1 (0.9).</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Adam β2 (0.98).</summary>
    public double OptimizerBeta2 { get; set; } = 0.98;

    /// <summary>Text-to-speech's Adam ε (1e-8, the recipe's default).</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

    /// <summary>Text-to-speech's gradient-norm clip (25, the recipe's).</summary>
    public double MaxGradientNorm { get; set; } = 25.0;

    /// <summary>Text-to-speech updates during which the shared encoder stays frozen (20k, the recipe's
    /// <c>freeze_encoder_updates</c>).</summary>
    public int FreezeEncoderUpdates { get; set; } = 20000;

    /// <summary>Weight of the stop token's positive class in its BCE (5).</summary>
    public double StopPositiveWeight { get; set; } = 5.0;

    /// <summary>Heads of every decoder layer's cross-attention the guided attention loss applies to (2).</summary>
    public int GuidedAttentionHeads { get; set; } = 2;

    /// <summary>The guided attention loss's σ (0.4).</summary>
    public double GuidedAttentionSigma { get; set; } = 0.4;

    /// <summary>Seed of the training draws (dropout, masks, crops and the codebook's Gumbel noise).</summary>
    public int TrainingSeed { get; set; } = 1337;

    // ---------------------------------------------------------------- generation

    /// <summary>Stop probability that ends generation (0.5).</summary>
    public double StopThreshold { get; set; } = 0.5;

    /// <summary>The longest generation, in decoder steps per text token times r (20).</summary>
    public double MaxLengthRatio { get; set; } = 20.0;

    /// <summary>When generation stops (the official code's: any frame).</summary>
    public SpeechT5StopCriterion StopCriterion { get; set; } = SpeechT5StopCriterion.AnyFrame;

    /// <summary>Seed of the generation's pre-net dropout and the x-vector's pooling noise.</summary>
    public int SamplingSeed { get; set; } = 1;

    // ---------------------------------------------------------------- the x-vector extractor

    /// <summary>The x-vector's TDNN channels (512, 512, 512, 512, 1500).</summary>
    public int[] SpeakerEncoderChannels { get; set; } = { 512, 512, 512, 512, 1500 };

    // ---------------------------------------------------------------- the vocoder (microsoft/speecht5_hifigan)

    /// <summary>HiFi-GAN's first width (512).</summary>
    public int VocoderInitialChannels { get; set; } = 512;

    /// <summary>HiFi-GAN's upsampling rates (4, 4, 4, 4; their product is <see cref="TtsModelOptions.HopSize"/>).</summary>
    public int[] VocoderUpsampleRates { get; set; } = { 4, 4, 4, 4 };

    /// <summary>HiFi-GAN's transposed-convolution kernels (8, 8, 8, 8).</summary>
    public int[] VocoderUpsampleKernels { get; set; } = { 8, 8, 8, 8 };

    /// <summary>HiFi-GAN's residual block kernels (3, 7, 11).</summary>
    public int[] VocoderResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>HiFi-GAN's residual block dilations (1, 3, 5 each).</summary>
    public int[][] VocoderResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
}
