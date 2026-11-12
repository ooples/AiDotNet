using AiDotNet.TextToSpeech.Classic;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>How PromptTTS fine-tunes its BERT style encoder on the style-factor classification task (Guo et al. 2023,
/// §2.2, Table 5).</summary>
public enum PromptStyleFineTuning
{
    /// <summary>P-tuning v2 (Liu et al. 2022): BERT stays frozen; trainable key and value prefixes in every layer and the
    /// classification heads learn (the paper's choice, §4.1).</summary>
    PTuningV2,

    /// <summary>Standard fine-tuning (Devlin et al. 2019): every BERT weight and the heads learn (Table 5's
    /// "Standard").</summary>
    Standard,

    /// <summary>No fine-tuning: BERT's pretrained [CLS] state is the style representation (Table 5's ablation).</summary>
    None,
}

/// <summary>The network a PromptTTS training step updates.</summary>
public enum PromptTTSTrainingStage
{
    /// <summary>The style encoder, on the auxiliary style-factor classification task (§2.2).</summary>
    StyleClassification,

    /// <summary>The content encoder, variance adaptor, speech decoder and the style projection, on FastSpeech 2's
    /// objective.</summary>
    Synthesis,
}

/// <summary>Options for PromptTTS (Guo et al. 2023, "PromptTTS: Controllable Text-to-Speech with Text
/// Descriptions").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's (§4.1): a 12-layer BERT-base style encoder fine-tuned with P-tuning v2 on the five style
/// factors of PromptSpeech (gender 2, pitch 3, speaking speed 3, volume 3 and emotion 5 classes, §3), style and content
/// representations of width 256, a content encoder of 4 Transformer blocks with FastSpeech 2's variance adaptor, and a
/// speech decoder with the same blocks. Everything the paper takes from FastSpeech 2 (block sizes, the variance
/// adaptor, the loss and the optimizer) is <see cref="FastSpeech2Options"/>'. The P-tuning v2 settings the paper does
/// not give are the reference code's argument defaults (THUDM/P-tuning-v2: 4 prefix positions, no prefix projection,
/// dropout 0.1, Hugging Face Trainer's AdamW at 5e-5); its own task scripts use 16 to 128 positions at 5e-3 to 2e-2,
/// so set them for your prompts.
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of PromptTTS's networks and how its style encoder is tuned.
/// The defaults reproduce the paper.</para>
/// </remarks>
public class PromptTTSOptions : FastSpeech2Options
{
    /// <summary>Creates the paper's options.</summary>
    public PromptTTSOptions()
    {
        HiddenDim = 256;
        EncoderDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public PromptTTSOptions(PromptTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        StyleVocabularySize = other.StyleVocabularySize;
        StyleHiddenDim = other.StyleHiddenDim;
        StyleLayers = other.StyleLayers;
        StyleHeads = other.StyleHeads;
        StyleIntermediateDim = other.StyleIntermediateDim;
        StyleMaxPositions = other.StyleMaxPositions;
        StyleDropout = other.StyleDropout;
        StyleFineTuning = other.StyleFineTuning;
        PrefixLength = other.PrefixLength;
        PrefixProjection = other.PrefixProjection;
        PrefixHiddenSize = other.PrefixHiddenSize;
        StyleFactorClasses = (int[])other.StyleFactorClasses.Clone();
        StyleLearningRate = other.StyleLearningRate;
        StyleTrainingSteps = other.StyleTrainingSteps;
        TrainStyleEncoderWithSynthesis = other.TrainStyleEncoderWithSynthesis;
        SamplingSeed = other.SamplingSeed;
    }

    // ---------------------------------------------------------------- style encoder (BERT-base)

    /// <summary>BERT's vocabulary (30,522: bert-base-uncased's WordPiece table).</summary>
    [AiDotNet.Attributes.NotScaled("holds every id of the bert-base-uncased tokenizer that tokenizes the style prompt")]
    public int StyleVocabularySize { get; set; } = 30_522;

    /// <summary>BERT's width (768).</summary>
    public int StyleHiddenDim { get; set; } = 768;

    /// <summary>BERT's layers (12).</summary>
    public int StyleLayers { get; set; } = 12;

    /// <summary>BERT's attention heads (12).</summary>
    public int StyleHeads { get; set; } = 12;

    /// <summary>BERT's feed-forward width (3072).</summary>
    public int StyleIntermediateDim { get; set; } = 3072;

    /// <summary>BERT's position table (512).</summary>
    public int StyleMaxPositions { get; set; } = 512;

    /// <summary>BERT's dropout, also on the prefixes and before the classifiers (0.1).</summary>
    public double StyleDropout { get; set; } = 0.1;

    /// <summary>How the style encoder is fine-tuned (P-tuning v2, §4.1).</summary>
    public PromptStyleFineTuning StyleFineTuning { get; set; } = PromptStyleFineTuning.PTuningV2;

    /// <summary>P-tuning v2's prefix positions per layer (4, the reference code's default; the paper gives none).</summary>
    public int PrefixLength { get; set; } = 4;

    /// <summary>Whether P-tuning v2 reparameterizes the prefixes through a two-layer MLP (off, the reference's
    /// default).</summary>
    public bool PrefixProjection { get; set; }

    /// <summary>Hidden width of that MLP (512, the reference's default).</summary>
    public int PrefixHiddenSize { get; set; } = 512;

    /// <summary>Classes of each style factor the auxiliary task predicts: gender 2, pitch 3, speaking speed 3, volume 3,
    /// emotion 5 (PromptSpeech's synthesized version, §3; the real version has no emotion).</summary>
    public int[] StyleFactorClasses { get; set; } = { 2, 3, 3, 3, 5 };

    /// <summary>Learning rate of the style-classification stage (5e-5, Hugging Face Trainer's default, which the
    /// P-tuning v2 code trains with).</summary>
    public double StyleLearningRate { get; set; } = 5e-5;

    /// <summary>Updates over which the style stage's learning rate decays linearly to zero, as the Trainer schedules it
    /// over the run (0 keeps it constant: the run's length depends on your data).</summary>
    public int StyleTrainingSteps { get; set; }

    /// <summary>Whether the synthesis loss also updates the style encoder (off: the paper's ablation, where an
    /// unfine-tuned BERT loses most of the style accuracy, shows that synthesis training does not adapt it).</summary>
    public bool TrainStyleEncoderWithSynthesis { get; set; }

    /// <summary>Seed of the dropout draws.</summary>
    public int SamplingSeed { get; set; }
}
