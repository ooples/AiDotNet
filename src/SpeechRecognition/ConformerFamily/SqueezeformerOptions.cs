using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Configuration options for the Squeezeformer speech recognition model.
/// </summary>
/// <remarks>
/// <para>Squeezeformer (Kim et al., 2022) improves Conformer with a temporal U-Net structure,
/// micro-macro design (pre-norm instead of post-norm), and depthwise separable downsampling
/// for efficient computation with better accuracy.</para>
/// </remarks>
public class SqueezeformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SqueezeformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SqueezeformerOptions(SqueezeformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardExpansionFactor = other.FeedForwardExpansionFactor;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        PeakLearningRate = other.PeakLearningRate;
        WarmupSteps = other.WarmupSteps;
        WeightDecay = other.WeightDecay;
        Language = other.Language;
        // CLONED, NOT SHARED. `Vocabulary = other.Vocabulary` hands the copy the SAME array
        // instance, so a later write through either options object is seen by both -- the copy
        // constructor exists precisely to prevent that coupling, and for a reference type a bare
        // assignment does not. Null is preserved as null rather than becoming an empty array,
        // which would silently change "unset" into "set to nothing".
        Vocabulary = (string[])other.Vocabulary.Clone();
        UseLayerNormalization = other.UseLayerNormalization;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 16;
    public int NumAttentionHeads { get; set; } = 8;
    public int FeedForwardExpansionFactor { get; set; } = 4;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the PEAK learning rate for the Noam-annealed schedule.
    /// </summary>
    /// <value>
    /// Defaults to 2e-3, the paper's peak rate for the SMALL variant (appendix A.1 gives 2e-3, 1.5e-3
    /// and {1, 0.5}e-3 for the small, medium and large variants respectively).
    /// </value>
    /// <remarks>
    /// In the paper this is the PEAK of a warmup-then-decay schedule rather than a constant rate. With
    /// <see cref="WarmupSteps"/> at its default of 0 it is applied flat, so for long training runs set
    /// WarmupSteps as well — the paper warms up over 20 epochs before reaching this value, and reports
    /// the architecture failing to converge at comparable peak rates when a stabilizing component is
    /// removed. Gradient clipping is enabled alongside it as a safeguard.
    /// </remarks>
    public double PeakLearningRate { get; set; } = 2e-3;

    /// <summary>
    /// Gets or sets the warmup steps for the Noam-annealed learning-rate schedule.
    /// </summary>
    /// <remarks>
    /// The paper specifies warmup in EPOCHS (20, then holding the peak for a further 160, decaying with
    /// d = 1), which cannot be converted to steps without knowing the dataset size and batch size — it
    /// trained on LibriSpeech-960h at batch 1024/2048. Defaults to 0, meaning NO warmup: a non-zero
    /// default would hold the learning rate near zero for the whole of any short run. Set it from your
    /// own dataset size (20 epochs worth of steps) to reproduce the paper exactly.
    /// </remarks>
    public int WarmupSteps { get; set; } = 0;

    /// <summary>
    /// Gets or sets AdamW's decoupled weight decay.
    /// </summary>
    /// <value>
    /// Defaults to 5e-4 — "We use AdamW optimizer with weight decay 5e-4 for all models"
    /// (appendix A.1). Note this is an order of magnitude below AdamW's usual 0.01 default.
    /// </value>
    public double WeightDecay { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets whether the native encoder normalises with LayerNorm rather
    /// than BatchNorm. Defaults to true, which is what the paper specifies and
    /// what every other normalisation stage in this encoder already uses.
    /// </summary>
    /// <remarks>
    /// BatchNormalization needs a batch to have statistics. At batch size 1 it
    /// falls back to an affine pass-through, so it damps nothing -- and the three
    /// BatchNorm stages in this encoder were the only ones not already LayerNorm.
    /// Measured: the generated fixture trains to a NaN forward pass
    /// (DifferentInputs_AfterTraining reported "L2 distance = NaN", the collapse
    /// signature). The sibling EfficientConformer threads the same flag and does
    /// not fail. Set false to restore the previous BatchNorm topology.
    /// </remarks>
    public bool UseLayerNormalization { get; set; } = true;

    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };
}
