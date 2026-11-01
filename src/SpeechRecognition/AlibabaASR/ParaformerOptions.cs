using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Paraformer: fast and accurate parallel Transformer for non-autoregressive ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Paraformer model. Defaults are FunASR's AISHELL-1 Paraformer recipe: a 12-block Conformer encoder of width 256, a 6-block parallel decoder, CTC weight 0.3, label smoothing 0.1 and glancing ratio 0.4.</para>
/// </remarks>
public class ParaformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ParaformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParaformerOptions(ParaformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        ConvKernel = other.ConvKernel;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        CtcWeight = other.CtcWeight;
        LabelSmoothing = other.LabelSmoothing;
        SamplingRatio = other.SamplingRatio;
        PredictorWeight = other.PredictorWeight;
        TailThreshold = other.TailThreshold;
        WarmupSteps = other.WarmupSteps;
        FrequencyMasks = other.FrequencyMasks;
        FrequencyMaskWidth = other.FrequencyMaskWidth;
        TimeMasks = other.TimeMasks;
        TimeMaskRatio = other.TimeMaskRatio;
        // Copy the array itself, not the reference: a clone that shares the original's vocabulary
        // would see later edits to it, which is not what copying an options object should mean.
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>The encoder's width (256), which the predictor and the decoder share.</summary>
    public int EncoderDim { get; set; } = 256;
    /// <summary>Conformer encoder blocks (12).</summary>
    public int NumEncoderLayers { get; set; } = 12;
    /// <summary>Parallel SAN decoder blocks (6).</summary>
    public int NumDecoderLayers { get; set; } = 6;
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>The encoder's and decoder's feed-forward width (2048), a multiple of <see cref="EncoderDim"/>.</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>The Conformer convolution module's kernel (15).</summary>
    public int ConvKernel { get; set; } = 15;
    public int NumMels { get; set; } = 80;
    /// <summary>Output vocabulary, blank (0), start (1) and end (2) included. The recipe builds its character list from
    /// the training data; the default is the released Mandarin Paraformer's 8,404 tokens.</summary>
    public int VocabSize { get; set; } = 8404;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public string Language { get; set; } = "zh";
    /// <summary>The encoder CTC loss's weight (0.3).</summary>
    public double CtcWeight { get; set; } = 0.3;
    /// <summary>The decoder loss's label smoothing (0.1).</summary>
    public double LabelSmoothing { get; set; } = 0.1;
    /// <summary>The glancing sampler's λ (0.4).</summary>
    public double SamplingRatio { get; set; } = 0.4;
    /// <summary>The MAE token-count loss's weight (1).</summary>
    public double PredictorWeight { get; set; } = 1.0;
    /// <summary>The CIF tail weight at inference (0.45).</summary>
    public double TailThreshold { get; set; } = 0.45;
    /// <summary>The learning rate's warmup steps (30,000, FunASR's warmuplr).</summary>
    public int WarmupSteps { get; set; } = 30000;
    /// <summary>SpecAugment frequency masks (2) of up to <see cref="FrequencyMaskWidth"/> bins (30).</summary>
    public int FrequencyMasks { get; set; } = 2;
    /// <inheritdoc cref="FrequencyMasks"/>
    public int FrequencyMaskWidth { get; set; } = 30;
    /// <summary>SpecAugment time masks (2), each up to <see cref="TimeMaskRatio"/> of the utterance.</summary>
    public int TimeMasks { get; set; } = 2;
    /// <inheritdoc cref="TimeMasks"/>
    public double TimeMaskRatio { get; set; } = 0.05;
    /// <summary>Optional vocabulary mapping token IDs to text tokens (e.g., SentencePiece vocabulary).</summary>
    public string[]? Vocabulary { get; set; }
}
