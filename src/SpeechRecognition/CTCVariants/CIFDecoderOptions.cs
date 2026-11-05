using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.CTCVariants;

/// <summary>Options for the CIF-based recognizer of Dong and Xu (2020).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the paper's Mandarin (AISHELL-2) model: 40 filterbanks with deltas, a 64-filter
/// convolutional front-end, five self-attention layers at each of three time resolutions of width 640, a two-layer
/// non-autoregressive decoder and 5,230 output labels. For the LibriSpeech base model set <see cref="EncoderDim"/> 512,
/// <see cref="FeedForwardDim"/> 2048, <see cref="PredictorWindow"/> 5, <see cref="NumDecoderLayers"/> 3,
/// <see cref="CtcWeight"/> 0.25 and <see cref="VocabSize"/> 3725.</para>
/// </remarks>
public class CIFDecoderOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CIFDecoderOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CIFDecoderOptions(CIFDecoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        NumFilters = other.NumFilters;
        EncoderDim = other.EncoderDim;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        NumLayersPerResolution = other.NumLayersPerResolution;
        NumResolutions = other.NumResolutions;
        PredictorWindow = other.PredictorWindow;
        NumDecoderLayers = other.NumDecoderLayers;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        Language = other.Language;
        LabelSmoothing = other.LabelSmoothing;
        CtcWeight = other.CtcWeight;
        QuantityWeight = other.QuantityWeight;
        TailThreshold = other.TailThreshold;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Filterbank bands; deltas and delta-deltas are added as channels (40).</summary>
    public int NumMels { get; set; } = 40;
    /// <summary>The front-end convolutions' filters (64).</summary>
    public int NumFilters { get; set; } = 64;
    /// <summary>The model width d (640).</summary>
    public int EncoderDim { get; set; } = 640;
    /// <summary>Gets or sets the number of attention heads. Defaults to <c>4</c>.</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>The self-attention layers' feed-forward width (2,560).</summary>
    public int FeedForwardDim { get; set; } = 2560;
    /// <summary>Self-attention layers at each temporal resolution of the encoder (n = 5).</summary>
    public int NumLayersPerResolution { get; set; } = 5;
    /// <summary>Temporal resolutions of the encoder, separated by pooling of width 2 (3).</summary>
    public int NumResolutions { get; set; } = 3;
    /// <summary>The weight predictor's convolution window (3).</summary>
    public int PredictorWindow { get; set; } = 3;
    /// <summary>Self-attention layers in the non-autoregressive decoder (2).</summary>
    public int NumDecoderLayers { get; set; } = 2;
    /// <summary>Output labels, blank (0), end of sentence (1) and pad (2) included (5,230).</summary>
    public int VocabSize { get; set; } = 5230;
    /// <summary>Gets or sets the maximum number of text tokens. Defaults to <c>512</c>.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>The AdamW learning rate used when no optimizer is injected (2e-4; the paper states no optimizer).</summary>
    public double LearningRate { get; set; } = 2e-4;
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"zh"</c>.</summary>
    public string Language { get; set; } = "zh";
    /// <summary>The uniform label smoothing of the cross-entropy (0.2).</summary>
    public double LabelSmoothing { get; set; } = 0.2;
    /// <summary>λ1, the encoder CTC loss's weight (0.5 for Mandarin, 0.25 for LibriSpeech).</summary>
    public double CtcWeight { get; set; } = 0.5;
    /// <summary>λ2, the quantity loss's weight (1).</summary>
    public double QuantityWeight { get; set; } = 1.0;
    /// <summary>The residual weight above which inference fires once more at the tail (0.5).</summary>
    public double TailThreshold { get; set; } = 0.5;
    /// <summary>Optional vocabulary mapping label IDs to text.</summary>
    public string[]? Vocabulary { get; set; }
}
