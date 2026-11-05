using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SenseVoice-Large: scaled multi-task speech model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the native SenseVoice-Large approximation.
/// The paper specifies an autoregressive encoder-decoder supporting more than 50 languages, but does
/// not publish its exact layer dimensions or optimizer hyperparameters. Every native scale and training
/// value used by this implementation is therefore exposed for customization.</para>
/// </remarks>
public class SenseVoiceLargeOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SenseVoiceLargeOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SenseVoiceLargeOptions(SenseVoiceLargeOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // INHERITED FROM ModelOptions, AND THEREFORE EASY TO MISS. Every declared property is copied
        // below; Seed is not declared here, so it was silently dropped and a copied configuration
        // produced a DIFFERENT model from the one it was copied from -- the failure mode that costs
        // the most to diagnose, because the two configurations compare equal on everything visible.
        Seed = other.Seed;

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
        FeedForwardDim = other.FeedForwardDim;
        UseCifAlignment = other.UseCifAlignment;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 16;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 25000;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the hidden width of the decoder.</summary>
    public int DecoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;
    /// <summary>Gets or sets the inner width of each feed-forward block.</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>
    /// Whether to insert a CIF (continuous integrate-and-fire) monotonic-alignment stage between
    /// the encoder and decoder.
    /// </summary>
    /// <value>
    /// Defaults to <c>false</c>. SenseVoice-Large is an AUTOREGRESSIVE ENCODER-DECODER
    /// (An et al., "FunAudioLLM", arXiv:2407.04051), and the paper describes no CIF stage and no
    /// Paraformer-style non-autoregressive decoder for either SenseVoice variant. The
    /// autoregressive decoder performs the alignment itself, so a CIF stage is not part of this
    /// architecture.
    /// </value>
    /// <remarks>
    /// <para>This previously defaulted to <c>true</c>, inherited from the shared Paraformer layer
    /// factory that SenseVoice is built on. CIF genuinely belongs to Paraformer (Gao et al. 2022)
    /// and to the CIF paper (Dong &amp; Xu 2020) — not to SenseVoice. SenseVoice-Small already
    /// defaulted to <c>false</c>, so only the Large variant carried the extra stage.</para>
    /// <para>Kept as a public option rather than removed, so callers who deliberately want a
    /// Paraformer-style non-autoregressive variant can still opt in.</para>
    /// </remarks>
    public bool UseCifAlignment { get; set; } = false;
    /// <summary>Gets or sets the learning rate of the default optimizer.</summary>
    public double LearningRate { get; set; } = 1e-4;
    /// <summary>Gets or sets the weight decay of the default optimizer.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
