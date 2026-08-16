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

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1024;
    public int NumEncoderLayers { get; set; } = 50;
    public int NumAttentionHeads { get; set; } = 16;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 25000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public int DecoderDim { get; set; } = 512;
    public int NumDecoderLayers { get; set; } = 6;
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
    public double LearningRate { get; set; } = 1e-4;
    public double WeightDecay { get; set; } = 0.01;
}
