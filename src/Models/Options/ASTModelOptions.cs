namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AST (Audio Spectrogram Transformer) models
/// (Gong et al. 2021).
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the published AST-Base recipe (Gong et al. 2021 §2):
/// 128-mel input × ~1024 frames, 12 transformer layers × 12 heads × 768
/// embedding dim, 16×16 patches (paper §2.2), trained on AudioSet
/// (527 classes).
/// </para>
/// <para><b>For Beginners:</b> AST treats a mel spectrogram (a 2-D
/// time/frequency picture of an audio clip) as a sequence of small image
/// patches and runs a vision transformer over them. The default knob values
/// here reproduce AST-Base; you usually only need to change them when fine-
/// tuning on a different dataset (e.g., setting <c>NumClasses</c>) or
/// hardware budget (e.g., shrinking <c>EmbeddingDim</c> for mobile).</para>
/// </remarks>
public class ASTModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Initializes a new instance with AST-Base defaults.</summary>
    public ASTModelOptions() { }

    /// <summary>
    /// Initializes a new instance by copying every property from
    /// <paramref name="other"/>. Throws when <paramref name="other"/> is null.
    /// </summary>
    /// <param name="other">Source options to copy. Must not be null.</param>
    public ASTModelOptions(ASTModelOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        // Inherited properties — base classes don't expose a copy ctor.
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        SampleRate = other.SampleRate;
        StftWindowSize = other.StftWindowSize;
        HopLength = other.HopLength;
        NumMelBands = other.NumMelBands;
        TargetLength = other.TargetLength;
        PatchSize = other.PatchSize;
        NumClasses = other.NumClasses;
        EmbeddingDim = other.EmbeddingDim;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FeedForwardDim = other.FeedForwardDim;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>Audio sample rate in Hz used by the STFT frontend.</summary>
    /// <value>Default 16000 — the AST-Base training rate (Gong et al. 2021).</value>
    /// <remarks><para><b>For Beginners:</b> How many audio samples per
    /// second. Higher values capture more high-frequency detail but cost more
    /// compute. AST was trained at 16 kHz, so leave this alone unless your
    /// data is resampled.</para></remarks>
    public int SampleRate { get; init; } = 16_000;

    /// <summary>STFT window size in samples (analysis frame length).</summary>
    /// <value>Default 400 (≈ 25 ms at 16 kHz, AST paper §2.1).</value>
    /// <remarks><para><b>For Beginners:</b> The STFT slides a window across
    /// the waveform and runs an FFT on each window. Bigger windows give
    /// sharper frequency resolution but smear out time; 25 ms is the
    /// standard speech / general-audio default.</para></remarks>
    public int StftWindowSize { get; init; } = 400;

    /// <summary>STFT hop length in samples between successive frames.</summary>
    /// <value>Default 160 (= 10 ms at 16 kHz, AST paper §2.1).</value>
    /// <remarks><para><b>For Beginners:</b> How far the analysis window
    /// shifts between frames. Smaller hops give more frames per second (and
    /// higher cost); 10 ms is the canonical default.</para></remarks>
    public int HopLength { get; init; } = 160;

    /// <summary>Number of mel filterbank bands per spectrogram frame.</summary>
    /// <value>Default 128 (AST paper §2.1).</value>
    /// <remarks><para><b>For Beginners:</b> The mel filterbank turns each
    /// FFT frame into a perceptually-spaced frequency vector; 128 bands is
    /// the AudioSet / AST default. Reduce to 64 for smaller models.</para></remarks>
    public int NumMelBands { get; init; } = 128;

    /// <summary>Target spectrogram length in frames (time axis).</summary>
    /// <value>Default 1024 (≈ 10 s of audio at 10 ms hop).</value>
    /// <remarks><para><b>For Beginners:</b> Each AST input is padded /
    /// trimmed to this many frames. Increase for longer clips, decrease for
    /// shorter ones.</para></remarks>
    public int TargetLength { get; init; } = 1024;

    /// <summary>Patch size H × W for the ViT patch embedding.</summary>
    /// <value>Default 16 (AST paper §2.2, matching ViT-B/16).</value>
    /// <remarks><para><b>For Beginners:</b> AST chops the spectrogram into
    /// 16×16 squares and treats each as a token. Smaller patches mean more
    /// tokens (better detail, higher cost).</para></remarks>
    public int PatchSize { get; init; } = 16;

    /// <summary>Number of output classes for the classification head.</summary>
    /// <value>Default 527 (the AudioSet ontology AST-Base trained on).</value>
    /// <remarks><para><b>For Beginners:</b> Set this to your label count
    /// when fine-tuning on a non-AudioSet dataset.</para></remarks>
    public int NumClasses { get; init; } = 527;

    /// <summary>Transformer hidden / embedding dimension.</summary>
    /// <value>Default 768 (AST-Base, matches ViT-B).</value>
    /// <remarks><para><b>For Beginners:</b> Width of every transformer
    /// layer. Common alternatives: 192 (Tiny), 384 (Small), 1024 (Large).
    /// Wider = more capacity, more memory.</para></remarks>
    public int EmbeddingDim { get; init; } = 768;

    /// <summary>Number of stacked transformer encoder layers.</summary>
    /// <value>Default 12 (AST-Base, matches ViT-B).</value>
    /// <remarks><para><b>For Beginners:</b> Depth of the model. More layers
    /// = more capacity for long-range patterns, at the cost of compute.
    /// Common alternatives: 12 (Base), 24 (Large).</para></remarks>
    public int NumLayers { get; init; } = 12;

    /// <summary>Number of attention heads per transformer block.</summary>
    /// <value>Default 12 (AST-Base, matches ViT-B; head_dim = 64).</value>
    /// <remarks><para><b>For Beginners:</b> Multi-head attention runs N
    /// smaller attention computations in parallel and concatenates them. By
    /// convention <c>EmbeddingDim / NumHeads</c> should be 64.</para></remarks>
    public int NumHeads { get; init; } = 12;

    /// <summary>Hidden dimension of the per-block feed-forward MLP.</summary>
    /// <value>Default 3072 (= 4× EmbeddingDim per Vaswani et al. 2017).</value>
    /// <remarks><para><b>For Beginners:</b> Each transformer block applies
    /// a two-layer MLP after the attention step. Keep this at
    /// <c>4 × EmbeddingDim</c> to match standard transformer scaling.</para></remarks>
    public int FeedForwardDim { get; init; } = 3072;

    /// <summary>Dropout rate inside the transformer blocks.</summary>
    /// <value>Default 0.0 (AST-Base does no in-block dropout).</value>
    /// <remarks><para><b>For Beginners:</b> Dropout randomly zeroes a
    /// fraction of activations during training to discourage overfitting.
    /// AST relies on SpecAugment data augmentation instead, so the default
    /// is 0. Try 0.1 if you see overfitting on a small dataset.</para></remarks>
    public double DropoutRate { get; init; } = 0.0;
}
