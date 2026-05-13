namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CLAP (Contrastive Language-Audio Pretraining) models.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the published CLAP "HTSAT + RoBERTa" recipe (Wu et al. 2023
/// "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion
/// and Keyword-to-Caption Augmentation"). The audio encoder is an HTSAT-based
/// Swin Transformer (Chen et al. 2022; Liu et al. 2021); the text encoder
/// is a RoBERTa-style transformer stack.
/// </para>
/// <para>
/// <b>For Beginners:</b> These knobs control how big and deep the audio and
/// text encoders are. The defaults match the published CLAP checkpoint and
/// produce 512-dim contrastive embeddings. Override individual fields to
/// build a smaller / faster variant.
/// </para>
/// </remarks>
public class CLAPModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Initializes a new instance with CLAP HTSAT+RoBERTa defaults.</summary>
    public CLAPModelOptions() { }

    /// <summary>
    /// Initializes a new instance by copying every property from
    /// <paramref name="other"/>. Throws when <paramref name="other"/> is null.
    /// </summary>
    /// <param name="other">Source options to copy. Must not be null.</param>
    public CLAPModelOptions(CLAPModelOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        NumMelBands = other.NumMelBands;
        StftWindowSize = other.StftWindowSize;
        HopLength = other.HopLength;
        SampleRate = other.SampleRate;
        AudioPatchSize = other.AudioPatchSize;
        AudioHiddenDim = other.AudioHiddenDim;
        AudioEncoderLayers = other.AudioEncoderLayers;
        AudioEncoderHeads = other.AudioEncoderHeads;
        SwinWindowSize = other.SwinWindowSize;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        TextHiddenDim = other.TextHiddenDim;
        TextEncoderLayers = other.TextEncoderLayers;
        TextEncoderHeads = other.TextEncoderHeads;
        ProjectionDim = other.ProjectionDim;
        InitialTemperature = other.InitialTemperature;
        DropoutRate = other.DropoutRate;
    }

    // ── Mel-spectrogram front-end (Wu 2023 §3.1: 64-mel, 1024-sample window,
    //    480-sample hop at 48 kHz → 50 Hz frame rate, matches Audio Spectrogram
    //    Transformer / HTSAT conventions). ──────────────────────────────────

    /// <summary>Number of mel-frequency bands extracted before the encoder.</summary>
    /// <value>Default 64 (CLAP paper §3.1).</value>
    /// <remarks><para><b>For Beginners:</b> Frequency bins in the
    /// mel-spectrogram passed to HTSAT. Match the pretrained checkpoint
    /// (64) unless retraining from scratch.</para></remarks>
    public int NumMelBands { get; init; } = 64;

    /// <summary>STFT window size in audio samples.</summary>
    /// <value>Default 1024 (CLAP paper §3.1).</value>
    /// <remarks><para><b>For Beginners:</b> Wider window = better frequency
    /// resolution, blurrier time. 1024 at 48 kHz ≈ 21 ms.</para></remarks>
    public int StftWindowSize { get; init; } = 1024;

    /// <summary>STFT hop length in audio samples.</summary>
    /// <value>Default 480 (= 10 ms at 48 kHz → 50 Hz frame rate, paper §3.1).</value>
    /// <remarks><para><b>For Beginners:</b> Distance between successive
    /// STFT frames. Smaller = more frames per second, higher cost.</para></remarks>
    public int HopLength { get; init; } = 480;

    /// <summary>Audio sample rate in Hz the encoder is configured for.</summary>
    /// <value>Default 48000 (CLAP training rate).</value>
    /// <remarks><para><b>For Beginners:</b> CLAP was trained at 48 kHz —
    /// downsampling will discard the high-frequency content the encoder
    /// was trained on.</para></remarks>
    public int SampleRate { get; init; } = 48_000;

    // ── HTSAT audio encoder (Chen et al. 2022 §3 — Swin Transformer over
    //    mel-spectrogram patches). Defaults match the HTSAT-S "small"
    //    variant CLAP uses. ──────────────────────────────────────────────────

    /// <summary>
    /// HTSAT patch size — height and width of the 2D mel-spectrogram patch
    /// that gets embedded into one token. Paper §3.1 uses 4 for HTSAT-S /
    /// HTSAT-T (96-channel base) and 4 for HTSAT-B (128-channel base).
    /// </summary>
    /// <value>Default 4 (HTSAT-S, matches the released CLAP checkpoint).</value>
    /// <remarks><para><b>For Beginners:</b> Bigger patches = fewer tokens
    /// = faster but coarser. Smaller patches give finer time-frequency
    /// resolution at higher cost.</para></remarks>
    public int AudioPatchSize { get; init; } = 4;

    /// <summary>Hidden / embedding dimension of the audio Swin blocks.</summary>
    /// <value>Default 768 (HTSAT-S).</value>
    /// <remarks><para><b>For Beginners:</b> Width of the audio
    /// transformer. Match the pretrained checkpoint when fine-tuning.</para></remarks>
    public int AudioHiddenDim { get; init; } = 768;

    /// <summary>Number of Swin Transformer blocks stacked in the audio encoder.</summary>
    /// <value>Default 4 (HTSAT-S).</value>
    /// <remarks><para><b>For Beginners:</b> Depth of the audio encoder.
    /// Increase for a deeper, slower, more accurate variant.</para></remarks>
    public int AudioEncoderLayers { get; init; } = 4;

    /// <summary>Number of attention heads per Swin block.</summary>
    /// <value>Default 12 (HTSAT-S; head_dim = AudioHiddenDim/heads = 64).</value>
    /// <remarks><para><b>For Beginners:</b> Multi-head attention runs N
    /// smaller attention computations in parallel. Keep
    /// <c>AudioHiddenDim / AudioEncoderHeads</c> at 64 for standard
    /// transformer scaling.</para></remarks>
    public int AudioEncoderHeads { get; init; } = 12;

    /// <summary>Swin window size (W-MSA / SW-MSA) — Liu et al. 2021 §3.2 uses 7.</summary>
    /// <value>Default 7 (Swin V1 default).</value>
    /// <remarks><para><b>For Beginners:</b> Swin attention is computed in
    /// local windows then shifted. 7 is the standard window size.</para></remarks>
    public int SwinWindowSize { get; init; } = 7;

    // ── RoBERTa-style text encoder (CLAP §3.2). Defaults match the
    //    RoBERTa-base size used in the released checkpoint. ─────────────────

    /// <summary>Text vocabulary size — RoBERTa BPE vocab.</summary>
    /// <value>Default 50265 (standard RoBERTa vocab).</value>
    /// <remarks><para><b>For Beginners:</b> The text encoder embedding
    /// table has this many rows. Don't change unless retraining with a
    /// different tokenizer.</para></remarks>
    public int VocabSize { get; init; } = 50_265;

    /// <summary>Maximum text token sequence length (CLAP truncates / pads to this).</summary>
    /// <value>Default 77 (same as CLIP).</value>
    /// <remarks><para><b>For Beginners:</b> Captions longer than this get
    /// truncated; shorter captions get padded. 77 matches CLIP.</para></remarks>
    public int MaxTextLength { get; init; } = 77;

    /// <summary>Hidden / embedding dimension of the text encoder.</summary>
    /// <value>Default 768 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> Width of the text encoder.
    /// Match the pretrained checkpoint when fine-tuning.</para></remarks>
    public int TextHiddenDim { get; init; } = 768;

    /// <summary>Number of transformer encoder layers in the text encoder.</summary>
    /// <value>Default 12 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> Depth of the text encoder.
    /// Match the pretrained checkpoint when fine-tuning.</para></remarks>
    public int TextEncoderLayers { get; init; } = 12;

    /// <summary>Number of attention heads per text-encoder layer.</summary>
    /// <value>Default 12 (RoBERTa-base; head_dim = 64).</value>
    /// <remarks><para><b>For Beginners:</b> Multi-head attention; keep
    /// <c>TextHiddenDim / TextEncoderHeads = 64</c> for standard scaling.</para></remarks>
    public int TextEncoderHeads { get; init; } = 12;

    // ── Shared projection + contrastive learning ──────────────────────────

    /// <summary>
    /// Final shared-embedding-space dimension that both encoders project into.
    /// CLAP §3.2 uses 512.
    /// </summary>
    /// <value>Default 512 (CLAP paper §3.2, matches CLIP).</value>
    /// <remarks><para><b>For Beginners:</b> Both encoders end with a linear
    /// projection into this shared dimension where audio and text vectors
    /// can be compared via cosine similarity.</para></remarks>
    public int ProjectionDim { get; init; } = 512;

    /// <summary>
    /// Initial temperature τ for the contrastive softmax. CLIP / CLAP both
    /// start at 0.07 (Radford 2021 §2.5; Wu 2023 §3.2); τ is learnable.
    /// </summary>
    /// <value>Default 0.07 (CLIP / CLAP papers).</value>
    /// <remarks><para><b>For Beginners:</b> Temperature scales the logits
    /// inside the contrastive softmax. Smaller τ ⇒ sharper similarity
    /// distribution. CLAP learns this jointly with the encoders; 0.07 is
    /// just the init.</para></remarks>
    public double InitialTemperature { get; init; } = 0.07;

    /// <summary>Dropout rate inside the transformer blocks. CLAP §3.2: 0.1.</summary>
    /// <value>Default 0.1 (CLAP paper §3.2).</value>
    /// <remarks><para><b>For Beginners:</b> Dropout randomly zeroes a
    /// fraction of activations during training to discourage overfitting.
    /// 0.1 is the CLAP paper default.</para></remarks>
    public double DropoutRate { get; init; } = 0.1;
}
