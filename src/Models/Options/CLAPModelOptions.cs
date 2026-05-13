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
    // ── Mel-spectrogram front-end (Wu 2023 §3.1: 64-mel, 1024-sample window,
    //    480-sample hop at 48 kHz → 50 Hz frame rate, matches Audio Spectrogram
    //    Transformer / HTSAT conventions). ──────────────────────────────────
    /// <summary>Number of mel-frequency bands extracted before the encoder.</summary>
    public int NumMelBands { get; init; } = 64;

    /// <summary>STFT window size in audio samples.</summary>
    public int StftWindowSize { get; init; } = 1024;

    /// <summary>STFT hop length in audio samples.</summary>
    public int HopLength { get; init; } = 480;

    /// <summary>Audio sample rate in Hz the encoder is configured for.</summary>
    public int SampleRate { get; init; } = 48_000;

    // ── HTSAT audio encoder (Chen et al. 2022 §3 — Swin Transformer over
    //    mel-spectrogram patches). Defaults match the HTSAT-S "small"
    //    variant CLAP uses. ──────────────────────────────────────────────────
    /// <summary>
    /// HTSAT patch size — height and width of the 2D mel-spectrogram patch
    /// that gets embedded into one token. Paper §3.1 uses 4 for HTSAT-S /
    /// HTSAT-T (96-channel base) and 4 for HTSAT-B (128-channel base).
    /// </summary>
    public int AudioPatchSize { get; init; } = 4;

    /// <summary>Hidden / embedding dimension of the audio Swin blocks.</summary>
    public int AudioHiddenDim { get; init; } = 768;

    /// <summary>Number of Swin Transformer blocks stacked in the audio encoder.</summary>
    public int AudioEncoderLayers { get; init; } = 4;

    /// <summary>Number of attention heads per Swin block.</summary>
    public int AudioEncoderHeads { get; init; } = 12;

    /// <summary>Swin window size (W-MSA / SW-MSA) — Liu et al. 2021 §3.2 uses 7.</summary>
    public int SwinWindowSize { get; init; } = 7;

    // ── RoBERTa-style text encoder (CLAP §3.2). Defaults match the
    //    RoBERTa-base size used in the released checkpoint. ─────────────────
    /// <summary>Text vocabulary size — RoBERTa BPE vocab.</summary>
    public int VocabSize { get; init; } = 50_265;

    /// <summary>Maximum text token sequence length (CLAP truncates / pads to this).</summary>
    public int MaxTextLength { get; init; } = 77;

    /// <summary>Hidden / embedding dimension of the text encoder.</summary>
    public int TextHiddenDim { get; init; } = 768;

    /// <summary>Number of transformer encoder layers in the text encoder.</summary>
    public int TextEncoderLayers { get; init; } = 12;

    /// <summary>Number of attention heads per text-encoder layer.</summary>
    public int TextEncoderHeads { get; init; } = 12;

    // ── Shared projection + contrastive learning ──────────────────────────
    /// <summary>
    /// Final shared-embedding-space dimension that both encoders project into.
    /// CLAP §3.2 uses 512.
    /// </summary>
    public int ProjectionDim { get; init; } = 512;

    /// <summary>
    /// Initial temperature τ for the contrastive softmax. CLIP / CLAP both
    /// start at 0.07 (Radford 2021 §2.5; Wu 2023 §3.2); τ is learnable.
    /// </summary>
    public double InitialTemperature { get; init; } = 0.07;

    /// <summary>Dropout rate inside the transformer blocks. CLAP §3.2: 0.1.</summary>
    public double DropoutRate { get; init; } = 0.1;
}
