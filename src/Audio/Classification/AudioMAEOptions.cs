using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the Audio-MAE (Masked Autoencoders for Audio) model.
/// </summary>
/// <remarks>
/// <para>
/// Audio-MAE (Xu et al., NeurIPS 2022) applies the Masked Autoencoder framework to audio
/// spectrograms for self-supervised pre-training. By masking 80% of spectrogram patches and
/// reconstructing them, Audio-MAE learns rich audio representations that achieve 47.3% mAP
/// on AudioSet and 97.0% on ESC-50 after fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio-MAE learns by playing "fill in the blanks" with spectrograms.
/// It hides 80% of the audio picture and tries to reconstruct what was hidden. This forces
/// the model to truly understand audio patterns. After this pre-training, it can be fine-tuned
/// for classification with very little labeled data.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Masked Autoencoders that Listen" (Xu et al., NeurIPS 2022)</item>
/// <item>Repository: https://github.com/facebookresearch/AudioMAE</item>
/// </list>
/// </para>
/// </remarks>
public class AudioMAEOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of mel bands.
    /// </summary>
    public int NumMels { get; set; } = 128;

    /// <summary>
    /// Gets or sets the minimum frequency.
    /// </summary>
    public int FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency.
    /// </summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region Transformer Architecture

    /// <summary>
    /// Gets or sets the encoder embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>Audio-MAE uses 768 for the encoder (ViT-Base).</para>
    /// </remarks>
    public int EncoderEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of encoder attention heads.
    /// </summary>
    public int NumEncoderHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the decoder embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>Audio-MAE uses a lightweight decoder (512-dim) for reconstruction during pre-training.
    /// The decoder is discarded after pre-training; only the encoder is used for classification.</para>
    /// <para><b>For Beginners:</b> The decoder is only needed during the "learning" phase.
    /// Once the model learns, only the encoder (the understanding part) is kept.</para>
    /// </remarks>
    public int DecoderEmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    public int NumDecoderLayers { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of decoder attention heads.
    /// </summary>
    public int NumDecoderHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the feed-forward dimension ratio.
    /// </summary>
    public double FeedForwardRatio { get; set; } = 4.0;

    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride.
    /// </summary>
    public int PatchStride { get; set; } = 16;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region MAE Pre-Training

    /// <summary>
    /// Gets or sets the mask ratio for pre-training.
    /// </summary>
    /// <remarks>
    /// <para>Audio-MAE uses 80% masking (higher than BEATs' 75%) because audio spectrograms
    /// are highly redundant and reconstructing from fewer visible patches forces deeper learning.</para>
    /// <para><b>For Beginners:</b> The model hides 80% of the spectrogram and tries to
    /// reconstruct it. This high masking ratio forces the model to learn strong patterns.</para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to use local attention in the encoder.
    /// </summary>
    /// <remarks>
    /// <para>Audio-MAE optionally uses local (windowed) attention in the encoder to reduce
    /// computation. Local attention restricts each patch to attend only to nearby patches.</para>
    /// </remarks>
    public bool UseLocalAttention { get; set; } = false;

    /// <summary>
    /// Gets or sets the local attention window size.
    /// </summary>
    public int LocalAttentionWindow { get; set; } = 4;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold.
    /// </summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds.
    /// </summary>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio.
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels.
    /// </summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 1.5e-4;

    /// <summary>
    /// Gets or sets the number of warm-up epochs.
    /// </summary>
    public int WarmUpEpochs { get; set; } = 40;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.1;

    #endregion
}
