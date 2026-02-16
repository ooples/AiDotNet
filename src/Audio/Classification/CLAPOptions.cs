using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the CLAP (Contrastive Language-Audio Pre-training) model.
/// </summary>
/// <remarks>
/// <para>
/// CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive
/// learning, similar to CLIP for images. It can classify audio using natural language descriptions
/// without task-specific training (zero-shot classification), achieving 26.7% zero-shot accuracy
/// on ESC-50 and 46.8% mAP on AudioSet with fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> CLAP is special because it understands both audio and text. Instead
/// of having fixed labels like "dog bark" or "siren", you can describe any sound in plain
/// English and CLAP will find it in audio. For example, you can search for "the sound of
/// rain hitting a tin roof" without ever training on that specific label.
///
/// How it works:
/// <list type="number">
/// <item>CLAP has two encoders: one for audio and one for text</item>
/// <item>Both encoders map their inputs to the same shared space</item>
/// <item>Matching audio-text pairs are placed close together in this space</item>
/// <item>To classify audio, compare it with text descriptions and find the closest match</item>
/// </list>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/LAION-AI/CLAP</item>
/// </list>
/// </para>
/// </remarks>
public class CLAPOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// <para>CLAP uses 48 kHz audio for maximum frequency coverage.</para>
    /// </remarks>
    public int SampleRate { get; set; } = 48000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 480;

    /// <summary>
    /// Gets or sets the number of mel bands.
    /// </summary>
    public int NumMels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the minimum frequency.
    /// </summary>
    public int FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency.
    /// </summary>
    public int FMax { get; set; } = 24000;

    #endregion

    #region Audio Encoder

    /// <summary>
    /// Gets or sets the audio encoder type.
    /// </summary>
    /// <remarks>
    /// <para>CLAP supports multiple audio encoders: "HTSAT" (default, best performance)
    /// and "PANN" (CNN14-based, faster).</para>
    /// <para><b>For Beginners:</b> The audio encoder processes the audio spectrogram.
    /// HTS-AT is more accurate; PANN is faster.</para>
    /// </remarks>
    public string AudioEncoderType { get; set; } = "HTSAT";

    /// <summary>
    /// Gets or sets the audio embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>The dimension of the shared audio-text embedding space.
    /// CLAP uses 512 dimensions for the joint space.</para>
    /// </remarks>
    public int AudioEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the projection dimension for the joint space.
    /// </summary>
    public int ProjectionDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of audio encoder layers.
    /// </summary>
    public int NumAudioEncoderLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of audio attention heads.
    /// </summary>
    public int NumAudioAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Text Encoder

    /// <summary>
    /// Gets or sets the text encoder type.
    /// </summary>
    /// <remarks>
    /// <para>CLAP uses a RoBERTa-based text encoder by default. The text encoder
    /// processes natural language descriptions of sounds.</para>
    /// </remarks>
    public string TextEncoderType { get; set; } = "RoBERTa";

    /// <summary>
    /// Gets or sets the text embedding dimension.
    /// </summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the maximum text token length.
    /// </summary>
    public int MaxTextLength { get; set; } = 77;

    #endregion

    #region Contrastive Learning

    /// <summary>
    /// Gets or sets the temperature parameter for contrastive loss.
    /// </summary>
    /// <remarks>
    /// <para>Controls the sharpness of the similarity distribution. Lower values make the
    /// model more discriminative but harder to train.</para>
    /// <para><b>For Beginners:</b> Temperature controls how "picky" the model is about
    /// matching audio to text. Lower = more picky, higher = more lenient.</para>
    /// </remarks>
    public double Temperature { get; set; } = 0.07;

    /// <summary>
    /// Gets or sets whether to enable feature fusion.
    /// </summary>
    /// <remarks>
    /// <para>Feature fusion combines features from multiple encoder layers for richer representations.</para>
    /// </remarks>
    public bool UseFeatureFusion { get; set; } = true;

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

    /// <summary>
    /// Gets or sets text prompts for zero-shot classification.
    /// </summary>
    /// <remarks>
    /// <para>For zero-shot classification, provide text descriptions of sounds to search for.
    /// CLAP compares audio embeddings with text embeddings to classify without fine-tuning.</para>
    /// <para><b>For Beginners:</b> Instead of fixed labels, describe sounds in natural language:
    /// "a dog barking loudly", "rain falling on a roof", "a car engine starting".</para>
    /// </remarks>
    public string[]? TextPrompts { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the text encoder ONNX model.
    /// </summary>
    public string? TextEncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the warm-up steps.
    /// </summary>
    public int WarmUpSteps { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.1;

    #endregion
}
