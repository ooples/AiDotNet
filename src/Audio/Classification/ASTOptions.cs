using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the AST (Audio Spectrogram Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// AST (Gong et al., Interspeech 2021) is the first purely attention-based model for audio
/// classification that directly applies a Vision Transformer to audio spectrograms. It achieves
/// 45.9% mAP on AudioSet and 95.6% accuracy on ESC-50.
/// </para>
/// <para>
/// <b>Architecture Overview:</b>
/// <list type="number">
/// <item>Audio waveform is converted to a mel spectrogram (128 bins)</item>
/// <item>The spectrogram is split into 16x16 patches with overlap</item>
/// <item>Patches are linearly projected to embedding vectors with [CLS] token prepended</item>
/// <item>A stack of Transformer encoder layers processes the patch embeddings</item>
/// <item>The [CLS] token output is used for classification</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> AST treats audio spectrograms exactly like images. It cuts the
/// spectrogram into small patches (like puzzle pieces), then uses a Transformer to understand
/// the relationships between patches. This approach works surprisingly well because spectrograms
/// are 2D representations of sound that look similar to images.
///
/// You can use AST in two ways:
/// <list type="bullet">
/// <item><b>ONNX mode</b>: Load a pre-trained model for instant inference</item>
/// <item><b>Native mode</b>: Train from scratch on your own audio data</item>
/// </list>
///
/// Example usage:
/// <code>
/// var options = new ASTOptions { ModelPath = "ast_audioset.onnx" };
/// var ast = new AST&lt;float&gt;(options);
/// var result = ast.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)</item>
/// <item>Repository: https://github.com/YuanGongND/ast</item>
/// </list>
/// </para>
/// </remarks>
public class ASTOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// <para>AST expects 16 kHz mono audio input for AudioSet tasks.</para>
    /// <para><b>For Beginners:</b> Sample rate is how many audio measurements per second.
    /// 16000 Hz is standard for general audio classification.</para>
    /// </remarks>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size for spectrogram computation.
    /// </summary>
    /// <remarks>
    /// <para>AST uses a 25ms Hanning window (400 samples at 16 kHz).</para>
    /// <para><b>For Beginners:</b> Controls the trade-off between frequency and time detail.</para>
    /// </remarks>
    public int FftSize { get; set; } = 400;

    /// <summary>
    /// Gets or sets the hop length between consecutive FFT frames.
    /// </summary>
    /// <remarks>
    /// <para>AST uses a 10ms hop (160 samples at 16 kHz), providing 100 frames per second.</para>
    /// <para><b>For Beginners:</b> How far to slide the analysis window between frames.</para>
    /// </remarks>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    /// <remarks>
    /// <para>AST uses 128 mel bands for fine frequency resolution.</para>
    /// <para><b>For Beginners:</b> How many frequency bands to divide audio into.</para>
    /// </remarks>
    public int NumMels { get; set; } = 128;

    /// <summary>
    /// Gets or sets the minimum frequency for the mel filterbank in Hz.
    /// </summary>
    public int FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency for the mel filterbank in Hz.
    /// </summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region Patch Embedding

    /// <summary>
    /// Gets or sets the patch size for spectrogram patching.
    /// </summary>
    /// <remarks>
    /// <para>AST uses 16x16 patches by default. The original paper experiments with
    /// both 16x16 and overlapping patches (stride 10) for improved performance.</para>
    /// <para><b>For Beginners:</b> Each small tile of the spectrogram. Smaller patches
    /// give more detail but increase computation.</para>
    /// </remarks>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride for overlapping patches.
    /// </summary>
    /// <remarks>
    /// <para>AST achieves best results with stride 10 (overlapping patches).
    /// Stride equal to patch size means no overlap.</para>
    /// <para><b>For Beginners:</b> How far apart the patches are. A stride smaller than
    /// patch size means patches overlap, which usually improves accuracy.</para>
    /// </remarks>
    public int PatchStride { get; set; } = 10;

    #endregion

    #region Transformer Architecture

    /// <summary>
    /// Gets or sets the embedding dimension of the Transformer encoder.
    /// </summary>
    /// <remarks>
    /// <para>AST-Base uses 768 dimensions (matching DeiT-Base/ViT-Base).
    /// AST-Small uses 384 dimensions.</para>
    /// <para><b>For Beginners:</b> How many numbers describe each patch. Larger = more
    /// capacity but more computation.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of Transformer encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>AST-Base uses 12 layers, matching ViT-Base. Each layer adds deeper understanding
    /// of the audio spectrogram through self-attention and feed-forward processing.</para>
    /// <para><b>For Beginners:</b> More layers means deeper understanding but slower inference.</para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads in each Transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>AST-Base uses 12 heads, each attending to different aspects of the audio.</para>
    /// <para><b>For Beginners:</b> Multiple heads let the model focus on different audio
    /// patterns simultaneously (pitch, rhythm, timbre, etc.).</para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward network dimension in each Transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>Typically 4x the embedding dimension. For AST-Base: 3072.</para>
    /// <para><b>For Beginners:</b> A larger internal processing layer within each Transformer block.</para>
    /// </remarks>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>AST uses 0.0 dropout when fine-tuning pre-trained models, and 0.1 when training
    /// from scratch to prevent overfitting.</para>
    /// <para><b>For Beginners:</b> Randomly disables neurons during training to prevent memorization.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// </summary>
    public double AttentionDropoutRate { get; set; } = 0.0;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold for event detection.
    /// </summary>
    /// <remarks>
    /// <para>Events with confidence below this threshold are not reported.
    /// AST uses sigmoid activation for multi-label AudioSet classification.</para>
    /// <para><b>For Beginners:</b> How confident the model must be to report a sound.</para>
    /// </remarks>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds for event detection.
    /// </summary>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio (0-1) for event detection.
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels. If null, uses AudioSet-527 labels.
    /// </summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have a pre-trained model file, specify its path here
    /// for fast inference without training.</para>
    /// </remarks>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate for training.
    /// </summary>
    /// <remarks>
    /// <para>AST fine-tuning uses 1e-5 learning rate with warm-up. Training from scratch
    /// uses a higher rate of 1e-4.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the number of warm-up epochs for learning rate scheduling.
    /// </summary>
    public int WarmUpEpochs { get; set; } = 5;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use ImageNet-pretrained weights for initialization.
    /// </summary>
    /// <remarks>
    /// <para>AST initializes from DeiT (Data-efficient Image Transformer) weights trained on
    /// ImageNet, then fine-tunes on audio. This transfer learning dramatically improves results.</para>
    /// <para><b>For Beginners:</b> Starting from image-trained weights works because spectrograms
    /// look like images. The model already knows how to process 2D patterns.</para>
    /// </remarks>
    public bool UseImageNetPretrain { get; set; } = true;

    #endregion
}
