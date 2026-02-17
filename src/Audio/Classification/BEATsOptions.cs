using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the BEATs (Audio Pre-Training with Acoustic Tokenizers) model.
/// </summary>
/// <remarks>
/// <para>
/// BEATs (Chen et al., ICML 2023) is a state-of-the-art audio classification model that uses
/// iterative self-distillation between an acoustic tokenizer and an audio SSL model. It achieves
/// 50.6% mAP on AudioSet-2M and 98.1% accuracy on ESC-50, setting new benchmarks for audio
/// event detection and classification tasks.
/// </para>
/// <para>
/// <b>Architecture Overview:</b>
/// <list type="number">
/// <item>Audio waveform is converted to a mel spectrogram</item>
/// <item>Spectrogram patches are extracted and linearly projected to embedding vectors</item>
/// <item>Positional embeddings are added to preserve spatial information</item>
/// <item>A stack of Transformer encoder layers processes the patch embeddings</item>
/// <item>A classification head maps the aggregated features to event labels</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BEATs is a powerful model for recognizing sounds in audio.
/// It works by:
/// <list type="bullet">
/// <item>Breaking audio into small patches (like cutting a spectrogram into tiles)</item>
/// <item>Using a Transformer (the same architecture behind ChatGPT) to understand relationships between patches</item>
/// <item>Learning what sounds are present by comparing against known audio patterns</item>
/// </list>
///
/// You can use BEATs in two ways:
/// <list type="bullet">
/// <item><b>ONNX mode</b>: Load a pre-trained model for instant inference</item>
/// <item><b>Native mode</b>: Train from scratch on your own audio data</item>
/// </list>
///
/// Example usage:
/// <code>
/// // ONNX inference mode
/// var options = new BEATsOptions { ModelPath = "beats_iter3.onnx" };
/// var beats = new BEATs&lt;float&gt;(options);
/// var result = beats.Detect(audioTensor);
///
/// // Native training mode
/// var options = new BEATsOptions { EmbeddingDim = 768, NumEncoderLayers = 12 };
/// var beats = new BEATs&lt;float&gt;(options);
/// beats.Train(features, labels);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (Chen et al., ICML 2023)</item>
/// <item>Repository: https://github.com/microsoft/unilm/tree/master/beats</item>
/// </list>
/// </para>
/// </remarks>
public class BEATsOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// <para>BEATs expects 16 kHz mono audio input, consistent with AudioSet conventions.</para>
    /// <para><b>For Beginners:</b> Sample rate is how many audio measurements per second.
    /// 16000 Hz is standard for speech and general audio classification tasks.</para>
    /// </remarks>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size for spectrogram computation.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses a 25ms window at 16 kHz, which is 400 samples. This provides good
    /// frequency resolution while maintaining sufficient temporal detail.</para>
    /// <para><b>For Beginners:</b> FFT size controls the trade-off between frequency and time resolution.
    /// Larger values give better frequency detail but coarser time detail.</para>
    /// </remarks>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between consecutive FFT frames.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses a 10ms hop (160 samples at 16 kHz), providing 100 frames per second.
    /// This ensures sufficient temporal resolution for detecting transient audio events.</para>
    /// <para><b>For Beginners:</b> Hop length is how many samples to skip between frames.
    /// Smaller values give finer temporal resolution but more computation.</para>
    /// </remarks>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses 128 mel bands, which provides finer frequency resolution than the
    /// typical 64 or 80 bands used by older models. This helps capture subtle spectral
    /// differences between similar-sounding events.</para>
    /// <para><b>For Beginners:</b> Mel bands group frequencies in a way that matches human
    /// hearing. More bands means the model can distinguish more subtle frequency differences.</para>
    /// </remarks>
    public int NumMels { get; set; } = 128;

    /// <summary>
    /// Gets or sets the minimum frequency for the mel filterbank in Hz.
    /// </summary>
    /// <remarks>
    /// <para>Setting to 0 Hz ensures the model captures the full frequency range, including
    /// low-frequency environmental sounds like traffic rumble and bass instruments.</para>
    /// </remarks>
    public int FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency for the mel filterbank in Hz.
    /// </summary>
    /// <remarks>
    /// <para>8000 Hz is the Nyquist frequency for 16 kHz audio. This captures the full
    /// useful frequency range of the input signal.</para>
    /// </remarks>
    public int FMax { get; set; } = 8000;

    #endregion

    #region Patch Embedding

    /// <summary>
    /// Gets or sets the patch size (height) for spectrogram patching.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses 16x16 patches on the mel spectrogram. The patch height corresponds
    /// to the number of mel bands covered by each patch. With 128 mel bands and patch size 16,
    /// there are 8 patches along the frequency axis.</para>
    /// <para><b>For Beginners:</b> The spectrogram is divided into small tiles (patches).
    /// Each patch captures a small region of time and frequency, like pixels in an image.</para>
    /// </remarks>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride for spectrogram patching.
    /// </summary>
    /// <remarks>
    /// <para>Stride controls the overlap between patches. A stride equal to patch size means
    /// non-overlapping patches. BEATs uses stride 16 (no overlap) by default.</para>
    /// </remarks>
    public int PatchStride { get; set; } = 16;

    #endregion

    #region Transformer Architecture

    /// <summary>
    /// Gets or sets the embedding dimension of the Transformer encoder.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses 768-dimensional embeddings for the base model (BEATs_iter3) and 512 for
    /// smaller variants. This determines the representational capacity of the model.</para>
    /// <para><b>For Beginners:</b> Each audio patch is represented as a vector of this many numbers.
    /// Larger values can capture more detail but need more computation and data to train.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of Transformer encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>BEATs_iter3 (the best-performing variant) uses 12 encoder layers.
    /// Each layer contains multi-head self-attention and a feed-forward network.
    /// More layers allow the model to learn more complex audio patterns.</para>
    /// <para><b>For Beginners:</b> Think of each layer as a level of understanding.
    /// Early layers learn simple patterns (pitch, loudness), while later layers learn
    /// complex concepts (speech vs. music, types of instruments).</para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads in each Transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses 12 attention heads (one per layer in the base model).
    /// Each head can independently focus on different aspects of the audio signal,
    /// such as onset timing, harmonic structure, or spectral envelope.</para>
    /// <para><b>For Beginners:</b> Multiple attention heads let the model look at the
    /// audio from different perspectives simultaneously, similar to having multiple
    /// listeners each focusing on different aspects of a sound.</para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward network dimension in each Transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>The feed-forward dimension is typically 4x the embedding dimension.
    /// For BEATs with 768-dim embeddings, this is 3072. This expansion allows the
    /// feed-forward network to learn complex nonlinear transformations.</para>
    /// <para><b>For Beginners:</b> After the attention mechanism combines information,
    /// this larger network processes the combined information to extract useful features.</para>
    /// </remarks>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses 0.1 dropout during training. During inference, dropout is disabled.
    /// This helps prevent overfitting by randomly deactivating neurons during training.</para>
    /// <para><b>For Beginners:</b> Dropout randomly turns off some neurons during training
    /// to prevent the model from memorizing the training data. A value of 0.1 means 10%
    /// of neurons are randomly disabled in each training step.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>Separate dropout applied to attention weights. This regularizes the attention
    /// patterns and prevents the model from relying too heavily on specific positions.</para>
    /// </remarks>
    public double AttentionDropoutRate { get; set; } = 0.1;

    #endregion

    #region Pre-Training (Masked Prediction)

    /// <summary>
    /// Gets or sets the mask probability for masked patch prediction pre-training.
    /// </summary>
    /// <remarks>
    /// <para>During pre-training, this fraction of patches is randomly masked.
    /// BEATs uses 0.75 (75%) masking, which is higher than typical NLP models (15%)
    /// because audio spectrograms have high spatial redundancy.</para>
    /// <para><b>For Beginners:</b> During training, the model hides 75% of the audio patches
    /// and tries to predict what was hidden. This forces the model to learn meaningful
    /// audio representations, similar to a fill-in-the-blank exercise.</para>
    /// </remarks>
    public double MaskProbability { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the minimum span length for contiguous masking.
    /// </summary>
    /// <remarks>
    /// <para>Instead of masking individual patches randomly, BEATs masks contiguous spans
    /// of patches. This prevents the model from trivially interpolating from neighbors.</para>
    /// </remarks>
    public int MinMaskSpanLength { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of acoustic tokenizer iterations.
    /// </summary>
    /// <remarks>
    /// <para>BEATs uses iterative self-distillation between the audio model and the acoustic
    /// tokenizer. BEATs_iter3 uses 3 iterations, where each iteration produces better
    /// discrete labels for pre-training the next version of the audio model.</para>
    /// <para><b>For Beginners:</b> The model improves itself in cycles. In each cycle:
    /// <list type="number">
    /// <item>The current model creates labels for audio patches (tokenizer)</item>
    /// <item>A new model is trained to predict those labels (audio SSL)</item>
    /// <item>The improved model creates better labels for the next cycle</item>
    /// </list>
    /// Three iterations (iter3) gives the best results.</para>
    /// </remarks>
    public int TokenizerIterations { get; set; } = 3;

    /// <summary>
    /// Gets or sets the codebook size for the acoustic tokenizer.
    /// </summary>
    /// <remarks>
    /// <para>The acoustic tokenizer maps each audio patch to one of K discrete codes.
    /// BEATs uses 8192 codes, providing a rich vocabulary for describing audio patterns.</para>
    /// <para><b>For Beginners:</b> Think of this as the size of the "audio vocabulary."
    /// Each audio patch is assigned one of 8192 possible labels, like categorizing sounds
    /// into 8192 different types.</para>
    /// </remarks>
    public int CodebookSize { get; set; } = 8192;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold for event detection.
    /// </summary>
    /// <remarks>
    /// <para>Events with confidence below this threshold are not reported.
    /// A threshold of 0.3 balances precision and recall for multi-label detection.
    /// Lower values detect more events but with more false positives.</para>
    /// <para><b>For Beginners:</b> This controls how confident the model must be before
    /// reporting a sound. 0.3 means the model must be at least 30% sure.</para>
    /// </remarks>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds for event detection.
    /// </summary>
    /// <remarks>
    /// <para>Audio is processed in overlapping windows of this duration.
    /// BEATs processes 10-second segments by default, matching AudioSet clip length.</para>
    /// </remarks>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio (0-1) for event detection.
    /// </summary>
    /// <remarks>
    /// <para>Controls how much adjacent windows overlap. 0.5 means 50% overlap,
    /// which provides good temporal continuity for event detection.</para>
    /// </remarks>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels. If null, uses AudioSet-527 labels.
    /// </summary>
    /// <remarks>
    /// <para>When fine-tuning on a custom dataset, provide your own class labels here.
    /// If null, the model uses the standard AudioSet ontology with 527 event classes.</para>
    /// </remarks>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    /// <remarks>
    /// <para>When provided, the model operates in ONNX inference mode.
    /// Pre-trained BEATs checkpoints can be exported to ONNX format for deployment.</para>
    /// <para><b>For Beginners:</b> If you have a pre-trained model file, specify its path here
    /// and the model will use it for fast inference without any training needed.</para>
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
    /// <para>BEATs uses a peak learning rate of 5e-4 with warm-up and cosine decay.
    /// This is the peak rate after warm-up completes.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets the number of warm-up steps for learning rate scheduling.
    /// </summary>
    /// <remarks>
    /// <para>During warm-up, the learning rate linearly increases from 0 to the peak rate.
    /// BEATs uses 32000 warm-up steps to stabilize early training.</para>
    /// </remarks>
    public int WarmUpSteps { get; set; } = 32000;

    /// <summary>
    /// Gets or sets the label smoothing factor for classification.
    /// </summary>
    /// <remarks>
    /// <para>Label smoothing prevents the model from becoming overconfident by
    /// softening the target distribution. BEATs uses 0.1 label smoothing for fine-tuning.</para>
    /// </remarks>
    public double LabelSmoothing { get; set; } = 0.1;

    #endregion
}
