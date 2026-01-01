using AiDotNet.Onnx;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// Configuration options for language identification models.
/// </summary>
/// <remarks>
/// <para>
/// These options configure how language identification models process audio
/// and return predictions.
/// </para>
/// <para><b>For Beginners:</b> Language identification (LID) determines which
/// language is being spoken in an audio recording.
///
/// Key settings:
/// - SampleRate: Must match your audio (16000 Hz is common for speech)
/// - MinConfidence: Minimum confidence to report a detection
/// - TopK: Number of top language predictions to return
///
/// Example:
/// <code>
/// var options = new LanguageIdentifierOptions
/// {
///     SampleRate = 16000,
///     MinConfidence = 0.7,
///     TopK = 3
/// };
/// var model = new ECAPATDNNLanguageIdentifier&lt;float&gt;(options);
/// var result = model.IdentifyLanguage(audio);
/// Console.WriteLine($"Language: {result.LanguageCode} ({result.Confidence:P0})");
/// </code>
/// </para>
/// </remarks>
public class LanguageIdentifierOptions
{
    /// <summary>
    /// Gets or sets the sample rate of input audio in Hz.
    /// </summary>
    /// <value>Default is 16000 Hz.</value>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT size for spectrogram computation.
    /// </summary>
    /// <value>Default is 512.</value>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between frames.
    /// </summary>
    /// <value>Default is 160 (10ms at 16kHz).</value>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    /// <value>Default is 80.</value>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <value>Default is 192 (ECAPA-TDNN standard).</value>
    public int EmbeddingDimension { get; set; } = 192;

    /// <summary>
    /// Gets or sets the minimum confidence threshold for valid detection.
    /// </summary>
    /// <value>Default is 0.5.</value>
    public double MinConfidence { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of top language predictions to return.
    /// </summary>
    /// <value>Default is 5.</value>
    public int TopK { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum audio duration in seconds required for reliable detection.
    /// </summary>
    /// <value>Default is 1.0 second.</value>
    public double MinDurationSeconds { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the window size in milliseconds for segmented language detection.
    /// </summary>
    /// <value>Default is 2000ms.</value>
    public int SegmentWindowMs { get; set; } = 2000;

    /// <summary>
    /// Gets or sets whether to apply voice activity detection before language identification.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseVoiceActivityDetection { get; set; } = true;

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    /// <value>Null for native training mode, or path to pre-trained ONNX model.</value>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model options.
    /// </summary>
    public OnnxModelOptions? OnnxOptions { get; set; }
}

/// <summary>
/// Configuration options specific to ECAPA-TDNN language identification.
/// </summary>
public class ECAPATDNNOptions : LanguageIdentifierOptions
{
    /// <summary>
    /// Gets or sets the number of TDNN channels.
    /// </summary>
    /// <value>Default is 1024.</value>
    public int TdnnChannels { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Default is 8.</value>
    public int AttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets whether to use channel attention.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseChannelAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets the dilation factors for TDNN layers.
    /// </summary>
    /// <value>Default is [1, 2, 3, 4, 1].</value>
    public int[] Dilations { get; set; } = [1, 2, 3, 4, 1];
}

/// <summary>
/// Configuration options specific to Wav2Vec2 language identification.
/// </summary>
public class Wav2Vec2LidOptions : LanguageIdentifierOptions
{
    /// <summary>
    /// Gets or sets the hidden size of the transformer.
    /// </summary>
    /// <value>Default is 768.</value>
    public int HiddenSize { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Default is 12.</value>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Default is 12.</value>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the intermediate size for feed-forward layers.
    /// </summary>
    /// <value>Default is 3072.</value>
    public int IntermediateSize { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the feature projection dropout rate.
    /// </summary>
    /// <value>Default is 0.0.</value>
    public double FeatureProjectionDropout { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the hidden dropout rate.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double HiddenDropout { get; set; } = 0.1;
}

/// <summary>
/// Configuration options specific to VoxLingua107 language identification.
/// </summary>
public class VoxLingua107Options : ECAPATDNNOptions
{
    /// <summary>
    /// Initializes a new instance with VoxLingua107 defaults.
    /// </summary>
    public VoxLingua107Options()
    {
        // VoxLingua107 uses standard ECAPA-TDNN architecture
        // but is specifically trained on 107 languages
        EmbeddingDimension = 256;
        TdnnChannels = 1024;
    }

    /// <summary>
    /// Gets the number of supported languages (107 for VoxLingua107).
    /// </summary>
    public int NumLanguages => 107;
}
