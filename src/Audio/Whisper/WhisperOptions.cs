using AiDotNet.Onnx;

namespace AiDotNet.Audio.Whisper;

/// <summary>
/// Configuration options for the Whisper speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// Whisper is a speech recognition model developed by OpenAI that can
/// transcribe audio in multiple languages and perform translation.
/// </para>
/// <para><b>For Beginners:</b> Whisper comes in different sizes (tiny to large).
/// Smaller models are faster but less accurate. Larger models are more accurate but slower.
/// <list type="bullet">
/// <item><b>Tiny</b>: ~39M parameters, fastest, good for quick transcription</item>
/// <item><b>Base</b>: ~74M parameters, balanced speed/accuracy</item>
/// <item><b>Small</b>: ~244M parameters, good accuracy</item>
/// <item><b>Medium</b>: ~769M parameters, high accuracy</item>
/// <item><b>Large</b>: ~1.5B parameters, best accuracy, slow</item>
/// </list>
/// </para>
/// </remarks>
public class WhisperOptions
{
    /// <summary>
    /// Gets or sets the model size to use.
    /// </summary>
    public WhisperModelSize ModelSize { get; set; } = WhisperModelSize.Base;

    /// <summary>
    /// Gets or sets the language code for transcription (e.g., "en", "es", "fr").
    /// Null for auto-detection.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets whether to translate to English.
    /// If true, non-English audio will be translated to English.
    /// </summary>
    public bool Translate { get; set; } = false;

    /// <summary>
    /// Gets or sets the sample rate expected by the model.
    /// Whisper expects 16kHz audio.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// Whisper uses 80 mel channels.
    /// </summary>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the maximum length of audio to process in seconds.
    /// Whisper processes 30-second chunks.
    /// </summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>
    /// Gets or sets the ONNX execution options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the path to the encoder ONNX model.
    /// If null, the model will be downloaded automatically.
    /// </summary>
    public string? EncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the decoder ONNX model.
    /// If null, the model will be downloaded automatically.
    /// </summary>
    public string? DecoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens to generate.
    /// </summary>
    public int MaxTokens { get; set; } = 448;

    /// <summary>
    /// Gets or sets the beam size for beam search decoding.
    /// Higher values give better results but are slower.
    /// </summary>
    public int BeamSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Lower values make output more deterministic.
    /// </summary>
    public double Temperature { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to return timestamps with the transcription.
    /// </summary>
    public bool ReturnTimestamps { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to include word-level timestamps.
    /// </summary>
    public bool WordTimestamps { get; set; } = false;
}

/// <summary>
/// Available Whisper model sizes.
/// </summary>
public enum WhisperModelSize
{
    /// <summary>Tiny model (~39M parameters) - fastest, least accurate.</summary>
    Tiny,

    /// <summary>Base model (~74M parameters) - good balance of speed and accuracy.</summary>
    Base,

    /// <summary>Small model (~244M parameters) - good accuracy.</summary>
    Small,

    /// <summary>Medium model (~769M parameters) - high accuracy.</summary>
    Medium,

    /// <summary>Large model (~1.5B parameters) - highest accuracy, slowest.</summary>
    Large,

    /// <summary>Large-v2 model - improved large model.</summary>
    LargeV2,

    /// <summary>Large-v3 model - latest large model with best accuracy.</summary>
    LargeV3
}

/// <summary>
/// Result of Whisper transcription.
/// </summary>
public class WhisperResult
{
    /// <summary>
    /// Gets or sets the transcribed text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the detected language code.
    /// </summary>
    public string? DetectedLanguage { get; set; }

    /// <summary>
    /// Gets or sets the language detection probability.
    /// </summary>
    public double LanguageProbability { get; set; }

    /// <summary>
    /// Gets or sets the word-level timestamps if requested.
    /// </summary>
    public List<WhisperWord> Words { get; set; } = [];

    /// <summary>
    /// Gets or sets the segment-level timestamps.
    /// </summary>
    public List<WhisperSegment> Segments { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}

/// <summary>
/// A transcribed word with timing information.
/// </summary>
public class WhisperWord
{
    /// <summary>
    /// Gets or sets the word text.
    /// </summary>
    public string Word { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public double Confidence { get; set; }
}

/// <summary>
/// A segment of transcribed speech with timing.
/// </summary>
public class WhisperSegment
{
    /// <summary>
    /// Gets or sets the segment text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the words in this segment.
    /// </summary>
    public List<WhisperWord> Words { get; set; } = [];
}
