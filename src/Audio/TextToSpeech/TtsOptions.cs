using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for text-to-speech models.
/// </summary>
/// <remarks>
/// <para>
/// TTS (Text-to-Speech) converts written text into natural-sounding speech.
/// Modern TTS uses a two-stage pipeline:
/// 1. Acoustic Model (e.g., FastSpeech2): Converts text to mel spectrogram
/// 2. Vocoder (e.g., HiFi-GAN): Converts mel spectrogram to audio waveform
/// </para>
/// <para><b>For Beginners:</b> Think of TTS as the opposite of speech recognition.
/// The acoustic model decides "what should this text sound like" (intonation, timing),
/// and the vocoder makes it actually sound like speech.
/// </para>
/// </remarks>
public class TtsOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the output sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the speaking rate multiplier.
    /// 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed.
    /// </summary>
    public double SpeakingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the pitch shift in semitones.
    /// 0 = normal, negative = lower, positive = higher.
    /// </summary>
    public double PitchShift { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the energy/volume level.
    /// 1.0 = normal.
    /// </summary>
    public double Energy { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the speaker ID for multi-speaker models.
    /// </summary>
    public int? SpeakerId { get; set; }

    /// <summary>
    /// Gets or sets the language code for multi-lingual models.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets the path to the acoustic model (FastSpeech2) ONNX file.
    /// </summary>
    public string? AcousticModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the vocoder (HiFi-GAN) ONNX file.
    /// </summary>
    public string? VocoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets whether to use Griffin-Lim as a fallback vocoder.
    /// </summary>
    public bool UseGriffinLimFallback { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of Griffin-Lim iterations if used.
    /// </summary>
    public int GriffinLimIterations { get; set; } = 60;

    /// <summary>
    /// Gets or sets the ONNX execution options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the number of mel channels.
    /// </summary>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 256;
}
