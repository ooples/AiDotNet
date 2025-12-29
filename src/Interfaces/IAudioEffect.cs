using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for audio effects processors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio effects modify sound in creative or corrective ways:
/// <list type="bullet">
/// <item><description>Dynamics: Compressor, Limiter, Gate, Expander</description></item>
/// <item><description>EQ: Parametric EQ, Graphic EQ, Filters</description></item>
/// <item><description>Time-based: Reverb, Delay, Echo</description></item>
/// <item><description>Modulation: Chorus, Flanger, Phaser, Tremolo</description></item>
/// <item><description>Pitch: Pitch Shifter, Auto-Tune, Harmonizer</description></item>
/// <item><description>Distortion: Overdrive, Fuzz, Saturation</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Audio effects are like Instagram filters for sound!
///
/// Common effects explained:
/// - Reverb: Adds room ambience (makes it sound like you're in a hall)
/// - Delay: Creates echoes of the sound
/// - Compressor: Evens out loud and quiet parts (used in podcasts)
/// - EQ: Boosts or cuts certain frequencies (more bass, less treble)
/// - Pitch Shift: Makes voice higher or lower
///
/// Effects can be:
/// - Chained: One after another (guitar -> distortion -> reverb -> amp)
/// - Real-time: Process audio live as it plays
/// - Offline: Process entire files for best quality
/// </para>
/// </remarks>
public interface IAudioEffect<T>
{
    /// <summary>
    /// Gets the name of this effect.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the sample rate this effect operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets or sets whether the effect is bypassed (disabled).
    /// </summary>
    bool Bypass { get; set; }

    /// <summary>
    /// Gets or sets the dry/wet mix (0.0 = dry only, 1.0 = wet only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// - Dry = original unprocessed sound
    /// - Wet = fully processed sound
    /// - Mix 0.5 = 50% original + 50% processed (parallel processing)
    /// </para>
    /// </remarks>
    double Mix { get; set; }

    /// <summary>
    /// Processes audio through the effect.
    /// </summary>
    /// <param name="input">Input audio tensor.</param>
    /// <returns>Processed audio tensor.</returns>
    Tensor<T> Process(Tensor<T> input);

    /// <summary>
    /// Processes a single sample (for real-time use).
    /// </summary>
    /// <param name="sample">Input sample value.</param>
    /// <returns>Processed sample value.</returns>
    T ProcessSample(T sample);

    /// <summary>
    /// Processes audio in-place for efficiency.
    /// </summary>
    /// <param name="buffer">Audio buffer to process in-place.</param>
    void ProcessInPlace(Span<T> buffer);

    /// <summary>
    /// Resets the effect's internal state.
    /// </summary>
    /// <remarks>
    /// Call this when starting a new audio stream to prevent artifacts
    /// from previous audio bleeding through.
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the processing latency in samples.
    /// </summary>
    int LatencySamples { get; }

    /// <summary>
    /// Gets the tail length in samples (how long the effect rings out after input stops).
    /// </summary>
    /// <remarks>
    /// Important for reverb/delay. When input stops, you need to continue
    /// processing for this many samples to capture the tail.
    /// </remarks>
    int TailSamples { get; }

    /// <summary>
    /// Gets all adjustable parameters for this effect.
    /// </summary>
    IReadOnlyDictionary<string, AudioEffectParameter<T>> Parameters { get; }

    /// <summary>
    /// Sets a parameter value by name.
    /// </summary>
    /// <param name="name">Parameter name.</param>
    /// <param name="value">New value.</param>
    void SetParameter(string name, T value);

    /// <summary>
    /// Gets a parameter value by name.
    /// </summary>
    /// <param name="name">Parameter name.</param>
    /// <returns>Current value.</returns>
    T GetParameter(string name);
}

/// <summary>
/// Represents an adjustable parameter for an audio effect.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class AudioEffectParameter<T>
{
    /// <summary>
    /// Gets the parameter name.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the display name for UI.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets the minimum allowed value.
    /// </summary>
    public required T MinValue { get; init; }

    /// <summary>
    /// Gets the maximum allowed value.
    /// </summary>
    public required T MaxValue { get; init; }

    /// <summary>
    /// Gets the default value.
    /// </summary>
    public required T DefaultValue { get; init; }

    /// <summary>
    /// Gets or sets the current value.
    /// </summary>
    public required T CurrentValue { get; set; }

    /// <summary>
    /// Gets the unit of measurement (e.g., "dB", "Hz", "ms", "%").
    /// </summary>
    public string Unit { get; init; } = string.Empty;

    /// <summary>
    /// Gets the description of what this parameter does.
    /// </summary>
    public string Description { get; init; } = string.Empty;
}
