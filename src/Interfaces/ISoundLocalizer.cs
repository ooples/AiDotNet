namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for sound localization models that estimate the spatial position of sound sources.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sound localization estimates where sound is coming from in 3D space. This requires
/// multi-channel audio (stereo or more) and uses differences in timing, loudness, and
/// spectral content between channels to determine direction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sound localization is like closing your eyes and pointing
/// to where a sound is coming from.
///
/// How it works (like human hearing):
/// 1. Sound reaches one ear slightly before the other (ITD - Interaural Time Difference)
/// 2. Sound is slightly louder in the closer ear (ILD - Interaural Level Difference)
/// 3. Head shape affects high frequencies differently for each ear
/// 4. Brain combines all cues to determine direction
///
/// What's measured:
/// - Azimuth: Left-right angle (0° = front, 90° = right, -90° = left)
/// - Elevation: Up-down angle (0° = level, 90° = above)
/// - Distance: How far away (harder to estimate from audio alone)
///
/// Use cases:
/// - Spatial audio for VR/AR (place sounds correctly in 3D)
/// - Smart speakers (know which direction user is speaking from)
/// - Security (detect where intruder sounds come from)
/// - Robotics (navigate toward or away from sounds)
/// - Audio surveillance (track moving sound sources)
/// - Hearing aids (enhance sounds from specific directions)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SoundLocalizer")]
public interface ISoundLocalizer<T>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the number of audio channels required.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Minimum 2 for stereo. More channels (e.g., 4+ for arrays) enable better accuracy.
    /// </para>
    /// </remarks>
    int RequiredChannels { get; }

    /// <summary>
    /// Gets the microphone array geometry if applicable.
    /// </summary>
    MicrophoneArrayConfig<T>? ArrayConfig { get; }

    /// <summary>
    /// Gets whether this model can estimate distance (not just direction).
    /// </summary>
    bool SupportsDistanceEstimation { get; }

    /// <summary>
    /// Gets whether this model can track multiple simultaneous sources.
    /// </summary>
    bool SupportsMultipleSourceTracking { get; }

    /// <summary>
    /// Localizes sound sources in multi-channel audio.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor [channels, samples].</param>
    /// <returns>Localization result with source positions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for finding where sounds are.
    /// - Pass in stereo (or multi-channel) audio
    /// - Get back the direction(s) sounds are coming from
    /// </para>
    /// </remarks>
    LocalizationResult<T> Localize(Tensor<T> audio);

    /// <summary>
    /// Localizes sound sources asynchronously.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Localization result.</returns>
    Task<LocalizationResult<T>> LocalizeAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Tracks sound source positions over time.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor.</param>
    /// <param name="windowDuration">Duration of each analysis window in seconds.</param>
    /// <returns>Tracking result showing source positions over time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For moving sound sources (like a person walking while
    /// talking), this tracks how the position changes over time.
    /// </para>
    /// </remarks>
    SoundTrackingResult<T> TrackSources(Tensor<T> audio, double windowDuration = 0.1);

    /// <summary>
    /// Estimates direction of arrival (DOA) for dominant sources.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor.</param>
    /// <param name="maxSources">Maximum number of sources to detect.</param>
    /// <returns>List of direction estimates.</returns>
    IReadOnlyList<DirectionEstimate<T>> EstimateDirections(Tensor<T> audio, int maxSources = 3);

    /// <summary>
    /// Computes spatial power spectrum for visualization.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor.</param>
    /// <param name="azimuthResolution">Resolution in degrees for azimuth.</param>
    /// <returns>Power values for each direction [num_directions].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a "heat map" of where sounds are coming from.
    /// Peaks in the spectrum indicate sound source directions.
    /// </para>
    /// </remarks>
    Tensor<T> ComputeSpatialSpectrum(Tensor<T> audio, double azimuthResolution = 5.0);

    /// <summary>
    /// Beamforms audio to focus on a specific direction.
    /// </summary>
    /// <param name="audio">Multi-channel audio tensor.</param>
    /// <param name="targetAzimuth">Target azimuth angle in degrees.</param>
    /// <param name="targetElevation">Target elevation angle in degrees.</param>
    /// <returns>Beamformed single-channel audio focused on target direction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Beamforming is like a "zoom" for audio - it enhances
    /// sounds from one direction while reducing sounds from other directions.
    /// </para>
    /// </remarks>
    Tensor<T> Beamform(Tensor<T> audio, double targetAzimuth, double targetElevation = 0.0);
}

/// <summary>
/// Result of sound localization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LocalizationResult<T>
{
    /// <summary>
    /// Gets or sets detected sound sources.
    /// </summary>
    public IReadOnlyList<SoundSource<T>> Sources { get; set; } = Array.Empty<SoundSource<T>>();

    /// <summary>
    /// Gets or sets the number of detected sources.
    /// </summary>
    public int NumSources => Sources.Count;

    /// <summary>
    /// Gets or sets the dominant source (highest energy).
    /// </summary>
    public SoundSource<T>? DominantSource { get; set; }

    /// <summary>
    /// Gets or sets the analysis duration in seconds.
    /// </summary>
    public double Duration { get; set; }
}

/// <summary>
/// A detected sound source with position.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SoundSource<T>
{
    /// <summary>
    /// Gets or sets the azimuth angle in degrees (-180 to 180, 0 = front).
    /// </summary>
    public T Azimuth { get; set; } = default!;

    /// <summary>
    /// Gets or sets the elevation angle in degrees (-90 to 90, 0 = level).
    /// </summary>
    public T Elevation { get; set; } = default!;

    /// <summary>
    /// Gets or sets the estimated distance (if supported).
    /// </summary>
    public T? Distance { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (0.0 to 1.0).
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the relative energy level.
    /// </summary>
    public T Energy { get; set; } = default!;

    /// <summary>
    /// Gets or sets the frequency range of this source.
    /// </summary>
    public FrequencyRange<T>? FrequencyRange { get; set; }
}

/// <summary>
/// A direction estimate without full source information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DirectionEstimate<T>
{
    /// <summary>
    /// Gets or sets the azimuth angle in degrees.
    /// </summary>
    public T Azimuth { get; set; } = default!;

    /// <summary>
    /// Gets or sets the elevation angle in degrees.
    /// </summary>
    public T Elevation { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// Result of sound source tracking over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SoundTrackingResult<T>
{
    /// <summary>
    /// Gets or sets tracked source trajectories.
    /// </summary>
    public IReadOnlyList<SourceTrajectory<T>> Trajectories { get; set; } = Array.Empty<SourceTrajectory<T>>();

    /// <summary>
    /// Gets or sets the total duration tracked.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the number of unique sources tracked.
    /// </summary>
    public int NumSourcesTracked { get; set; }
}

/// <summary>
/// A tracked trajectory of a sound source.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SourceTrajectory<T>
{
    /// <summary>
    /// Gets or sets the source ID.
    /// </summary>
    public int SourceId { get; set; }

    /// <summary>
    /// Gets or sets the trajectory points over time.
    /// </summary>
    public IReadOnlyList<TrajectoryPoint<T>> Points { get; set; } = Array.Empty<TrajectoryPoint<T>>();

    /// <summary>
    /// Gets or sets the start time of this trajectory.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time of this trajectory.
    /// </summary>
    public double EndTime { get; set; }
}

/// <summary>
/// A single point in a source trajectory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrajectoryPoint<T>
{
    /// <summary>
    /// Gets or sets the time in seconds.
    /// </summary>
    public double Time { get; set; }

    /// <summary>
    /// Gets or sets the azimuth angle.
    /// </summary>
    public T Azimuth { get; set; } = default!;

    /// <summary>
    /// Gets or sets the elevation angle.
    /// </summary>
    public T Elevation { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence at this point.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// Configuration for microphone array geometry.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MicrophoneArrayConfig<T>
{
    /// <summary>
    /// Gets or sets the number of microphones.
    /// </summary>
    public int NumMicrophones { get; set; }

    /// <summary>
    /// Gets or sets the array type.
    /// </summary>
    public ArrayType Type { get; set; } = ArrayType.Linear;

    /// <summary>
    /// Gets or sets microphone positions [num_mics, 3] (x, y, z).
    /// </summary>
    public Tensor<T>? Positions { get; set; }

    /// <summary>
    /// Gets or sets the spacing between microphones (for regular arrays).
    /// </summary>
    public T Spacing { get; set; } = default!;
}

/// <summary>
/// Types of microphone arrays.
/// </summary>
public enum ArrayType
{
    /// <summary>Linear array (microphones in a line).</summary>
    Linear,
    /// <summary>Circular array (microphones in a circle).</summary>
    Circular,
    /// <summary>Spherical array (microphones on a sphere).</summary>
    Spherical,
    /// <summary>Custom/irregular array geometry.</summary>
    Custom
}

/// <summary>
/// A frequency range.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FrequencyRange<T>
{
    /// <summary>
    /// Gets or sets the lower frequency bound in Hz.
    /// </summary>
    public T LowFrequency { get; set; } = default!;

    /// <summary>
    /// Gets or sets the upper frequency bound in Hz.
    /// </summary>
    public T HighFrequency { get; set; } = default!;
}
