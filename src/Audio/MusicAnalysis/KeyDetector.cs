using AiDotNet.Audio.Features;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Detects the musical key of audio using chromagram analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Key detection uses the Krumhansl-Kessler key profiles to match average
/// chroma features against major and minor key templates. This provides
/// the most likely key and its relative minor/major.
/// </para>
/// <para><b>For Beginners:</b> The "key" of a song tells you which notes are
/// emphasized and how the music "feels." For example:
/// - C major: Bright, happy sound (uses mainly white keys on piano)
/// - A minor: Sadder, darker sound (also mainly white keys, but starts on A)
///
/// Knowing the key helps with:
/// - Playing along with songs
/// - Transposing music to different keys
/// - Understanding the harmonic structure
///
/// Usage:
/// <code>
/// var detector = new KeyDetector&lt;float&gt;();
/// var key = detector.Detect(audioTensor);
/// Console.WriteLine($"Key: {key.Name} ({key.Mode})");
/// Console.WriteLine($"Confidence: {key.Confidence:P0}");
/// </code>
/// </para>
/// </remarks>
public class KeyDetector<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ChromaExtractor<T> _chromaExtractor;
    private readonly KeyDetectorOptions _options;
    private readonly double[,] _majorProfiles;
    private readonly double[,] _minorProfiles;

    private static readonly string[] NoteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    // Krumhansl-Kessler key profiles (normalized)
    private static readonly double[] MajorProfile =
    [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ];

    private static readonly double[] MinorProfile =
    [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ];

    /// <summary>
    /// Creates a new key detector.
    /// </summary>
    /// <param name="options">Key detection options.</param>
    public KeyDetector(KeyDetectorOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new KeyDetectorOptions();

        _chromaExtractor = new ChromaExtractor<T>(new ChromaOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });

        // Build rotated profiles for all 12 keys
        _majorProfiles = BuildRotatedProfiles(MajorProfile);
        _minorProfiles = BuildRotatedProfiles(MinorProfile);
    }

    /// <summary>
    /// Detects the musical key of the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Key detection result.</returns>
    public KeyDetectionResult Detect(Tensor<T> audio)
    {
        // Extract chroma features
        var chroma = _chromaExtractor.Extract(audio);

        // Compute average chroma over entire audio
        var avgChroma = ComputeAverageChroma(chroma);

        // Match against all key profiles
        var results = MatchKeyProfiles(avgChroma);

        // Return best match
        return results.OrderByDescending(r => r.Correlation).First();
    }

    /// <summary>
    /// Detects the musical key of the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Key detection result.</returns>
    public KeyDetectionResult Detect(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Detect(tensor);
    }

    /// <summary>
    /// Gets all key detection results ranked by correlation.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>List of all key results, ordered by likelihood.</returns>
    public List<KeyDetectionResult> DetectAll(Tensor<T> audio)
    {
        var chroma = _chromaExtractor.Extract(audio);
        var avgChroma = ComputeAverageChroma(chroma);
        var results = MatchKeyProfiles(avgChroma);
        return results.OrderByDescending(r => r.Correlation).ToList();
    }

    private static double[,] BuildRotatedProfiles(double[] profile)
    {
        var rotated = new double[12, 12];

        // Normalize the profile
        double sum = profile.Sum();
        var normalized = profile.Select(p => p / sum).ToArray();

        // Create rotations for each key
        for (int key = 0; key < 12; key++)
        {
            for (int i = 0; i < 12; i++)
            {
                rotated[key, i] = normalized[(i - key + 12) % 12];
            }
        }

        return rotated;
    }

    private double[] ComputeAverageChroma(Tensor<T> chroma)
    {
        int numFrames = chroma.Shape[0];
        var avgChroma = new double[12];

        for (int f = 0; f < numFrames; f++)
        {
            for (int c = 0; c < 12; c++)
            {
                avgChroma[c] += _numOps.ToDouble(chroma[f, c]);
            }
        }

        // Normalize
        double sum = avgChroma.Sum();
        if (sum > 0)
        {
            for (int c = 0; c < 12; c++)
            {
                avgChroma[c] /= sum;
            }
        }

        return avgChroma;
    }

    private List<KeyDetectionResult> MatchKeyProfiles(double[] chroma)
    {
        var results = new List<KeyDetectionResult>();

        // Match against all major keys
        for (int key = 0; key < 12; key++)
        {
            var profile = new double[12];
            for (int i = 0; i < 12; i++)
            {
                profile[i] = _majorProfiles[key, i];
            }

            double correlation = PearsonCorrelation(chroma, profile);

            results.Add(new KeyDetectionResult
            {
                KeyIndex = key,
                Name = $"{NoteNames[key]} major",
                RootNote = NoteNames[key],
                Mode = KeyMode.Major,
                Correlation = correlation,
                Confidence = (correlation + 1) / 2, // Normalize to 0-1
                RelativeKey = $"{NoteNames[(key + 9) % 12]} minor"
            });
        }

        // Match against all minor keys
        for (int key = 0; key < 12; key++)
        {
            var profile = new double[12];
            for (int i = 0; i < 12; i++)
            {
                profile[i] = _minorProfiles[key, i];
            }

            double correlation = PearsonCorrelation(chroma, profile);

            results.Add(new KeyDetectionResult
            {
                KeyIndex = key,
                Name = $"{NoteNames[key]} minor",
                RootNote = NoteNames[key],
                Mode = KeyMode.Minor,
                Correlation = correlation,
                Confidence = (correlation + 1) / 2,
                RelativeKey = $"{NoteNames[(key + 3) % 12]} major"
            });
        }

        return results;
    }

    private static double PearsonCorrelation(double[] x, double[] y)
    {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        int n = x.Length;

        for (int i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        if (denominator < 1e-10) return 0;
        return numerator / denominator;
    }
}

/// <summary>
/// Result of key detection.
/// </summary>
public class KeyDetectionResult
{
    /// <summary>
    /// Gets or sets the key index (0 = C, 1 = C#, etc.).
    /// </summary>
    public int KeyIndex { get; set; }

    /// <summary>
    /// Gets or sets the full key name (e.g., "C major", "A minor").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the root note name (e.g., "C", "A").
    /// </summary>
    public string RootNote { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the key mode (major or minor).
    /// </summary>
    public KeyMode Mode { get; set; }

    /// <summary>
    /// Gets or sets the Pearson correlation with the key profile (-1 to 1).
    /// </summary>
    public double Correlation { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the relative major/minor key.
    /// </summary>
    public string RelativeKey { get; set; } = string.Empty;
}

/// <summary>
/// Musical key mode.
/// </summary>
public enum KeyMode
{
    /// <summary>Major key (happy, bright sound).</summary>
    Major,

    /// <summary>Minor key (sad, dark sound).</summary>
    Minor
}

/// <summary>
/// Configuration options for key detection.
/// </summary>
public class KeyDetectorOptions
{
    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 2048;
}
