using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
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
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("A Robust Method for Key Estimation in Music", "https://doi.org/10.1109/TASLP.2014.2316141")]
public class KeyDetector<T> : MusicAnalysisBase<T>
{
    private readonly ChromaExtractor<T> _chromaExtractor;
    private readonly KeyDetectorOptions _options;
    private readonly double[,] _majorProfiles;
    private readonly double[,] _minorProfiles;
    private Tensor<T>? _keyProfileTensor;

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
        _options = options ?? new KeyDetectorOptions();

        // Set base class properties
        SampleRate = _options.SampleRate;
        HopLength = _options.HopLength;
        FftSize = _options.FftSize;

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
    public KeyDetectionResult<T> Detect(Tensor<T> audio)
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
    public KeyDetectionResult<T> Detect(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length], audio);
        return Detect(tensor);
    }

    /// <summary>
    /// Gets all key detection results ranked by correlation.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>List of all key results, ordered by likelihood.</returns>
    public List<KeyDetectionResult<T>> DetectAll(Tensor<T> audio)
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

    private Tensor<T> ComputeAverageChroma(Tensor<T> chroma)
    {
        int numFrames = chroma.Shape[0];
        if (numFrames == 0)
            return new Tensor<T>([12]);

        // Sum over frames, then normalize to a distribution when the total is positive
        var avgChroma = Engine.ReduceSum(Engine.TensorSlice(chroma, [0, 0], [numFrames, 12]), [0], keepDims: false);
        T sum = Engine.TensorSum(avgChroma);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
            avgChroma = Engine.TensorDivideScalar(avgChroma, sum);

        return avgChroma;
    }

    private List<KeyDetectionResult<T>> MatchKeyProfiles(Tensor<T> chroma)
    {
        // Pearson correlation of the chroma with all 24 key profiles (12 major, then 12 minor):
        // (n * sum(xy) - sum(x) sum(y)) / sqrt((n * sum(x^2) - sum(x)^2) * (n * sum(y^2) - sum(y)^2)),
        // 0 when that denominator is below 1e-10
        var profiles = KeyProfileTensor();
        int keys = profiles.Shape[0];
        T n = NumOps.FromDouble(12);
        T sumX = Engine.TensorSum(chroma);
        T sumX2 = Engine.TensorSum(Engine.TensorSquare(chroma));
        var sumY = Engine.ReduceSum(profiles, [1], keepDims: false);
        var sumY2 = Engine.ReduceSum(Engine.TensorSquare(profiles), [1], keepDims: false);
        var sumXY = Engine.Reshape(Engine.TensorMatMul(profiles, Engine.Reshape(chroma, [12, 1])), [keys]);

        var numerator = Engine.TensorSubtract(Engine.TensorMultiplyScalar(sumXY, n), Engine.TensorMultiplyScalar(sumY, sumX));
        T chromaSpread = NumOps.Subtract(NumOps.Multiply(n, sumX2), NumOps.Multiply(sumX, sumX));
        var profileSpread = Engine.TensorSubtract(Engine.TensorMultiplyScalar(sumY2, n), Engine.TensorSquare(sumY));
        var denominator = Engine.TensorSqrt(Engine.TensorMultiplyScalar(profileSpread, chromaSpread));
        var correlationTensor = Engine.TensorWhere(
            Engine.TensorLessThan(denominator, NumOps.FromDouble(1e-10)),
            new Tensor<T>([keys]),
            Engine.TensorDivide(numerator, denominator));

        // Confidence: the correlation normalized to 0-1
        T[] correlations = correlationTensor.ToArray();
        T[] confidences = Engine.TensorDivideScalar(Engine.TensorAddScalar(correlationTensor, NumOps.One), NumOps.FromDouble(2)).ToArray();

        var results = new List<KeyDetectionResult<T>>();

        // Major keys
        for (int key = 0; key < 12; key++)
        {
            results.Add(new KeyDetectionResult<T>
            {
                KeyIndex = key,
                Name = $"{NoteNames[key]} major",
                RootNote = NoteNames[key],
                Mode = KeyMode.Major,
                Correlation = correlations[key],
                Confidence = confidences[key],
                RelativeKey = $"{NoteNames[(key + 9) % 12]} minor"
            });
        }

        // Minor keys
        for (int key = 0; key < 12; key++)
        {
            results.Add(new KeyDetectionResult<T>
            {
                KeyIndex = key,
                Name = $"{NoteNames[key]} minor",
                RootNote = NoteNames[key],
                Mode = KeyMode.Minor,
                Correlation = correlations[12 + key],
                Confidence = confidences[12 + key],
                RelativeKey = $"{NoteNames[(key + 3) % 12]} major"
            });
        }

        return results;
    }

    /// <summary>
    /// The 24 rotated key profiles (12 major, then 12 minor) as a <c>[24, 12]</c> tensor, converted from
    /// the fixed Krumhansl-Kessler tables to T once, in bulk.
    /// </summary>
    private Tensor<T> KeyProfileTensor()
    {
        if (_keyProfileTensor is null)
        {
            var values = new double[24 * 12];
            for (int key = 0; key < 12; key++)
            {
                for (int i = 0; i < 12; i++)
                {
                    values[key * 12 + i] = _majorProfiles[key, i];
                    values[(12 + key) * 12 + i] = _minorProfiles[key, i];
                }
            }
            _keyProfileTensor = new Tensor<double>(values, [24, 12]).Cast<T>();
        }
        return _keyProfileTensor;
    }
}
