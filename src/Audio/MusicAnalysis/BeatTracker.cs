using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Extracts beat and tempo information from audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Beat tracking involves detecting the tempo (beats per minute) and the
/// specific times when beats occur in the audio. This is fundamental for
/// music synchronization and rhythm analysis.
/// </para>
/// <para><b>For Beginners:</b> The "beat" is what you tap your foot to when
/// listening to music. This algorithm finds:
/// - Tempo: How fast the music is (measured in BPM - beats per minute)
/// - Beat times: Exactly when each beat occurs in the song
///
/// For example, a typical pop song is around 120 BPM, meaning there are
/// 120 beats per minute (2 beats per second).
///
/// Usage:
/// <code>
/// var tracker = new BeatTracker&lt;float&gt;();
/// var result = tracker.Track(audioTensor);
/// // Result is available in the returned value
/// foreach (var beat in result.BeatTimes.Take(10))
///     Console.WriteLine($"Beat at {beat:F3}s");
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.SignalProcessing)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Beat Tracking by Dynamic Programming", "https://doi.org/10.1006/jnca.2006.0150")]
public class BeatTracker<T> : MusicAnalysisBase<T>
{
    private readonly SpectralFeatureExtractor<T> _spectralExtractor;
    private readonly BeatTrackerOptions _options;
    private readonly int _fluxIndex;

    /// <summary>
    /// Gets the minimum BPM for beat detection.
    /// </summary>
    public double MinBPM => _options.MinTempo;

    /// <summary>
    /// Gets the maximum BPM for beat detection.
    /// </summary>
    public double MaxBPM => _options.MaxTempo;

    /// <summary>
    /// Creates a new beat tracker.
    /// </summary>
    /// <param name="options">Beat tracking options.</param>
    public BeatTracker(BeatTrackerOptions? options = null)
    {
        _options = options ?? new BeatTrackerOptions();
        ValidateOptions(_options);

        // Set base class properties
        SampleRate = _options.SampleRate;
        HopLength = _options.HopLength;
        FftSize = _options.FftSize;

        // Create spectral extractor - ensure Flux is included for beat tracking
        // We use Flux (spectral change) as the onset strength signal
        _spectralExtractor = new SpectralFeatureExtractor<T>(new SpectralFeatureOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            // Ensure Flux is enabled - it's part of Basic by default
            FeatureTypes = SpectralFeatureType.Flux
        });

        // Get the actual index of Flux in the feature tensor
        _fluxIndex = _spectralExtractor.GetFeatureIndex(SpectralFeatureType.Flux);

        if (_fluxIndex < 0)
        {
            throw new InvalidOperationException(
                "BeatTracker requires spectral flux feature but it is not available. " +
                "Ensure SpectralFeatureType.Flux is enabled in the extractor options.");
        }
    }

    /// <summary>
    /// Tracks beats in the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Beat tracking result with tempo and beat times.</returns>
    public BeatTrackingResult<T> Track(Tensor<T> audio)
    {
        ValidateOptions(_options);
        // Compute spectral flux (onset strength)
        var onsetEnvelope = ComputeOnsetEnvelopeInternal(audio);

        // Estimate tempo from onset autocorrelation
        double tempo = EstimateTempoFromEnvelope(onsetEnvelope);

        // Track beats using dynamic programming
        var beatTimes = TrackBeats(onsetEnvelope, tempo);

        return new BeatTrackingResult<T>
        {
            Tempo = NumOps.FromDouble(tempo),
            TempoConfidence = ComputeConfidence(onsetEnvelope, beatTimes),
            BeatTimes = beatTimes,
            BeatInterval = tempo > 0 ? 60.0 / tempo : 0.0
        };
    }

    /// <summary>
    /// Tracks beats in the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Beat tracking result.</returns>
    public BeatTrackingResult<T> Track(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length], audio);
        return Track(tensor);
    }

    private Tensor<T> ComputeOnsetEnvelopeInternal(Tensor<T> audio)
    {
        // Use spectral flux as onset strength
        var features = _spectralExtractor.Extract(audio);

        // Features shape: [numFrames, numFeatures]
        int numFrames = features.Shape[0];
        if (numFrames == 0)
            return new Tensor<T>([0]);

        // Spectral flux (dynamically determined column), half-wave rectified to only keep increases:
        // max(0, flux), NaN propagating as Math.Max does
        var zeros = new Tensor<T>([numFrames]);
        var flux = Engine.Reshape(Engine.TensorSlice(features, [0, _fluxIndex], [numFrames, 1]), [numFrames]);
        var onsetEnvelope = Engine.TensorWhere(Engine.TensorLessThan(flux, NumOps.Zero), zeros, flux);

        // Smoothing: mean over the frames i - w .. i + w that exist
        int smoothWindow = _options.SmoothingWindow;
        var sum = zeros;
        var count = new int[numFrames];
        for (int i = 0; i < numFrames; i++)
            count[i] = Math.Max(0, Math.Min(numFrames - 1, i + smoothWindow) - Math.Max(0, i - smoothWindow) + 1);
        for (int k = -smoothWindow; k <= smoothWindow; k++)
        {
            if (Math.Abs(k) >= numFrames) continue;
            var shifted = k >= 0
                ? Engine.Concat(new[] { Engine.TensorSlice(onsetEnvelope, [k], [numFrames - k]), new Tensor<T>([k]) }, 0)
                : Engine.Concat(new[] { new Tensor<T>([-k]), Engine.TensorSlice(onsetEnvelope, [0], [numFrames + k]) }, 0);
            sum = Engine.TensorAdd(sum, shifted);
        }
        return Engine.TensorDivide(sum, new Tensor<int>(count, [numFrames]).Cast<T>());
    }

    /// <summary>
    /// Rejects option values that leave no valid lag range: the frame rate and tempo bounds must be
    /// positive, the maximum tempo at least the minimum, and the lag at the minimum tempo must fit in
    /// an <see cref="int"/>.
    /// </summary>
    private static void ValidateOptions(BeatTrackerOptions options)
    {
        if (options.SampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.SampleRate, "BeatTrackerOptions.SampleRate must be positive.");
        if (options.HopLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.HopLength, "BeatTrackerOptions.HopLength must be positive.");
        if (!(options.MinTempo > 0))
            throw new ArgumentOutOfRangeException(nameof(options), options.MinTempo, "BeatTrackerOptions.MinTempo must be positive.");
        if (!(options.MaxTempo >= options.MinTempo))
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxTempo, "BeatTrackerOptions.MaxTempo must be at least MinTempo.");
        double maxLag = options.SampleRate / (double)options.HopLength * 60.0 / options.MinTempo;
        if (maxLag > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(options), options.MinTempo, "BeatTrackerOptions.MinTempo is too small: the beat lag in frames does not fit in an int.");
    }

    private double EstimateTempoFromEnvelope(Tensor<T> onsetEnvelope)
    {
        // Compute autocorrelation of onset envelope
        int maxLag = (int)(_options.SampleRate / (double)_options.HopLength * 60.0 / _options.MinTempo);
        int minLag = (int)(_options.SampleRate / (double)_options.HopLength * 60.0 / _options.MaxTempo);

        // The best lag is the first whose correlation beats 0 and every earlier lag (NaN never wins)
        int bestLag = minLag;
        int length = onsetEnvelope.Length;
        int lagEnd = Math.Min(maxLag, length);
        if (lagEnd > minLag)
        {
            var correlations = new T[lagEnd - minLag];
            for (int lag = minLag; lag < lagEnd; lag++)
            {
                int overlap = length - lag;
                correlations[lag - minLag] = Engine.TensorSum(Engine.TensorMultiply(
                    Engine.TensorSlice(onsetEnvelope, [0], [overlap]), Engine.TensorSlice(onsetEnvelope, [lag], [overlap])));
            }

            var candidates = Engine.Concat(new[] { new Tensor<T>([1]), new Tensor<T>(correlations, [correlations.Length]) }, 0);
            int best = AudioEngineOps.FirstArgMaxSequential(Engine, candidates);
            if (best > 0)
                bestLag = minLag + best - 1;
        }

        // Convert lag to tempo (BPM)
        double frameRate = _options.SampleRate / (double)_options.HopLength;
        double beatPeriod = bestLag / frameRate; // seconds
        double tempo = 60.0 / beatPeriod;

        // Clamp to valid range
        return Math.Max(_options.MinTempo, Math.Min(tempo, _options.MaxTempo));
    }

    private List<double> TrackBeats(Tensor<T> onsetEnvelope, double tempo)
    {
        var beatTimes = new List<double>();

        double frameRate = _options.SampleRate / (double)_options.HopLength;
        double beatPeriod = 60.0 / tempo; // seconds
        double beatPeriodFrames = beatPeriod * frameRate;

        // Dynamic programming beat tracking
        int numFrames = onsetEnvelope.Length;
        var score = new T[numFrames];
        var predecessor = new int[numFrames];

        // Initialize
        for (int i = 0; i < numFrames; i++)
        {
            predecessor[i] = -1;
        }

        // Gaussian penalty for tempo deviation by beat distance d (penalize deviation from the expected
        // beat period), laid out for d = lookBack .. 1 so a window of previous frames lines up with it
        int lookBack = (int)(beatPeriodFrames * 2);
        int maxWindow = Math.Max(0, lookBack);
        Tensor<T> penaltyByDistance = new Tensor<T>([0]);
        if (maxWindow > 0)
        {
            var distance = Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, maxWindow, 1), NumOps.FromDouble(-1)), NumOps.FromDouble(maxWindow));
            T expectedDist = NumOps.FromDouble(beatPeriodFrames);
            var deviation = Engine.TensorDivideScalar(Engine.TensorAbs(Engine.TensorSubtractScalar(distance, expectedDist)), expectedDist);
            T flexibility = NumOps.FromDouble(_options.TempoFlexibility * _options.TempoFlexibility);
            penaltyByDistance = Engine.TensorExp(Engine.TensorDivideScalar(
                Engine.TensorMultiplyScalar(Engine.TensorSquare(deviation), NumOps.FromDouble(-0.5)), flexibility));
        }

        // Forward pass: score[i] = envelope[i] + the first largest positive score[j] * penalty(i - j)
        var noPrevious = new Tensor<T>([1]);
        for (int i = 0; i < numFrames; i++)
        {
            var current = Engine.TensorSlice(onsetEnvelope, [i], [1]);
            int start = Math.Max(0, i - lookBack);
            int window = i - start;
            if (window > 0)
            {
                var previous = new T[window];
                Array.Copy(score, start, previous, 0, window);
                var transition = Engine.TensorMultiply(
                    new Tensor<T>(previous, [window]), Engine.TensorSlice(penaltyByDistance, [maxWindow - window], [window]));
                int best = AudioEngineOps.FirstArgMaxSequential(Engine, Engine.Concat(new[] { noPrevious, transition }, 0));
                if (best > 0)
                {
                    current = Engine.TensorAdd(current, Engine.TensorSlice(transition, [best - 1], [1]));
                    predecessor[i] = start + best - 1;
                }
            }
            score[i] = Engine.TensorSum(current);
        }

        // Backtrack to find beats: start from the last frame (scanning backwards) with the largest positive score
        int last = 0;
        int firstScanned = Math.Max(0, numFrames - (int)beatPeriodFrames);
        int scanned = numFrames - firstScanned;
        if (scanned > 0)
        {
            var backwards = new T[scanned];
            Array.Copy(score, firstScanned, backwards, 0, scanned);
            Array.Reverse(backwards);
            int best = AudioEngineOps.FirstArgMaxSequential(
                Engine, Engine.Concat(new[] { noPrevious, new Tensor<T>(backwards, [scanned]) }, 0));
            if (best > 0)
                last = numFrames - best;
        }

        var beatFrames = new List<int>();
        int frame = last;
        while (frame >= 0)
        {
            beatFrames.Add(frame);
            frame = predecessor[frame];
        }

        beatFrames.Reverse();

        // Convert to times
        foreach (int beatFrame in beatFrames)
        {
            beatTimes.Add(beatFrame / frameRate);
        }

        return beatTimes;
    }

    private T ComputeConfidence(Tensor<T> onsetEnvelope, List<double> beatTimes)
    {
        if (beatTimes.Count < 2) return NumOps.Zero;

        double frameRate = _options.SampleRate / (double)_options.HopLength;

        // Compute how well beats align with onset peaks: the mean of envelope[beat] / max(envelope)
        // (the maximum skips NaN, as Enumerable.Max does, and is NaN only when every value is)
        int length = onsetEnvelope.Length;
        var isNaN = Engine.TensorIsNan(onsetEnvelope);
        var ranked = Engine.TensorWhere(isNaN, Tensor<T>.CreateDefault([length], AudioEngineOps.NegativeInfinityOrMin<T>()), onsetEnvelope);
        int nanCount = Engine.TensorNonzero(Engine.TensorWhere(isNaN, Tensor<T>.CreateDefault([length], NumOps.One), new Tensor<T>([length]))).Length;
        T maxEnvelope = nanCount == length ? NumOps.FromDouble(double.NaN) : Engine.TensorMaxValue(ranked);
        if (NumOps.LessThan(maxEnvelope, NumOps.FromDouble(1e-10))) return NumOps.Zero;

        var frames = new List<int>();
        foreach (double beatTime in beatTimes)
        {
            int frame = (int)(beatTime * frameRate);
            if (frame >= 0 && frame < length)
                frames.Add(frame);
        }

        T totalScore = frames.Count == 0
            ? NumOps.Zero
            : Engine.TensorSum(Engine.TensorDivideScalar(
                Engine.TensorIndexSelect(onsetEnvelope, new Tensor<int>(frames.ToArray(), [frames.Count]), 0), maxEnvelope));
        return NumOps.Divide(totalScore, NumOps.FromDouble(beatTimes.Count));
    }
}
