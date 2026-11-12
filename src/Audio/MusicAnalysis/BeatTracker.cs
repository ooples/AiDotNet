using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
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
    public BeatTrackingResult Track(Tensor<T> audio)
    {
        // Compute spectral flux (onset strength)
        var onsetEnvelope = ComputeOnsetEnvelopeInternal(audio);

        // Estimate tempo from onset autocorrelation
        double tempo = EstimateTempoFromEnvelope(onsetEnvelope);

        // Track beats using dynamic programming
        var beatTimes = TrackBeats(onsetEnvelope, tempo);

        return new BeatTrackingResult
        {
            Tempo = tempo,
            BeatTimes = beatTimes,
            ConfidenceScore = ComputeConfidence(onsetEnvelope, beatTimes)
        };
    }

    /// <summary>
    /// Tracks beats in the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Beat tracking result.</returns>
    public BeatTrackingResult Track(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length], audio);
        return Track(tensor);
    }

    private double[] ComputeOnsetEnvelopeInternal(Tensor<T> audio)
    {
        // Use spectral flux as onset strength
        var features = _spectralExtractor.Extract(audio);

        // Features shape: [numFrames, numFeatures]
        int numFrames = features.Shape[0];
        if (numFrames == 0)
            return new double[0];

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
        var smoothed = Engine.TensorDivide(sum, new Tensor<int>(count, [numFrames]).Cast<T>());

        // The beat tracker works on the double envelope
        return smoothed.Cast<double>().ToArray();
    }

    private double EstimateTempoFromEnvelope(double[] onsetEnvelope)
    {
        // Compute autocorrelation of onset envelope
        int maxLag = (int)(_options.SampleRate / (double)_options.HopLength * 60.0 / _options.MinTempo);
        int minLag = (int)(_options.SampleRate / (double)_options.HopLength * 60.0 / _options.MaxTempo);

        var autocorr = new double[maxLag];
        double maxCorr = 0;
        int bestLag = minLag;

        for (int lag = minLag; lag < maxLag && lag < onsetEnvelope.Length; lag++)
        {
            double corr = 0;
            for (int i = 0; i < onsetEnvelope.Length - lag; i++)
            {
                corr += onsetEnvelope[i] * onsetEnvelope[i + lag];
            }

            autocorr[lag] = corr;

            if (corr > maxCorr)
            {
                maxCorr = corr;
                bestLag = lag;
            }
        }

        // Convert lag to tempo (BPM)
        double frameRate = _options.SampleRate / (double)_options.HopLength;
        double beatPeriod = bestLag / frameRate; // seconds
        double tempo = 60.0 / beatPeriod;

        // Clamp to valid range
        return Math.Max(_options.MinTempo, Math.Min(tempo, _options.MaxTempo));
    }

    private List<double> TrackBeats(double[] onsetEnvelope, double tempo)
    {
        var beatTimes = new List<double>();

        double frameRate = _options.SampleRate / (double)_options.HopLength;
        double beatPeriod = 60.0 / tempo; // seconds
        double beatPeriodFrames = beatPeriod * frameRate;

        // Dynamic programming beat tracking
        int numFrames = onsetEnvelope.Length;
        var score = new double[numFrames];
        var predecessor = new int[numFrames];

        // Initialize
        for (int i = 0; i < numFrames; i++)
        {
            predecessor[i] = -1;
        }

        // Forward pass
        for (int i = 0; i < numFrames; i++)
        {
            score[i] = onsetEnvelope[i];

            // Look back for previous beat
            int lookBack = (int)(beatPeriodFrames * 2);
            double bestPrevScore = 0;
            int bestPrev = -1;

            for (int j = Math.Max(0, i - lookBack); j < i; j++)
            {
                // Compute transition score (penalize deviation from expected beat period)
                double expectedDist = beatPeriodFrames;
                double actualDist = i - j;
                double deviation = Math.Abs(actualDist - expectedDist) / expectedDist;

                // Gaussian penalty for tempo deviation
                double penalty = Math.Exp(-0.5 * deviation * deviation / (_options.TempoFlexibility * _options.TempoFlexibility));

                double transitionScore = score[j] * penalty;

                if (transitionScore > bestPrevScore)
                {
                    bestPrevScore = transitionScore;
                    bestPrev = j;
                }
            }

            if (bestPrev >= 0)
            {
                score[i] += bestPrevScore;
                predecessor[i] = bestPrev;
            }
        }

        // Backtrack to find beats
        int current = 0;
        double maxScore = 0;
        for (int i = numFrames - 1; i >= Math.Max(0, numFrames - (int)beatPeriodFrames); i--)
        {
            if (score[i] > maxScore)
            {
                maxScore = score[i];
                current = i;
            }
        }

        var beatFrames = new List<int>();
        while (current >= 0)
        {
            beatFrames.Add(current);
            current = predecessor[current];
        }

        beatFrames.Reverse();

        // Convert to times
        foreach (int frame in beatFrames)
        {
            beatTimes.Add(frame / frameRate);
        }

        return beatTimes;
    }

    private double ComputeConfidence(double[] onsetEnvelope, List<double> beatTimes)
    {
        if (beatTimes.Count < 2) return 0;

        double frameRate = _options.SampleRate / (double)_options.HopLength;

        // Compute how well beats align with onset peaks
        double totalScore = 0;
        double maxEnvelope = onsetEnvelope.Max();
        if (maxEnvelope < 1e-10) return 0;

        foreach (double beatTime in beatTimes)
        {
            int frame = (int)(beatTime * frameRate);
            if (frame >= 0 && frame < onsetEnvelope.Length)
            {
                totalScore += onsetEnvelope[frame] / maxEnvelope;
            }
        }

        return totalScore / beatTimes.Count;
    }
}
