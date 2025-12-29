using AiDotNet.Audio.Features;
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
/// Console.WriteLine($"Tempo: {result.Tempo} BPM");
/// foreach (var beat in result.BeatTimes.Take(10))
///     Console.WriteLine($"Beat at {beat:F3}s");
/// </code>
/// </para>
/// </remarks>
public class BeatTracker<T> : MusicAnalysisBase<T>
{
    private readonly SpectralFeatureExtractor<T> _spectralExtractor;
    private readonly BeatTrackerOptions _options;

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

        _spectralExtractor = new SpectralFeatureExtractor<T>(new SpectralFeatureOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });
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
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Track(tensor);
    }

    private double[] ComputeOnsetEnvelopeInternal(Tensor<T> audio)
    {
        // Use spectral flux as onset strength
        var features = _spectralExtractor.Extract(audio);

        // Features shape: [numFrames, numFeatures]
        int numFrames = features.Shape[0];
        var onsetEnvelope = new double[numFrames];

        // Get spectral flux (column 4 in our SpectralFeatureExtractor)
        for (int f = 0; f < numFrames; f++)
        {
            // Use spectral flux (index 4) as onset strength
            onsetEnvelope[f] = Math.Max(0, NumOps.ToDouble(features[f, 4]));
        }

        // Apply half-wave rectification and smoothing
        var smoothed = new double[numFrames];
        int smoothWindow = _options.SmoothingWindow;

        for (int i = 0; i < numFrames; i++)
        {
            double sum = 0;
            int count = 0;
            for (int j = Math.Max(0, i - smoothWindow); j <= Math.Min(numFrames - 1, i + smoothWindow); j++)
            {
                sum += onsetEnvelope[j];
                count++;
            }
            smoothed[i] = sum / count;
        }

        return smoothed;
    }

    private double EstimateTempoFromEnvelope(double[] onsetEnvelope)
    {
        // Compute autocorrelation of onset envelope
        int maxLag = (int)(_options.SampleRate / _options.HopLength * 60.0 / _options.MinTempo);
        int minLag = (int)(_options.SampleRate / _options.HopLength * 60.0 / _options.MaxTempo);

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
