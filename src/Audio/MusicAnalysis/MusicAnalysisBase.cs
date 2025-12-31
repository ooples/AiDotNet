using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Base class for music analysis algorithms (beat tracking, chord recognition, key detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Music analysis algorithms extract musical information from audio signals.
/// Unlike neural network models, many traditional music analysis methods use
/// signal processing techniques that don't require training.
/// </para>
/// <para>
/// <b>For Beginners:</b> Music analysis is about understanding music computationally:
/// - Beat tracking: Finding the rhythm/pulse of music
/// - Chord recognition: Identifying the harmony
/// - Key detection: Finding the musical key (C major, A minor, etc.)
///
/// This base class provides:
/// - Common spectral feature extractors
/// - Chromagram computation for harmonic analysis
/// - Onset detection utilities
/// </para>
/// </remarks>
public abstract class MusicAnalysisBase<T>
{
    /// <summary>
    /// Operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets or sets the expected sample rate for input audio.
    /// </summary>
    public int SampleRate { get; protected set; } = 22050;

    /// <summary>
    /// Gets or sets the hop length for frame-based analysis.
    /// </summary>
    protected int HopLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the FFT size for spectral analysis.
    /// </summary>
    protected int FftSize { get; set; } = 2048;

    /// <summary>
    /// Gets the spectral feature extractor.
    /// </summary>
    protected SpectralFeatureExtractor<T>? SpectralExtractor { get; set; }

    /// <summary>
    /// Gets the chromagram extractor for harmonic analysis.
    /// </summary>
    protected ChromaExtractor<T>? ChromaExtractor { get; set; }

    /// <summary>
    /// Initializes a new instance of the MusicAnalysisBase class.
    /// </summary>
    protected MusicAnalysisBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes onset strength envelope for beat/onset detection.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Array of onset strength values over time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Onset strength measures how "event-like" each moment in the
    /// audio is. Peaks in the onset strength often correspond to beats or note onsets.
    /// This implementation uses spectral flux - the change in spectral energy over time.
    /// </para>
    /// </remarks>
    protected virtual T[] ComputeOnsetStrength(Tensor<T> audio)
    {
        // Compute onset strength using spectral flux
        // First get spectral features, then compute differences
        if (SpectralExtractor is null)
        {
            var options = new SpectralFeatureOptions
            {
                SampleRate = SampleRate,
                FftSize = FftSize,
                HopLength = HopLength,
                FeatureTypes = SpectralFeatureType.Flux
            };
            SpectralExtractor = new SpectralFeatureExtractor<T>(options);
        }

        // Extract spectral features which includes flux
        var features = SpectralExtractor.Extract(audio);

        // Convert the flux feature (first column) to array
        int numFrames = features.Shape[0];
        T[] onsetStrength = new T[numFrames];

        for (int i = 0; i < numFrames; i++)
        {
            onsetStrength[i] = features[i, 0];
        }

        return onsetStrength;
    }

    /// <summary>
    /// Computes tempogram for tempo estimation.
    /// </summary>
    /// <param name="onsetStrength">Onset strength envelope.</param>
    /// <param name="minBpm">Minimum BPM to consider.</param>
    /// <param name="maxBpm">Maximum BPM to consider.</param>
    /// <returns>Tempogram showing tempo strength at each BPM.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tempogram shows how strong the rhythm is at different
    /// tempos (BPMs). The peak in the tempogram indicates the most likely tempo.
    /// </para>
    /// </remarks>
    protected virtual T[] ComputeTempogram(T[] onsetStrength, double minBpm = 30, double maxBpm = 300)
    {
        // Autocorrelation-based tempogram
        int numBins = (int)(maxBpm - minBpm) + 1;
        T[] tempogram = new T[numBins];

        double hopTime = (double)HopLength / SampleRate;

        for (int i = 0; i < numBins; i++)
        {
            double bpm = minBpm + i;
            double period = 60.0 / bpm; // Period in seconds
            int lag = (int)(period / hopTime);

            if (lag >= onsetStrength.Length)
            {
                tempogram[i] = NumOps.Zero;
                continue;
            }

            // Compute autocorrelation at this lag
            T correlation = NumOps.Zero;
            int count = 0;
            for (int j = 0; j < onsetStrength.Length - lag; j++)
            {
                correlation = NumOps.Add(correlation,
                    NumOps.Multiply(onsetStrength[j], onsetStrength[j + lag]));
                count++;
            }

            if (count > 0)
            {
                tempogram[i] = NumOps.Divide(correlation, NumOps.FromDouble(count));
            }
        }

        return tempogram;
    }

    /// <summary>
    /// Extracts chromagram (pitch class profile) from audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Chromagram tensor [time_frames, 12].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A chromagram shows the energy in each of the 12 musical
    /// pitch classes (C, C#, D, ..., B) over time. It's fundamental for chord
    /// and key detection because it captures harmonic content.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> ExtractChromagram(Tensor<T> audio)
    {
        if (ChromaExtractor is null)
        {
            var options = new ChromaOptions
            {
                SampleRate = SampleRate,
                FftSize = FftSize,
                HopLength = HopLength
            };
            ChromaExtractor = new ChromaExtractor<T>(options);
        }

        return ChromaExtractor.Extract(audio);
    }

    /// <summary>
    /// Finds peaks in a signal (for beat detection, etc.).
    /// </summary>
    /// <param name="signal">Input signal.</param>
    /// <param name="threshold">Minimum peak height (relative to max).</param>
    /// <param name="minDistance">Minimum distance between peaks.</param>
    /// <returns>Indices of detected peaks.</returns>
    protected virtual int[] FindPeaks(T[] signal, double threshold = 0.3, int minDistance = 1)
    {
        var peaks = new List<int>();

        if (signal.Length == 0) return peaks.ToArray();

        // Find global max for threshold
        T maxVal = signal[0];
        for (int i = 1; i < signal.Length; i++)
        {
            if (NumOps.GreaterThan(signal[i], maxVal))
            {
                maxVal = signal[i];
            }
        }

        T absThreshold = NumOps.Multiply(maxVal, NumOps.FromDouble(threshold));

        for (int i = 1; i < signal.Length - 1; i++)
        {
            // Check if this is a local maximum above threshold
            if (NumOps.GreaterThan(signal[i], signal[i - 1]) &&
                NumOps.GreaterThan(signal[i], signal[i + 1]) &&
                NumOps.GreaterThan(signal[i], absThreshold))
            {
                // Check minimum distance from previous peak
                if (peaks.Count == 0 || (i - peaks[^1]) >= minDistance)
                {
                    peaks.Add(i);
                }
                else if (NumOps.GreaterThan(signal[i], signal[peaks[^1]]))
                {
                    // Replace previous peak if this one is higher
                    peaks[^1] = i;
                }
            }
        }

        return peaks.ToArray();
    }

    /// <summary>
    /// Converts frame index to time in seconds.
    /// </summary>
    /// <param name="frameIndex">Frame index.</param>
    /// <param name="hopLength">Hop length in samples. If null, uses the class default.</param>
    /// <returns>Time in seconds.</returns>
    protected double FrameToTime(int frameIndex, int? hopLength = null)
    {
        int hop = hopLength ?? HopLength;
        return (double)(frameIndex * hop) / SampleRate;
    }

    /// <summary>
    /// Converts time in seconds to frame index.
    /// </summary>
    /// <param name="time">Time in seconds.</param>
    /// <param name="hopLength">Hop length in samples. If null, uses the class default.</param>
    /// <returns>Frame index.</returns>
    protected int TimeToFrame(double time, int? hopLength = null)
    {
        int hop = hopLength ?? HopLength;
        return (int)(time * SampleRate / hop);
    }
}
