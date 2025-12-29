using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Spectrogram peak-based audio fingerprinter (Shazam-style).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This fingerprinter uses spectral peak detection similar to the Shazam algorithm.
/// It finds prominent peaks in the spectrogram and creates hash codes from peak
/// pairs, providing robustness to noise and speed variations.
/// </para>
/// <para><b>For Beginners:</b> This algorithm finds the loudest frequency points
/// in the audio (like mountain peaks on a landscape) and remembers their positions.
/// By comparing peak patterns, it can identify songs even with background noise
/// or slight speed changes.
/// </para>
/// </remarks>
public class SpectrogramFingerprinter<T> : AudioFingerprinterBase<T>
{
    private readonly ShortTimeFourierTransform<T> _stft;
    private readonly SpectrogramFingerprintOptions _options;

    /// <summary>
    /// Gets the name of the fingerprinting algorithm.
    /// </summary>
    public override string Name => "SpectrogramPeaks";

    /// <summary>
    /// Creates a new spectrogram-based fingerprinter.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public SpectrogramFingerprinter(SpectrogramFingerprintOptions? options = null)
    {
        _options = options ?? new SpectrogramFingerprintOptions();

        // Set base class properties
        SampleRate = _options.SampleRate;
        FingerprintLength = 32;

        _stft = new ShortTimeFourierTransform<T>(
            nFft: _options.FftSize,
            hopLength: _options.HopLength);
    }

    /// <summary>
    /// Generates a fingerprint from audio tensor.
    /// </summary>
    public override AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // Compute STFT
        var stftResult = _stft.Forward(audio);

        // Extract magnitude spectrogram
        var magnitude = ComputeMagnitude(stftResult);

        // Find spectral peaks
        var peaks = FindPeaks(magnitude);

        // Generate hashes from peak pairs
        return CreateFingerprintFromPeaks(peaks, audio.Length);
    }

    /// <summary>
    /// Generates a fingerprint from audio vector.
    /// </summary>
    public override AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Fingerprint(tensor);
    }

    private double[,] ComputeMagnitude(Tensor<Complex<T>> stft)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];

        var magnitude = new double[numFrames, numBins];

        for (int f = 0; f < numFrames; f++)
        {
            for (int b = 0; b < numBins; b++)
            {
                var complex = stft[f, b];
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                magnitude[f, b] = Math.Sqrt(real * real + imag * imag);
            }
        }

        return magnitude;
    }

    private List<SpectralPeak> FindPeaks(double[,] magnitude)
    {
        int numFrames = magnitude.GetLength(0);
        int numBins = magnitude.GetLength(1);

        var peaks = new List<SpectralPeak>();

        // Find local maxima in the spectrogram
        for (int f = _options.PeakNeighborhood; f < numFrames - _options.PeakNeighborhood; f++)
        {
            for (int b = _options.PeakNeighborhood; b < numBins - _options.PeakNeighborhood; b++)
            {
                double value = magnitude[f, b];

                if (value < _options.PeakThreshold)
                    continue;

                // Check if this is a local maximum
                bool isPeak = true;
                for (int df = -_options.PeakNeighborhood; df <= _options.PeakNeighborhood && isPeak; df++)
                {
                    for (int db = -_options.PeakNeighborhood; db <= _options.PeakNeighborhood && isPeak; db++)
                    {
                        if (df == 0 && db == 0) continue;

                        if (magnitude[f + df, b + db] >= value)
                        {
                            isPeak = false;
                        }
                    }
                }

                if (isPeak)
                {
                    peaks.Add(new SpectralPeak
                    {
                        Frame = f,
                        Bin = b,
                        Magnitude = value
                    });
                }
            }
        }

        // Keep only the strongest peaks per time window
        var filteredPeaks = new List<SpectralPeak>();
        int windowSize = _options.PeaksPerSecond;

        for (int start = 0; start < numFrames; start += windowSize)
        {
            var windowPeaks = peaks
                .Where(p => p.Frame >= start && p.Frame < start + windowSize)
                .OrderByDescending(p => p.Magnitude)
                .Take(_options.MaxPeaksPerWindow)
                .ToList();

            filteredPeaks.AddRange(windowPeaks);
        }

        return filteredPeaks.OrderBy(p => p.Frame).ThenBy(p => p.Bin).ToList();
    }

    private AudioFingerprint<T> CreateFingerprintFromPeaks(List<SpectralPeak> peaks, int audioLength)
    {
        var hashes = new List<uint>();
        var hashToTime = new Dictionary<uint, double>();

        // Create hashes from peak pairs (anchor + target)
        for (int i = 0; i < peaks.Count; i++)
        {
            var anchor = peaks[i];

            // Find target peaks within the target zone
            for (int j = i + 1; j < peaks.Count; j++)
            {
                var target = peaks[j];

                int timeDiff = target.Frame - anchor.Frame;
                if (timeDiff < _options.TargetZoneStart)
                    continue;
                if (timeDiff > _options.TargetZoneEnd)
                    break;

                // Create hash: freq1 (10 bits) | freq2 (10 bits) | time delta (12 bits)
                uint hash = CreateHash(anchor.Bin, target.Bin, timeDiff);
                hashes.Add(hash);

                // Store time for this hash
                double time = anchor.Frame * _options.HopLength / (double)_options.SampleRate;
                if (!hashToTime.ContainsKey(hash))
                {
                    hashToTime[hash] = time;
                }
            }
        }

        // Convert to fingerprint data
        var fpData = hashes.Select(h => NumOps.FromDouble(h)).ToArray();

        return new AudioFingerprint<T>
        {
            Data = fpData,
            Hash = [.. hashes],
            Duration = (double)audioLength / _options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = Name,
            FrameCount = peaks.Count,
            Metadata = new Dictionary<string, object>
            {
                ["peak_count"] = peaks.Count,
                ["hash_count"] = hashes.Count
            }
        };
    }

    private static uint CreateHash(int freq1, int freq2, int timeDelta)
    {
        // Clamp values to fit in bit fields
        freq1 = Math.Max(0, Math.Min(freq1, 1023));
        freq2 = Math.Max(0, Math.Min(freq2, 1023));
        timeDelta = Math.Max(0, Math.Min(timeDelta, 4095));

        return ((uint)freq1 << 22) | ((uint)freq2 << 12) | (uint)timeDelta;
    }

    /// <summary>
    /// Computes similarity between two fingerprints.
    /// </summary>
    public override double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        if (fp1.Hash is null || fp2.Hash is null)
            return 0;

        // Build hash set from first fingerprint
        var hashSet = new HashSet<uint>(fp1.Hash);

        // Count matching hashes
        int matches = fp2.Hash.Count(h => hashSet.Contains(h));

        // Jaccard similarity
        int union = hashSet.Count + fp2.Hash.Length - matches;
        if (union == 0) return 0;

        return (double)matches / union;
    }

    /// <summary>
    /// Finds matching segments between fingerprints.
    /// </summary>
    public override IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10)
    {
        var matches = new List<FingerprintMatch>();

        if (query.Hash is null || reference.Hash is null)
            return matches;

        // Build hash-to-position index for reference
        var refIndex = new Dictionary<uint, List<int>>();
        for (int i = 0; i < reference.Hash.Length; i++)
        {
            uint h = reference.Hash[i];
            if (!refIndex.TryGetValue(h, out var positions))
            {
                positions = [];
                refIndex[h] = positions;
            }
            positions.Add(i);
        }

        // Track offset histogram
        var offsetCounts = new Dictionary<int, int>();

        for (int q = 0; q < query.Hash.Length; q++)
        {
            if (refIndex.TryGetValue(query.Hash[q], out var refPositions))
            {
                foreach (int r in refPositions)
                {
                    int offset = r - q;
                    offsetCounts[offset] = offsetCounts.GetValueOrDefault(offset) + 1;
                }
            }
        }

        // Find significant offset peaks
        double frameRate = _options.SampleRate / (double)_options.HopLength;
        var significantOffsets = offsetCounts
            .Where(kv => kv.Value >= minMatchLength)
            .OrderByDescending(kv => kv.Value);

        foreach (var (offset, matchCount) in significantOffsets)
        {
            // Estimate match region
            double refStartTime = Math.Max(0, offset) / frameRate;
            double queryStartTime = Math.Max(0, -offset) / frameRate;
            double duration = Math.Min(query.Duration - queryStartTime, reference.Duration - refStartTime);

            if (duration > 0)
            {
                matches.Add(new FingerprintMatch
                {
                    QueryStartTime = queryStartTime,
                    ReferenceStartTime = refStartTime,
                    Duration = duration,
                    Confidence = (double)matchCount / query.Hash.Length,
                    MatchCount = matchCount
                });
            }
        }

        return matches;
    }
}

/// <summary>
/// Represents a spectral peak.
/// </summary>
internal class SpectralPeak
{
    public int Frame { get; set; }
    public int Bin { get; set; }
    public double Magnitude { get; set; }
}
