using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("An Industrial-Strength Audio Search Algorithm", "https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf", Year = 2003, Authors = "Avery Li-Chun Wang")]
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
        var tensor = new Tensor<T>([audio.Length], audio);
        return Fingerprint(tensor);
    }

    private Tensor<T> ComputeMagnitude(Tensor<Complex<T>> stft)
    {
        // |X| = sqrt(re^2 + im^2), shape [numFrames, numBins]
        var real = Engine.TensorReal(stft);
        var imag = Engine.TensorImag(stft);
        return Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(real), Engine.TensorSquare(imag)));
    }

    private List<SpectralPeak> FindPeaks(Tensor<T> magnitude)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        int radius = _options.PeakNeighborhood;
        if (radius < 0)
            throw new IndexOutOfRangeException("PeakNeighborhood must be non-negative.");

        var peaks = new List<SpectralPeak>();
        Tensor<T> peakValues = new Tensor<T>([0]);

        // Find local maxima in the spectrogram: not below the threshold and strictly greater than
        // every neighbour (a comparison with NaN never disqualifies, as before)
        int innerFrames = numFrames - 2 * radius;
        int innerBins = numBins - 2 * radius;
        if (innerFrames > 0 && innerBins > 0)
        {
            var center = Engine.TensorSlice(magnitude, [radius, radius], [innerFrames, innerBins]);
            var ones = Tensor<T>.CreateDefault([innerFrames, innerBins], NumOps.One);
            var isPeak = Engine.TensorSubtract(ones, Engine.TensorLessThan(center, NumOps.FromDouble(_options.PeakThreshold)));
            for (int df = -radius; df <= radius; df++)
            {
                for (int db = -radius; db <= radius; db++)
                {
                    if (df == 0 && db == 0) continue;
                    var neighbor = Engine.TensorSlice(magnitude, [radius + df, radius + db], [innerFrames, innerBins]);
                    isPeak = Engine.TensorMultiply(isPeak, Engine.TensorSubtract(ones, Engine.TensorGreaterEqual(neighbor, center)));
                }
            }

            // Peaks in frame-then-bin order (row-major), with their magnitudes
            var coords = Engine.TensorNonzero(isPeak).ToArray();
            var flatIndex = new int[coords.Length / 2];
            for (int k = 0; k < flatIndex.Length; k++)
            {
                peaks.Add(new SpectralPeak { Frame = coords[2 * k] + radius, Bin = coords[2 * k + 1] + radius });
                flatIndex[k] = coords[2 * k] * innerBins + coords[2 * k + 1];
            }
            if (flatIndex.Length > 0)
            {
                peakValues = Engine.TensorIndexSelect(
                    AudioEngineOps.Flatten(Engine, center), new Tensor<int>(flatIndex, [flatIndex.Length]), 0);
            }
        }

        // Keep only the strongest peaks per time window (stable: equal magnitudes keep peak order,
        // NaN magnitudes rank last)
        var filteredPeaks = new List<SpectralPeak>();
        int windowSize = _options.PeakWindowSizeFrames;

        for (int start = 0; start < numFrames; start += windowSize)
        {
            int windowStart = start;
            var members = Enumerable.Range(0, peaks.Count)
                .Where(i => peaks[i].Frame >= windowStart && peaks[i].Frame < windowStart + windowSize)
                .ToArray();
            if (members.Length == 0) continue;

            var windowValues = Engine.TensorIndexSelect(peakValues, new Tensor<int>(members, [members.Length]), 0);
            filteredPeaks.AddRange(AudioEngineOps.StableDescendingOrder(Engine, windowValues)
                .Take(_options.MaxPeaksPerWindow)
                .Select(rank => peaks[members[rank]]));
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
        var fpData = hashes.Count == 0
            ? new T[0]
            : new Tensor<uint>(hashes.ToArray(), [hashes.Count]).Cast<T>().ToArray();

        // Calculate actual frame count from audio length
        int actualFrameCount = Math.Max(1, (audioLength - _options.FftSize) / _options.HopLength + 1);

        return new AudioFingerprint<T>
        {
            Data = fpData,
            Hash = [.. hashes],
            Duration = (double)audioLength / _options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = Name,
            FrameCount = actualFrameCount,
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
}
