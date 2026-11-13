using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.Metrics;

/// <summary>
/// Word Error Rate (WER) metric for speech recognition evaluation.
/// </summary>
/// <remarks>
/// <para>
/// WER measures the accuracy of automatic speech recognition (ASR) systems by computing
/// the edit distance between the predicted transcription and the reference transcription.
/// </para>
/// <para>
/// Formula: WER = (Substitutions + Insertions + Deletions) / Number of words in reference
/// </para>
/// <para>
/// Typical WER values:
/// - &lt;5%: Excellent (human-level performance)
/// - 5-10%: Very good
/// - 10-20%: Good
/// - 20-30%: Acceptable
/// - &gt;30%: Poor
/// </para>
/// </remarks>
public class WordErrorRate
{
    /// <summary>
    /// Computes the Word Error Rate between a hypothesis and reference transcription.
    /// </summary>
    /// <param name="hypothesis">The predicted transcription.</param>
    /// <param name="reference">The ground truth transcription.</param>
    /// <returns>WER as a value between 0 (perfect) and potentially &gt;1 for very poor predictions.</returns>
    public double Compute(string hypothesis, string reference)
    {
        if (string.IsNullOrEmpty(reference))
        {
            return string.IsNullOrEmpty(hypothesis) ? 0.0 : 1.0;
        }

        var hypWords = TokenizeWords(hypothesis);
        var refWords = TokenizeWords(reference);

        if (refWords.Length == 0)
        {
            return hypWords.Length == 0 ? 0.0 : 1.0;
        }

        var (substitutions, insertions, deletions) = ComputeEditOperations(hypWords, refWords);
        return (double)(substitutions + insertions + deletions) / refWords.Length;
    }

    /// <summary>
    /// Computes WER for a batch of hypothesis-reference pairs.
    /// </summary>
    /// <param name="hypotheses">Array of predicted transcriptions.</param>
    /// <param name="references">Array of ground truth transcriptions.</param>
    /// <returns>Average WER across all pairs.</returns>
    public double ComputeBatch(string[] hypotheses, string[] references)
    {
        if (hypotheses.Length != references.Length)
        {
            throw new ArgumentException("Number of hypotheses must match number of references");
        }

        if (hypotheses.Length == 0)
        {
            return 0.0;
        }

        double totalWer = 0.0;
        for (int i = 0; i < hypotheses.Length; i++)
        {
            totalWer += Compute(hypotheses[i], references[i]);
        }

        return totalWer / hypotheses.Length;
    }

    /// <summary>
    /// Computes detailed error statistics including substitutions, insertions, and deletions.
    /// </summary>
    /// <param name="hypothesis">The predicted transcription.</param>
    /// <param name="reference">The ground truth transcription.</param>
    /// <returns>Tuple of (WER, substitutions, insertions, deletions, reference word count).</returns>
    public (double wer, int substitutions, int insertions, int deletions, int refWordCount) ComputeDetailed(
        string hypothesis, string reference)
    {
        var hypWords = TokenizeWords(hypothesis);
        var refWords = TokenizeWords(reference);

        if (refWords.Length == 0)
        {
            int ins = hypWords.Length;
            return (hypWords.Length == 0 ? 0.0 : 1.0, 0, ins, 0, 0);
        }

        var (substitutions, insertions, deletions) = ComputeEditOperations(hypWords, refWords);
        double wer = (double)(substitutions + insertions + deletions) / refWords.Length;

        return (wer, substitutions, insertions, deletions, refWords.Length);
    }

    /// <summary>
    /// Tokenizes a string into words.
    /// </summary>
    private static string[] TokenizeWords(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<string>();
        }

        return text.ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>
    /// Computes the minimum edit operations using dynamic programming (Levenshtein distance variant).
    /// </summary>
    private static (int substitutions, int insertions, int deletions) ComputeEditOperations(
        string[] hypothesis, string[] reference)
    {
        int m = reference.Length;
        int n = hypothesis.Length;

        // dp[i,j] stores (cost, substitutions, insertions, deletions) to transform ref[0:i] to hyp[0:j]
        var dp = new int[m + 1, n + 1];
        var ops = new (int sub, int ins, int del)[m + 1, n + 1];

        // Initialize base cases
        for (int i = 0; i <= m; i++)
        {
            dp[i, 0] = i; // Delete all reference words
            ops[i, 0] = (0, 0, i);
        }

        for (int j = 0; j <= n; j++)
        {
            dp[0, j] = j; // Insert all hypothesis words
            ops[0, j] = (0, j, 0);
        }

        // Fill the DP table
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                if (reference[i - 1].Equals(hypothesis[j - 1], StringComparison.OrdinalIgnoreCase))
                {
                    // Words match - no operation needed
                    dp[i, j] = dp[i - 1, j - 1];
                    ops[i, j] = ops[i - 1, j - 1];
                }
                else
                {
                    // Find minimum cost operation
                    int subCost = dp[i - 1, j - 1] + 1;
                    int insCost = dp[i, j - 1] + 1;
                    int delCost = dp[i - 1, j] + 1;

                    if (subCost <= insCost && subCost <= delCost)
                    {
                        dp[i, j] = subCost;
                        var prev = ops[i - 1, j - 1];
                        ops[i, j] = (prev.sub + 1, prev.ins, prev.del);
                    }
                    else if (insCost <= delCost)
                    {
                        dp[i, j] = insCost;
                        var prev = ops[i, j - 1];
                        ops[i, j] = (prev.sub, prev.ins + 1, prev.del);
                    }
                    else
                    {
                        dp[i, j] = delCost;
                        var prev = ops[i - 1, j];
                        ops[i, j] = (prev.sub, prev.ins, prev.del + 1);
                    }
                }
            }
        }

        return ops[m, n];
    }
}

/// <summary>
/// Character Error Rate (CER) metric for speech recognition and OCR evaluation.
/// </summary>
/// <remarks>
/// <para>
/// CER is similar to WER but operates at the character level rather than word level.
/// It's particularly useful for languages without clear word boundaries or for OCR evaluation.
/// </para>
/// <para>
/// Formula: CER = (Substitutions + Insertions + Deletions) / Number of characters in reference
/// </para>
/// </remarks>
public class CharacterErrorRate
{
    /// <summary>
    /// Computes the Character Error Rate between a hypothesis and reference.
    /// </summary>
    /// <param name="hypothesis">The predicted text.</param>
    /// <param name="reference">The ground truth text.</param>
    /// <param name="ignoreWhitespace">Whether to ignore whitespace characters.</param>
    /// <returns>CER as a value between 0 (perfect) and potentially &gt;1.</returns>
    public double Compute(string hypothesis, string reference, bool ignoreWhitespace = false)
    {
        if (string.IsNullOrEmpty(reference))
        {
            return string.IsNullOrEmpty(hypothesis) ? 0.0 : 1.0;
        }

        string hyp = ignoreWhitespace ? RemoveWhitespace(hypothesis) : hypothesis;
        string refText = ignoreWhitespace ? RemoveWhitespace(reference) : reference;

        if (refText.Length == 0)
        {
            return hyp.Length == 0 ? 0.0 : 1.0;
        }

        int editDistance = ComputeLevenshteinDistance(hyp, refText);
        return (double)editDistance / refText.Length;
    }

    /// <summary>
    /// Computes CER for a batch of hypothesis-reference pairs.
    /// </summary>
    public double ComputeBatch(string[] hypotheses, string[] references, bool ignoreWhitespace = false)
    {
        if (hypotheses.Length != references.Length)
        {
            throw new ArgumentException("Number of hypotheses must match number of references");
        }

        if (hypotheses.Length == 0)
        {
            return 0.0;
        }

        double totalCer = 0.0;
        for (int i = 0; i < hypotheses.Length; i++)
        {
            totalCer += Compute(hypotheses[i], references[i], ignoreWhitespace);
        }

        return totalCer / hypotheses.Length;
    }

    private static string RemoveWhitespace(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        var chars = new List<char>(text.Length);
        foreach (char c in text)
        {
            if (!char.IsWhiteSpace(c))
            {
                chars.Add(c);
            }
        }

        return new string(chars.ToArray());
    }

    private static int ComputeLevenshteinDistance(string s1, string s2)
    {
        int m = s1.Length;
        int n = s2.Length;

        var dp = new int[m + 1, n + 1];

        for (int i = 0; i <= m; i++)
        {
            dp[i, 0] = i;
        }

        for (int j = 0; j <= n; j++)
        {
            dp[0, j] = j;
        }

        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(
                    Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1),
                    dp[i - 1, j - 1] + cost);
            }
        }

        return dp[m, n];
    }
}

/// <summary>
/// Short-Time Objective Intelligibility (STOI) metric for speech intelligibility assessment.
/// </summary>
/// <remarks>
/// <para>
/// STOI predicts the intelligibility of degraded speech signals. It correlates well with
/// human listening tests and is commonly used to evaluate speech enhancement and separation.
/// </para>
/// <para>
/// Values are correlations, so they lie in [-1, 1]; in practice they fall between 0 and 1, where higher
/// values indicate better intelligibility.
/// - &gt;0.9: Excellent intelligibility
/// - 0.7-0.9: Good intelligibility
/// - 0.5-0.7: Fair intelligibility
/// - &lt;0.5: Poor intelligibility
/// </para>
/// <para>
/// Based on "A short-time objective intelligibility measure for time-frequency weighted noisy speech"
/// by Taal et al. (2011), following the pystoi reference implementation: both signals are resampled to
/// 10 kHz, frames more than 40 dB below the loudest clean frame are removed, the Hann-windowed STFT
/// is grouped into one-third octave bands from 150 Hz, and the clipped, normalized degraded envelopes
/// are correlated with the clean envelopes over 30-frame (384 ms) segments.
/// </para>
/// <para>
/// When fewer than 30 frames remain after silence removal there is nothing to correlate, and the
/// score is 1e-5 (as in pystoi).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ShortTimeObjectiveIntelligibility<T> where T : struct
{
    // Constants of Taal et al. (2011) as used by pystoi.
    private const int InternalSampleRate = 10000;
    private const int SegmentFrames = 30;
    private const double LowestBandCenterHz = 150.0;
    private const double ClipLowerBoundDb = -15.0;
    private const double DynamicRangeDb = 40.0;
    private const double NotEnoughFramesScore = 1e-5;

    // numpy's float64 machine epsilon, the guard pystoi adds to every norm.
    private const double Epsilon = 2.220446049250313e-16;

    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;
    private readonly int _frameLength;
    private readonly int _hopLength;
    private readonly int _numBands;
    private readonly int _fftLength;
    private readonly Tensor<T> _window;
    private readonly Tensor<T> _bandMatrix;
    private readonly int _upFactor;
    private readonly int _downFactor;
    private readonly int _filterHalfLength;
    private readonly List<(Tensor<T> Taps, int TapCount, int Offset)> _phases = new List<(Tensor<T> Taps, int TapCount, int Offset)>();

    /// <summary>
    /// Initializes a new instance of STOI calculator.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000. Signals are resampled to 10 kHz.</param>
    /// <param name="frameLength">Analysis frame length in samples at 10 kHz. Default is 256 (25.6 ms).</param>
    /// <param name="hopLength">Hop length between frames at 10 kHz. Default is 128.</param>
    /// <param name="numBands">Number of one-third octave bands. Default is 15.</param>
    public ShortTimeObjectiveIntelligibility(
        int sampleRate = 16000,
        int frameLength = 256,
        int hopLength = 128,
        int numBands = 15)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        }

        if (frameLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frameLength), "Frame length must be positive.");
        }

        if (hopLength <= 0 || hopLength > frameLength)
        {
            throw new ArgumentOutOfRangeException(nameof(hopLength), "Hop length must be positive and not exceed frame length.");
        }

        if (numBands <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numBands), "Number of bands must be positive.");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _sampleRate = sampleRate;
        _frameLength = frameLength;
        _hopLength = hopLength;
        _numBands = numBands;

        // pystoi zero-pads each frame to an FFT twice the frame length (512 for 256-sample frames).
        _fftLength = 1;
        while (_fftLength < 2 * frameLength)
        {
            _fftLength *= 2;
        }

        var engine = AiDotNetEngine.Current;

        // numpy.hanning(N + 2)[1:-1], the MATLAB hanning(N) window.
        _window = engine.TensorNarrow(engine.TensorHannWindow<T>(frameLength + 2, false), 0, 1, frameLength);
        _bandMatrix = BuildBandMatrix(engine);

        int divisor = GreatestCommonDivisor(InternalSampleRate, sampleRate);
        _upFactor = InternalSampleRate / divisor;
        _downFactor = sampleRate / divisor;
        _filterHalfLength = _upFactor != _downFactor ? BuildResamplingPhases(engine) : 0;
    }

    /// <summary>
    /// Computes STOI between degraded and clean speech signals.
    /// </summary>
    /// <param name="degraded">The degraded/processed speech signal.</param>
    /// <param name="clean">The clean reference speech signal.</param>
    /// <returns>STOI score; 1 for identical signals and lower for less intelligible ones.</returns>
    public T Compute(Tensor<T> degraded, Tensor<T> clean)
    {
        if (degraded == null) throw new ArgumentNullException(nameof(degraded));
        if (clean == null) throw new ArgumentNullException(nameof(clean));

        if (degraded.Length != clean.Length)
        {
            throw new ArgumentException("Degraded and clean signals must have the same length");
        }

        if (clean.Length < 2)
        {
            return _numOps.FromDouble(NotEnoughFramesScore);
        }

        var engine = AiDotNetEngine.Current;
        var x = engine.Reshape(clean, new[] { clean.Length });
        var y = engine.Reshape(degraded, new[] { degraded.Length });
        if (_upFactor != _downFactor)
        {
            x = Resample(engine, x);
            y = Resample(engine, y);
        }

        // Remove the frames of both signals where the clean signal is more than 40 dB below its loudest frame.
        var cleanFrames = WindowedFrames(engine, x);
        var degradedFrames = WindowedFrames(engine, y);
        if (cleanFrames is null || degradedFrames is null)
        {
            return _numOps.FromDouble(NotEnoughFramesScore);
        }

        var energies = engine.TensorMultiplyScalar(
            engine.TensorLog10(engine.TensorAddScalar(engine.TensorNorm(cleanFrames, 1, false), _numOps.FromDouble(Epsilon))),
            _numOps.FromDouble(20.0));
        var threshold = _numOps.Subtract(engine.TensorMaxValue(energies), _numOps.FromDouble(DynamicRangeDb));
        var coordinates = engine.TensorNonzero(engine.TensorGreaterThan(energies, threshold));
        var active = engine.Reshape(coordinates, new[] { coordinates.Shape[0] });
        x = OverlapAdd(engine, engine.TensorIndexSelect(cleanFrames, active, 0));
        y = OverlapAdd(engine, engine.TensorIndexSelect(degradedFrames, active, 0));

        var cleanSpectrumFrames = WindowedFrames(engine, x);
        var degradedSpectrumFrames = WindowedFrames(engine, y);
        if (cleanSpectrumFrames is null || degradedSpectrumFrames is null || cleanSpectrumFrames.Shape[0] < SegmentFrames)
        {
            return _numOps.FromDouble(NotEnoughFramesScore);
        }

        // One-third octave band envelopes [frames, bands], then 30-frame segments [segments, bands, 30].
        var cleanSegments = engine.TensorUnfold(OctaveBandEnvelopes(engine, cleanSpectrumFrames), 0, SegmentFrames, 1);
        var degradedSegments = engine.TensorUnfold(OctaveBandEnvelopes(engine, degradedSpectrumFrames), 0, SegmentFrames, 1);
        int[] segmentShape = cleanSegments.Shape.ToArray();

        // Scale the degraded envelope to the clean energy, then clip it at -15 dB signal-to-distortion.
        var scale = engine.TensorDivide(
            engine.TensorNorm(cleanSegments, 2, true),
            engine.TensorAddScalar(engine.TensorNorm(degradedSegments, 2, true), _numOps.FromDouble(Epsilon)));
        var scaledDegraded = engine.TensorMultiply(degradedSegments, engine.TensorBroadcastTo(scale, segmentShape));
        var clipFactor = _numOps.FromDouble(1.0 + Math.Pow(10.0, -ClipLowerBoundDb / 20.0));
        var clippedDegraded = engine.TensorMin(scaledDegraded, engine.TensorMultiplyScalar(cleanSegments, clipFactor));

        var correlations = engine.TensorMultiply(
            CenterAndNormalize(engine, clippedDegraded, segmentShape),
            CenterAndNormalize(engine, cleanSegments, segmentShape));
        return _numOps.Divide(engine.TensorSum(correlations), _numOps.FromDouble(segmentShape[0] * segmentShape[1]));
    }

    /// <summary>
    /// Subtracts each segment's mean and divides by its norm (plus epsilon), along the last axis.
    /// </summary>
    private Tensor<T> CenterAndNormalize(IEngine engine, Tensor<T> segments, int[] shape)
    {
        var centered = engine.TensorSubtract(
            segments, engine.TensorBroadcastTo(engine.ReduceMean(segments, new[] { 2 }, true), shape));
        var norms = engine.TensorAddScalar(engine.TensorNorm(centered, 2, true), _numOps.FromDouble(Epsilon));
        return engine.TensorDivide(centered, engine.TensorBroadcastTo(norms, shape));
    }

    /// <summary>
    /// Hann-windowed frames x[i:i + frameLength] for i in range(0, len(x) - frameLength, hop), as in pystoi.
    /// </summary>
    /// <returns>The [frames, frameLength] matrix, or null when the signal holds no frame.</returns>
    private Tensor<T>? WindowedFrames(IEngine engine, Tensor<T> signal)
    {
        // range(0, len - frameLength, hop) stops before len - frameLength, which is unfolding x[0:len-1].
        int usable = signal.Length - 1;
        if (usable < _frameLength)
        {
            return null;
        }

        var frames = engine.TensorUnfold(engine.TensorNarrow(signal, 0, 0, usable), 0, _frameLength, _hopLength);
        return engine.TensorMultiply(frames, engine.TensorBroadcastTo(_window, frames.Shape.ToArray()));
    }

    /// <summary>
    /// Overlap-adds frames spaced by the hop length back into a signal.
    /// </summary>
    private Tensor<T> OverlapAdd(IEngine engine, Tensor<T> frames)
    {
        int frameCount = frames.Shape[0];
        int frameLength = _frameLength;
        int hop = _hopLength;
        int outputLength = (frameCount - 1) * hop + frameLength;
        var positions = Enumerable.Range(0, frameCount * frameLength)
            .Select(k => k / frameLength * hop + k % frameLength)
            .ToArray();
        return engine.TensorIndexAdd(
            new Tensor<T>(new[] { outputLength }),
            0,
            new Tensor<int>(positions, new[] { positions.Length }),
            engine.Reshape(frames, new[] { frameCount * frameLength }));
    }

    /// <summary>
    /// sqrt(OBM |X|^2): the one-third octave band magnitudes of each zero-padded frame's spectrum.
    /// </summary>
    private Tensor<T> OctaveBandEnvelopes(IEngine engine, Tensor<T> frames)
    {
        int frameCount = frames.Shape[0];
        int bins = _fftLength / 2 + 1;
        var padded = engine.Concat(
            new[] { frames, new Tensor<T>(new[] { frameCount, _fftLength - _frameLength }) }, 1);

        // RFFT returns interleaved (re, im) pairs per bin.
        var spectrum = engine.Reshape(engine.RFFT(padded), new[] { frameCount, bins, 2 });
        var power = engine.ReduceSum(engine.TensorSquare(spectrum), new[] { 2 }, false);
        return engine.TensorSqrt(engine.TensorMatMul(power, _bandMatrix));
    }

    /// <summary>
    /// The [bins, bands] one-third octave band matrix of pystoi's thirdoct: band k covers the FFT bins from
    /// the bin nearest 150 * 2^((2k - 1) / 6) Hz up to, not including, the bin nearest 150 * 2^((2k + 1) / 6) Hz.
    /// </summary>
    private Tensor<T> BuildBandMatrix(IEngine engine)
    {
        int bins = _fftLength / 2 + 1;
        double binWidth = InternalSampleRate / (double)_fftLength;
        var entries = new List<int>();
        for (int band = 0; band < _numBands; band++)
        {
            int low = NearestBin(LowestBandCenterHz * Math.Pow(2.0, (2.0 * band - 1) / 6.0), bins, binWidth);
            int high = NearestBin(LowestBandCenterHz * Math.Pow(2.0, (2.0 * band + 1) / 6.0), bins, binWidth);
            for (int bin = low; bin < high; bin++)
            {
                entries.Add(bin * _numBands + band);
            }
        }

        var matrix = new Tensor<T>(new[] { bins * _numBands });
        if (entries.Count > 0)
        {
            matrix = engine.TensorIndexAdd(
                matrix,
                0,
                new Tensor<int>(entries.ToArray(), new[] { entries.Count }),
                Tensor<T>.CreateDefault(new[] { entries.Count }, _numOps.One));
        }

        return engine.Reshape(matrix, new[] { bins, _numBands });
    }

    /// <summary>
    /// numpy.argmin((f - frequency)^2) over the bin frequencies f = k * binWidth (first index on ties).
    /// </summary>
    private static int NearestBin(double frequency, int bins, double binWidth)
    {
        int best = 0;
        double bestDistance = double.PositiveInfinity;
        for (int bin = 0; bin < bins; bin++)
        {
            double difference = bin * binWidth - frequency;
            double distance = difference * difference;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = bin;
            }
        }

        return best;
    }

    private static int GreatestCommonDivisor(int a, int b)
    {
        while (b != 0)
        {
            int remainder = a % b;
            a = b;
            b = remainder;
        }

        return a;
    }

    /// <summary>
    /// Builds the polyphase form of pystoi's Octave-compatible resampler (resample_oct): a Kaiser-windowed
    /// sinc low-pass h with 60 dB rejection, normalized to unit sum and scaled by the up factor, as passed to
    /// scipy.signal.resample_poly. Output k is sum_m h[m] * up(x)[k * down + m - half], where up(x) inserts
    /// up - 1 zeros between samples; only the taps m with k * down + m - half divisible by up touch a sample,
    /// so each output phase k mod up is a strided correlation of x with every up-th tap.
    /// </summary>
    private int BuildResamplingPhases(IEngine engine)
    {
        int up = _upFactor;
        int down = _downFactor;
        double cutoff = 1.0 / (2.0 * Math.Max(up, down));
        double rejectionDb = 60.0;
        double rollOffWidth = cutoff / 10.0;
        int half = (int)Math.Ceiling((rejectionDb - 8.0) / (28.714 * rollOffWidth));
        double beta = 0.1102 * (rejectionDb - 8.7);
        int taps = 2 * half + 1;

        var positions = engine.TensorArange<T>(-half, half + 1, 1);
        var ideal = engine.TensorMultiplyScalar(
            engine.TensorSinc(engine.TensorMultiplyScalar(positions, _numOps.FromDouble(2 * cutoff))),
            _numOps.FromDouble(2.0 * up * cutoff));
        var filter = engine.TensorMultiply(engine.TensorKaiserWindow<T>(taps, false, beta), ideal);
        filter = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(filter, engine.TensorSum(filter)), _numOps.FromDouble(up));

        for (int phase = 0; phase < up; phase++)
        {
            int firstTap = ((half - phase * down) % up + up) % up;
            var tapIndices = Enumerable.Range(0, (taps - firstTap + up - 1) / up)
                .Select(j => firstTap + j * up)
                .ToArray();
            var phaseTaps = engine.Reshape(
                engine.TensorIndexSelect(filter, new Tensor<int>(tapIndices, new[] { tapIndices.Length }), 0),
                new[] { tapIndices.Length, 1 });
            _phases.Add((phaseTaps, tapIndices.Length, (phase * down + firstTap - half) / up));
        }

        return half;
    }

    /// <summary>
    /// Resamples a signal from the configured rate to 10 kHz (scipy resample_poly with pystoi's filter).
    /// </summary>
    private Tensor<T> Resample(IEngine engine, Tensor<T> signal)
    {
        int up = _upFactor;
        int down = _downFactor;
        int length = signal.Length;
        int outputLength = (int)(((long)length * up + down - 1) / down);
        int perPhase = (outputLength + up - 1) / up;
        int leftPad = (_filterHalfLength + up - 1) / up + 1;

        var spans = new List<(int Start, int Count, int Span)>();
        int needed = leftPad + length;
        for (int phase = 0; phase < up; phase++)
        {
            int count = phase < outputLength ? (outputLength - phase + up - 1) / up : 0;
            int tapCount = _phases[phase].TapCount;
            int start = _phases[phase].Offset + leftPad;
            int span = count > 0 ? (count - 1) * down + tapCount : 0;
            spans.Add((start, count, span));
            needed = Math.Max(needed, start + span);
        }

        var padded = engine.Concat(
            new[]
            {
                new Tensor<T>(new[] { leftPad }),
                signal,
                new Tensor<T>(new[] { needed - leftPad - length + 1 }),
            },
            0);

        var phaseOutputs = new Tensor<T>[up];
        for (int phase = 0; phase < up; phase++)
        {
            var (start, count, span) = spans[phase];
            var output = new Tensor<T>(new[] { perPhase });
            if (count > 0)
            {
                var windows = engine.TensorUnfold(
                    engine.TensorNarrow(padded, 0, start, span), 0, _phases[phase].TapCount, down);
                var values = engine.Reshape(engine.TensorMatMul(windows, _phases[phase].Taps), new[] { count });
                output = count == perPhase
                    ? values
                    : engine.Concat(new[] { values, new Tensor<T>(new[] { perPhase - count }) }, 0);
            }

            phaseOutputs[phase] = output;
        }

        // Interleave the phases: output k = phase k mod up, position k / up.
        var interleaved = engine.Reshape(engine.TensorStack(phaseOutputs, 1), new[] { perPhase * up });
        return engine.TensorNarrow(interleaved, 0, 0, outputLength);
    }
}

/// <summary>
/// Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric for source separation evaluation.
/// </summary>
/// <remarks>
/// <para>
/// SI-SDR is the standard metric for evaluating source separation quality. It measures
/// how well the estimated signal matches the target signal, ignoring scale differences.
/// </para>
/// <para>
/// Formula: SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
/// where s_target is the projection of the estimate onto the target, and e_noise is the residual.
/// </para>
/// <para>
/// Higher values indicate better separation quality. Typical values:
/// - &gt;15 dB: Excellent separation
/// - 10-15 dB: Good separation
/// - 5-10 dB: Fair separation
/// - &lt;5 dB: Poor separation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ScaleInvariantSignalToDistortionRatio<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of SI-SDR calculator.
    /// </summary>
    public ScaleInvariantSignalToDistortionRatio()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes SI-SDR between an estimated signal and the target signal.
    /// </summary>
    /// <param name="estimated">The estimated/separated signal.</param>
    /// <param name="target">The ground truth target signal.</param>
    /// <returns>SI-SDR in decibels. Higher is better.</returns>
    public T Compute(Tensor<T> estimated, Tensor<T> target)
    {
        if (estimated == null) throw new ArgumentNullException(nameof(estimated));
        if (target == null) throw new ArgumentNullException(nameof(target));

        if (estimated.Length != target.Length)
        {
            throw new ArgumentException("Estimated and target signals must have the same length");
        }

        // Handle empty tensors - cannot compute SI-SDR with no samples
        if (estimated.Length == 0)
        {
            return _numOps.FromDouble(-100.0); // Return very low SI-SDR for empty signals
        }

        // Zero-mean the signals
        var engine = AiDotNetEngine.Current;
        var estCentered = engine.TensorSubtractScalar(engine.Reshape(estimated, new[] { estimated.Length }), ComputeMean(estimated));
        var tarCentered = engine.TensorSubtractScalar(engine.Reshape(target, new[] { target.Length }), ComputeMean(target));

        // Compute <s_est, s_tar> / <s_tar, s_tar>
        T dotProduct = engine.TensorSum(engine.TensorMultiply(estCentered, tarCentered));
        T tarNormSq = engine.TensorSum(engine.TensorSquare(tarCentered));

        if (_numOps.LessThan(tarNormSq, _numOps.FromDouble(1e-10)))
        {
            return _numOps.FromDouble(-100.0); // Return very low SI-SDR for zero target
        }

        T alpha = _numOps.Divide(dotProduct, tarNormSq);

        // s_target = alpha * s_tar (projection of estimate onto target)
        // e_noise = s_est - s_target
        var sTarget = engine.TensorMultiplyScalar(tarCentered, alpha);
        var eNoise = engine.TensorSubtract(estCentered, sTarget);
        T sTargetNormSq = engine.TensorSum(engine.TensorSquare(sTarget));
        T eNoiseNormSq = engine.TensorSum(engine.TensorSquare(eNoise));

        if (_numOps.LessThan(eNoiseNormSq, _numOps.FromDouble(1e-10)))
        {
            return _numOps.FromDouble(100.0); // Near-perfect reconstruction
        }

        // SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        return Decibels(_numOps, _numOps.Divide(sTargetNormSq, eNoiseNormSq));
    }

    /// <summary>
    /// Computes SI-SDR improvement relative to a baseline (typically the mixture).
    /// </summary>
    /// <param name="estimated">The estimated/separated signal.</param>
    /// <param name="target">The ground truth target signal.</param>
    /// <param name="baseline">The baseline signal (e.g., input mixture).</param>
    /// <returns>SI-SDR improvement in dB.</returns>
    public T ComputeImprovement(Tensor<T> estimated, Tensor<T> target, Tensor<T> baseline)
    {
        T siSdrEst = Compute(estimated, target);
        T siSdrBaseline = Compute(baseline, target);

        return _numOps.Subtract(siSdrEst, siSdrBaseline);
    }

    private T ComputeMean(Tensor<T> tensor)
        => _numOps.Divide(AiDotNetEngine.Current.TensorSum(tensor), _numOps.FromDouble(tensor.Length));

    /// <summary>
    /// 10 * log10(ratio).
    /// </summary>
    internal static T Decibels(INumericOperations<T> numOps, T ratio)
        => numOps.Multiply(numOps.FromDouble(10.0 / Math.Log(10.0)), numOps.Log(ratio));
}

/// <summary>
/// Signal-to-Noise Ratio (SNR) metric for audio quality assessment.
/// </summary>
/// <remarks>
/// <para>
/// SNR measures the ratio of signal power to noise power. Higher values indicate cleaner audio.
/// </para>
/// <para>
/// Formula: SNR = 10 * log10(P_signal / P_noise)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SignalToNoiseRatio<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of SNR calculator.
    /// </summary>
    public SignalToNoiseRatio()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes SNR between a clean signal and a noisy signal.
    /// </summary>
    /// <param name="clean">The clean reference signal.</param>
    /// <param name="noisy">The noisy signal.</param>
    /// <returns>SNR in decibels. Higher is better.</returns>
    public T Compute(Tensor<T> clean, Tensor<T> noisy)
    {
        if (clean == null) throw new ArgumentNullException(nameof(clean));
        if (noisy == null) throw new ArgumentNullException(nameof(noisy));

        if (clean.Length != noisy.Length)
        {
            throw new ArgumentException("Clean and noisy signals must have the same length");
        }

        var engine = AiDotNetEngine.Current;
        var cleanFlat = engine.Reshape(clean, new[] { clean.Length });
        var noise = engine.TensorSubtract(engine.Reshape(noisy, new[] { noisy.Length }), cleanFlat);
        T signalPower = engine.TensorSum(engine.TensorSquare(cleanFlat));
        T noisePower = engine.TensorSum(engine.TensorSquare(noise));

        if (_numOps.LessThan(noisePower, _numOps.FromDouble(1e-10)))
        {
            return _numOps.FromDouble(100.0); // Near-perfect match
        }

        return ScaleInvariantSignalToDistortionRatio<T>.Decibels(_numOps, _numOps.Divide(signalPower, noisePower));
    }

    /// <summary>
    /// Computes segmental SNR (average SNR over short segments).
    /// </summary>
    /// <param name="clean">The clean reference signal.</param>
    /// <param name="noisy">The noisy signal.</param>
    /// <param name="frameLength">Length of each segment in samples.</param>
    /// <returns>Average segmental SNR in decibels.</returns>
    public T ComputeSegmental(Tensor<T> clean, Tensor<T> noisy, int frameLength = 256)
    {
        int numFrames = clean.Length / frameLength;
        if (numFrames == 0)
        {
            return Compute(clean, noisy);
        }

        // Non-overlapping frames [numFrames, frameLength]; a trailing partial frame is ignored.
        var engine = AiDotNetEngine.Current;
        int used = numFrames * frameLength;
        var cleanFrames = engine.Reshape(
            engine.TensorNarrow(engine.Reshape(clean, new[] { clean.Length }), 0, 0, used), new[] { numFrames, frameLength });
        var noisyFrames = engine.Reshape(
            engine.TensorNarrow(engine.Reshape(noisy, new[] { noisy.Length }), 0, 0, used), new[] { numFrames, frameLength });
        var signalPower = engine.ReduceSum(engine.TensorSquare(cleanFrames), new[] { 1 }, false);
        var noisePower = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(noisyFrames, cleanFrames)), new[] { 1 }, false);

        // Skip silent frames (and frames reproduced exactly)
        T floor = _numOps.FromDouble(1e-10);
        var valid = engine.TensorMultiply(engine.TensorGreaterThan(signalPower, floor), engine.TensorGreaterThan(noisePower, floor));
        T validFrames = engine.TensorSum(valid);
        if (_numOps.Equals(validFrames, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        var ones = Tensor<T>.CreateDefault(new[] { numFrames }, _numOps.One);
        var ratios = engine.TensorDivide(engine.TensorWhere(valid, signalPower, ones), engine.TensorWhere(valid, noisePower, ones));
        var snr = engine.TensorMultiplyScalar(engine.TensorLog10(ratios), _numOps.FromDouble(10.0));

        // Clamp to reasonable range [-10, 35] dB
        snr = engine.TensorClamp(snr, _numOps.FromDouble(-10.0), _numOps.FromDouble(35.0));
        return _numOps.Divide(engine.TensorSum(engine.TensorMultiply(snr, valid)), validFrames);
    }
}

/// <summary>
/// A PESQ-like speech quality approximation. This is NOT ITU-T P.862 (PESQ) and does not reproduce its scores.
/// </summary>
/// <remarks>
/// <para>
/// This metric blends STOI and segmental SNR into a number on the MOS-LQO range (-0.5 to 4.5) using an
/// empirical mapping. It does not implement the ITU-T P.862 / P.862.2 perceptual model (level and IRS
/// filtering, time alignment, auditory transform, disturbance aggregation), and its values do not match
/// the official reference implementation or the python <c>pesq</c> package. Do not report its output as
/// PESQ or compare it with published PESQ results.
/// </para>
/// <para>
/// Use it only as a rough relative quality indicator; higher values indicate better quality.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ApproximatePerceptualSpeechQuality<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;
    private readonly ShortTimeObjectiveIntelligibility<T> _stoi;
    private readonly SignalToNoiseRatio<T> _snr;

    /// <summary>
    /// Initializes a new instance of the PESQ-like approximation (not ITU-T P.862).
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (8000 or 16000 Hz supported).</param>
    public ApproximatePerceptualSpeechQuality(int sampleRate = 16000)
    {
        if (sampleRate != 8000 && sampleRate != 16000)
        {
            throw new ArgumentException("PESQ only supports 8000 Hz (narrowband) or 16000 Hz (wideband) sample rates");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _sampleRate = sampleRate;
        _stoi = new ShortTimeObjectiveIntelligibility<T>(sampleRate);
        _snr = new SignalToNoiseRatio<T>();
    }

    /// <summary>
    /// Computes the PESQ-like approximation between degraded and reference speech (not an ITU-T P.862 score).
    /// </summary>
    /// <param name="degraded">The degraded speech signal.</param>
    /// <param name="reference">The clean reference speech signal.</param>
    /// <returns>A PESQ-like score on the MOS-LQO range (-0.5 to 4.5); not comparable with P.862 scores.</returns>
    public T Compute(Tensor<T> degraded, Tensor<T> reference)
    {
        // This is a simplified approximation combining STOI and segmental SNR
        // It is not ITU-T P.862 and does not match P.862 scores

        // Get STOI score (0-1)
        T stoiScore = _stoi.Compute(degraded, reference);
        double stoi = _numOps.ToDouble(stoiScore);

        // Get segmental SNR
        T snrScore = _snr.ComputeSegmental(reference, degraded);
        double snr = _numOps.ToDouble(snrScore);

        // Approximate PESQ using empirical mapping from STOI and SNR
        // This is based on observed correlations between PESQ and these metrics
        double pesqApprox = MapToPesqLikeScale(stoi, snr);

        return _numOps.FromDouble(pesqApprox);
    }

    /// <summary>
    /// Maps STOI and SNR to a PESQ-like score.
    /// </summary>
    private double MapToPesqLikeScale(double stoi, double snr)
    {
        // Normalize SNR to [0, 1] range (assuming -5 to 30 dB range)
        double snrNorm = Math.Max(0, Math.Min(1, (snr + 5) / 35));

        // Combine STOI (weight 0.7) and SNR (weight 0.3)
        double combined = 0.7 * stoi + 0.3 * snrNorm;

        // Map to PESQ scale (-0.5 to 4.5)
        // Using a sigmoid-like mapping for more realistic distribution
        double pesq = 4.5 * combined - 0.5 * (1 - combined);

        return Math.Max(-0.5, Math.Min(4.5, pesq));
    }
}
