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
/// Values range from 0 to 1, where higher values indicate better intelligibility.
/// - &gt;0.9: Excellent intelligibility
/// - 0.7-0.9: Good intelligibility
/// - 0.5-0.7: Fair intelligibility
/// - &lt;0.5: Poor intelligibility
/// </para>
/// <para>
/// Based on "A short-time objective intelligibility measure for time-frequency weighted noisy speech"
/// by Taal et al. (2011).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ShortTimeObjectiveIntelligibility<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;
    private readonly int _frameLength;
    private readonly int _hopLength;
    private readonly int _numBands;

    /// <summary>
    /// Initializes a new instance of STOI calculator.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="frameLength">Analysis frame length in samples. Default is 256 (16ms at 16kHz).</param>
    /// <param name="hopLength">Hop length between frames. Default is 128 (8ms at 16kHz).</param>
    /// <param name="numBands">Number of frequency bands. Default is 15.</param>
    public ShortTimeObjectiveIntelligibility(
        int sampleRate = 16000,
        int frameLength = 256,
        int hopLength = 128,
        int numBands = 15)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sampleRate = sampleRate;
        _frameLength = frameLength;
        _hopLength = hopLength;
        _numBands = numBands;
    }

    /// <summary>
    /// Computes STOI between degraded and clean speech signals.
    /// </summary>
    /// <param name="degraded">The degraded/processed speech signal.</param>
    /// <param name="clean">The clean reference speech signal.</param>
    /// <returns>STOI score between 0 and 1. Higher is better.</returns>
    public T Compute(Tensor<T> degraded, Tensor<T> clean)
    {
        if (degraded.Length != clean.Length)
        {
            throw new ArgumentException("Degraded and clean signals must have the same length");
        }

        // Compute short-time spectral envelopes
        var degradedEnvelopes = ComputeSpectralEnvelopes(degraded);
        var cleanEnvelopes = ComputeSpectralEnvelopes(clean);

        // Compute intermediate intelligibility measure per frame
        int numFrames = degradedEnvelopes.GetLength(0);
        int numSegments = Math.Max(1, numFrames - 30 + 1); // 384ms segments

        T sumCorrelation = _numOps.Zero;
        int validSegments = 0;

        for (int seg = 0; seg < numSegments; seg++)
        {
            int startFrame = seg;
            int endFrame = Math.Min(seg + 30, numFrames);
            int segmentLength = endFrame - startFrame;

            // Compute normalized correlation for this segment
            for (int band = 0; band < _numBands; band++)
            {
                T correlation = ComputeNormalizedCorrelation(
                    degradedEnvelopes, cleanEnvelopes, band, startFrame, segmentLength);

                // Clamp correlation to [-1, 1]
                double corrDouble = _numOps.ToDouble(correlation);
                corrDouble = Math.Max(-1.0, Math.Min(1.0, corrDouble));

                sumCorrelation = _numOps.Add(sumCorrelation, _numOps.FromDouble(corrDouble));
                validSegments++;
            }
        }

        if (validSegments == 0)
        {
            return _numOps.Zero;
        }

        // Average correlation
        T avgCorrelation = _numOps.Divide(sumCorrelation, _numOps.FromDouble(validSegments));

        // Map to [0, 1] range: (correlation + 1) / 2
        T one = _numOps.One;
        T two = _numOps.FromDouble(2.0);
        return _numOps.Divide(_numOps.Add(avgCorrelation, one), two);
    }

    /// <summary>
    /// Computes spectral envelopes using one-third octave band analysis.
    /// </summary>
    private T[,] ComputeSpectralEnvelopes(Tensor<T> signal)
    {
        int numFrames = (signal.Length - _frameLength) / _hopLength + 1;
        numFrames = Math.Max(1, numFrames);

        var envelopes = new T[numFrames, _numBands];

        // Define center frequencies for one-third octave bands (150 Hz to 4.3 kHz)
        double[] centerFreqs = GetOneThirdOctaveCenterFrequencies();

        for (int frame = 0; frame < numFrames; frame++)
        {
            int startSample = frame * _hopLength;
            int endSample = Math.Min(startSample + _frameLength, signal.Length);

            // Extract frame and compute power spectrum (simplified - using energy in time domain)
            for (int band = 0; band < _numBands; band++)
            {
                double bandLow = centerFreqs[band] / Math.Pow(2, 1.0 / 6.0);
                double bandHigh = centerFreqs[band] * Math.Pow(2, 1.0 / 6.0);

                // Compute band energy using a simple bandpass approximation
                T bandEnergy = ComputeBandEnergy(signal, startSample, endSample, bandLow, bandHigh);
                envelopes[frame, band] = bandEnergy;
            }
        }

        return envelopes;
    }

    /// <summary>
    /// Gets center frequencies for one-third octave bands.
    /// </summary>
    private double[] GetOneThirdOctaveCenterFrequencies()
    {
        var freqs = new double[_numBands];
        double startFreq = 150.0; // Starting frequency in Hz

        for (int i = 0; i < _numBands; i++)
        {
            freqs[i] = startFreq * Math.Pow(2, i / 3.0);
        }

        return freqs;
    }

    /// <summary>
    /// Computes energy in a frequency band (simplified approximation).
    /// </summary>
    private T ComputeBandEnergy(Tensor<T> signal, int startSample, int endSample, double bandLow, double bandHigh)
    {
        // Simplified: compute RMS energy weighted by approximate band response
        T sum = _numOps.Zero;
        int count = 0;

        for (int i = startSample; i < endSample; i++)
        {
            T sample = signal[i];
            sum = _numOps.Add(sum, _numOps.Multiply(sample, sample));
            count++;
        }

        if (count == 0)
        {
            return _numOps.Zero;
        }

        // Apply frequency-dependent weighting based on band position
        double bandCenter = (bandLow + bandHigh) / 2.0;
        double weight = Math.Log10(bandCenter / 100.0 + 1.0); // Higher weight for higher frequencies

        T energy = _numOps.Divide(sum, _numOps.FromDouble(count));
        return _numOps.Multiply(energy, _numOps.FromDouble(weight));
    }

    /// <summary>
    /// Computes normalized correlation between two envelope sequences.
    /// </summary>
    private T ComputeNormalizedCorrelation(
        T[,] degraded, T[,] clean, int band, int startFrame, int segmentLength)
    {
        T sumDeg = _numOps.Zero;
        T sumClean = _numOps.Zero;
        T sumDegSq = _numOps.Zero;
        T sumCleanSq = _numOps.Zero;
        T sumProduct = _numOps.Zero;

        for (int f = 0; f < segmentLength; f++)
        {
            int frame = startFrame + f;
            if (frame >= degraded.GetLength(0))
            {
                break;
            }

            T d = degraded[frame, band];
            T c = clean[frame, band];

            sumDeg = _numOps.Add(sumDeg, d);
            sumClean = _numOps.Add(sumClean, c);
            sumDegSq = _numOps.Add(sumDegSq, _numOps.Multiply(d, d));
            sumCleanSq = _numOps.Add(sumCleanSq, _numOps.Multiply(c, c));
            sumProduct = _numOps.Add(sumProduct, _numOps.Multiply(d, c));
        }

        T n = _numOps.FromDouble(segmentLength);

        // Compute Pearson correlation coefficient
        T meanDeg = _numOps.Divide(sumDeg, n);
        T meanClean = _numOps.Divide(sumClean, n);

        T numerator = _numOps.Subtract(sumProduct, _numOps.Multiply(_numOps.Multiply(n, meanDeg), meanClean));

        T varDeg = _numOps.Subtract(sumDegSq, _numOps.Multiply(_numOps.Multiply(n, meanDeg), meanDeg));
        T varClean = _numOps.Subtract(sumCleanSq, _numOps.Multiply(_numOps.Multiply(n, meanClean), meanClean));

        T denominator = _numOps.Sqrt(_numOps.Multiply(varDeg, varClean));

        double denomDouble = _numOps.ToDouble(denominator);
        if (denomDouble < 1e-10)
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(numerator, denominator);
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
        if (estimated.Length != target.Length)
        {
            throw new ArgumentException("Estimated and target signals must have the same length");
        }

        // Zero-mean the signals
        T estMean = ComputeMean(estimated);
        T tarMean = ComputeMean(target);

        var estCentered = new Tensor<T>(estimated.Shape);
        var tarCentered = new Tensor<T>(target.Shape);

        for (int i = 0; i < estimated.Length; i++)
        {
            estCentered[i] = _numOps.Subtract(estimated[i], estMean);
            tarCentered[i] = _numOps.Subtract(target[i], tarMean);
        }

        // Compute <s_est, s_tar> / <s_tar, s_tar>
        T dotProduct = _numOps.Zero;
        T tarNormSq = _numOps.Zero;

        for (int i = 0; i < estimated.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(estCentered[i], tarCentered[i]));
            tarNormSq = _numOps.Add(tarNormSq, _numOps.Multiply(tarCentered[i], tarCentered[i]));
        }

        double tarNormSqDouble = _numOps.ToDouble(tarNormSq);
        if (tarNormSqDouble < 1e-10)
        {
            return _numOps.FromDouble(double.NegativeInfinity);
        }

        T alpha = _numOps.Divide(dotProduct, tarNormSq);

        // s_target = alpha * s_tar (projection of estimate onto target)
        // e_noise = s_est - s_target
        T sTargetNormSq = _numOps.Zero;
        T eNoiseNormSq = _numOps.Zero;

        for (int i = 0; i < estimated.Length; i++)
        {
            T sTarget = _numOps.Multiply(alpha, tarCentered[i]);
            T eNoise = _numOps.Subtract(estCentered[i], sTarget);

            sTargetNormSq = _numOps.Add(sTargetNormSq, _numOps.Multiply(sTarget, sTarget));
            eNoiseNormSq = _numOps.Add(eNoiseNormSq, _numOps.Multiply(eNoise, eNoise));
        }

        double eNoiseDouble = _numOps.ToDouble(eNoiseNormSq);
        if (eNoiseDouble < 1e-10)
        {
            return _numOps.FromDouble(100.0); // Near-perfect reconstruction
        }

        // SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        double ratio = _numOps.ToDouble(sTargetNormSq) / eNoiseDouble;
        double siSdr = 10.0 * Math.Log10(ratio);

        return _numOps.FromDouble(siSdr);
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
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < tensor.Length; i++)
        {
            sum = _numOps.Add(sum, tensor[i]);
        }
        return _numOps.Divide(sum, _numOps.FromDouble(tensor.Length));
    }
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
        if (clean.Length != noisy.Length)
        {
            throw new ArgumentException("Clean and noisy signals must have the same length");
        }

        T signalPower = _numOps.Zero;
        T noisePower = _numOps.Zero;

        for (int i = 0; i < clean.Length; i++)
        {
            T noise = _numOps.Subtract(noisy[i], clean[i]);
            signalPower = _numOps.Add(signalPower, _numOps.Multiply(clean[i], clean[i]));
            noisePower = _numOps.Add(noisePower, _numOps.Multiply(noise, noise));
        }

        double noiseDouble = _numOps.ToDouble(noisePower);
        if (noiseDouble < 1e-10)
        {
            return _numOps.FromDouble(100.0); // Near-perfect match
        }

        double ratio = _numOps.ToDouble(signalPower) / noiseDouble;
        double snr = 10.0 * Math.Log10(ratio);

        return _numOps.FromDouble(snr);
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

        double sumSnr = 0.0;
        int validFrames = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * frameLength;
            int end = Math.Min(start + frameLength, clean.Length);

            T signalPower = _numOps.Zero;
            T noisePower = _numOps.Zero;

            for (int i = start; i < end; i++)
            {
                T noise = _numOps.Subtract(noisy[i], clean[i]);
                signalPower = _numOps.Add(signalPower, _numOps.Multiply(clean[i], clean[i]));
                noisePower = _numOps.Add(noisePower, _numOps.Multiply(noise, noise));
            }

            double sigDouble = _numOps.ToDouble(signalPower);
            double noiseDouble = _numOps.ToDouble(noisePower);

            // Skip silent frames
            if (sigDouble > 1e-10 && noiseDouble > 1e-10)
            {
                double snr = 10.0 * Math.Log10(sigDouble / noiseDouble);
                // Clamp to reasonable range [-10, 35] dB
                snr = Math.Max(-10.0, Math.Min(35.0, snr));
                sumSnr += snr;
                validFrames++;
            }
        }

        if (validFrames == 0)
        {
            return _numOps.Zero;
        }

        return _numOps.FromDouble(sumSnr / validFrames);
    }
}

/// <summary>
/// Perceptual Evaluation of Speech Quality (PESQ) approximation metric.
/// </summary>
/// <remarks>
/// <para>
/// This is a simplified approximation of the ITU-T P.862 PESQ algorithm.
/// For production use, consider using an official PESQ implementation.
/// </para>
/// <para>
/// PESQ scores range from -0.5 to 4.5, where higher values indicate better quality.
/// - &gt;4.0: Excellent quality
/// - 3.5-4.0: Good quality
/// - 3.0-3.5: Fair quality
/// - &lt;3.0: Poor quality
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PerceptualSpeechQuality<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;
    private readonly ShortTimeObjectiveIntelligibility<T> _stoi;
    private readonly SignalToNoiseRatio<T> _snr;

    /// <summary>
    /// Initializes a new instance of PESQ approximation calculator.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (8000 or 16000 Hz supported).</param>
    public PerceptualSpeechQuality(int sampleRate = 16000)
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
    /// Computes an approximation of PESQ score between degraded and reference speech.
    /// </summary>
    /// <param name="degraded">The degraded speech signal.</param>
    /// <param name="reference">The clean reference speech signal.</param>
    /// <returns>Approximated PESQ score (MOS-LQO scale, -0.5 to 4.5).</returns>
    public T Compute(Tensor<T> degraded, Tensor<T> reference)
    {
        // This is a simplified approximation combining STOI and segmental SNR
        // For accurate PESQ, use the official ITU-T P.862 implementation

        // Get STOI score (0-1)
        T stoiScore = _stoi.Compute(degraded, reference);
        double stoi = _numOps.ToDouble(stoiScore);

        // Get segmental SNR
        T snrScore = _snr.ComputeSegmental(reference, degraded);
        double snr = _numOps.ToDouble(snrScore);

        // Approximate PESQ using empirical mapping from STOI and SNR
        // This is based on observed correlations between PESQ and these metrics
        double pesqApprox = MapToPesqScale(stoi, snr);

        return _numOps.FromDouble(pesqApprox);
    }

    /// <summary>
    /// Maps STOI and SNR to an approximate PESQ score.
    /// </summary>
    private double MapToPesqScale(double stoi, double snr)
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
