using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings against cloning using psychoacoustic masking — adding noise
/// that is hidden beneath audible content but disrupts speaker embedding extraction.
/// </summary>
/// <remarks>
/// <para>
/// Applies frequency-domain noise shaped to the psychoacoustic masking curve of the audio.
/// The masking curve determines the threshold below which noise is inaudible to humans;
/// this protector adds maximum disruption noise just below this threshold. The disruption
/// specifically targets the frequency bands used by speaker verification systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> Our ears have blind spots — when a loud sound is playing, we can't
/// hear quiet sounds nearby. This module exploits those blind spots to hide anti-cloning noise
/// where your ears can't detect it, but AI cloning systems can "hear" and get confused by.
/// </para>
/// <para>
/// <b>References:</b>
/// - VocalCrypt: Pseudo-timbre jamming for voice protection (2025, arxiv:2502.10329)
/// - Psychoacoustic masking models for audio steganography (2023)
/// - MPEG psychoacoustic model for perceptual coding (ISO/IEC 11172-3)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MaskingVoiceProtector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _maskingStrength;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "MaskingVoiceProtector";

    /// <summary>
    /// Initializes a new masking-based voice protector.
    /// </summary>
    /// <param name="maskingStrength">
    /// Masking noise strength relative to masking threshold (0-1). Default: 0.8.
    /// 1.0 = add noise at exactly the masking threshold; 0.5 = half threshold.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public MaskingVoiceProtector(double maskingStrength = 0.8, int sampleRate = 16000)
        : base(sampleRate)
    {
        _maskingStrength = NumOps.FromDouble(maskingStrength);
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Applies psychoacoustic masking protection and returns protected audio.
    /// </summary>
    public Vector<T> ProtectAudio(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 512) return audioSamples;

        int frameSize = 512;
        int hopSize = 256;
        double strength = NumOps.ToDouble(_maskingStrength);

        var result = new Vector<T>(audioSamples.Length);
        var windowSum = new Vector<T>(audioSamples.Length);

        for (int start = 0; start + frameSize <= audioSamples.Length; start += hopSize)
        {
            // Window the frame
            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                frame[i] = NumOps.Multiply(audioSamples[start + i], NumOps.FromDouble(w));
            }

            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;

            // Compute magnitude spectrum
            var magnitudes = new double[halfLen];
            for (int k = 0; k < halfLen; k++)
            {
                magnitudes[k] = NumOps.ToDouble(spectrum[k].Magnitude);
            }

            // Compute simplified masking curve
            var maskingThresholds = ComputeMaskingCurve(magnitudes, halfLen, sampleRate, frameSize);

            // Add masking noise in speaker-embedding-critical bands
            for (int k = 1; k < halfLen; k++)
            {
                double freq = (double)k * sampleRate / frameSize;

                // Focus on formant region (300-3400 Hz) and speaker-characteristic region (1-6 kHz)
                double bandWeight = 1.0;
                if (freq < 300 || freq > 6000) bandWeight = 0.2;
                else if (freq >= 1000 && freq <= 4000) bandWeight = 1.0;
                else bandWeight = 0.6;

                // Noise amplitude = masking threshold * strength * band weight
                double noiseAmp = maskingThresholds[k] * strength * bandWeight;

                // Pseudo-random phase
                int hash = HashInt(start * 1000 + k);
                double phase = (hash % 6283) / 1000.0;

                T noiseReal = NumOps.FromDouble(noiseAmp * Math.Cos(phase));
                T noiseImag = NumOps.FromDouble(noiseAmp * Math.Sin(phase));

                spectrum[k] = new Complex<T>(
                    NumOps.Add(spectrum[k].Real, noiseReal),
                    NumOps.Add(spectrum[k].Imaginary, noiseImag));
            }

            var protectedFrame = _fft.Inverse(spectrum);

            // Overlap-add
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                T windowed = NumOps.Multiply(protectedFrame[i], NumOps.FromDouble(w));
                result[start + i] = NumOps.Add(result[start + i], windowed);
                windowSum[start + i] = NumOps.Add(windowSum[start + i], NumOps.FromDouble(w * w));
            }
        }

        // Normalize
        for (int i = 0; i < result.Length; i++)
        {
            double ws = NumOps.ToDouble(windowSum[i]);
            if (ws > 1e-10)
                result[i] = NumOps.Divide(result[i], windowSum[i]);
            else
                result[i] = audioSamples[i];
        }

        return result;
    }

    /// <summary>
    /// Computes a simplified psychoacoustic masking curve.
    /// Based on the spreading function model from MPEG psychoacoustic analysis.
    /// </summary>
    private static double[] ComputeMaskingCurve(double[] magnitudes, int halfLen,
        int sampleRate, int frameSize)
    {
        var thresholds = new double[halfLen];

        // Convert to dB scale
        var dB = new double[halfLen];
        for (int k = 0; k < halfLen; k++)
        {
            dB[k] = magnitudes[k] > 1e-20 ? 20 * Math.Log10(magnitudes[k]) : -100;
        }

        // Simultaneous masking: each spectral peak masks nearby frequencies
        for (int k = 0; k < halfLen; k++)
        {
            double maxMask = -100;

            // Each nearby bin contributes masking
            int spreadRange = Math.Min(32, halfLen / 4);
            for (int j = Math.Max(0, k - spreadRange); j < Math.Min(halfLen, k + spreadRange); j++)
            {
                double distance = Math.Abs(k - j);
                // Simplified spreading function: -10 dB per critical band
                double spreading = dB[j] - distance * 2.0;
                if (spreading > maxMask) maxMask = spreading;
            }

            // Absolute hearing threshold (simplified) — varies by frequency
            double freq = (double)k * sampleRate / frameSize;
            double absThreshold = ComputeAbsoluteThreshold(freq);

            // Masking threshold = max of simultaneous masking and absolute threshold
            // Subtract offset (masking is ~15 dB below masker for tonal, ~5 dB for noise)
            double maskingOffset = 10.0; // dB below masker
            double maskedThreshold = Math.Max(maxMask - maskingOffset, absThreshold);

            // Convert back to linear
            thresholds[k] = Math.Pow(10, maskedThreshold / 20);
        }

        return thresholds;
    }

    /// <summary>
    /// Simplified absolute hearing threshold (ISO 226).
    /// </summary>
    private static double ComputeAbsoluteThreshold(double freq)
    {
        if (freq < 20) return 80;
        if (freq > 20000) return 80;

        // Simplified approximation of the minimum audibility curve
        double f = freq / 1000.0; // kHz
        double threshold = 3.64 * Math.Pow(f, -0.8)
                          - 6.5 * Math.Exp(-0.6 * (f - 3.3) * (f - 3.3))
                          + 0.001 * Math.Pow(f, 4);

        return Math.Max(-60, Math.Min(80, threshold));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // Detect if audio has psychoacoustic masking-based protection
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024) return findings;

        // Look for noise below masking threshold (unusual for natural audio)
        double protectionScore = DetectMaskingProtection(audioSamples, sampleRate);

        if (protectionScore >= 0.5)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, protectionScore),
                Description = $"Psychoacoustic masking protection detected (score: {protectionScore:F3}). " +
                              $"Audio appears to have voice protection applied.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double DetectMaskingProtection(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        if (audio.Length < frameSize) return 0;

        // Analyze noise floor in typically quiet frequency regions
        var frame = new Vector<T>(frameSize);
        int mid = (audio.Length - frameSize) / 2;
        for (int i = 0; i < frameSize; i++)
        {
            double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
            frame[i] = NumOps.Multiply(audio[mid + i], NumOps.FromDouble(w));
        }

        var spectrum = _fft.Forward(frame);
        int halfLen = spectrum.Length / 2;

        var magnitudes = new double[halfLen];
        for (int k = 0; k < halfLen; k++)
        {
            magnitudes[k] = NumOps.ToDouble(spectrum[k].Magnitude);
        }

        // Check if energy in masking-threshold-level regions is higher than expected
        var maskCurve = ComputeMaskingCurve(magnitudes, halfLen, sampleRate, frameSize);
        int nearThresholdBins = 0;
        int totalBins = 0;

        for (int k = 5; k < halfLen; k++)
        {
            if (maskCurve[k] > 1e-10)
            {
                double ratio = magnitudes[k] / maskCurve[k];
                if (ratio > 0.5 && ratio < 2.0) nearThresholdBins++;
                totalBins++;
            }
        }

        return totalBins > 0 ? Math.Min(1.0, (double)nearThresholdBins / totalBins * 3) : 0;
    }

    private static int HashInt(int x)
    {
        unchecked
        {
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = (x >> 16) ^ x;
            return x & 0x7FFFFFFF;
        }
    }
}
