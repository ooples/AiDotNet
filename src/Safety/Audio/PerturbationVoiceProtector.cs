using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings against cloning by adding imperceptible adversarial perturbations.
/// </summary>
/// <remarks>
/// <para>
/// Adds carefully crafted perturbations to the audio that disrupt voice cloning systems while
/// remaining inaudible to human listeners. The perturbations target the frequency bands most
/// important for speaker embedding extraction (typically 300-3400 Hz for formants) and add
/// noise shaped to psychoacoustic masking thresholds.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like adding an invisible "anti-copying" pattern to a voice
/// recording. Humans can't hear the difference, but if someone tries to clone the voice using
/// AI, the protection disrupts the cloning process. Think of it like a DRM for your voice.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeSpeech: SPEC perturbation framework against voice cloning (2025, arxiv:2504.09839)
/// - Adversarial examples for speech recognition (Carlini &amp; Wagner, 2018)
/// - Voice protection via perturbation (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PerturbationVoiceProtector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _perturbationStrength;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "PerturbationVoiceProtector";

    /// <summary>
    /// Initializes a new perturbation-based voice protector.
    /// </summary>
    /// <param name="perturbationStrength">
    /// Perturbation strength (0-1). Default: 0.02. Higher values provide more protection
    /// but may become audible.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public PerturbationVoiceProtector(double perturbationStrength = 0.02, int sampleRate = 16000)
        : base(sampleRate)
    {
        _perturbationStrength = NumOps.FromDouble(perturbationStrength);
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Applies voice protection perturbations and returns protected audio with safety findings.
    /// </summary>
    public Vector<T> ProtectAudio(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 512)
        {
            return audioSamples;
        }

        var protected_ = new Vector<T>(audioSamples.Length);
        for (int i = 0; i < audioSamples.Length; i++)
        {
            protected_[i] = audioSamples[i];
        }

        int frameSize = 512;
        int hopSize = 256;
        double strength = NumOps.ToDouble(_perturbationStrength);

        // Process in overlapping frames
        for (int start = 0; start + frameSize <= audioSamples.Length; start += hopSize)
        {
            // Extract frame
            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                frame[i] = audioSamples[start + i];
            }

            // FFT
            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;

            // Add perturbation in formant regions (300-3400 Hz)
            int lowBin = (int)(300.0 * frameSize / sampleRate);
            int highBin = (int)(3400.0 * frameSize / sampleRate);
            highBin = Math.Min(highBin, halfLen - 1);

            // Hash-based pseudo-random perturbation (deterministic for reproducibility)
            for (int k = lowBin; k <= highBin; k++)
            {
                T magnitude = spectrum[k].Magnitude;
                double magD = NumOps.ToDouble(magnitude);

                // Perturbation proportional to local magnitude (masking-aware)
                int hash = HashInt(start * 1000 + k);
                double pertPhase = (hash % 628) / 100.0; // Random phase [0, 2*pi)
                double pertMag = magD * strength * ((hash % 100) / 100.0);

                T newReal = NumOps.Add(spectrum[k].Real,
                    NumOps.FromDouble(pertMag * Math.Cos(pertPhase)));
                T newImag = NumOps.Add(spectrum[k].Imaginary,
                    NumOps.FromDouble(pertMag * Math.Sin(pertPhase)));

                spectrum[k] = new Complex<T>(newReal, newImag);
            }

            // Inverse FFT
            var perturbedFrame = _fft.Inverse(spectrum);

            // Overlap-add with Hann window
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                T weighted = NumOps.Multiply(perturbedFrame[i], NumOps.FromDouble(w));
                protected_[start + i] = NumOps.Add(protected_[start + i],
                    NumOps.Multiply(NumOps.Subtract(weighted, NumOps.Multiply(frame[i], NumOps.FromDouble(w))),
                        NumOps.One));
            }
        }

        return protected_;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // This module is primarily a protection tool, but can detect if audio
        // already has perturbation-like artifacts
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024) return findings;

        // Check for existing perturbation artifacts in formant band
        double artifactScore = DetectPerturbationArtifacts(audioSamples, sampleRate);

        if (artifactScore >= 0.5)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, artifactScore),
                Description = $"Voice perturbation artifacts detected (score: {artifactScore:F3}). " +
                              $"Audio may have been processed with anti-cloning protection.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double DetectPerturbationArtifacts(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        if (audio.Length < frameSize * 2) return 0;

        // Compare adjacent frames for unnatural spectral discontinuities in formant band
        int numPairs = Math.Min(4, audio.Length / frameSize - 1);
        double anomalySum = 0;

        for (int p = 0; p < numPairs; p++)
        {
            int start1 = p * frameSize;
            int start2 = start1 + frameSize;

            var frame1 = new Vector<T>(frameSize);
            var frame2 = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                frame1[i] = audio[start1 + i];
                frame2[i] = audio[start2 + i];
            }

            var spec1 = _fft.Forward(frame1);
            var spec2 = _fft.Forward(frame2);
            int halfLen = spec1.Length / 2;

            int lowBin = (int)(300.0 * frameSize / sampleRate);
            int highBin = Math.Min((int)(3400.0 * frameSize / sampleRate), halfLen - 1);

            double diffSum = 0;
            double magSum = 0;
            for (int k = lowBin; k <= highBin; k++)
            {
                double m1 = NumOps.ToDouble(spec1[k].Magnitude);
                double m2 = NumOps.ToDouble(spec2[k].Magnitude);
                diffSum += Math.Abs(m1 - m2);
                magSum += (m1 + m2) / 2;
            }

            double relDiff = magSum > 1e-10 ? diffSum / magSum : 0;
            anomalySum += relDiff;
        }

        double meanAnomaly = anomalySum / numPairs;
        return Math.Min(1.0, Math.Max(0, (meanAnomaly - 0.3) * 3));
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
