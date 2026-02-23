using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Detects AI-generated/deepfake images by analyzing frequency domain artifacts.
/// </summary>
/// <remarks>
/// <para>
/// AI-generated images (GANs, diffusion models) leave characteristic artifacts in the
/// frequency domain that are invisible to the human eye. This detector applies FFT to
/// image rows and columns, then analyzes the resulting spectrum for anomalies such as
/// periodic peaks (GAN fingerprints), unusual spectral roll-off patterns, and frequency
/// band energy distribution abnormalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> Every image can be decomposed into waves of different frequencies
/// (like separating sound into bass, mid, and treble). AI-generated images have unusual
/// patterns in these frequency waves — like a fingerprint left by the AI. This module
/// detects those fingerprints.
/// </para>
/// <para>
/// <b>References:</b>
/// - Frequency analysis for deepfake detection via spectral artifacts (2020, arxiv:2003.08685)
/// - Generalizable deepfake detection across benchmarks (CVPR 2025, arxiv:2508.06248)
/// - NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024, arxiv:2407.10550)
/// - AI-generated media detection survey: non-MLLM to MLLM (2025, arxiv:2502.05240)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FrequencyDeepfakeDetector<T> : ImageSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly FastFourierTransform<T> _fft;

    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "FrequencyDeepfakeDetector";

    /// <summary>
    /// Initializes a new frequency-domain deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public FrequencyDeepfakeDetector(double threshold = 0.5)
    {
        _threshold = threshold;
        _fft = new FastFourierTransform<T>();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var span = image.Data.Span;
        if (span.Length == 0) return findings;

        var layout = DetermineLayout(image.Shape, span.Length);
        if (layout.Height < 8 || layout.Width < 8) return findings;

        // Find nearest power of 2 for FFT
        int fftSize = 1;
        while (fftSize < layout.Width && fftSize < 512) fftSize *= 2;
        if (fftSize > layout.Width) fftSize /= 2;
        if (fftSize < 8) return findings;

        // Analyze multiple rows from different parts of the image
        int numRows = Math.Min(16, layout.Height);
        int rowStride = layout.Height / numRows;

        double spectralFlatnessSum = 0;
        double periodicPeakSum = 0;
        double highFreqEnergySum = 0;
        int analyzedRows = 0;

        for (int r = 0; r < numRows; r++)
        {
            int y = r * rowStride;
            if (y >= layout.Height) break;

            // Extract row as vector
            var rowData = new Vector<T>(fftSize);
            for (int x = 0; x < fftSize && x < layout.Width; x++)
            {
                T val = GetChannelDouble(span, layout, y, x, 0);
                if (NumOps.LessThanOrEquals(val, One))
                    val = NumOps.Multiply(val, TwoFiftyFive);
                rowData[x] = val;
            }

            // FFT
            var spectrum = _fft.Forward(rowData);
            int halfLen = spectrum.Length / 2;

            // Compute magnitude spectrum
            var magnitudes = new double[halfLen];
            for (int k = 1; k < halfLen; k++)
            {
                magnitudes[k] = NumOps.ToDouble(spectrum[k].Magnitude);
            }

            // 1. Spectral flatness: geometric mean / arithmetic mean
            //    Low flatness = concentrated spectrum (natural)
            //    High flatness = flat spectrum (can indicate GAN artifacts)
            double flatness = ComputeSpectralFlatness(magnitudes, 1, halfLen);

            // 2. Periodic peak detection (GAN fingerprints create periodic patterns)
            double periodicPeak = DetectPeriodicPeaks(magnitudes, halfLen);

            // 3. High-frequency energy ratio
            int midBand = halfLen / 2;
            double lowEnergy = 0, highEnergy = 0;
            for (int k = 1; k < halfLen; k++)
            {
                if (k < midBand) lowEnergy += magnitudes[k] * magnitudes[k];
                else highEnergy += magnitudes[k] * magnitudes[k];
            }
            double totalEnergy = lowEnergy + highEnergy;
            double highFreqRatio = totalEnergy > 1e-10 ? highEnergy / totalEnergy : 0;

            spectralFlatnessSum += flatness;
            periodicPeakSum += periodicPeak;
            highFreqEnergySum += highFreqRatio;
            analyzedRows++;
        }

        if (analyzedRows == 0) return findings;

        double avgFlatness = spectralFlatnessSum / analyzedRows;
        double avgPeriodic = periodicPeakSum / analyzedRows;
        double avgHighFreq = highFreqEnergySum / analyzedRows;

        // Combine signals:
        // - Spectral flatness deviation from natural images (~0.1-0.3)
        //   GAN images often have flatness > 0.5
        double flatnessAnomaly = Math.Max(0, (avgFlatness - 0.3) / 0.7);

        // - Periodic peaks (natural images rarely have them)
        double periodicAnomaly = Math.Min(1.0, avgPeriodic);

        // - High frequency energy (AI images often have too much or too little)
        //   Natural: ~0.2-0.4; GAN: often < 0.1 or > 0.5
        double hfAnomaly = avgHighFreq < 0.1 ? (0.1 - avgHighFreq) * 10 :
                           avgHighFreq > 0.5 ? (avgHighFreq - 0.5) * 2 : 0;
        hfAnomaly = Math.Min(1.0, hfAnomaly);

        double finalScore = 0.35 * flatnessAnomaly + 0.40 * periodicAnomaly + 0.25 * hfAnomaly;

        if (finalScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Frequency domain analysis: potential AI-generated image (score: {finalScore:F3}). " +
                              $"Spectral flatness: {avgFlatness:F3}, periodic peaks: {avgPeriodic:F3}, " +
                              $"high-freq energy ratio: {avgHighFreq:F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static double ComputeSpectralFlatness(double[] magnitudes, int start, int end)
    {
        int n = end - start;
        if (n <= 0) return 0;

        double logSum = 0;
        double arithmeticSum = 0;
        int validCount = 0;

        for (int k = start; k < end; k++)
        {
            double m = magnitudes[k];
            if (m > 1e-20)
            {
                logSum += Math.Log(m);
                arithmeticSum += m;
                validCount++;
            }
        }

        if (validCount == 0 || arithmeticSum < 1e-20) return 0;

        double geometricMean = Math.Exp(logSum / validCount);
        double arithmeticMean = arithmeticSum / validCount;

        return arithmeticMean > 1e-20 ? geometricMean / arithmeticMean : 0;
    }

    private static double DetectPeriodicPeaks(double[] magnitudes, int length)
    {
        if (length < 8) return 0;

        // Compute autocorrelation of the magnitude spectrum to find periodicity
        // Periodic GAN artifacts create peaks in autocorrelation at regular intervals
        double meanMag = 0;
        for (int k = 1; k < length; k++) meanMag += magnitudes[k];
        meanMag /= (length - 1);

        double maxCorr = 0;
        double variance = 0;

        for (int k = 1; k < length; k++)
        {
            double diff = magnitudes[k] - meanMag;
            variance += diff * diff;
        }
        variance /= (length - 1);
        if (variance < 1e-20) return 0;

        // Check for autocorrelation peaks at lags 2-16 (common GAN periodicities)
        for (int lag = 2; lag <= Math.Min(16, length / 4); lag++)
        {
            double corr = 0;
            int pairs = 0;
            for (int k = 1; k < length - lag; k++)
            {
                corr += (magnitudes[k] - meanMag) * (magnitudes[k + lag] - meanMag);
                pairs++;
            }
            if (pairs > 0) corr /= pairs;
            double normalizedCorr = corr / variance;
            if (normalizedCorr > maxCorr) maxCorr = normalizedCorr;
        }

        return Math.Max(0, Math.Min(1.0, maxCorr));
    }

    private static T GetChannelDouble(ReadOnlySpan<T> data, ImageLayout layout, int y, int x, int c)
    {
        int idx;
        if (layout.Format == PixFmt.CHW)
            idx = c * layout.Height * layout.Width + y * layout.Width + x;
        else
            idx = (y * layout.Width + x) * layout.Channels + c;

        if (idx < 0 || idx >= data.Length) return Zero;
        return data[idx];
    }

    private static ImageLayout DetermineLayout(int[] shape, int dataLength)
    {
        if (shape.Length >= 4)
        {
            if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4)
                return new ImageLayout { Channels = shape[1], Height = shape[2], Width = shape[3], Format = PixFmt.CHW };
            if (shape[3] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[3], Height = shape[1], Width = shape[2], Format = PixFmt.HWC };
        }
        if (shape.Length == 3)
        {
            if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[0], Height = shape[1], Width = shape[2], Format = PixFmt.CHW };
            if (shape[2] <= 4 && shape[0] > 4 && shape[1] > 4)
                return new ImageLayout { Channels = shape[2], Height = shape[0], Width = shape[1], Format = PixFmt.HWC };
        }
        if (shape.Length == 2)
            return new ImageLayout { Channels = 1, Height = shape[0], Width = shape[1], Format = PixFmt.CHW };

        int side = (int)Math.Sqrt(dataLength);
        return new ImageLayout { Channels = 1, Height = side, Width = side > 0 ? dataLength / side : dataLength, Format = PixFmt.CHW };
    }

    private enum PixFmt { CHW, HWC }

    private struct ImageLayout
    {
        public int Channels, Height, Width;
        public PixFmt Format;
    }
}
