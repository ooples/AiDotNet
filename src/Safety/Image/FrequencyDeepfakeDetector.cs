using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected to arXiv 1911.00686, the actual id for this title. The recorded 2003.08685
// is "Leveraging Frequency Analysis for Deep Fake Image Recognition" (Frank et al.) — a real and
// closely related paper, but a different one, and it detects via the DCT. This implementation uses the
// DFT power spectrum with no DCT anywhere, which matches Durall et al., so the recorded title is the
// correct attribution and only the identifier was wrong.
[ResearchPaper("Unmasking DeepFakes with simple Features",
    "https://arxiv.org/abs/1911.00686",
    Year = 2019,
    Authors = "Ricard Durall, Margret Keuper, Franz-Josef Pfreundt, Janis Keuper")]
public class FrequencyDeepfakeDetector<T> : DeepfakeDetectorBase<T>
{

    private static readonly T Zero = NumOps.Zero;
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "FrequencyDeepfakeDetector";

    /// <summary>
    /// Initializes a new frequency-domain deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public FrequencyDeepfakeDetector(double threshold = 0.5)
        : base(threshold)
    {
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var analysis = Analyze(image);
        if (analysis is null) return findings;

        double finalScore = NumOps.ToDouble(analysis.Score);
        if (finalScore >= Threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = finalScore,
                Description = $"Frequency domain analysis: potential AI-generated image (score: {finalScore:F3}). " +
                              $"Spectral flatness: {NumOps.ToDouble(analysis.Flatness):F3}, " +
                              $"periodic peaks: {NumOps.ToDouble(analysis.Periodic):F3}, " +
                              $"high-freq energy ratio: {NumOps.ToDouble(analysis.HighFrequency):F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// The image is smaller than 8x8 pixels. A score of 0 would read as "authentic" for an image that
    /// was never analysed.
    /// </exception>
    public override double GetDeepfakeScore(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var analysis = Analyze(image);
        if (analysis is null)
        {
            throw new ArgumentException(
                "Frequency analysis needs an image of at least 8x8 pixels.", nameof(image));
        }

        return NumOps.ToDouble(analysis.Score);
    }

    /// <summary>The row-averaged spectral measurements and their weighted score.</summary>
    private sealed class Analysis
    {
        public T Flatness { get; set; } = Zero;
        public T Periodic { get; set; } = Zero;
        public T HighFrequency { get; set; } = Zero;
        public T Score { get; set; } = Zero;
    }

    /// <summary>Measures the image's row spectra, or returns null when it is too small to analyse.</summary>
    private Analysis? Analyze(Tensor<T> image)
    {
        if (image.Length == 0) return null;

        var layout = DetermineLayout(image._shape, image.Length);
        if (layout.Height < 8 || layout.Width < 8) return null;

        // Find nearest power of 2 for FFT
        int fftSize = 1;
        while (fftSize < layout.Width && fftSize < 512) fftSize *= 2;
        if (fftSize > layout.Width) fftSize /= 2;
        if (fftSize < 8) return null;

        using var noGrad = new NoGradScope<T>();
        var engine = Engine;

        // Analyze multiple rows from different parts of the image: rows y = r * rowStride
        // of channel 0, first fftSize columns, as [numRows, fftSize].
        int numRows = Math.Min(16, layout.Height);
        int rowStride = layout.Height / numRows;
        var rows = engine.TensorNarrow(
            engine.TensorSliceAxis(
                engine.Reshape(
                    engine.TensorNarrow(ChannelPlane(image, layout), 0, 0, numRows * rowStride).Contiguous(),
                    new[] { numRows, rowStride, layout.Width }),
                1, 0),
            1, 0, fftSize).Contiguous();
        rows = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Le(rows, 1.0),
            engine.TensorMultiplyScalar(rows, TwoFiftyFive), rows);

        // FFT and magnitude spectrum, bins 1 .. halfLen-1
        int halfLen = fftSize / 2;
        var magnitudes = engine.TensorNarrow(SafetyTensorOps<T>.FftMagnitude(rows), 1, 1, halfLen - 1);

        // 1. Spectral flatness: geometric mean / arithmetic mean
        //    Low flatness = concentrated spectrum (natural)
        //    High flatness = flat spectrum (can indicate GAN artifacts)
        var flatness = ComputeSpectralFlatness(magnitudes);

        // 2. Periodic peak detection (GAN fingerprints create periodic patterns)
        var periodicPeak = DetectPeriodicPeaks(magnitudes, halfLen);

        // 3. High-frequency energy ratio (low band: bins 1 .. midBand-1)
        int midBand = halfLen / 2;
        int lowCount = Math.Max(0, midBand - 1);
        var power = engine.TensorSquare(magnitudes);
        var axis1 = new[] { 1 };
        var lowEnergy = lowCount > 0
            ? engine.ReduceSum(engine.TensorNarrow(power, 1, 0, lowCount), axis1, false)
            : Tensor<T>.CreateZeros(numRows);
        var highEnergy = engine.ReduceSum(engine.TensorNarrow(power, 1, lowCount, halfLen - 1 - lowCount), axis1, false);
        var totalEnergy = engine.TensorAdd(lowEnergy, highEnergy);
        var hasEnergy = SafetyTensorOps<T>.Gt(totalEnergy, 1e-10);
        var highFreqRatio = SafetyTensorOps<T>.Where(hasEnergy,
            engine.TensorDivide(highEnergy, SafetyTensorOps<T>.Where(hasEnergy, totalEnergy, 1.0)), 0.0);

        T avgFlatness = engine.TensorMean(flatness);
        T avgPeriodic = engine.TensorMean(periodicPeak);
        T avgHighFreq = engine.TensorMean(highFreqRatio);

        // Combine signals:
        // - Spectral flatness deviation from natural images (~0.1-0.3)
        //   GAN images often have flatness > 0.5
        T flatnessAnomaly = SafetyTensorOps<T>.MaxZero(
            NumOps.Divide(NumOps.Subtract(avgFlatness, NumOps.FromDouble(0.3)), NumOps.FromDouble(0.7)));

        // - Periodic peaks (natural images rarely have them)
        T periodicAnomaly = SafetyTensorOps<T>.MinOne(avgPeriodic);

        // - High frequency energy (AI images often have too much or too little)
        //   Natural: ~0.2-0.4; GAN: often < 0.1 or > 0.5
        T hfAnomaly = NumOps.LessThan(avgHighFreq, NumOps.FromDouble(0.1))
            ? NumOps.Multiply(NumOps.Subtract(NumOps.FromDouble(0.1), avgHighFreq), NumOps.FromDouble(10))
            : NumOps.GreaterThan(avgHighFreq, NumOps.FromDouble(0.5))
                ? NumOps.Multiply(NumOps.Subtract(avgHighFreq, NumOps.FromDouble(0.5)), NumOps.FromDouble(2))
                : Zero;
        hfAnomaly = SafetyTensorOps<T>.MinOne(hfAnomaly);

        T score = NumOps.Add(NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(0.35), flatnessAnomaly),
            NumOps.Multiply(NumOps.FromDouble(0.40), periodicAnomaly)),
            NumOps.Multiply(NumOps.FromDouble(0.25), hfAnomaly));

        return new Analysis
        {
            Flatness = avgFlatness,
            Periodic = avgPeriodic,
            HighFrequency = avgHighFreq,
            Score = SafetyTensorOps<T>.MinOne(score),
        };
    }

    /// <summary>Per-row spectral flatness of <c>[rows, bins]</c> magnitudes over the bins above 1e-20.</summary>
    private static Tensor<T> ComputeSpectralFlatness(Tensor<T> magnitudes)
    {
        var engine = AiDotNetEngine.Current;
        int rows = magnitudes.Shape[0];
        if (magnitudes.Shape[1] <= 0) return Tensor<T>.CreateZeros(rows);

        var axis1 = new[] { 1 };
        var valid = SafetyTensorOps<T>.Gt(magnitudes, 1e-20);
        var logSum = engine.ReduceSum(SafetyTensorOps<T>.Where(valid,
            engine.TensorLog(SafetyTensorOps<T>.Where(valid, magnitudes, 1.0)), 0.0), axis1, false);
        var arithmeticSum = engine.ReduceSum(SafetyTensorOps<T>.Where(valid, magnitudes, 0.0), axis1, false);
        var validCount = engine.ReduceSum(valid, axis1, false);

        var usable = SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(validCount, 0.0),
            SafetyTensorOps<T>.Not(SafetyTensorOps<T>.Lt(arithmeticSum, 1e-20)));
        var safeCount = SafetyTensorOps<T>.Where(usable, validCount, 1.0);
        var geometricMean = engine.TensorExp(engine.TensorDivide(logSum, safeCount));
        var arithmeticMean = engine.TensorDivide(arithmeticSum, safeCount);
        usable = SafetyTensorOps<T>.And(usable, SafetyTensorOps<T>.Gt(arithmeticMean, 1e-20));

        return SafetyTensorOps<T>.Where(usable,
            engine.TensorDivide(geometricMean, SafetyTensorOps<T>.Where(usable, arithmeticMean, 1.0)), 0.0);
    }

    /// <summary>
    /// Per-row peak normalised autocorrelation of the magnitude spectrum (bins 1 .. length-1,
    /// given as <c>[rows, length - 1]</c>) over lags 2 .. min(16, length / 4).
    /// </summary>
    private static Tensor<T> DetectPeriodicPeaks(Tensor<T> magnitudes, int length)
    {
        var engine = AiDotNetEngine.Current;
        int rows = magnitudes.Shape[0];
        if (length < 8) return Tensor<T>.CreateZeros(rows);

        // Compute autocorrelation of the magnitude spectrum to find periodicity
        // Periodic GAN artifacts create peaks in autocorrelation at regular intervals
        var axis1 = new[] { 1 };
        T count = NumOps.FromDouble(length - 1);
        var meanMag = engine.TensorDivideScalar(engine.ReduceSum(magnitudes, axis1, true), count);
        var centered = engine.TensorSubtract(magnitudes, meanMag);
        var variance = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(centered), axis1, false), count);
        var hasVariance = SafetyTensorOps<T>.Not(SafetyTensorOps<T>.Lt(variance, 1e-20));

        // Check for autocorrelation peaks at lags 2-16 (common GAN periodicities)
        int maxLag = Math.Min(16, length / 4);
        if (maxLag < 2) return Tensor<T>.CreateZeros(rows);
        SafetyTensorOps<T>.LaggedProducts(centered, 2, maxLag + 1, out var corrSum, out _, out _);
        var pairs = engine.ScalarMinusTensor(count, engine.TensorArange<T>(2, maxLag + 1, 1));
        var corr = engine.TensorDivide(corrSum, engine.Reshape(pairs, new[] { 1, maxLag - 1 }));
        var safeVariance = engine.Reshape(SafetyTensorOps<T>.Where(hasVariance, variance, 1.0), new[] { rows, 1 });
        var normalizedCorr = engine.TensorDivide(corr, safeVariance);
        var maxCorr = SafetyTensorOps<T>.FirstStrictMax(normalizedCorr, 0.0).Best;

        return SafetyTensorOps<T>.Where(hasVariance, SafetyTensorOps<T>.MinOne(maxCorr), 0.0);
    }

    /// <summary>Channel 0 of the image as a <c>[height, width]</c> plane.</summary>
    private static Tensor<T> ChannelPlane(Tensor<T> image, ImageLayout layout)
    {
        var engine = AiDotNetEngine.Current;
        int pixels = layout.Height * layout.Width;
        var flat = engine.Reshape(image.Contiguous(), new[] { image.Length });
        var planeShape = new[] { layout.Height, layout.Width };
        if (layout.Format == PixFmt.CHW)
            return engine.Reshape(engine.TensorNarrow(flat, 0, 0, pixels), planeShape);

        var interleaved = engine.Reshape(engine.TensorNarrow(flat, 0, 0, pixels * layout.Channels),
            new[] { pixels, layout.Channels });
        return engine.Reshape(engine.TensorNarrow(interleaved, 1, 0, 1).Contiguous(), planeShape);
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
