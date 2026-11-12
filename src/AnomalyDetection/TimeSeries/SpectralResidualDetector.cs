using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies in time series using Spectral Residual method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Spectral Residual is inspired by visual saliency detection in images.
/// It transforms the time series to the frequency domain, finds what's "unusual" in the
/// spectrum (the saliency), and transforms back to identify anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Apply FFT to convert time series to frequency domain
/// 2. Compute the log amplitude spectrum
/// 3. Extract spectral residual by subtracting smoothed spectrum
/// 4. Transform back to get saliency map (anomaly scores)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with recurring patterns at multiple frequencies
/// - When anomalies disrupt the normal frequency pattern
/// - Large-scale time series (efficient O(n log n) complexity)
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Window size: 3 (for spectrum smoothing)
/// - Score threshold: Based on contamination
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Microsoft's SR-CNN: Spectral Residual based anomaly detection.
/// Originally from Hou, X., Zhang, L. (2007). "Saliency Detection: A Spectral Residual Approach."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Saliency Detection: A Spectral Residual Approach", "https://doi.org/10.1109/CVPR.2007.383267", Year = 2007, Authors = "Xiaodi Hou, Liqing Zhang")]
public class SpectralResidualDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _windowSize;

    /// <summary>
    /// Gets the window size for spectrum smoothing.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Creates a new Spectral Residual anomaly detector.
    /// </summary>
    /// <param name="windowSize">
    /// Window size for smoothing the log amplitude spectrum. Default is 3.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SpectralResidualDetector(int windowSize = 3, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (windowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "WindowSize must be at least 1. Recommended is 3.");
        }

        _windowSize = windowSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Spectral Residual expects univariate time series (1 column).",
                nameof(X));
        }

        // Compute baseline statistics from training data
        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Spectral Residual expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;
        var engine = Engine;

        // Zero-pad to the next power of two for the FFT.
        int fftSize = 1;
        while (fftSize < n) fftSize *= 2;

        var values = X.GetColumn(0).AsTensor();
        var padded = fftSize > n
            ? engine.TensorConcatenate(new[] { values, Tensor<T>.CreateZeros(new[] { fftSize - n }) }, 0)
            : values;
        engine.FFT(padded, Tensor<T>.CreateZeros(new[] { fftSize }), out var real, out var imag);

        // Log amplitude spectrum.
        var amplitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(real), engine.TensorSquare(imag)));
        var logAmplitude = engine.TensorLog(engine.TensorAddScalar(amplitude, NumOps.FromDouble(1e-10)));

        // Spectral residual: log amplitude minus its centred moving average (window truncated at the ends).
        int halfWindow = _windowSize / 2;
        var windows = AnomalyTensorOps.SlidingWindows(logAmplitude, halfWindow, halfWindow, out var mask);
        var avgLogAmplitude = engine.TensorDivide(
            engine.ReduceSum(windows, new[] { 1 }, false), engine.ReduceSum(mask, new[] { 1 }, false));
        var spectralResidual = engine.TensorSubtract(logAmplitude, avgLogAmplitude);

        // Reconstruct with the residual amplitude and the original phase, then invert.
        var phase = engine.TensorAtan2(imag, real);
        var newAmplitude = engine.TensorExp(spectralResidual);
        var newReal = engine.TensorMultiply(newAmplitude, engine.TensorCos(phase));
        var newImag = engine.TensorMultiply(newAmplitude, engine.TensorSin(phase));
        engine.IFFT(newReal, newImag, out var saliencyReal, out _);

        // Saliency: squared magnitude of the real part, trimmed to the original length.
        var trimmed = fftSize > n ? engine.TensorSlice(saliencyReal, new[] { 0 }, new[] { n }) : saliencyReal;
        return engine.TensorSquare(trimmed).ToVector();
    }
}
