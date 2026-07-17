using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DriftDetection;

/// <summary>
/// A drift monitor that watches two lenses in a single pass and attributes drift to <b>concept</b>
/// (the error stream) or <b>covariate</b> (the prediction-distribution) shift.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The configured <see cref="IDriftDetector{T}"/> is the concept lens: it consumes the per-observation
/// error <c>|predicted - actual|</c>, the industry-standard signal for "the model is getting wrong"
/// (P(y|x) changed). Layered on top, in the same pass, is a lightweight windowed mean-shift test on the
/// predictions themselves — the covariate lens — which flags "the inputs shifted" (P(x) changed) even
/// before accuracy visibly degrades. Each observation updates both lenses once; there is no second full
/// detector and no second pass.
/// </para>
/// <para><b>For Beginners:</b> Prime the monitor on the training data (so it knows what "normal" looks
/// like), then stream held-out or live <c>(predicted, actual)</c> pairs through <see cref="Observe"/>.
/// Ask <see cref="IsInDrift"/>/<see cref="Source"/> at any time, or take a <see cref="DriftReport"/> from
/// a batch check with <see cref="Check"/>.</para>
/// </remarks>
public sealed class DriftMonitor<T>
{
    private readonly IDriftDetector<T> _conceptDetector;
    private readonly INumericOperations<T> _numOps;

    // Covariate lens: reference distribution from priming (Welford) + sliding window during checking.
    private readonly int _windowSize;
    private readonly double _warningZ;
    private readonly double _driftZ;
    private readonly int _persistence;
    private double _refMean;
    private double _refStd;
    private long _refCount;
    private double _refM2;
    private readonly Queue<double> _window = new();
    private double _windowSum;
    private int _covariateBreachRun;

    // Concept lens: the error stream is turned into a binary "worse than training-typical" indicator so
    // rate-based detectors (DDM/EDDM) and mean-shift detectors (Page-Hinkley/ADWIN) alike see a
    // meaningful signal — a constant nonzero error would otherwise read as a constant 100% error rate.
    private readonly double _errorSigma;
    private double _errThreshold;
    private double _errMean;
    private long _errCount;
    private double _errM2;

    private bool _conceptDrift;
    private bool _conceptWarning;
    private bool _covariateDrift;
    private bool _covariateWarning;

    /// <summary>Creates a drift monitor over a configured concept detector.</summary>
    /// <param name="conceptDetector">The configured detector fed the error stream (concept lens).</param>
    /// <param name="windowSize">The covariate lens's sliding-window length. Defaults to 30.</param>
    /// <param name="warningZ">Standardized window-mean shift that raises a covariate warning. Defaults to 2.0.</param>
    /// <param name="driftZ">Standardized window-mean shift that flags covariate drift. Defaults to 3.0.</param>
    /// <param name="persistence">
    /// Consecutive windows whose standardized mean-shift must exceed <paramref name="driftZ"/> before
    /// covariate drift is flagged. Requiring a run rather than a single crossing keeps a noisy stream from
    /// tripping the lens by chance. Defaults to 3.
    /// </param>
    /// <param name="errorSigma">
    /// How many standard deviations above the training-typical error a residual must be to count as an
    /// "error" for the concept lens's binary indicator. Defaults to 3 (a standard outlier threshold).
    /// </param>
    public DriftMonitor(IDriftDetector<T> conceptDetector, int windowSize = 30, double warningZ = 2.0, double driftZ = 3.0, int persistence = 3, double errorSigma = 3.0)
    {
        _conceptDetector = conceptDetector ?? throw new ArgumentNullException(nameof(conceptDetector));
        _numOps = MathHelper.GetNumericOperations<T>();
        _windowSize = windowSize < 1 ? 1 : windowSize;
        _warningZ = warningZ;
        _driftZ = driftZ;
        _persistence = persistence < 1 ? 1 : persistence;
        _errorSigma = errorSigma;
    }

    /// <summary>Gets whether either lens is currently in drift.</summary>
    public bool IsInDrift => _conceptDrift || _covariateDrift;

    /// <summary>Gets whether either lens is currently in a warning state.</summary>
    public bool IsInWarning => _conceptWarning || _covariateWarning;

    /// <summary>Gets which shift the current drift is attributed to.</summary>
    public DriftSource Source =>
        (_conceptDrift, _covariateDrift) switch
        {
            (true, true) => DriftSource.Both,
            (true, false) => DriftSource.Concept,
            (false, true) => DriftSource.Covariate,
            _ => DriftSource.None,
        };

    /// <summary>
    /// Calibrates both lenses on a reference (training) stream: feeds the concept detector the training
    /// errors and accumulates the prediction reference distribution for the covariate lens.
    /// </summary>
    /// <param name="predicted">Reference predictions.</param>
    /// <param name="actual">Reference targets (same length as <paramref name="predicted"/>).</param>
    public void Prime(Vector<T> predicted, Vector<T> actual)
    {
        int n = Math.Min(predicted.Length, actual.Length);

        // Pass 1: accumulate the prediction reference (covariate lens) and the error baseline (concept
        // lens), so the concept threshold is known before any indicator is fed to the detector.
        for (int i = 0; i < n; i++)
        {
            double x = _numOps.ToDouble(predicted[i]);
            _refCount++;
            double dx = x - _refMean;
            _refMean += dx / _refCount;
            _refM2 += dx * (x - _refMean);

            double e = Math.Abs(_numOps.ToDouble(predicted[i]) - _numOps.ToDouble(actual[i]));
            _errCount++;
            double de = e - _errMean;
            _errMean += de / _errCount;
            _errM2 += de * (e - _errMean);
        }

        _refStd = _refCount > 1 ? Math.Sqrt(_refM2 / (_refCount - 1)) : 0.0;
        double errStd = _errCount > 1 ? Math.Sqrt(_errM2 / (_errCount - 1)) : 0.0;
        // A residual counts as an "error" when it exceeds the training-typical level by errorSigma std.
        // Floor above the mean so a zero-variance error stream (constant residual) still needs a genuine
        // increase to trip, rather than firing on residuals equal to the baseline.
        _errThreshold = _errMean + Math.Max(_errorSigma * errStd, Math.Abs(_errMean) * 1e-9 + 1e-12);

        // Pass 2: prime the detector's baseline rate with the training exceedance indicators.
        for (int i = 0; i < n; i++)
        {
            _conceptDetector.AddObservation(ExceedanceIndicator(predicted[i], actual[i]));
        }

        // Priming establishes the baseline; drift state accrues only during Observe/Check.
        _conceptDrift = _conceptWarning = _covariateDrift = _covariateWarning = false;
        _covariateBreachRun = 0;
        _window.Clear();
        _windowSum = 0.0;
    }

    /// <summary>
    /// Streams one observation through both lenses, updating drift/warning state.
    /// </summary>
    /// <param name="predicted">The prediction for this observation.</param>
    /// <param name="actual">The target for this observation.</param>
    /// <returns><c>true</c> if either lens is in drift after this observation.</returns>
    public bool Observe(T predicted, T actual)
    {
        // Concept lens: the configured detector on the binary "worse than training-typical" indicator.
        bool conceptFlag = _conceptDetector.AddObservation(ExceedanceIndicator(predicted, actual));
        _conceptDrift = _conceptDrift || conceptFlag || _conceptDetector.IsInDrift;
        _conceptWarning = _conceptWarning || _conceptDetector.IsInWarning;

        // Covariate lens: windowed mean-shift z-test on the prediction against the primed reference.
        double x = _numOps.ToDouble(predicted);
        _window.Enqueue(x);
        _windowSum += x;
        if (_window.Count > _windowSize)
        {
            _windowSum -= _window.Dequeue();
        }

        if (_window.Count >= _windowSize)
        {
            double windowMean = _windowSum / _window.Count;
            // A zero-variance reference (e.g. a constant-prediction model) has no natural scale, so any
            // real shift is definite drift: floor the std to a tiny value relative to the reference level.
            double effectiveStd = _refStd > 0.0 ? _refStd : Math.Max(1e-12, Math.Abs(_refMean) * 1e-9);
            double standardError = effectiveStd / Math.Sqrt(_window.Count);
            double z = Math.Abs((windowMean - _refMean) / standardError);
            if (z >= _driftZ)
            {
                // Require a run of breaching windows so single-window noise doesn't trip the lens.
                _covariateBreachRun++;
                if (_covariateBreachRun >= _persistence)
                {
                    _covariateDrift = true;
                }
            }
            else
            {
                _covariateBreachRun = 0;
                if (z >= _warningZ)
                {
                    _covariateWarning = true;
                }
            }
        }

        return IsInDrift;
    }

    /// <summary>
    /// Streams a whole (checking) batch through the monitor and returns an attributed report.
    /// </summary>
    /// <param name="predicted">Predictions to check.</param>
    /// <param name="actual">Targets to check (same length as <paramref name="predicted"/>).</param>
    /// <returns>The drift assessment over the batch.</returns>
    public DriftReport Check(Vector<T> predicted, Vector<T> actual)
    {
        int n = Math.Min(predicted.Length, actual.Length);
        int firstDriftIndex = -1;
        for (int i = 0; i < n; i++)
        {
            bool wasInDrift = IsInDrift;
            Observe(predicted[i], actual[i]);
            if (!wasInDrift && IsInDrift && firstDriftIndex < 0)
            {
                firstDriftIndex = i;
            }
        }

        return new DriftReport
        {
            DriftDetected = IsInDrift,
            WarningDetected = IsInWarning,
            Source = Source,
            FirstDriftIndex = firstDriftIndex,
            ObservationsChecked = n,
        };
    }

    /// <summary>
    /// The concept lens's per-observation signal: <c>1</c> when the residual exceeds the training-typical
    /// error threshold, else <c>0</c>. Turning the error stream into an exceedance rate lets both
    /// rate-based (DDM/EDDM) and mean-shift (Page-Hinkley/ADWIN) detectors register concept drift.
    /// </summary>
    private T ExceedanceIndicator(T predicted, T actual)
    {
        double e = Math.Abs(_numOps.ToDouble(predicted) - _numOps.ToDouble(actual));
        return _numOps.FromDouble(e > _errThreshold ? 1.0 : 0.0);
    }
}
