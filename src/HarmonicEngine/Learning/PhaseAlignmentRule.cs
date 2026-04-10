using AiDotNet.HarmonicEngine.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Learning;

/// <summary>
/// Learning rule that aligns the phase of internal oscillators with the input signal
/// by adjusting carrier phases to maximize correlation with target patterns.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each spectral coefficient has a magnitude (how strong) and phase (where in the cycle).
/// Phase alignment adjusts only the phases to synchronize with recurring data patterns,
/// leaving magnitudes alone. When the model's oscillators are "in sync" with the data's rhythms,
/// prediction accuracy is maximized.
///
/// The update rule at each frequency k:
///   phase_new(k) = phase_old(k) + eta * sin(phase_target(k) - phase_old(k))
///
/// The sine function ensures:
/// - When phases are nearly aligned (small difference): small correction
/// - When phases are opposite (pi difference): maximum correction
/// - Updates wrap naturally around 2pi (no discontinuity)
///
/// This is especially effective for periodic/cyclical data like financial markets,
/// seasonal patterns, or biological rhythms.
/// </para>
/// </remarks>
public class PhaseAlignmentRule<T> : ISpectralLearningRule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _learningRate;

    /// <inheritdoc/>
    public string Name => "PhaseAlignment";

    /// <inheritdoc/>
    public double LearningRate => _learningRate;

    /// <summary>
    /// Initializes a new PhaseAlignmentRule.
    /// </summary>
    /// <param name="learningRate">Learning rate for phase updates.</param>
    public PhaseAlignmentRule(double learningRate = 0.05)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _learningRate = learningRate;
    }

    /// <summary>
    /// Updates the spectral filter by aligning phases toward the target spectrum.
    /// Magnitudes are adjusted via a slow exponential moving average.
    /// </summary>
    public void Update(Vector<Complex<T>> filter, Vector<Complex<T>> inputSpectrum, Vector<Complex<T>> targetSpectrum)
    {
        int n = filter.Length;

        for (int k = 0; k < n; k++)
        {
            // Current filter phase and magnitude
            double filterReal = _numOps.ToDouble(filter[k].Real);
            double filterImag = _numOps.ToDouble(filter[k].Imaginary);
            double filterMag = Math.Sqrt(filterReal * filterReal + filterImag * filterImag);
            double filterPhase = Math.Atan2(filterImag, filterReal);

            // Compute desired filter: H_desired = Y(k) / X(k)
            double xr = _numOps.ToDouble(inputSpectrum[k].Real);
            double xi = _numOps.ToDouble(inputSpectrum[k].Imaginary);
            double xPower = xr * xr + xi * xi;

            if (xPower < 1e-12)
            {
                continue; // Skip frequencies with no input energy
            }

            double yr = _numOps.ToDouble(targetSpectrum[k].Real);
            double yi = _numOps.ToDouble(targetSpectrum[k].Imaginary);

            // H_desired = Y * conj(X) / |X|^2
            double desiredReal = (yr * xr + yi * xi) / xPower;
            double desiredImag = (yi * xr - yr * xi) / xPower;
            double desiredPhase = Math.Atan2(desiredImag, desiredReal);
            double desiredMag = Math.Sqrt(desiredReal * desiredReal + desiredImag * desiredImag);

            // Phase update: move toward desired phase via sin(difference)
            double phaseDiff = desiredPhase - filterPhase;
            double phaseUpdate = _learningRate * Math.Sin(phaseDiff);
            double newPhase = filterPhase + phaseUpdate;

            // Magnitude update: slow exponential moving average toward desired
            double magRate = _learningRate * 0.1; // Slower for magnitude
            double newMag = filterMag + magRate * (desiredMag - filterMag);

            // Reconstruct complex filter coefficient
            filter[k] = new Complex<T>(
                _numOps.FromDouble(newMag * Math.Cos(newPhase)),
                _numOps.FromDouble(newMag * Math.Sin(newPhase)));
        }
    }
}
