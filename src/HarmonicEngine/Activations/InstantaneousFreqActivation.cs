using AiDotNet.ActivationFunctions;
using AiDotNet.HarmonicEngine.Transforms;

namespace AiDotNet.HarmonicEngine.Activations;

/// <summary>
/// Instantaneous Frequency activation function that modulates a signal by the derivative of its phase.
/// This captures rate-of-change information about oscillatory patterns that traditional activations miss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Most activation functions (ReLU, sigmoid) look at each value independently.
/// The Instantaneous Frequency activation is different — it looks at the entire signal to understand
/// how fast the oscillations are changing at each point.
///
/// Imagine watching a spinning top: the regular speed of spinning is the "carrier frequency,"
/// but if it speeds up or slows down, that change is the "instantaneous frequency deviation."
/// This activation extracts that deviation and uses it to modulate the signal.
///
/// For financial data, this means the activation can detect when market cycles are
/// accelerating or decelerating — information that ReLU simply cannot capture.
///
/// The scalar Activate/Derivative methods use a simple soft-threshold for compatibility.
/// The vector-level Activate method performs the full Hilbert-transform-based computation.
/// </para>
/// </remarks>
public class InstantaneousFreqActivation<T> : ActivationFunctionBase<T>
{
    private readonly AnalyticSignal<T> _analyticSignal;
    private readonly T _modulationStrength;

    /// <summary>
    /// Initializes a new InstantaneousFreqActivation.
    /// </summary>
    /// <param name="modulationStrength">
    /// Controls how strongly the instantaneous frequency modulates the signal.
    /// Values near 1.0 produce strong modulation; near 0.0 produces minimal effect.
    /// </param>
    public InstantaneousFreqActivation(double modulationStrength = 0.5)
    {
        if (modulationStrength < 0.0 || modulationStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(modulationStrength),
                $"Modulation strength must be in [0, 1], got {modulationStrength}.");

        _analyticSignal = new AnalyticSignal<T>();
        _modulationStrength = NumOps.FromDouble(modulationStrength);
    }

    /// <inheritdoc/>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Scalar activation: uses a soft tanh nonlinearity as a fallback when
    /// the full signal context is not available.
    /// </summary>
    public override T Activate(T input)
    {
        // Fallback: tanh-like nonlinearity for scalar context
        var x = NumOps.ToDouble(input);
        return NumOps.FromDouble(x * Math.Tanh(x));
    }

    /// <summary>
    /// Scalar derivative of the tanh fallback: d/dx[x * tanh(x)] = tanh(x) + x * sech^2(x).
    /// </summary>
    public override T Derivative(T input)
    {
        var x = NumOps.ToDouble(input);
        var tanhX = Math.Tanh(x);
        var sech2X = 1.0 - tanhX * tanhX;
        return NumOps.FromDouble(tanhX + x * sech2X);
    }

    /// <summary>
    /// Applies the full instantaneous frequency activation to a vector signal.
    /// The signal is modulated by the deviation of its instantaneous frequency from the mean.
    /// </summary>
    /// <param name="input">The input signal vector. Length should be a power of 2.</param>
    /// <returns>The modulated signal.</returns>
    /// <remarks>
    /// <para>
    /// Algorithm:
    /// 1. Compute instantaneous frequency via Hilbert transform
    /// 2. Subtract mean frequency to get deviation (how much faster/slower than average)
    /// 3. Apply sigmoid to deviation to get modulation factor in [0, 1]
    /// 4. Output = input * (1 - alpha + alpha * sigmoid(freq_deviation))
    ///    where alpha is the modulation strength
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        int n = input.Length;

        // Need at least 4 samples for meaningful instantaneous frequency
        if (n < 4 || (n & (n - 1)) != 0)
        {
            return input.Transform(Activate);
        }

        // Compute instantaneous frequency (length n-1)
        var instFreq = _analyticSignal.InstantaneousFrequency(input);

        // Compute mean frequency
        double meanFreq = 0;
        for (int i = 0; i < instFreq.Length; i++)
        {
            meanFreq += NumOps.ToDouble(instFreq[i]);
        }
        meanFreq /= instFreq.Length;

        // Modulate input by frequency deviation
        var output = new Vector<T>(n);
        var alpha = NumOps.ToDouble(_modulationStrength);

        for (int i = 0; i < n; i++)
        {
            double freqDev;
            if (i < instFreq.Length)
            {
                freqDev = NumOps.ToDouble(instFreq[i]) - meanFreq;
            }
            else
            {
                // Last sample: use the last available frequency deviation
                freqDev = NumOps.ToDouble(instFreq[instFreq.Length - 1]) - meanFreq;
            }

            // Sigmoid of frequency deviation
            var sigVal = 1.0 / (1.0 + Math.Exp(-freqDev));
            var modFactor = NumOps.FromDouble(1.0 - alpha + alpha * sigVal);

            output[i] = NumOps.Multiply(input[i], modFactor);
        }

        return output;
    }

    /// <inheritdoc/>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        // Approximate Jacobian: diagonal with scalar derivatives
        int n = input.Length;
        var jacobian = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            jacobian[i, i] = Derivative(input[i]);
        }
        return jacobian;
    }

    /// <inheritdoc/>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        var output = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }
        return output;
    }

    /// <inheritdoc/>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        var output = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }
        return output;
    }
}
