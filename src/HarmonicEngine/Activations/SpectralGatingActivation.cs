using AiDotNet.ActivationFunctions;

namespace AiDotNet.HarmonicEngine.Activations;

/// <summary>
/// Spectral Gating activation function that applies an input-dependent multiplicative gate.
/// Each value is scaled by a learned sigmoid gate based on its own magnitude.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Spectral Gating is like a smart volume control for each frequency in a signal.
/// Instead of a fixed threshold (like ReLU), each value gets its own gate that learns how much
/// to let through. The gate value is between 0 (completely block) and 1 (fully pass).
///
/// Formula: f(x) = x * sigmoid(w * |x| + b)
///
/// The sigmoid function ensures the gate is always between 0 and 1.
/// The learnable parameters w and b control the gate's sensitivity and threshold.
///
/// This is nonlinear because the gate depends on the input — it's not just a fixed scaling.
/// Analogous to Gated Linear Units (GLU) used in transformers, but operating element-wise.
/// </para>
/// </remarks>
public class SpectralGatingActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _weight;
    private readonly T _bias;

    /// <summary>
    /// Initializes a new SpectralGating activation with specified gate parameters.
    /// </summary>
    /// <param name="weight">Controls gate sensitivity. Higher values make the gate sharper.</param>
    /// <param name="bias">Controls the gate threshold. Negative values require larger inputs to open the gate.</param>
    public SpectralGatingActivation(double weight = 5.0, double bias = -1.0)
    {
        _weight = NumOps.FromDouble(weight);
        _bias = NumOps.FromDouble(bias);
    }

    /// <inheritdoc/>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies spectral gating: f(x) = x * sigmoid(w * |x| + b).
    /// </summary>
    public override T Activate(T input)
    {
        var absVal = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(input)));
        var gateInput = NumOps.Add(NumOps.Multiply(_weight, absVal), _bias);
        var gate = Sigmoid(gateInput);
        return NumOps.Multiply(input, gate);
    }

    /// <summary>
    /// Computes the derivative of spectral gating.
    /// d/dx [x * sigma(w|x| + b)] = sigma + x * sigma' * w * sign(x)
    /// </summary>
    public override T Derivative(T input)
    {
        var x = NumOps.ToDouble(input);
        var absX = Math.Abs(x);
        var gateIn = NumOps.ToDouble(_weight) * absX + NumOps.ToDouble(_bias);
        var sig = 1.0 / (1.0 + Math.Exp(-gateIn));
        var sigPrime = sig * (1.0 - sig);
        var sign = x >= 0 ? 1.0 : -1.0;

        // Product rule: sigma + x * sigma' * w * sign(x)
        var deriv = sig + x * sigPrime * NumOps.ToDouble(_weight) * sign;
        return NumOps.FromDouble(deriv);
    }

    /// <inheritdoc/>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    /// <inheritdoc/>
    public override Matrix<T> Derivative(Vector<T> input)
    {
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

    private T Sigmoid(T x)
    {
        var expNeg = NumOps.FromDouble(Math.Exp(-NumOps.ToDouble(x)));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
    }
}
