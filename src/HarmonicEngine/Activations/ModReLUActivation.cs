using AiDotNet.ActivationFunctions;

namespace AiDotNet.HarmonicEngine.Activations;

/// <summary>
/// Modified Rectified Linear Unit (modReLU) activation for spectral signal processing.
/// Applies a learnable threshold based on signal magnitude while preserving sign/phase.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> modReLU is designed for signals that have both positive and negative values
/// (like oscillating waves). Unlike standard ReLU which kills all negative values, modReLU
/// thresholds based on the absolute magnitude. Small oscillations (below the threshold) are
/// zeroed out as noise, while large oscillations pass through with their sign preserved.
///
/// Formula: f(x) = x * max(0, |x| + b) / max(|x|, epsilon)
///
/// When b is negative, values with |x| less than |b| are zeroed.
/// When b is positive, all values pass through (scaled slightly).
///
/// Reference: Arjovsky et al., "Unitary Evolution Recurrent Neural Networks" (2016).
/// </para>
/// </remarks>
public class ModReLUActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _bias;
    private readonly T _epsilon;

    /// <summary>
    /// Initializes a new ModReLU activation with a specified bias threshold.
    /// </summary>
    /// <param name="bias">
    /// The learnable bias parameter. Negative values create a dead zone for small magnitudes.
    /// Default is -0.1, which zeros out signals with magnitude less than 0.1.
    /// </param>
    public ModReLUActivation(double bias = -0.1)
    {
        _bias = NumOps.FromDouble(bias);
        _epsilon = NumOps.FromDouble(1e-8);
    }

    /// <inheritdoc/>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies modReLU to a single value: f(x) = x * max(0, |x| + b) / max(|x|, epsilon).
    /// </summary>
    public override T Activate(T input)
    {
        var absVal = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(input)));
        var gateInput = NumOps.Add(absVal, _bias);

        // max(0, |x| + b)
        var gate = NumOps.GreaterThan(gateInput, NumOps.Zero) ? gateInput : NumOps.Zero;

        // Normalize by |x| (with epsilon for stability)
        var denom = NumOps.GreaterThan(absVal, _epsilon) ? absVal : _epsilon;

        return NumOps.Multiply(input, NumOps.Divide(gate, denom));
    }

    /// <summary>
    /// Computes the derivative of modReLU.
    /// </summary>
    public override T Derivative(T input)
    {
        var absVal = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(input)));
        var gateInput = NumOps.Add(absVal, _bias);

        // In the active region (|x| + b > 0): derivative is approximately 1
        // In the dead zone: derivative is 0
        return NumOps.GreaterThan(gateInput, NumOps.Zero) ? NumOps.One : NumOps.Zero;
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
}
