using AiDotNet.Helpers;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// The exact Gaussian Error Linear Unit, GELU(x) = x·Φ(x) = 0.5·x·erfc(−x/√2) (Hendrycks and Gimpel 2016).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <see cref="GELUActivation{T}"/> computes the tanh approximation, which differs from the exact function by up to
/// about 5e-4. Use this class when a model must reproduce a checkpoint trained with the exact form, as released BERT
/// checkpoints are (Hugging Face's <c>hidden_act = "gelu"</c>, PyTorch's <c>nn.GELU()</c> default).
/// </para>
/// <para><b>For Beginners:</b> GELU lets a value through in proportion to how likely a standard normal variable is to
/// be smaller than it: large positive values pass almost unchanged, large negative values become almost zero.
/// </para>
/// </remarks>
public class ExactGELUActivation<T> : ActivationFunctionBase<T>
{
    private static readonly double InverseSqrt2 = 1.0 / Math.Sqrt(2.0);
    private static readonly double InverseSqrt2Pi = 1.0 / Math.Sqrt(2.0 * Math.PI);

    /// <inheritdoc/>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>Computes 0.5·x·erfc(−x/√2) for one value.</summary>
    /// <param name="input">The value.</param>
    /// <returns>GELU of the value.</returns>
    public override T Activate(T input) => Activate(Scalar(input)).Data.Span[0];

    /// <summary>Computes the derivative Φ(x) + x·φ(x) for one value.</summary>
    /// <param name="input">The value.</param>
    /// <returns>The derivative at the value.</returns>
    public override T Derivative(T input) => Derivative(Scalar(input)).Data.Span[0];

    /// <summary>Computes 0.5·x·erfc(−x/√2) elementwise, through engine operations the gradient tape records.</summary>
    /// <param name="input">The values.</param>
    /// <returns>GELU of each value.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return ActivationHelper.ExactGelu(input, Engine);
    }

    /// <summary>Computes the derivative Φ(x) + x·φ(x) elementwise, with Φ(x) = 0.5·erfc(−x/√2) and
    /// φ(x) = exp(−x²/2)/√(2π).</summary>
    /// <param name="input">The values.</param>
    /// <returns>The derivative at each value.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0)
            return new Tensor<T>(input._shape);

        var engine = Engine;
        var cdf = engine.TensorMultiplyScalar(
            engine.TensorErfc(engine.TensorMultiplyScalar(input, NumOps.FromDouble(-InverseSqrt2))), NumOps.FromDouble(0.5));
        var pdf = engine.TensorMultiplyScalar(
            engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorMultiply(input, input), NumOps.FromDouble(-0.5))),
            NumOps.FromDouble(InverseSqrt2Pi));
        return engine.TensorAdd(cdf, engine.TensorMultiply(input, pdf));
    }

    private static Tensor<T> Scalar(T value) => new(new[] { 1 }, new Vector<T>(new[] { value }));
}
