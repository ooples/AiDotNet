using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff.Testing;

/// <summary>
/// Provides numerical gradient computation using finite differences for gradient verification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This utility class computes gradients numerically using the central difference method.
/// It serves as a ground truth for verifying that automatic differentiation produces correct gradients.
/// </para>
/// <para><b>For Beginners:</b> This class helps verify that our gradient calculations are correct.
///
/// The idea is simple:
/// 1. We want to know how much f(x) changes when we change x slightly
/// 2. We compute f(x+h) and f(x-h) where h is a tiny number
/// 3. The gradient is approximately: (f(x+h) - f(x-h)) / (2h)
///
/// This is called the "central difference" method. It's slow but reliable.
/// We use it to check that our fast autodiff gradients are correct.
///
/// Example:
/// - For f(x) = x^2, the true gradient is 2x
/// - At x=3: numerical gradient = ((3+h)^2 - (3-h)^2) / (2h) ≈ 6
/// - Autodiff should also give 6
/// </para>
/// </remarks>
public static class NumericalGradient<T>
{
    /// <summary>
    /// The numeric operations appropriate for the generic type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Default configuration for numerical gradient computation.
    /// </summary>
    public static class Defaults
    {
        /// <summary>Default step size for finite differences.</summary>
        public const double Epsilon = 1e-5;

        /// <summary>Default relative tolerance for gradient comparison.</summary>
        public const double RelativeTolerance = 1e-4;

        /// <summary>Default absolute tolerance for gradient comparison.</summary>
        public const double AbsoluteTolerance = 1e-6;
    }

    /// <summary>
    /// Computes numerical gradient for a scalar-valued function of a tensor.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <param name="scalarFunction">A function that takes a tensor and returns a scalar value.</param>
    /// <param name="epsilon">Step size for finite differences (default: 1e-5).</param>
    /// <returns>A tensor of the same shape as input containing numerical gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method computes df/dx for each element x in the input tensor, where f is the
    /// scalar-valued function. The central difference formula is used:
    /// df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    /// </para>
    /// <para><b>For Beginners:</b> This computes how much the function output changes
    /// when each input element changes slightly.
    ///
    /// For each element in the input:
    /// 1. Increase it by a tiny amount (epsilon)
    /// 2. Compute the function
    /// 3. Decrease it by the same amount
    /// 4. Compute the function again
    /// 5. The gradient is (result1 - result2) / (2 * epsilon)
    /// </para>
    /// </remarks>
    public static Tensor<T> ComputeForScalarFunction(
        Tensor<T> input,
        Func<Tensor<T>, T> scalarFunction,
        double epsilon = Defaults.Epsilon)
    {
        var gradient = new Tensor<T>(input.Shape);
        var h = NumOps.FromDouble(epsilon);
        var twoH = NumOps.FromDouble(2 * epsilon);

        for (int i = 0; i < input.Length; i++)
        {
            var originalValue = input[i];

            // f(x + h)
            input[i] = NumOps.Add(originalValue, h);
            var fPlus = scalarFunction(input);

            // f(x - h)
            input[i] = NumOps.Subtract(originalValue, h);
            var fMinus = scalarFunction(input);

            // Restore original value
            input[i] = originalValue;

            // Central difference: (f(x+h) - f(x-h)) / (2h)
            gradient[i] = NumOps.Divide(NumOps.Subtract(fPlus, fMinus), twoH);
        }

        return gradient;
    }

    /// <summary>
    /// Computes numerical gradient for a tensor-valued function, given an output gradient.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <param name="outputGradient">The gradient flowing back from the output (upstream gradient).</param>
    /// <param name="tensorFunction">A function that takes a tensor and returns a tensor.</param>
    /// <param name="epsilon">Step size for finite differences (default: 1e-5).</param>
    /// <returns>A tensor of the same shape as input containing numerical gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of a loss function L with respect to the input,
    /// where L = sum(output * outputGradient). This matches how gradients flow in backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> In neural networks, gradients flow backwards through layers.
    ///
    /// When we have a layer y = f(x), we receive the gradient dL/dy from the next layer.
    /// We need to compute dL/dx = dL/dy * dy/dx (chain rule).
    ///
    /// This method computes dL/dx numerically by:
    /// 1. Perturbing each input element
    /// 2. Computing how the output changes
    /// 3. Multiplying by the output gradient (chain rule)
    /// </para>
    /// </remarks>
    public static Tensor<T> ComputeForTensorFunction(
        Tensor<T> input,
        Tensor<T> outputGradient,
        Func<Tensor<T>, Tensor<T>> tensorFunction,
        double epsilon = Defaults.Epsilon)
    {
        // Convert to scalar function by taking dot product with output gradient
        T ScalarFunction(Tensor<T> x)
        {
            var output = tensorFunction(x);
            return DotProduct(output, outputGradient);
        }

        return ComputeForScalarFunction(input, ScalarFunction, epsilon);
    }

    /// <summary>
    /// Computes numerical gradient using ComputationNode operations for direct comparison with autodiff.
    /// </summary>
    /// <param name="inputValue">The input tensor value.</param>
    /// <param name="outputGradient">The gradient flowing back from the output.</param>
    /// <param name="operation">A function that takes a ComputationNode and returns a ComputationNode.</param>
    /// <param name="epsilon">Step size for finite differences (default: 1e-5).</param>
    /// <returns>A tensor containing numerical gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method is specifically designed for testing TensorOperations. It wraps inputs in
    /// ComputationNodes and applies the operation, making it directly comparable to autodiff results.
    /// </para>
    /// <para><b>For Beginners:</b> This is used to verify TensorOperations gradients.
    ///
    /// TensorOperations like ReLU, Sigmoid, etc. work with ComputationNodes.
    /// This method:
    /// 1. Creates a ComputationNode from the input tensor
    /// 2. Applies the operation (like ReLU)
    /// 3. Computes numerical gradients
    /// 4. These can be compared with the autodiff gradients
    /// </para>
    /// </remarks>
    public static Tensor<T> ComputeForOperation(
        Tensor<T> inputValue,
        Tensor<T> outputGradient,
        Func<ComputationNode<T>, ComputationNode<T>> operation,
        double epsilon = Defaults.Epsilon)
    {
        Tensor<T> TensorFunction(Tensor<T> x)
        {
            var node = TensorOperations<T>.Variable(x, requiresGradient: false);
            var result = operation(node);
            return result.Value;
        }

        return ComputeForTensorFunction(inputValue, outputGradient, TensorFunction, epsilon);
    }

    /// <summary>
    /// Computes numerical gradient for a binary operation (two inputs).
    /// </summary>
    /// <param name="input1">The first input tensor.</param>
    /// <param name="input2">The second input tensor.</param>
    /// <param name="outputGradient">The gradient flowing back from the output.</param>
    /// <param name="operation">A function that takes two ComputationNodes and returns a ComputationNode.</param>
    /// <param name="epsilon">Step size for finite differences (default: 1e-5).</param>
    /// <returns>A tuple containing gradients for both inputs.</returns>
    /// <remarks>
    /// <para>
    /// This method computes numerical gradients for operations with two inputs, like Add or Multiply.
    /// </para>
    /// <para><b>For Beginners:</b> Some operations like addition (a + b) have two inputs.
    ///
    /// We need to compute gradients for both:
    /// - dL/da: How does loss change when we change 'a'?
    /// - dL/db: How does loss change when we change 'b'?
    ///
    /// This method computes both gradients numerically.
    /// </para>
    /// </remarks>
    public static (Tensor<T> grad1, Tensor<T> grad2) ComputeForBinaryOperation(
        Tensor<T> input1,
        Tensor<T> input2,
        Tensor<T> outputGradient,
        Func<ComputationNode<T>, ComputationNode<T>, ComputationNode<T>> operation,
        double epsilon = Defaults.Epsilon)
    {
        // Gradient for input1
        Tensor<T> TensorFunction1(Tensor<T> x)
        {
            var node1 = TensorOperations<T>.Variable(x, requiresGradient: false);
            var node2 = TensorOperations<T>.Variable(input2.Clone(), requiresGradient: false);
            var result = operation(node1, node2);
            return result.Value;
        }

        // Gradient for input2
        Tensor<T> TensorFunction2(Tensor<T> x)
        {
            var node1 = TensorOperations<T>.Variable(input1.Clone(), requiresGradient: false);
            var node2 = TensorOperations<T>.Variable(x, requiresGradient: false);
            var result = operation(node1, node2);
            return result.Value;
        }

        var grad1 = ComputeForTensorFunction(input1.Clone(), outputGradient, TensorFunction1, epsilon);
        var grad2 = ComputeForTensorFunction(input2.Clone(), outputGradient, TensorFunction2, epsilon);

        return (grad1, grad2);
    }

    /// <summary>
    /// Compares two tensors and returns the maximum relative error.
    /// </summary>
    /// <param name="expected">The expected (numerical) gradient.</param>
    /// <param name="actual">The actual (autodiff) gradient.</param>
    /// <param name="relativeTolerance">Relative tolerance for comparison.</param>
    /// <param name="absoluteTolerance">Absolute tolerance for comparison.</param>
    /// <returns>Comparison result with detailed error information.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two gradient tensors and reports any discrepancies.
    /// Both relative and absolute tolerances are considered to handle both large
    /// and near-zero gradient values appropriately.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if two gradient tensors are "close enough".
    ///
    /// We use two types of tolerances:
    /// - Relative: For large values, we allow small percentage differences
    /// - Absolute: For values near zero, we allow small absolute differences
    ///
    /// A gradient passes if EITHER tolerance is satisfied.
    /// </para>
    /// </remarks>
    public static ComparisonResult Compare(
        Tensor<T> expected,
        Tensor<T> actual,
        double relativeTolerance = Defaults.RelativeTolerance,
        double absoluteTolerance = Defaults.AbsoluteTolerance)
    {
        if (!expected.Shape.SequenceEqual(actual.Shape))
        {
            return new ComparisonResult
            {
                Passed = false,
                MaxRelativeError = double.MaxValue,
                Errors = { $"Shape mismatch: expected {FormatShape(expected.Shape)}, got {FormatShape(actual.Shape)}" }
            };
        }

        var result = new ComparisonResult();
        var errors = new List<double>();

        for (int i = 0; i < expected.Length; i++)
        {
            var expectedVal = NumOps.ToDouble(expected[i]);
            var actualVal = NumOps.ToDouble(actual[i]);

            var relativeError = ComputeRelativeError(expectedVal, actualVal);
            var absoluteError = Math.Abs(expectedVal - actualVal);

            errors.Add(relativeError);

            // Fail if both tolerances are exceeded
            if (relativeError > relativeTolerance && absoluteError > absoluteTolerance)
            {
                result.FailedElements++;
                result.Errors.Add(
                    $"Index {i}: expected={expectedVal:E6}, actual={actualVal:E6}, " +
                    $"relError={relativeError:E4}, absError={absoluteError:E6}");
            }

            result.TotalElementsChecked++;
        }

        result.MaxRelativeError = errors.Count > 0 ? errors.Max() : 0;
        result.AverageRelativeError = errors.Count > 0 ? errors.Average() : 0;
        result.Passed = result.FailedElements == 0;

        return result;
    }

    /// <summary>
    /// Computes relative error between two values.
    /// </summary>
    private static double ComputeRelativeError(double expected, double actual)
    {
        var maxAbs = Math.Max(Math.Abs(expected), Math.Abs(actual));
        if (maxAbs < 1e-10)
            return 0; // Both essentially zero

        return Math.Abs(expected - actual) / maxAbs;
    }

    /// <summary>
    /// Computes dot product of two tensors (sum of element-wise products).
    /// </summary>
    private static T DotProduct(Tensor<T> a, Tensor<T> b)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    /// <summary>
    /// Formats a shape array for display.
    /// </summary>
    private static string FormatShape(int[] shape)
    {
        return $"[{string.Join(", ", shape)}]";
    }

    /// <summary>
    /// Result of comparing numerical and analytical gradients.
    /// </summary>
    public class ComparisonResult
    {
        /// <summary>Whether all gradients passed verification.</summary>
        public bool Passed { get; set; }

        /// <summary>Maximum relative error observed.</summary>
        public double MaxRelativeError { get; set; }

        /// <summary>Average relative error.</summary>
        public double AverageRelativeError { get; set; }

        /// <summary>Number of elements that failed verification.</summary>
        public int FailedElements { get; set; }

        /// <summary>Total elements checked.</summary>
        public int TotalElementsChecked { get; set; }

        /// <summary>Detailed error messages for failed elements.</summary>
        public List<string> Errors { get; set; } = new();

        /// <summary>
        /// Returns a summary string of the comparison result.
        /// </summary>
        public override string ToString()
        {
            return $"GradientComparison: {(Passed ? "PASSED" : "FAILED")} " +
                   $"(MaxError: {MaxRelativeError:E4}, AvgError: {AverageRelativeError:E4}, " +
                   $"Failed: {FailedElements}/{TotalElementsChecked})";
        }
    }
}
