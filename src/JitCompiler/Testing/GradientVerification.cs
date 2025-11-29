using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.JitCompiler.Testing;

/// <summary>
/// Utility for verifying gradient formulas using numerical differentiation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>Important:</b> This class verifies mathematical gradient formulas, NOT the actual
/// TensorOperations autodiff implementations. For verifying TensorOperations gradients,
/// use <see cref="AiDotNet.Autodiff.Testing.TensorOperationsVerification{T}"/> instead.
/// </para>
/// <para>
/// This class is useful for:
/// - Testing that hand-written gradient formulas are mathematically correct
/// - Verifying gradient formulas before implementing them in TensorOperations
/// - Educational purposes to understand how numerical gradient checking works
/// </para>
/// <para><b>For Beginners:</b> This tests that gradient formulas are mathematically correct.
///
/// The idea:
/// 1. You provide a forward function (e.g., f(x) = x^2)
/// 2. You provide a gradient function (e.g., df/dx = 2x)
/// 3. We compute numerical gradient using finite differences: (f(x+h) - f(x-h)) / (2h)
/// 4. We compare your gradient formula with the numerical gradient
///
/// This is different from <see cref="AiDotNet.Autodiff.Testing.TensorOperationsVerification{T}"/>
/// which tests the actual autodiff implementation in TensorOperations.
///
/// Example:
/// - f(x) = x^2
/// - Your gradient formula: df/dx = 2x
/// - Numerical: (f(x+h) - f(x-h)) / (2h) = ((x+h)^2 - (x-h)^2) / (2h) = 2x
/// - They match! Your gradient formula is correct.
/// </para>
/// </remarks>
[Obsolete("For testing TensorOperations autodiff gradients, use AiDotNet.Autodiff.Testing.TensorOperationsVerification<T> instead. " +
          "This class is retained for testing raw gradient formulas and IR operation semantics.")]
public class GradientVerification<T>
{
    /// <summary>
    /// The numeric operations appropriate for the generic type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Configuration for gradient verification.
    /// </summary>
    public class VerificationConfig
    {
        /// <summary>Step size for finite differences (default: 1e-5).</summary>
        public double Epsilon { get; set; } = 1e-5;

        /// <summary>Relative tolerance for gradient comparison (default: 1e-4).</summary>
        public double RelativeTolerance { get; set; } = 1e-4;

        /// <summary>Absolute tolerance for gradient comparison (default: 1e-6).</summary>
        public double AbsoluteTolerance { get; set; } = 1e-6;

        /// <summary>Maximum number of elements to check for large tensors (default: 1000).</summary>
        public int MaxElementsToCheck { get; set; } = 1000;

        /// <summary>Whether to print detailed results (default: false).</summary>
        public bool Verbose { get; set; } = false;
    }

    private readonly VerificationConfig _config;

    /// <summary>
    /// Result of gradient verification.
    /// </summary>
    public class VerificationResult
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
        /// Returns a summary string of the verification result.
        /// </summary>
        public override string ToString()
        {
            return $"GradientVerification: {(Passed ? "PASSED" : "FAILED")} " +
                   $"(MaxError: {MaxRelativeError:E4}, AvgError: {AverageRelativeError:E4}, " +
                   $"Failed: {FailedElements}/{TotalElementsChecked})";
        }
    }

    /// <summary>
    /// Initializes with default configuration.
    /// </summary>
    public GradientVerification() : this(new VerificationConfig()) { }

    /// <summary>
    /// Initializes with custom configuration.
    /// </summary>
    /// <param name="config">The verification configuration.</param>
    public GradientVerification(VerificationConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Verifies that a gradient formula matches numerical differentiation.
    /// </summary>
    /// <param name="operation">The IR operation being verified (for metadata purposes only).</param>
    /// <param name="inputs">Input arrays to test.</param>
    /// <param name="gradientFunc">Your gradient formula implementation.</param>
    /// <param name="forwardFunc">Your forward pass implementation.</param>
    /// <returns>Verification result with detailed error information.</returns>
    /// <remarks>
    /// <para>
    /// This method tests whether your provided gradient function produces results
    /// that match numerical differentiation. It does NOT test the actual autodiff
    /// system - use <see cref="AiDotNet.Autodiff.Testing.TensorOperationsVerification{T}"/> for that.
    /// </para>
    /// <para><b>For Beginners:</b> This tests your hand-written gradient formula.
    ///
    /// You provide:
    /// - forwardFunc: How to compute the output from inputs (forward pass)
    /// - gradientFunc: Your gradient formula (backward pass)
    ///
    /// The method will:
    /// 1. Compute gradients using your formula
    /// 2. Compute gradients numerically (the "ground truth")
    /// 3. Compare them and report any differences
    /// </para>
    /// </remarks>
    public VerificationResult VerifyOperation(
        IROp operation,
        T[][] inputs,
        Func<T[][], T[], T[][]> gradientFunc,
        Func<T[][], T[]> forwardFunc)
    {
        var result = new VerificationResult();
        var errors = new List<double>();

        // Compute analytical gradients using provided gradient function
        var outputGrad = CreateOnesLike(forwardFunc(inputs));
        var analyticalGradients = gradientFunc(inputs, outputGrad);

        // Compute numerical gradients for each input
        for (int inputIdx = 0; inputIdx < inputs.Length; inputIdx++)
        {
            var input = inputs[inputIdx];
            var analyticalGrad = analyticalGradients[inputIdx];

            var elementsToCheck = Math.Min(input.Length, _config.MaxElementsToCheck);

            for (int i = 0; i < elementsToCheck; i++)
            {
                // Compute numerical gradient using central differences
                var numericalGrad = ComputeNumericalGradient(inputs, inputIdx, i, forwardFunc);
                var analyticVal = NumOps.ToDouble(analyticalGrad[i]);

                // Compute relative error
                var error = ComputeRelativeError(analyticVal, numericalGrad);
                errors.Add(error);

                if (error > _config.RelativeTolerance &&
                    Math.Abs(analyticVal - numericalGrad) > _config.AbsoluteTolerance)
                {
                    result.FailedElements++;
                    result.Errors.Add(
                        $"Input[{inputIdx}][{i}]: Analytic={analyticVal:E6}, Numeric={numericalGrad:E6}, Error={error:E4}");
                }

                result.TotalElementsChecked++;
            }
        }

        result.MaxRelativeError = errors.Count > 0 ? errors.Max() : 0;
        result.AverageRelativeError = errors.Count > 0 ? errors.Average() : 0;
        result.Passed = result.FailedElements == 0;

        return result;
    }

    /// <summary>
    /// Verifies gradient formulas without requiring an IROp (simplified API).
    /// </summary>
    /// <param name="inputs">Input arrays to test.</param>
    /// <param name="gradientFunc">Your gradient formula implementation.</param>
    /// <param name="forwardFunc">Your forward pass implementation.</param>
    /// <returns>Verification result with detailed error information.</returns>
    /// <remarks>
    /// <para>
    /// This is a simplified version that doesn't require an IROp parameter.
    /// Use this when you just want to verify a gradient formula.
    /// </para>
    /// </remarks>
    public VerificationResult VerifyGradientFormula(
        T[][] inputs,
        Func<T[][], T[], T[][]> gradientFunc,
        Func<T[][], T[]> forwardFunc)
    {
        return VerifyOperation(null!, inputs, gradientFunc, forwardFunc);
    }

    /// <summary>
    /// Computes numerical gradient using central differences.
    /// </summary>
    private double ComputeNumericalGradient(
        T[][] inputs,
        int inputIdx,
        int elementIdx,
        Func<T[][], T[]> forwardFunc)
    {
        var h = NumOps.FromDouble(_config.Epsilon);

        // Save original value
        var originalValue = inputs[inputIdx][elementIdx];

        // f(x + h)
        inputs[inputIdx][elementIdx] = NumOps.Add(originalValue, h);
        var outputPlus = forwardFunc(inputs);
        var fPlus = SumArray(outputPlus);

        // f(x - h)
        inputs[inputIdx][elementIdx] = NumOps.Subtract(originalValue, h);
        var outputMinus = forwardFunc(inputs);
        var fMinus = SumArray(outputMinus);

        // Restore original value
        inputs[inputIdx][elementIdx] = originalValue;

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        return (fPlus - fMinus) / (2 * _config.Epsilon);
    }

    /// <summary>
    /// Computes relative error between two values.
    /// </summary>
    private static double ComputeRelativeError(double analytical, double numerical)
    {
        var maxAbs = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
        if (maxAbs < 1e-10)
            return 0; // Both essentially zero

        return Math.Abs(analytical - numerical) / maxAbs;
    }

    /// <summary>
    /// Sums all elements in an array.
    /// </summary>
    private static double SumArray(T[] array)
    {
        double sum = 0;
        foreach (var value in array)
        {
            sum += NumOps.ToDouble(value);
        }
        return sum;
    }

    /// <summary>
    /// Creates an array of ones with the same length.
    /// </summary>
    private static T[] CreateOnesLike(T[] array)
    {
        var result = new T[array.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.One;
        }
        return result;
    }

    #region Built-in Gradient Formula Verifications

    /// <summary>
    /// Verifies gradient formulas for common operations.
    /// </summary>
    /// <returns>Overall verification result.</returns>
    /// <remarks>
    /// <para>
    /// This method verifies that the built-in gradient formulas for common operations
    /// are mathematically correct. It does NOT test the TensorOperations implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This runs tests on gradient formulas for common operations.
    ///
    /// It tests formulas like:
    /// - ReLU: gradient = 1 if x > 0, else 0
    /// - Sigmoid: gradient = sigmoid(x) * (1 - sigmoid(x))
    /// - Tanh: gradient = 1 - tanh(x)^2
    /// - Add: gradient = 1 for both inputs
    /// - Multiply: gradient = other input
    ///
    /// This verifies the math is correct, not the implementation in TensorOperations.
    /// </para>
    /// </remarks>
    [Obsolete("For testing TensorOperations autodiff, use TensorOperationsVerification<T>.VerifyAllOperations() instead.")]
    public static VerificationResult VerifyAllOperations()
    {
        var verifier = new GradientVerification<T>();
        var overallResult = new VerificationResult { Passed = true };

        // Test ReLU formula
        var reluResult = VerifyReLUFormula(verifier);
        MergeResults(overallResult, reluResult, "ReLU");

        // Test Sigmoid formula
        var sigmoidResult = VerifySigmoidFormula(verifier);
        MergeResults(overallResult, sigmoidResult, "Sigmoid");

        // Test Tanh formula
        var tanhResult = VerifyTanhFormula(verifier);
        MergeResults(overallResult, tanhResult, "Tanh");

        // Test Add formula
        var addResult = VerifyAddFormula(verifier);
        MergeResults(overallResult, addResult, "Add");

        // Test Multiply formula
        var mulResult = VerifyMultiplyFormula(verifier);
        MergeResults(overallResult, mulResult, "Multiply");

        // Test MatMul formula
        var matmulResult = VerifyMatMulFormula(verifier);
        MergeResults(overallResult, matmulResult, "MatMul");

        return overallResult;
    }

    private static void MergeResults(VerificationResult overall, VerificationResult specific, string opName)
    {
        if (!specific.Passed)
        {
            overall.Passed = false;
            overall.Errors.Add($"{opName}: FAILED");
            overall.Errors.AddRange(specific.Errors.Select(e => $"  {e}"));
        }
        else
        {
            overall.Errors.Add($"{opName}: PASSED (MaxError: {specific.MaxRelativeError:E4})");
        }

        overall.MaxRelativeError = Math.Max(overall.MaxRelativeError, specific.MaxRelativeError);
        overall.TotalElementsChecked += specific.TotalElementsChecked;
        overall.FailedElements += specific.FailedElements;
    }

    private static VerificationResult VerifyReLUFormula(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // ReLU gradient formula: gradOut * (input > 0 ? 1 : 0)
                var grad = new T[ins[0].Length];
                for (int i = 0; i < grad.Length; i++)
                {
                    grad[i] = NumOps.GreaterThan(ins[0][i], NumOps.Zero)
                        ? gradOut[i]
                        : NumOps.Zero;
                }
                return new T[][] { grad };
            },
            ins =>
            {
                // ReLU forward: max(0, x)
                var output = new T[ins[0].Length];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.GreaterThan(ins[0][i], NumOps.Zero) ? ins[0][i] : NumOps.Zero;
                }
                return output;
            });
    }

    private static VerificationResult VerifySigmoidFormula(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // Sigmoid gradient formula: gradOut * sigmoid(x) * (1 - sigmoid(x))
                var grad = new T[ins[0].Length];
                for (int i = 0; i < grad.Length; i++)
                {
                    var sig = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(ins[0][i]))));
                    grad[i] = NumOps.Multiply(gradOut[i], NumOps.Multiply(sig, NumOps.Subtract(NumOps.One, sig)));
                }
                return new T[][] { grad };
            },
            ins =>
            {
                // Sigmoid forward: 1 / (1 + exp(-x))
                var output = new T[ins[0].Length];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(ins[0][i]))));
                }
                return output;
            });
    }

    private static VerificationResult VerifyTanhFormula(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // Tanh gradient formula: gradOut * (1 - tanh(x)^2)
                var grad = new T[ins[0].Length];
                for (int i = 0; i < grad.Length; i++)
                {
                    var expX = NumOps.Exp(ins[0][i]);
                    var expNegX = NumOps.Exp(NumOps.Negate(ins[0][i]));
                    var t = NumOps.Divide(NumOps.Subtract(expX, expNegX), NumOps.Add(expX, expNegX));
                    grad[i] = NumOps.Multiply(gradOut[i], NumOps.Subtract(NumOps.One, NumOps.Multiply(t, t)));
                }
                return new T[][] { grad };
            },
            ins =>
            {
                // Tanh forward: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                var output = new T[ins[0].Length];
                for (int i = 0; i < output.Length; i++)
                {
                    var expX = NumOps.Exp(ins[0][i]);
                    var expNegX = NumOps.Exp(NumOps.Negate(ins[0][i]));
                    output[i] = NumOps.Divide(NumOps.Subtract(expX, expNegX), NumOps.Add(expX, expNegX));
                }
                return output;
            });
    }

    private static VerificationResult VerifyAddFormula(GradientVerification<T> verifier)
    {
        var input1 = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5) };
        var input2 = new T[] { NumOps.FromDouble(0.5), NumOps.FromDouble(1.5), NumOps.FromDouble(2.5), NumOps.FromDouble(3.5), NumOps.FromDouble(4.5) };
        var inputs = new T[][] { input1, input2 };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // Add gradient formula: gradOut for both inputs
                return new T[][] { gradOut.ToArray(), gradOut.ToArray() };
            },
            ins =>
            {
                // Add forward: a + b
                var output = new T[ins[0].Length];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.Add(ins[0][i], ins[1][i]);
                }
                return output;
            });
    }

    private static VerificationResult VerifyMultiplyFormula(GradientVerification<T> verifier)
    {
        var input1 = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5) };
        var input2 = new T[] { NumOps.FromDouble(0.5), NumOps.FromDouble(1.5), NumOps.FromDouble(2.5), NumOps.FromDouble(3.5), NumOps.FromDouble(4.5) };
        var inputs = new T[][] { input1, input2 };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // Multiply gradient formula: gradOut * other input
                var grad1 = new T[ins[0].Length];
                var grad2 = new T[ins[0].Length];
                for (int i = 0; i < ins[0].Length; i++)
                {
                    grad1[i] = NumOps.Multiply(gradOut[i], ins[1][i]);
                    grad2[i] = NumOps.Multiply(gradOut[i], ins[0][i]);
                }
                return new T[][] { grad1, grad2 };
            },
            ins =>
            {
                // Multiply forward: a * b
                var output = new T[ins[0].Length];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.Multiply(ins[0][i], ins[1][i]);
                }
                return output;
            });
    }

    private static VerificationResult VerifyMatMulFormula(GradientVerification<T> verifier)
    {
        // 2x3 * 3x2 = 2x2
        var a = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5), NumOps.FromDouble(6) };
        var b = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5), NumOps.FromDouble(6) };
        var inputs = new T[][] { a, b };

        return verifier.VerifyGradientFormula(
            inputs,
            (ins, gradOut) =>
            {
                // MatMul gradient formula:
                // dA = gradOut @ B^T
                // dB = A^T @ gradOut
                int m = 2, k = 3, n = 2;

                var gradA = new T[m * k];
                var gradB = new T[k * n];

                // dA = gradOut @ B^T
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        var sum = NumOps.Zero;
                        for (int l = 0; l < n; l++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(gradOut[i * n + l], ins[1][j * n + l]));
                        }
                        gradA[i * k + j] = sum;
                    }
                }

                // dB = A^T @ gradOut
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        var sum = NumOps.Zero;
                        for (int l = 0; l < m; l++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(ins[0][l * k + i], gradOut[l * n + j]));
                        }
                        gradB[i * n + j] = sum;
                    }
                }

                return new T[][] { gradA, gradB };
            },
            ins =>
            {
                // MatMul forward: A @ B
                int m = 2, k = 3, n = 2;
                var output = new T[m * n];

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        var sum = NumOps.Zero;
                        for (int l = 0; l < k; l++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(ins[0][i * k + l], ins[1][l * n + j]));
                        }
                        output[i * n + j] = sum;
                    }
                }

                return output;
            });
    }

    #endregion
}

/// <summary>
/// Extension methods for gradient verification.
/// </summary>
public static class GradientVerificationExtensions
{
    /// <summary>
    /// Runs gradient verification and prints results to console.
    /// </summary>
    public static void RunAndPrint<T>(this GradientVerification<T>.VerificationResult result)
    {
        Console.WriteLine(result.ToString());
        Console.WriteLine();

        foreach (var error in result.Errors)
        {
            Console.WriteLine(error);
        }
    }
}
