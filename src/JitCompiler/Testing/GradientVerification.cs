using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.JitCompiler.Testing;

/// <summary>
/// Utility for verifying gradient computations using numerical differentiation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gradient verification compares analytically computed gradients (from autodiff)
/// with numerically computed gradients (using finite differences). This is essential
/// for testing the correctness of backward pass implementations.
/// </para>
/// <para><b>For Beginners:</b> This tests that our gradients are computed correctly.
///
/// The idea:
/// 1. Compute gradient using autodiff (our implementation)
/// 2. Compute gradient using finite differences (slow but always correct)
/// 3. Compare them - they should match!
///
/// Finite difference gradient:
/// df/dx ≈ (f(x+h) - f(x-h)) / (2h)  where h is a small number
///
/// If our autodiff gradient matches the numerical gradient, our implementation is correct!
///
/// Example:
/// - f(x) = x^2
/// - Autodiff: df/dx = 2x
/// - Numerical: (f(x+h) - f(x-h)) / (2h) = ((x+h)^2 - (x-h)^2) / (2h) = 2x
/// - They match! Our gradient for x^2 is correct.
/// </para>
/// </remarks>
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
        /// <summary>Step size for finite differences.</summary>
        public double Epsilon { get; set; } = 1e-5;

        /// <summary>Relative tolerance for gradient comparison.</summary>
        public double RelativeTolerance { get; set; } = 1e-4;

        /// <summary>Absolute tolerance for gradient comparison.</summary>
        public double AbsoluteTolerance { get; set; } = 1e-6;

        /// <summary>Maximum number of elements to check (for large tensors).</summary>
        public int MaxElementsToCheck { get; set; } = 1000;

        /// <summary>Whether to print detailed results.</summary>
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

        /// <summary>Detailed error messages.</summary>
        public List<string> Errors { get; set; } = new();

        /// <summary>
        /// Returns a summary string.
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
    public GradientVerification(VerificationConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Verifies gradients for a single operation.
    /// </summary>
    /// <param name="operation">The operation to verify.</param>
    /// <param name="inputs">Input tensors.</param>
    /// <param name="gradientFunc">Function that computes gradients.</param>
    /// <param name="forwardFunc">Function that computes forward pass.</param>
    /// <returns>Verification result.</returns>
    public VerificationResult VerifyOperation(
        IROp operation,
        T[][] inputs,
        Func<T[][], T[], T[][]> gradientFunc,
        Func<T[][], T[]> forwardFunc)
    {
        var result = new VerificationResult();
        var errors = new List<double>();

        // Compute analytical gradients
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
    /// Creates an array of ones with the same shape.
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

    /// <summary>
    /// Verifies gradients for common operations.
    /// </summary>
    public static VerificationResult VerifyAllOperations()
    {
        var verifier = new GradientVerification<T>();
        var overallResult = new VerificationResult { Passed = true };

        // Test ReLU
        var reluResult = VerifyReLU(verifier);
        MergeResults(overallResult, reluResult, "ReLU");

        // Test Sigmoid
        var sigmoidResult = VerifySigmoid(verifier);
        MergeResults(overallResult, sigmoidResult, "Sigmoid");

        // Test Tanh
        var tanhResult = VerifyTanh(verifier);
        MergeResults(overallResult, tanhResult, "Tanh");

        // Test Add
        var addResult = VerifyAdd(verifier);
        MergeResults(overallResult, addResult, "Add");

        // Test Multiply
        var mulResult = VerifyMultiply(verifier);
        MergeResults(overallResult, mulResult, "Multiply");

        // Test MatMul
        var matmulResult = VerifyMatMul(verifier);
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

    private static VerificationResult VerifyReLU(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyOperation(
            new ReLUOp(),
            inputs,
            (ins, gradOut) =>
            {
                // ReLU gradient: gradOut * (input > 0 ? 1 : 0)
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

    private static VerificationResult VerifySigmoid(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyOperation(
            new SigmoidOp(),
            inputs,
            (ins, gradOut) =>
            {
                // Sigmoid gradient: gradOut * sigmoid(x) * (1 - sigmoid(x))
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

    private static VerificationResult VerifyTanh(GradientVerification<T> verifier)
    {
        var input = new T[] { NumOps.FromDouble(-2), NumOps.FromDouble(-1), NumOps.Zero, NumOps.One, NumOps.FromDouble(2) };
        var inputs = new T[][] { input };

        return verifier.VerifyOperation(
            new TanhOp(),
            inputs,
            (ins, gradOut) =>
            {
                // Tanh gradient: gradOut * (1 - tanh(x)^2)
                var grad = new T[ins[0].Length];
                for (int i = 0; i < grad.Length; i++)
                {
                    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
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

    private static VerificationResult VerifyAdd(GradientVerification<T> verifier)
    {
        var input1 = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5) };
        var input2 = new T[] { NumOps.FromDouble(0.5), NumOps.FromDouble(1.5), NumOps.FromDouble(2.5), NumOps.FromDouble(3.5), NumOps.FromDouble(4.5) };
        var inputs = new T[][] { input1, input2 };

        return verifier.VerifyOperation(
            new AddOp(),
            inputs,
            (ins, gradOut) =>
            {
                // Add gradient: gradOut for both inputs
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

    private static VerificationResult VerifyMultiply(GradientVerification<T> verifier)
    {
        var input1 = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5) };
        var input2 = new T[] { NumOps.FromDouble(0.5), NumOps.FromDouble(1.5), NumOps.FromDouble(2.5), NumOps.FromDouble(3.5), NumOps.FromDouble(4.5) };
        var inputs = new T[][] { input1, input2 };

        return verifier.VerifyOperation(
            new ElementwiseMultiplyOp(),
            inputs,
            (ins, gradOut) =>
            {
                // Multiply gradient: gradOut * other input
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

    private static VerificationResult VerifyMatMul(GradientVerification<T> verifier)
    {
        // 2x3 * 3x2 = 2x2
        var a = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5), NumOps.FromDouble(6) };  // 2x3
        var b = new T[] { NumOps.One, NumOps.FromDouble(2), NumOps.FromDouble(3), NumOps.FromDouble(4), NumOps.FromDouble(5), NumOps.FromDouble(6) };  // 3x2
        var inputs = new T[][] { a, b };

        return verifier.VerifyOperation(
            new MatMulOp(),
            inputs,
            (ins, gradOut) =>
            {
                // MatMul gradients:
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
}

/// <summary>
/// Extension methods for gradient verification.
/// </summary>
public static class GradientVerificationExtensions
{
    /// <summary>
    /// Runs gradient verification and prints results.
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
