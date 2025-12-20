using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff.Testing;

/// <summary>
/// Verifies that TensorOperations autodiff gradients match numerical gradients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class provides comprehensive verification of TensorOperations gradient implementations
/// by comparing autodiff results with numerically computed gradients using the central difference method.
/// </para>
/// <para><b>For Beginners:</b> This class tests that our automatic differentiation is correct.
///
/// The process:
/// 1. We have operations like ReLU, Sigmoid, Add, etc. in TensorOperations
/// 2. Each operation computes gradients using autodiff (our fast implementation)
/// 3. We also compute gradients numerically (slow but always correct)
/// 4. If they match, our autodiff is correct!
///
/// This is essential for:
/// - Testing new operations before using them in training
/// - Debugging gradient issues in neural networks
/// - Ensuring mathematical correctness of backward passes
///
/// Example usage:
/// <code>
/// var verifier = new TensorOperationsVerification&lt;float&gt;();
/// var result = verifier.VerifyReLU();
/// Console.WriteLine(result); // "PASSED" or "FAILED" with details
/// </code>
/// </para>
/// </remarks>
public class TensorOperationsVerification<T>
{
    /// <summary>
    /// The numeric operations appropriate for the generic type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly VerificationConfig _config;

    /// <summary>
    /// Configuration for gradient verification.
    /// </summary>
    public class VerificationConfig
    {
        /// <summary>Step size for finite differences (default: 1e-5).</summary>
        public double Epsilon { get; set; } = NumericalGradient<T>.Defaults.Epsilon;

        /// <summary>Relative tolerance for gradient comparison (default: 1e-4).</summary>
        public double RelativeTolerance { get; set; } = NumericalGradient<T>.Defaults.RelativeTolerance;

        /// <summary>Absolute tolerance for gradient comparison (default: 1e-6).</summary>
        public double AbsoluteTolerance { get; set; } = NumericalGradient<T>.Defaults.AbsoluteTolerance;

        /// <summary>Whether to print detailed results (default: false).</summary>
        public bool Verbose { get; set; } = false;

        /// <summary>Random seed for reproducible test data (default: 42).</summary>
        public int RandomSeed { get; set; } = 42;
    }

    /// <summary>
    /// Initializes with default configuration.
    /// </summary>
    public TensorOperationsVerification() : this(new VerificationConfig()) { }

    /// <summary>
    /// Initializes with custom configuration.
    /// </summary>
    /// <param name="config">The verification configuration.</param>
    public TensorOperationsVerification(VerificationConfig config)
    {
        _config = config;
    }

    #region Unary Operation Verification

    /// <summary>
    /// Verifies a unary operation's gradient computation.
    /// </summary>
    /// <param name="operation">The TensorOperations function to verify.</param>
    /// <param name="input">The input tensor.</param>
    /// <param name="operationName">Name of the operation for error messages.</param>
    /// <returns>Verification result with detailed error information.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a complete gradient verification cycle:
    /// 1. Computes forward pass and autodiff gradients using GradientTape
    /// 2. Computes numerical gradients using finite differences
    /// 3. Compares the two and reports any discrepancies
    /// </para>
    /// <para><b>For Beginners:</b> This tests a single operation like ReLU or Sigmoid.
    ///
    /// A unary operation takes one input and produces one output.
    /// We verify that dOutput/dInput is computed correctly by:
    /// - Running autodiff (fast, our implementation)
    /// - Running numerical differentiation (slow, ground truth)
    /// - Checking they match
    /// </para>
    /// </remarks>
    public NumericalGradient<T>.ComparisonResult VerifyUnaryOperation(
        Func<ComputationNode<T>, ComputationNode<T>> operation,
        Tensor<T> input,
        string operationName)
    {
        // Compute autodiff gradient
        Tensor<T> autodiffGradient;
        using (var tape = new GradientTape<T>())
        {
            var inputNode = TensorOperations<T>.Variable(input.Clone(), "input", requiresGradient: true);
            tape.Watch(inputNode);

            var outputNode = operation(inputNode);

            // Create output gradient (ones)
            var outputGradient = CreateOnes(outputNode.Value.Shape);
            outputNode.Gradient = outputGradient;

            // Run backward pass
            RunBackward(outputNode);

            autodiffGradient = inputNode.Gradient ?? new Tensor<T>(input.Shape);
        }

        // Compute numerical gradient
        var outputGrad = CreateOnes(input.Shape);
        var numericalGradient = NumericalGradient<T>.ComputeForOperation(
            input.Clone(),
            outputGrad,
            operation,
            _config.Epsilon);

        // Compare
        var result = NumericalGradient<T>.Compare(
            numericalGradient,
            autodiffGradient,
            _config.RelativeTolerance,
            _config.AbsoluteTolerance);

        if (_config.Verbose)
        {
            Console.WriteLine($"{operationName}: {result}");
            foreach (var error in result.Errors.Take(5))
            {
                Console.WriteLine($"  {error}");
            }
        }

        return result;
    }

    /// <summary>
    /// Verifies a binary operation's gradient computation.
    /// </summary>
    /// <param name="operation">The TensorOperations function to verify.</param>
    /// <param name="input1">The first input tensor.</param>
    /// <param name="input2">The second input tensor.</param>
    /// <param name="operationName">Name of the operation for error messages.</param>
    /// <returns>A tuple of verification results for both inputs.</returns>
    /// <remarks>
    /// <para>
    /// Binary operations like Add, Multiply, MatMul have two inputs, so we verify
    /// gradients for both inputs separately.
    /// </para>
    /// <para><b>For Beginners:</b> This tests operations with two inputs like a + b or a * b.
    ///
    /// For c = f(a, b), we need to verify both:
    /// - dc/da: How does c change when a changes?
    /// - dc/db: How does c change when b changes?
    /// </para>
    /// </remarks>
    public (NumericalGradient<T>.ComparisonResult grad1Result, NumericalGradient<T>.ComparisonResult grad2Result)
        VerifyBinaryOperation(
            Func<ComputationNode<T>, ComputationNode<T>, ComputationNode<T>> operation,
            Tensor<T> input1,
            Tensor<T> input2,
            string operationName)
    {
        // Compute autodiff gradients
        Tensor<T> autodiffGrad1, autodiffGrad2;
        using (var tape = new GradientTape<T>())
        {
            var node1 = TensorOperations<T>.Variable(input1.Clone(), "input1", requiresGradient: true);
            var node2 = TensorOperations<T>.Variable(input2.Clone(), "input2", requiresGradient: true);
            tape.Watch(node1);
            tape.Watch(node2);

            var outputNode = operation(node1, node2);

            // Create output gradient (ones)
            var outputGradient = CreateOnes(outputNode.Value.Shape);
            outputNode.Gradient = outputGradient;

            // Run backward pass
            RunBackward(outputNode);

            autodiffGrad1 = node1.Gradient ?? new Tensor<T>(input1.Shape);
            autodiffGrad2 = node2.Gradient ?? new Tensor<T>(input2.Shape);
        }

        // Compute numerical gradients
        var outputGrad = CreateOnes(input1.Shape);
        var (numericalGrad1, numericalGrad2) = NumericalGradient<T>.ComputeForBinaryOperation(
            input1.Clone(),
            input2.Clone(),
            outputGrad,
            operation,
            _config.Epsilon);

        // Compare
        var result1 = NumericalGradient<T>.Compare(
            numericalGrad1,
            autodiffGrad1,
            _config.RelativeTolerance,
            _config.AbsoluteTolerance);

        var result2 = NumericalGradient<T>.Compare(
            numericalGrad2,
            autodiffGrad2,
            _config.RelativeTolerance,
            _config.AbsoluteTolerance);

        if (_config.Verbose)
        {
            Console.WriteLine($"{operationName} (input1): {result1}");
            Console.WriteLine($"{operationName} (input2): {result2}");
        }

        return (result1, result2);
    }

    #endregion

    #region Specific Operation Verifications

    /// <summary>
    /// Verifies ReLU operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyReLU(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(TensorOperations<T>.ReLU, input, "ReLU");
    }

    /// <summary>
    /// Verifies Sigmoid operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifySigmoid(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -3.0, 3.0);
        return VerifyUnaryOperation(TensorOperations<T>.Sigmoid, input, "Sigmoid");
    }

    /// <summary>
    /// Verifies Tanh operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyTanh(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(TensorOperations<T>.Tanh, input, "Tanh");
    }

    /// <summary>
    /// Verifies Negate operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyNegate(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(TensorOperations<T>.Negate, input, "Negate");
    }

    /// <summary>
    /// Verifies Exp operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyExp(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        // Use smaller range to avoid overflow
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(TensorOperations<T>.Exp, input, "Exp");
    }

    /// <summary>
    /// Verifies Log operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyLog(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        // Use positive values only for log
        var input = CreateTestTensor(inputShape, 0.1, 5.0);
        return VerifyUnaryOperation(TensorOperations<T>.Log, input, "Log");
    }

    /// <summary>
    /// Verifies Sqrt operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifySqrt(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        // Use positive values only for sqrt
        var input = CreateTestTensor(inputShape, 0.1, 5.0);
        return VerifyUnaryOperation(TensorOperations<T>.Sqrt, input, "Sqrt");
    }

    /// <summary>
    /// Verifies Square operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifySquare(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(TensorOperations<T>.Square, input, "Square");
    }

    /// <summary>
    /// Verifies LeakyReLU operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensor (default: [5]).</param>
    /// <param name="alpha">Negative slope coefficient.</param>
    /// <returns>Verification result.</returns>
    public NumericalGradient<T>.ComparisonResult VerifyLeakyReLU(int[]? inputShape = null, double alpha = 0.01)
    {
        inputShape ??= new[] { 5 };
        var input = CreateTestTensor(inputShape, -2.0, 2.0);
        return VerifyUnaryOperation(
            node => TensorOperations<T>.LeakyReLU(node, alpha),
            input,
            $"LeakyReLU(alpha={alpha})");
    }

    /// <summary>
    /// Verifies Add operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensors (default: [5]).</param>
    /// <returns>Verification results for both inputs.</returns>
    public (NumericalGradient<T>.ComparisonResult, NumericalGradient<T>.ComparisonResult) VerifyAdd(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input1 = CreateTestTensor(inputShape, -2.0, 2.0);
        var input2 = CreateTestTensor(inputShape, -2.0, 2.0, seedOffset: 100);
        return VerifyBinaryOperation(TensorOperations<T>.Add, input1, input2, "Add");
    }

    /// <summary>
    /// Verifies Subtract operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensors (default: [5]).</param>
    /// <returns>Verification results for both inputs.</returns>
    public (NumericalGradient<T>.ComparisonResult, NumericalGradient<T>.ComparisonResult) VerifySubtract(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input1 = CreateTestTensor(inputShape, -2.0, 2.0);
        var input2 = CreateTestTensor(inputShape, -2.0, 2.0, seedOffset: 100);
        return VerifyBinaryOperation(TensorOperations<T>.Subtract, input1, input2, "Subtract");
    }

    /// <summary>
    /// Verifies ElementwiseMultiply operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensors (default: [5]).</param>
    /// <returns>Verification results for both inputs.</returns>
    public (NumericalGradient<T>.ComparisonResult, NumericalGradient<T>.ComparisonResult) VerifyElementwiseMultiply(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input1 = CreateTestTensor(inputShape, -2.0, 2.0);
        var input2 = CreateTestTensor(inputShape, -2.0, 2.0, seedOffset: 100);
        return VerifyBinaryOperation(TensorOperations<T>.ElementwiseMultiply, input1, input2, "ElementwiseMultiply");
    }

    /// <summary>
    /// Verifies ElementwiseDivide operation gradients.
    /// </summary>
    /// <param name="inputShape">Shape of test tensors (default: [5]).</param>
    /// <returns>Verification results for both inputs.</returns>
    public (NumericalGradient<T>.ComparisonResult, NumericalGradient<T>.ComparisonResult) VerifyElementwiseDivide(int[]? inputShape = null)
    {
        inputShape ??= new[] { 5 };
        var input1 = CreateTestTensor(inputShape, -2.0, 2.0);
        // Avoid division by zero with values away from zero
        var input2 = CreateTestTensor(inputShape, 0.5, 2.0, seedOffset: 100);
        return VerifyBinaryOperation(TensorOperations<T>.Divide, input1, input2, "ElementwiseDivide");
    }

    #endregion

    #region Comprehensive Verification

    /// <summary>
    /// Runs verification for all standard operations.
    /// </summary>
    /// <returns>A summary result containing all operation results.</returns>
    /// <remarks>
    /// <para>
    /// This method verifies gradients for a comprehensive set of TensorOperations:
    /// - Activation functions: ReLU, Sigmoid, Tanh, LeakyReLU
    /// - Arithmetic: Add, Subtract, Multiply, Divide
    /// - Math functions: Exp, Log, Sqrt, Square, Negate
    /// </para>
    /// <para><b>For Beginners:</b> This runs all gradient tests at once.
    ///
    /// Use this to verify the entire autodiff system is working correctly.
    /// Each operation is tested individually, and a summary report is generated.
    /// </para>
    /// </remarks>
    public VerificationSummary VerifyAllOperations()
    {
        var summary = new VerificationSummary();

        // Unary operations
        AddResult(summary, "ReLU", VerifyReLU());
        AddResult(summary, "Sigmoid", VerifySigmoid());
        AddResult(summary, "Tanh", VerifyTanh());
        AddResult(summary, "Negate", VerifyNegate());
        AddResult(summary, "Exp", VerifyExp());
        AddResult(summary, "Log", VerifyLog());
        AddResult(summary, "Sqrt", VerifySqrt());
        AddResult(summary, "Square", VerifySquare());
        AddResult(summary, "LeakyReLU", VerifyLeakyReLU());

        // Binary operations
        var (addResult1, addResult2) = VerifyAdd();
        AddResult(summary, "Add (input1)", addResult1);
        AddResult(summary, "Add (input2)", addResult2);

        var (subResult1, subResult2) = VerifySubtract();
        AddResult(summary, "Subtract (input1)", subResult1);
        AddResult(summary, "Subtract (input2)", subResult2);

        var (mulResult1, mulResult2) = VerifyElementwiseMultiply();
        AddResult(summary, "Multiply (input1)", mulResult1);
        AddResult(summary, "Multiply (input2)", mulResult2);

        var (divResult1, divResult2) = VerifyElementwiseDivide();
        AddResult(summary, "Divide (input1)", divResult1);
        AddResult(summary, "Divide (input2)", divResult2);

        return summary;
    }

    private static void AddResult(VerificationSummary summary, string name, NumericalGradient<T>.ComparisonResult result)
    {
        summary.Results[name] = result;
        if (!result.Passed)
        {
            summary.FailedOperations.Add(name);
        }
        summary.MaxRelativeError = Math.Max(summary.MaxRelativeError, result.MaxRelativeError);
        summary.TotalElementsChecked += result.TotalElementsChecked;
        summary.TotalFailedElements += result.FailedElements;
    }

    /// <summary>
    /// Summary of verification results for all operations.
    /// </summary>
    public class VerificationSummary
    {
        /// <summary>Individual results for each operation.</summary>
        public Dictionary<string, NumericalGradient<T>.ComparisonResult> Results { get; } = new();

        /// <summary>List of operations that failed verification.</summary>
        public List<string> FailedOperations { get; } = new();

        /// <summary>Maximum relative error across all operations.</summary>
        public double MaxRelativeError { get; set; }

        /// <summary>Total elements checked across all operations.</summary>
        public int TotalElementsChecked { get; set; }

        /// <summary>Total failed elements across all operations.</summary>
        public int TotalFailedElements { get; set; }

        /// <summary>Whether all operations passed.</summary>
        public bool AllPassed => FailedOperations.Count == 0;

        /// <summary>
        /// Returns a detailed summary string.
        /// </summary>
        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== TensorOperations Gradient Verification ===");
            sb.AppendLine($"Overall: {(AllPassed ? "ALL PASSED" : "SOME FAILED")}");
            sb.AppendLine($"Max Relative Error: {MaxRelativeError:E4}");
            sb.AppendLine($"Total Elements: {TotalElementsChecked}, Failed: {TotalFailedElements}");
            sb.AppendLine();

            foreach (var (name, result) in Results.OrderBy(r => r.Value.Passed ? 0 : 1))
            {
                var status = result.Passed ? "PASS" : "FAIL";
                sb.AppendLine($"  {status}: {name} (MaxError: {result.MaxRelativeError:E4})");
            }

            if (FailedOperations.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("Failed operations:");
                foreach (var op in FailedOperations)
                {
                    sb.AppendLine($"  - {op}");
                }
            }

            return sb.ToString();
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    private static Tensor<T> CreateOnes(int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.One;
        }
        return tensor;
    }

    /// <summary>
    /// Creates a test tensor with random values in the specified range.
    /// </summary>
    private Tensor<T> CreateTestTensor(int[] shape, double min, double max, int seedOffset = 0)
    {
        var tensor = new Tensor<T>(shape);
        var random = RandomHelper.CreateSeededRandom(_config.RandomSeed + seedOffset);
        var range = max - min;

        for (int i = 0; i < tensor.Length; i++)
        {
            var value = min + random.NextDouble() * range;
            tensor[i] = NumOps.FromDouble(value);
        }

        return tensor;
    }

    /// <summary>
    /// Runs the backward pass for a computation graph starting from the given node.
    /// </summary>
    private static void RunBackward(ComputationNode<T> root)
    {
        var topoOrder = GetTopologicalOrder(root);
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }
    }

    /// <summary>
    /// Gets topological order for gradient computation.
    /// </summary>
    private static List<ComputationNode<T>> GetTopologicalOrder(ComputationNode<T> root)
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                        stack.Push((parent, false));
                }
            }
        }

        return result;
    }

    #endregion
}

/// <summary>
/// Extension methods for TensorOperationsVerification.
/// </summary>
public static class TensorOperationsVerificationExtensions
{
    /// <summary>
    /// Runs verification and prints the summary to console.
    /// </summary>
    public static void RunAndPrint<T>(this TensorOperationsVerification<T>.VerificationSummary summary)
    {
        Console.WriteLine(summary.ToString());
    }
}
