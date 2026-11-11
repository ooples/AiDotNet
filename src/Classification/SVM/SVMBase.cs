using AiDotNet.Classification;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.SVM;

/// <summary>
/// Base class for Support Vector Machine classifiers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Support Vector Machines (SVMs) are powerful classifiers that find the optimal
/// hyperplane separating classes with maximum margin. This base class provides
/// common functionality for SVM implementations.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// SVMs are like finding the best possible line (or curve) to separate different groups.
/// Unlike other methods that just find "a" line that works, SVMs find "the best" line
/// by maximizing the gap (margin) between the line and the nearest points from each class.
///
/// Key SVM concepts:
/// - Margin: The gap between the decision boundary and the nearest training points
/// - Support Vectors: The training points closest to the decision boundary
/// - Kernel Trick: A way to handle non-linear boundaries without explicitly computing new features
/// </para>
/// </remarks>
public abstract partial class SVMBase<T> : ProbabilisticClassifierBase<T>, IDecisionFunctionClassifier<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <summary>
    /// Gets the SVM specific options.
    /// </summary>
    protected new SVMOptions<T> Options => (SVMOptions<T>)base.Options;

    /// <summary>
    /// The support vectors learned during training.
    /// </summary>
    protected Matrix<T>? _supportVectors;

    /// <summary>
    /// The dual coefficients for the support vectors.
    /// </summary>
    protected Matrix<T>? _dualCoef;

    /// <summary>
    /// The bias terms for each classifier.
    /// </summary>
    [FittedParameter]
    protected Vector<T>? _intercept;

    /// <inheritdoc/>
    public Matrix<T>? SupportVectors => _supportVectors;

    /// <inheritdoc/>
    public int NSupportVectors => _supportVectors?.Rows ?? 0;

    /// <summary>
    /// Initializes a new instance of the SVMBase class.
    /// </summary>
    /// <param name="options">Configuration options for the SVM.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    protected SVMBase(SVMOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new SVMOptions<T>(), regularization, new HingeLoss<T>())
    {
    }

    /// <inheritdoc/>
    public abstract Matrix<T> DecisionFunction(Matrix<T> input);

    /// <summary>
    /// Computes the kernel between two vectors.
    /// </summary>
    /// <param name="x">First vector.</param>
    /// <param name="y">Second vector.</param>
    /// <returns>The kernel value K(x, y).</returns>
    protected T ComputeKernel(Vector<T> x, Vector<T> y)
    {
        return Options.Kernel switch
        {
            KernelType.Linear => ComputeLinearKernel(x, y),
            KernelType.Polynomial => ComputePolynomialKernel(x, y),
            KernelType.RBF => ComputeRBFKernel(x, y),
            KernelType.Sigmoid => ComputeSigmoidKernel(x, y),
            KernelType.Laplacian => ComputeLaplacianKernel(x, y),
            _ => ComputeLinearKernel(x, y)
        };
    }

    /// <summary>
    /// Computes linear kernel: K(x, y) = x · y
    /// </summary>
    protected T ComputeLinearKernel(Vector<T> x, Vector<T> y)
    {
        // Use Engine for hardware-accelerated dot product (SIMD/vectorized)
        return Engine.DotProduct(x, y);
    }

    /// <summary>
    /// Computes polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
    /// </summary>
    protected T ComputePolynomialKernel(Vector<T> x, Vector<T> y)
    {
        T dot = ComputeLinearKernel(x, y);
        T gamma = GetGamma();
        T result = NumOps.Add(
            NumOps.Multiply(gamma, dot),
            NumOps.FromDouble(Options.Coef0)
        );
        return NumOps.Power(result, NumOps.FromDouble(Options.Degree));
    }

    /// <summary>
    /// Computes RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    /// </summary>
    protected T ComputeRBFKernel(Vector<T> x, Vector<T> y)
    {
        // Hot path inside SMO training: each sample's ComputeError calls
        // ComputeDecision, which loops the full training set evaluating this
        // kernel — at default opts that's tens of millions of element-level
        // hits per Train(). Profiling on PR #1184 showed every Vector<T>[i]
        // here acquired the deferred-materializer monitor, putting 99% of
        // SVC train wall-clock in Monitor.Enter_Slowpath. Materialising the
        // backing arrays once via .ToArray() and then indexing them keeps
        // the inner loop on plain memory access.
        var xArr = x.ToArray();
        var yArr = y.ToArray();
        return ComputeRBFKernelArrays(xArr, yArr);
    }

    /// <summary>
    /// Array-typed RBF kernel for callers that already have backing arrays
    /// (e.g. <see cref="SupportVectorClassifier{T}"/>'s SMO loop, which
    /// pre-extracts <c>_xTrain</c> rows once and reuses them across every
    /// inner iteration).
    /// </summary>
    protected T ComputeRBFKernelArrays(T[] x, T[] y)
    {
        // exp(-gamma * ||x - y||^2)
        var diff = Engine.Subtract(Vector<T>.Wrap(x), Vector<T>.Wrap(y));
        T squaredNorm = Engine.DotProduct(diff, diff);
        T gamma = GetGamma();
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, squaredNorm)));
    }

    /// <summary>
    /// The [a.Rows, b.Rows] kernel matrix K[i, j] = kernel(a_i, b_j) for the configured kernel.
    /// </summary>
    /// <param name="a">The first set of samples (rows).</param>
    /// <param name="b">The second set of samples (rows).</param>
    /// <returns>The kernel matrix.</returns>
    private protected Tensor<T> ComputeKernelMatrix(Matrix<T> a, Matrix<T> b)
    {
        var engine = Engine;
        int n = a.Rows;
        int m = b.Rows;
        if (n == 0 || m == 0)
        {
            return new Tensor<T>(new[] { n, m });
        }

        var left = Tensor<T>.FromMatrix(a);
        var right = Tensor<T>.FromMatrix(b);
        T gamma = GetGamma();
        T coef0 = NumOps.FromDouble(Options.Coef0);

        switch (Options.Kernel)
        {
            case KernelType.Polynomial:
                return engine.TensorPower(
                    engine.TensorAddScalar(engine.TensorMultiplyScalar(LinearKernelMatrix(left, right), gamma), coef0),
                    NumOps.FromDouble(Options.Degree));
            case KernelType.RBF:
                return engine.TensorExp(engine.TensorMultiplyScalar(PairwiseReduction(left, right, squared: true), NumOps.Negate(gamma)));
            case KernelType.Sigmoid:
            {
                // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                var result = engine.TensorAddScalar(engine.TensorMultiplyScalar(LinearKernelMatrix(left, right), gamma), coef0);
                var exp2x = engine.TensorExp(engine.TensorMultiplyScalar(result, NumOps.FromDouble(2.0)));
                return engine.TensorDivide(engine.TensorSubtractScalar(exp2x, NumOps.One), engine.TensorAddScalar(exp2x, NumOps.One));
            }
            case KernelType.Laplacian:
                return engine.TensorExp(engine.TensorMultiplyScalar(PairwiseReduction(left, right, squared: false), NumOps.Negate(gamma)));
            default:
                return LinearKernelMatrix(left, right);
        }
    }

    /// <summary>
    /// a * b^T.
    /// </summary>
    private Tensor<T> LinearKernelMatrix(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorMatMul(a, Engine.TensorTranspose(b));
    }

    /// <summary>
    /// sum_k (a_ik - b_jk)^2 (or sum_k |a_ik - b_jk|) for every pair, computed exactly in row blocks.
    /// </summary>
    private Tensor<T> PairwiseReduction(Tensor<T> a, Tensor<T> b, bool squared)
    {
        var engine = Engine;
        int n = a.Shape[0];
        int m = b.Shape[0];
        int d = a.Shape[1];
        var right = b.Reshape(1, m, d);
        int batch = Math.Max(1, (1 << 22) / Math.Max(1, m * Math.Max(1, d)));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < n; start += batch)
        {
            int rows = Math.Min(batch, n - start);
            var diff = engine.TensorSubtract(a.Slice(0, start, start + rows).Reshape(rows, 1, d), right);
            var terms = squared ? engine.TensorMultiply(diff, diff) : engine.TensorAbs(diff);
            blocks.Add(engine.ReduceSum(terms, new[] { 2 }, false));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <summary>
    /// Computes sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0)
    /// </summary>
    protected T ComputeSigmoidKernel(Vector<T> x, Vector<T> y)
    {
        T dot = ComputeLinearKernel(x, y);
        T gamma = GetGamma();
        T result = NumOps.Add(
            NumOps.Multiply(gamma, dot),
            NumOps.FromDouble(Options.Coef0)
        );
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        T exp2x = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(2.0), result));
        return NumOps.Divide(
            NumOps.Subtract(exp2x, NumOps.One),
            NumOps.Add(exp2x, NumOps.One)
        );
    }

    /// <summary>
    /// Computes Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
    /// </summary>
    protected T ComputeLaplacianKernel(Vector<T> x, Vector<T> y)
    {
        // exp(-gamma * ||x - y||_1)
        T manhattanNorm = x.Length == 0 ? NumOps.Zero : Engine.TensorSum(Engine.TensorAbs(Engine.Subtract(x, y).AsTensor()));
        T gamma = GetGamma();
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, manhattanNorm)));
    }

    /// <summary>
    /// +1 for the positive class (the last class label) and -1 for every other label.
    /// </summary>
    private protected Vector<T> SignedTargets(Vector<T> y)
    {
        if (ClassLabels is null || y.Length == 0)
        {
            return new Vector<T>(y.Length);
        }

        var isPositive = Engine.TensorEquals(y.AsTensor(), ClassLabels[ClassLabels.Length - 1]);
        return Engine.TensorSubtractScalar(Engine.TensorMultiplyScalar(isPositive, NumOps.FromDouble(2.0)), NumOps.One).ToVector();
    }

    /// <summary>
    /// Indices i with values[i] &gt; threshold, in increasing order.
    /// </summary>
    private protected int[] IndicesAbove(Vector<T> values, T threshold)
    {
        if (values.Length == 0)
        {
            return Array.Empty<int>();
        }

        var positions = Engine.TensorNonzero(Engine.TensorGreaterThan(values.AsTensor(), threshold));
        return positions.Length == 0 ? Array.Empty<int>() : positions.Reshape(positions.Length).ToArray();
    }

    /// <summary>
    /// bias + sum_i coefficients[i] * kernel(input_row, train_i) for every input row, as an [input.Rows, 1] matrix.
    /// </summary>
    private protected Matrix<T> KernelDecisions(Matrix<T> input, Matrix<T> train, Vector<T> coefficients, T bias)
    {
        var engine = Engine;
        if (input.Rows == 0)
        {
            return new Matrix<T>(0, 1);
        }

        if (train.Rows == 0)
        {
            var constant = new Tensor<T>(new[] { input.Rows, 1 });
            engine.TensorFill(constant, bias);
            return constant.ToMatrix();
        }

        var kernel = ComputeKernelMatrix(input, train);
        var sums = engine.TensorMatMul(kernel, coefficients.AsTensor().Reshape(coefficients.Length, 1));
        return engine.TensorAddScalar(sums, bias).ToMatrix();
    }

    /// <summary>
    /// Gets the gamma value, computing it automatically if not specified.
    /// </summary>
    protected T GetGamma()
    {
        if (Options.Gamma.HasValue)
        {
            return NumOps.FromDouble(Options.Gamma.Value);
        }

        // Default: 1 / n_features (scale)
        return NumOps.FromDouble(1.0 / NumFeatures);
    }

    /// <summary>
    /// Extracts a row from a matrix as a vector.
    /// </summary>
    protected Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        return matrix.GetRow(row);
    }

    /// <summary>
    /// Same as <see cref="GetRow"/> but returns a raw array — avoids the
    /// <see cref="Vector{T}"/> indexer's deferred-materializer monitor on hot
    /// paths. Callers that loop over the full training matrix should pre-cache
    /// these arrays ONCE rather than re-extracting per iteration.
    /// </summary>
    protected T[] GetRowArray(Matrix<T> matrix, int row)
    {
        return matrix.GetRow(row).ToArray();
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = Clone();
        if (newModel is SVMBase<T> svm)
        {
            svm.SetParameters(parameters);
        }
        return newModel;
    }



    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Kernel"] = Options.Kernel.ToString();
        metadata.AdditionalInfo["C"] = Options.C;
        metadata.AdditionalInfo["Gamma"] = Options.Gamma?.ToString() ?? "auto";
        metadata.AdditionalInfo["NSupportVectors"] = NSupportVectors;
        return metadata;
    }
}
