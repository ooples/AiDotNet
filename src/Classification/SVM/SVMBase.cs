using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

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
public abstract class SVMBase<T> : ProbabilisticClassifierBase<T>, IDecisionFunctionClassifier<T>
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
        T dot = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(x[i], y[i]));
        }
        return dot;
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
        T squaredNorm = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = NumOps.Subtract(x[i], y[i]);
            squaredNorm = NumOps.Add(squaredNorm, NumOps.Multiply(diff, diff));
        }
        T gamma = GetGamma();
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, squaredNorm)));
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
        T manhattanNorm = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = NumOps.Subtract(x[i], y[i]);
            manhattanNorm = NumOps.Add(manhattanNorm, NumOps.Abs(diff));
        }
        T gamma = GetGamma();
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, manhattanNorm)));
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
        var vector = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            vector[j] = matrix[row, j];
        }
        return vector;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Return intercepts as parameters
        return _intercept ?? new Vector<T>(0);
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
    public override void SetParameters(Vector<T> parameters)
    {
        if (_intercept != null && parameters.Length == _intercept.Length)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                _intercept[i] = parameters[i];
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // SVMs don't use gradient-based optimization in the typical sense
        // They use quadratic programming
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // SVMs don't use gradient-based optimization
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
