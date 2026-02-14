namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Interface for mean functions in Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A mean function defines the "expected" value of the GP at any point
/// before we observe any data. It represents our prior belief about the function's behavior.
///
/// Common choices:
/// - Zero mean: We expect the function to hover around zero
/// - Constant mean: We expect the function to hover around some constant value
/// - Linear mean: We expect a linear trend in the data
///
/// The GP then models deviations from this mean using the kernel function.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MeanFunction")]
public interface IMeanFunction<T>
{
    /// <summary>
    /// Computes the mean value at a given input point.
    /// </summary>
    /// <param name="x">The input point.</param>
    /// <returns>The mean value at x.</returns>
    T Evaluate(Vector<T> x);

    /// <summary>
    /// Computes the mean values at multiple input points.
    /// </summary>
    /// <param name="X">Matrix of input points (one per row).</param>
    /// <returns>Vector of mean values.</returns>
    Vector<T> Evaluate(Matrix<T> X);
}

/// <summary>
/// Implements a zero mean function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The zero mean function returns zero for all inputs.
/// This is the most common default choice for GPs.
///
/// m(x) = 0
///
/// Why use zero mean?
/// - Simplicity: No parameters to tune
/// - Flexibility: The GP can still learn any function through the kernel
/// - Data centering: Often we center our data to have zero mean anyway
///
/// When zero mean makes sense:
/// - Centered/normalized data
/// - When you want the GP to be flexible
/// - When you don't have strong prior beliefs about the trend
/// </para>
/// </remarks>
public class ZeroMean<T> : IMeanFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new zero mean function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a mean function that always returns zero.
    /// The simplest and most commonly used mean function.
    /// </para>
    /// </remarks>
    public ZeroMean()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Returns zero for any input point.
    /// </summary>
    /// <param name="x">The input point (ignored).</param>
    /// <returns>Zero.</returns>
    public T Evaluate(Vector<T> x) => _numOps.Zero;

    /// <summary>
    /// Returns a vector of zeros for all input points.
    /// </summary>
    /// <param name="X">Matrix of input points.</param>
    /// <returns>Vector of zeros with length equal to number of rows in X.</returns>
    public Vector<T> Evaluate(Matrix<T> X)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        return new Vector<T>(X.Rows);
    }
}

/// <summary>
/// Implements a constant mean function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The constant mean function returns the same value for all inputs.
///
/// m(x) = c
///
/// Why use constant mean?
/// - When your data has a known average value
/// - When you expect predictions far from training data to revert to this constant
/// - As a simple baseline trend
///
/// The constant can be:
/// - Set manually based on domain knowledge
/// - Learned during hyperparameter optimization
/// - Set to the empirical mean of your training targets
///
/// Example: If predicting house prices in thousands, you might set c = 300
/// to represent your prior belief that houses cost around $300k on average.
/// </para>
/// </remarks>
public class ConstantMean<T> : IMeanFunction<T>
{
    private readonly T _constant;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new constant mean function.
    /// </summary>
    /// <param name="constant">The constant mean value.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a mean function that always returns the specified constant.
    ///
    /// Tips for choosing the constant:
    /// - Use the mean of your training targets as a starting point
    /// - Adjust based on domain knowledge
    /// - Can be optimized during hyperparameter tuning
    ///
    /// Example:
    /// var meanFunc = new ConstantMean&lt;double&gt;(300.0);
    /// </para>
    /// </remarks>
    public ConstantMean(T constant)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _constant = constant;
    }

    /// <summary>
    /// Initializes a constant mean with a double value.
    /// </summary>
    /// <param name="constant">The constant as a double.</param>
    public ConstantMean(double constant)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _constant = _numOps.FromDouble(constant);
    }

    /// <summary>
    /// Gets the constant value.
    /// </summary>
    public T Constant => _constant;

    /// <summary>
    /// Returns the constant for any input point.
    /// </summary>
    /// <param name="x">The input point (ignored).</param>
    /// <returns>The constant value.</returns>
    public T Evaluate(Vector<T> x) => _constant;

    /// <summary>
    /// Returns a vector of the constant for all input points.
    /// </summary>
    /// <param name="X">Matrix of input points.</param>
    /// <returns>Vector filled with the constant.</returns>
    public Vector<T> Evaluate(Matrix<T> X)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));

        var result = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            result[i] = _constant;
        }
        return result;
    }
}

/// <summary>
/// Implements a linear mean function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The linear mean function returns a linear combination of input features.
///
/// m(x) = w^T × x + b = w_1×x_1 + w_2×x_2 + ... + w_d×x_d + b
///
/// Where:
/// - w is the weight vector (one weight per input dimension)
/// - b is the bias (intercept)
/// - x is the input point
///
/// Why use linear mean?
/// - When you expect a linear trend in your data
/// - Combines well with kernels that capture nonlinear deviations
/// - Can be initialized from linear regression on your data
///
/// Example use cases:
/// - Time series with a linear trend
/// - Data where features have known linear effects
/// - When you want to separate trend from residuals
///
/// The GP will model deviations from this linear trend using the kernel.
/// </para>
/// </remarks>
public class LinearMean<T> : IMeanFunction<T>
{
    private readonly Vector<T> _weights;
    private readonly T _bias;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new linear mean function with specified weights and bias.
    /// </summary>
    /// <param name="weights">The weight vector (one per input dimension).</param>
    /// <param name="bias">The bias term. Default is 0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a linear mean function m(x) = w^T × x + b.
    ///
    /// Example for 2D input with weights [1.5, -0.5] and bias 10:
    /// m([x1, x2]) = 1.5×x1 - 0.5×x2 + 10
    ///
    /// If you have training data, you can initialize weights from:
    /// 1. Linear regression on (X, y)
    /// 2. Domain knowledge about feature effects
    /// 3. Random initialization for later optimization
    /// </para>
    /// </remarks>
    public LinearMean(Vector<T> weights, T? bias = default)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));

        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = new Vector<T>(weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            _weights[i] = weights[i];
        }

        _bias = bias is not null ? bias : _numOps.Zero;
    }

    /// <summary>
    /// Initializes a linear mean function with double arrays.
    /// </summary>
    /// <param name="weights">The weights as doubles.</param>
    /// <param name="bias">The bias as a double. Default is 0.</param>
    public LinearMean(double[] weights, double bias = 0.0)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (weights.Length == 0)
            throw new ArgumentException("Weights array cannot be empty.", nameof(weights));

        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = new Vector<T>(weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            _weights[i] = _numOps.FromDouble(weights[i]);
        }

        _bias = _numOps.FromDouble(bias);
    }

    /// <summary>
    /// Initializes a zero-weight linear mean for the given number of dimensions.
    /// </summary>
    /// <param name="numDimensions">The number of input dimensions.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a linear mean with all weights initialized to zero.
    /// This is equivalent to a zero mean but can be optimized later.
    /// </para>
    /// </remarks>
    public LinearMean(int numDimensions)
    {
        if (numDimensions < 1)
            throw new ArgumentException("Must have at least one dimension.", nameof(numDimensions));

        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = new Vector<T>(numDimensions);
        _bias = _numOps.Zero;
    }

    /// <summary>
    /// Gets the weight vector.
    /// </summary>
    public Vector<T> Weights
    {
        get
        {
            var copy = new Vector<T>(_weights.Length);
            for (int i = 0; i < _weights.Length; i++)
            {
                copy[i] = _weights[i];
            }
            return copy;
        }
    }

    /// <summary>
    /// Gets the bias term.
    /// </summary>
    public T Bias => _bias;

    /// <summary>
    /// Gets the number of dimensions.
    /// </summary>
    public int NumDimensions => _weights.Length;

    /// <summary>
    /// Computes the linear mean value at the given input point.
    /// </summary>
    /// <param name="x">The input point.</param>
    /// <returns>The linear mean m(x) = w^T × x + b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the dot product of weights with input plus bias.
    ///
    /// For input x = [x1, x2, x3] and weights w = [w1, w2, w3]:
    /// m(x) = w1×x1 + w2×x2 + w3×x3 + bias
    /// </para>
    /// </remarks>
    public T Evaluate(Vector<T> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.Length != _weights.Length)
            throw new ArgumentException($"Input dimension ({x.Length}) must match weight dimension ({_weights.Length}).");

        T result = _bias;
        for (int i = 0; i < x.Length; i++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_weights[i], x[i]));
        }
        return result;
    }

    /// <summary>
    /// Computes the linear mean values at multiple input points.
    /// </summary>
    /// <param name="X">Matrix of input points (one per row).</param>
    /// <returns>Vector of linear mean values.</returns>
    public Vector<T> Evaluate(Matrix<T> X)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (X.Columns != _weights.Length)
            throw new ArgumentException($"Input dimension ({X.Columns}) must match weight dimension ({_weights.Length}).");

        var result = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            T value = _bias;
            for (int j = 0; j < X.Columns; j++)
            {
                value = _numOps.Add(value, _numOps.Multiply(_weights[j], X[i, j]));
            }
            result[i] = value;
        }
        return result;
    }

    /// <summary>
    /// Creates a linear mean function by fitting linear regression to training data.
    /// </summary>
    /// <param name="X">Training input matrix (N x D).</param>
    /// <param name="y">Training target vector (N).</param>
    /// <returns>A LinearMean with fitted weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fits a linear mean function to your data using least squares.
    ///
    /// This finds weights that minimize Σ(y_i - m(x_i))².
    /// The GP will then model the residuals (differences from this linear fit).
    ///
    /// Usage:
    /// var linearMean = LinearMean&lt;double&gt;.FromData(X_train, y_train);
    /// var gp = new StandardGaussianProcess&lt;double&gt;(kernel, linearMean);
    /// </para>
    /// </remarks>
    public static LinearMean<T> FromData(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (X.Rows != y.Length)
            throw new ArgumentException("X and y must have the same number of samples.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = X.Rows;
        int d = X.Columns;

        // Add bias column (all ones)
        var XBias = new Matrix<T>(n, d + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                XBias[i, j] = X[i, j];
            }
            XBias[i, d] = numOps.One; // Bias column
        }

        // Solve X^T X w = X^T y using simple pseudo-inverse
        // First compute X^T X
        var XtX = new Matrix<T>(d + 1, d + 1);
        for (int i = 0; i < d + 1; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(XBias[k, i], XBias[k, j]));
                }
                XtX[i, j] = sum;
                XtX[j, i] = sum;
            }
            // Add regularization
            XtX[i, i] = numOps.Add(XtX[i, i], numOps.FromDouble(1e-6));
        }

        // Compute X^T y
        var Xty = new Vector<T>(d + 1);
        for (int i = 0; i < d + 1; i++)
        {
            T sum = numOps.Zero;
            for (int k = 0; k < n; k++)
            {
                sum = numOps.Add(sum, numOps.Multiply(XBias[k, i], y[k]));
            }
            Xty[i] = sum;
        }

        // Solve using Cholesky decomposition or simple Gauss-Jordan
        var w = SolveLinearSystem(XtX, Xty);

        // Extract weights and bias
        var weights = new double[d];
        for (int i = 0; i < d; i++)
        {
            weights[i] = numOps.ToDouble(w[i]);
        }
        double bias = numOps.ToDouble(w[d]);

        return new LinearMean<T>(weights, bias);
    }

    /// <summary>
    /// Solves Ax = b using Gauss-Jordan elimination.
    /// </summary>
    private static Vector<T> SolveLinearSystem(Matrix<T> A, Vector<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = A.Rows;

        // Create augmented matrix
        var aug = new Matrix<T>(n, n + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                aug[i, j] = A[i, j];
            }
            aug[i, n] = b[i];
        }

        // Forward elimination
        for (int i = 0; i < n; i++)
        {
            // Find pivot
            int maxRow = i;
            double maxVal = Math.Abs(numOps.ToDouble(aug[i, i]));
            for (int k = i + 1; k < n; k++)
            {
                double val = Math.Abs(numOps.ToDouble(aug[k, i]));
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = k;
                }
            }

            // Swap rows
            if (maxRow != i)
            {
                for (int j = 0; j <= n; j++)
                {
                    T temp = aug[i, j];
                    aug[i, j] = aug[maxRow, j];
                    aug[maxRow, j] = temp;
                }
            }

            // Eliminate column
            T pivot = aug[i, i];
            if (Math.Abs(numOps.ToDouble(pivot)) > 1e-12)
            {
                for (int j = 0; j <= n; j++)
                {
                    aug[i, j] = numOps.Divide(aug[i, j], pivot);
                }

                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        T factor = aug[k, i];
                        for (int j = 0; j <= n; j++)
                        {
                            aug[k, j] = numOps.Subtract(aug[k, j],
                                numOps.Multiply(factor, aug[i, j]));
                        }
                    }
                }
            }
        }

        // Extract solution
        var x = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            x[i] = aug[i, n];
        }
        return x;
    }
}

/// <summary>
/// Implements a polynomial mean function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The polynomial mean function returns a polynomial of the input features.
///
/// For 1D input: m(x) = a_0 + a_1×x + a_2×x² + ... + a_d×x^d
///
/// Why use polynomial mean?
/// - When you expect a curved trend in your data
/// - For extrapolation beyond training data with a known trend shape
/// - When physics/domain knowledge suggests polynomial behavior
///
/// Warning: High-degree polynomials can cause problems:
/// - Extrapolation becomes unstable
/// - May overfit to noise
/// - Consider degree 2 or 3 as maximum for most applications
/// </para>
/// </remarks>
public class PolynomialMean<T> : IMeanFunction<T>
{
    private readonly double[] _coefficients;
    private readonly int _degree;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a polynomial mean function for 1D input.
    /// </summary>
    /// <param name="coefficients">Polynomial coefficients [a_0, a_1, ..., a_d].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a polynomial mean m(x) = a_0 + a_1×x + a_2×x² + ...
    ///
    /// The coefficients array should have length (degree + 1):
    /// - coefficients[0] = constant term
    /// - coefficients[1] = linear coefficient
    /// - coefficients[2] = quadratic coefficient
    /// - etc.
    ///
    /// Example for quadratic: m(x) = 1 + 2x + 3x²
    /// var poly = new PolynomialMean&lt;double&gt;(new[] { 1.0, 2.0, 3.0 });
    /// </para>
    /// </remarks>
    public PolynomialMean(double[] coefficients)
    {
        if (coefficients is null) throw new ArgumentNullException(nameof(coefficients));
        if (coefficients.Length == 0)
            throw new ArgumentException("Must have at least one coefficient.", nameof(coefficients));

        _coefficients = (double[])coefficients.Clone();
        _degree = coefficients.Length - 1;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the polynomial degree.
    /// </summary>
    public int Degree => _degree;

    /// <summary>
    /// Gets a copy of the coefficients.
    /// </summary>
    public double[] Coefficients => (double[])_coefficients.Clone();

    /// <summary>
    /// Computes the polynomial mean value at the given input point.
    /// </summary>
    /// <param name="x">The input point (uses first dimension only for multi-D).</param>
    /// <returns>The polynomial mean value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses Horner's method for efficient polynomial evaluation:
    /// m(x) = a_0 + x(a_1 + x(a_2 + ... + x×a_d))
    ///
    /// For multi-dimensional input, only the first dimension is used.
    /// For multi-dimensional polynomials, use a different approach.
    /// </para>
    /// </remarks>
    public T Evaluate(Vector<T> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.Length == 0)
            throw new ArgumentException("Input vector cannot be empty.", nameof(x));

        double xVal = _numOps.ToDouble(x[0]);

        // Horner's method for polynomial evaluation
        double result = _coefficients[_degree];
        for (int i = _degree - 1; i >= 0; i--)
        {
            result = result * xVal + _coefficients[i];
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the polynomial mean values at multiple input points.
    /// </summary>
    /// <param name="X">Matrix of input points.</param>
    /// <returns>Vector of polynomial mean values.</returns>
    public Vector<T> Evaluate(Matrix<T> X)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (X.Columns == 0)
            throw new ArgumentException("Input matrix must have at least one column.", nameof(X));

        var result = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            double xVal = _numOps.ToDouble(X[i, 0]);

            // Horner's method
            double value = _coefficients[_degree];
            for (int j = _degree - 1; j >= 0; j--)
            {
                value = value * xVal + _coefficients[j];
            }
            result[i] = _numOps.FromDouble(value);
        }

        return result;
    }
}
