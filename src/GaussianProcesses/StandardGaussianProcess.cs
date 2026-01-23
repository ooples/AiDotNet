namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a standard Gaussian Process regression model for making probabilistic predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Gaussian Process is a flexible machine learning method that can make predictions
/// with uncertainty estimates.
/// 
/// Think of it like drawing a line through data points, but instead of just one line, it gives you
/// a range of possible lines with a confidence level for each. This helps you understand not just
/// what the prediction is, but how certain the model is about that prediction.
/// 
/// Gaussian Processes are particularly useful when:
/// - You have a small to medium amount of data
/// - You need to know how confident the model is in its predictions
/// - Your data might have complex patterns that simpler models can't capture
/// 
/// Unlike many other machine learning methods, Gaussian Processes don't just learn a fixed set of
/// parameters - they use all training data when making predictions, which allows them to capture
/// complex patterns in your data.
/// </para>
/// </remarks>
public class StandardGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The kernel function that determines how similarity between data points is calculated.
    /// </summary>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The matrix of input features from the training data.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The vector of target values from the training data.
    /// </summary>
    private Vector<T> _y;

    /// <summary>
    /// The kernel matrix calculated from the training data.
    /// </summary>
    private Matrix<T> _K;

    /// <summary>
    /// Operations for performing numeric calculations with the generic type T.
    /// </summary>
    private INumericOperations<T> _numOps;

    /// <summary>
    /// The method used to decompose matrices for solving linear systems.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// Initializes a new instance of the StandardGaussianProcess class.
    /// </summary>
    /// <param name="kernel">The kernel function to use for measuring similarity between data points.</param>
    /// <param name="decompositionType">The matrix decomposition method to use for calculations.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The constructor sets up your Gaussian Process model with the essential components it needs.
    /// 
    /// The kernel function is particularly important - it defines how the model measures similarity between data points.
    /// Different kernels capture different types of patterns:
    /// - RBF (Radial Basis Function) kernel: Good for smooth patterns
    /// - Linear kernel: Good for linear relationships
    /// - Periodic kernel: Good for repeating patterns
    /// 
    /// The decomposition type is a technical detail about how the model solves certain mathematical equations.
    /// For most users, the default (Cholesky) works well and is efficient.
    /// </para>
    /// </remarks>
    public StandardGaussianProcess(IKernelFunction<T> kernel, MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        _kernel = kernel;
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _K = Matrix<T>.Empty();
        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionType = decompositionType;
    }

    /// <summary>
    /// Trains the Gaussian Process model on the provided data.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a data point and each column is a feature.</param>
    /// <param name="y">The target values corresponding to each row in X.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method "teaches" your model using your training data.
    /// 
    /// Unlike many machine learning models that learn specific parameters during training,
    /// Gaussian Processes actually store the entire training dataset. This is because they
    /// use all training points when making predictions.
    /// 
    /// During fitting, the model calculates how similar each training point is to every other
    /// training point (using the kernel function). This information is stored in the kernel
    /// matrix (_K) and will be used later when making predictions.
    /// 
    /// This approach allows Gaussian Processes to capture complex patterns in your data,
    /// but it also means they can become slow with very large datasets.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;
        _K = CalculateKernelMatrix(X, X);

        // Add small jitter term to diagonal for numerical stability
        // This prevents the kernel matrix from being singular or nearly singular
        // when data points are close together, which would cause Cholesky decomposition to fail
        var jitter = _numOps.FromDouble(1e-6);
        for (int i = 0; i < _K.Rows; i++)
        {
            _K[i, i] = _numOps.Add(_K[i, i], jitter);
        }
    }

    /// <summary>
    /// Makes a prediction for a new data point, returning both the predicted value and its uncertainty.
    /// </summary>
    /// <param name="x">The input features vector for which to make a prediction.</param>
    /// <returns>A tuple containing the predicted mean value and variance (uncertainty).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method makes predictions for new data points and tells you how confident it is.
    /// 
    /// The prediction process works by:
    /// 1. Calculating how similar the new point is to each training point
    /// 2. Using these similarities to create a weighted average of the training outputs
    /// 3. Calculating how uncertain the prediction is based on the data
    /// 
    /// The result gives you two important pieces of information:
    /// - mean: The actual prediction (best guess) for your input
    /// - variance: How uncertain the model is about this prediction
    /// 
    /// A low variance means the model is confident in its prediction, while a high variance
    /// means there's more uncertainty - perhaps because the new point is unlike any in the
    /// training data, or because that region of the data was noisy.
    /// 
    /// This uncertainty information is one of the biggest advantages of Gaussian Processes
    /// compared to many other machine learning methods.
    /// </para>
    /// </remarks>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        var k = CalculateKernelVector(_X, x);

        // Solve _K * alpha = _y
        var alpha = MatrixSolutionHelper.SolveLinearSystem(_K, _y, _decompositionType);
        var mean = k.DotProduct(alpha);

        // Solve _K * v = k
        var v = MatrixSolutionHelper.SolveLinearSystem(_K, k, _decompositionType);
        var variance = _numOps.Subtract(_kernel.Calculate(x, x), k.DotProduct(v));

        return (mean, variance);
    }

    /// <summary>
    /// Updates the kernel function used by the model and recalculates the kernel matrix if training data exists.
    /// </summary>
    /// <param name="kernel">The new kernel function to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you change how the model measures similarity between data points.
    /// 
    /// The kernel function is like the "lens" through which your model views the data. Different
    /// kernels can help the model see different types of patterns:
    /// 
    /// - An RBF kernel sees smooth, continuous patterns
    /// - A linear kernel sees straight-line relationships
    /// - A periodic kernel sees repeating patterns
    /// 
    /// By changing the kernel, you can help your model better capture the specific patterns in your data.
    /// 
    /// If you've already trained the model (using Fit), this method will automatically recalculate
    /// everything needed with the new kernel so your model stays up-to-date.
    /// </para>
    /// </remarks>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (_X != null && _y != null)
        {
            Fit(_X, _y);
        }
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of data points.
    /// </summary>
    /// <param name="X1">The first set of data points.</param>
    /// <param name="X2">The second set of data points.</param>
    /// <returns>A matrix where each element [i,j] represents the kernel value between the i-th point in X1 and the j-th point in X2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a "similarity table" between data points.
    /// 
    /// The kernel matrix is a table where each cell shows how similar two data points are to each other,
    /// according to the kernel function. For example, with an RBF kernel:
    /// - Points that are close together will have values close to 1
    /// - Points that are far apart will have values close to 0
    /// 
    /// This similarity information is crucial for the Gaussian Process to understand patterns in your data
    /// and make predictions based on those patterns.
    /// 
    /// This method computes this similarity for every possible pair of points between the two sets X1 and X2,
    /// resulting in a matrix of size (X1.Rows × X2.Rows).
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = _kernel.Calculate(X1.GetRow(i), X2.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Calculates the kernel values between a set of data points and a single point.
    /// </summary>
    /// <param name="X">A matrix where each row is a data point.</param>
    /// <param name="x">A single data point as a vector.</param>
    /// <returns>A vector where each element is the kernel value between the corresponding row in X and the point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how similar a new data point is to each of our training data points.
    /// 
    /// When making a prediction for a new point, we need to know how it relates to our training data.
    /// This method creates a vector of similarity scores between the new point and each training point.
    /// 
    /// These similarity scores are then used to weight the influence of each training point on the final prediction.
    /// Points that are more similar to our new point will have a stronger influence on the prediction.
    /// 
    /// This is a key part of how Gaussian Processes make predictions - by understanding the relationships
    /// between data points through these similarity measures.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateKernelVector(Matrix<T> X, Vector<T> x)
    {
        var k = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            k[i] = _kernel.Calculate(X.GetRow(i), x);
        }

        return k;
    }
}
