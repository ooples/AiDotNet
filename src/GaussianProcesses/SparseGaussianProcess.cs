namespace AiDotNet.GaussianProcesses;

/// <summary>
/// A sparse implementation of Gaussian Process regression that uses inducing points to reduce computational complexity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Gaussian Process is a flexible machine learning method that can model complex relationships in data.
/// 
/// The "sparse" version solves a common problem with Gaussian Processes - they can be very slow with large datasets.
/// Instead of using all data points for predictions (which can be computationally expensive), 
/// this implementation selects a smaller set of representative points called "inducing points" 
/// that capture the essential patterns in your data.
/// 
/// Think of it like summarizing a book: instead of reading every word (standard GP),
/// you read just the chapter summaries (sparse GP) to get the main ideas more efficiently.
/// 
/// This approach makes Gaussian Processes practical for larger datasets while maintaining most of their predictive power.
/// </para>
/// </remarks>
public class SparseGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The kernel function that defines the similarity between data points.
    /// </summary>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The training input data matrix.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The training output data vector.
    /// </summary>
    private Vector<T> _y;

    /// <summary>
    /// A subset of training points used to approximate the full Gaussian Process.
    /// </summary>
    private Matrix<T> _inducingPoints;

    /// <summary>
    /// Helper for performing numeric operations on type T.
    /// </summary>
    private INumericOperations<T> _numOps;

    /// <summary>
    /// The lower triangular matrix from Cholesky decomposition of the kernel matrix.
    /// </summary>
    private Matrix<T> _L;

    /// <summary>
    /// Intermediate matrix used for efficient predictions.
    /// </summary>
    private Matrix<T> _V;

    /// <summary>
    /// Diagonal elements used in the sparse approximation.
    /// </summary>
    private Vector<T> _D;

    /// <summary>
    /// Weights vector used for mean prediction.
    /// </summary>
    private Vector<T> _alpha;

    /// <summary>
    /// The type of matrix decomposition to use for solving linear systems.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// Initializes a new instance of the SparseGaussianProcess class.
    /// </summary>
    /// <param name="kernel">The kernel function to use for measuring similarity between data points.</param>
    /// <param name="decompositionType">The matrix decomposition method to use for numerical stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the sparse Gaussian Process model with your chosen kernel function.
    /// 
    /// The kernel function is like a "similarity measure" - it determines how the model understands 
    /// the relationships between different data points. Different kernels are suited for different types of data:
    /// - RBF/Gaussian kernel: Good for smooth data
    /// - Periodic kernel: Good for repeating patterns
    /// - Linear kernel: Good for linear relationships
    /// 
    /// The decompositionType parameter is a technical detail about how the math is handled internally
    /// for numerical stability. For most users, the default (Cholesky) works well.
    /// </para>
    /// </remarks>
    public SparseGaussianProcess(IKernelFunction<T> kernel, MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        _kernel = kernel;
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _L = Matrix<T>.Empty();
        _V = Matrix<T>.Empty();
        _D = Vector<T>.Empty();
        _alpha = Vector<T>.Empty();
        _inducingPoints = Matrix<T>.Empty();
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Trains the Gaussian Process model on the provided data.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a data point and each column is a feature.</param>
    /// <param name="y">The target values corresponding to each input data point.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method "teaches" the model using your training data.
    /// 
    /// The training process involves:
    /// 1. Selecting a smaller set of representative points (inducing points) from your data
    /// 2. Computing how these points relate to each other and to all your data
    /// 3. Storing this information in an efficient way for making predictions later
    /// 
    /// The algorithm used (FITC - Fully Independent Training Conditional) is a mathematical approach
    /// that balances accuracy with computational efficiency.
    /// 
    /// After calling this method, your model is ready to make predictions on new data points.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;
        _inducingPoints = SelectInducingPoints(X);

        // Sparse GP training algorithm (Fully Independent Training Conditional - FITC)
        var Kuu = CalculateKernelMatrix(_inducingPoints, _inducingPoints);
        var Kuf = CalculateKernelMatrix(_inducingPoints, X);
        var Kff_diag = CalculateKernelDiagonal(X);

        var choleskyKuu = new CholeskyDecomposition<T>(Kuu);
        var L = choleskyKuu.L;

        // Solve for each column of Kuf separately
        var V = new Matrix<T>(Kuu.Rows, Kuf.Columns);
        for (int i = 0; i < Kuf.Columns; i++)
        {
            var column = Kuf.GetColumn(i);
            var solvedColumn = choleskyKuu.Solve(column);
            V.SetColumn(i, solvedColumn);
        }

        // Calculate Qff_diag
        var Qff_diag = new Vector<T>(Kuf.Columns);
        for (int i = 0; i < Kuf.Columns; i++)
        {
            var column = V.GetColumn(i);
            Qff_diag[i] = _numOps.Square(column.Sum());
        }

        var Lambda = Kff_diag.Subtract(Qff_diag);

        var noise = _numOps.FromDouble(1e-6); // Small noise term for numerical stability
        var D = Lambda.Add(noise).Transform(v => Reciprocal(v));

        var Ky = Kuu.Add(Kuf.Multiply(D.CreateDiagonal()).Multiply(Kuf.Transpose()));
        var choleskyKy = new CholeskyDecomposition<T>(Ky);
        var alpha = choleskyKy.Solve(Kuf.Multiply(D.CreateDiagonal()).Multiply(y));

        // Store necessary components for prediction
        _L = L;
        _V = V;
        _D = D;
        _alpha = alpha;
    }

    /// <summary>
    /// Calculates the reciprocal (1/x) of a value.
    /// </summary>
    /// <param name="value">The value to calculate the reciprocal for.</param>
    /// <returns>The reciprocal of the input value.</returns>
    private T Reciprocal(T value)
    {
        return _numOps.Divide(_numOps.One, value);
    }

    /// <summary>
    /// Makes a prediction for a new input point, returning both the mean prediction and its variance.
    /// </summary>
    /// <param name="x">The input feature vector to predict for.</param>
    /// <returns>A tuple containing the predicted mean value and the variance (uncertainty) of the prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses your trained model to make predictions on new data points.
    /// 
    /// What makes Gaussian Processes special is that they don't just give you a prediction - 
    /// they also tell you how confident they are about that prediction through the variance value.
    /// 
    /// The mean is the model's best guess at the correct answer.
    /// The variance tells you how certain the model is:
    /// - Low variance = high confidence (the model is pretty sure about its prediction)
    /// - High variance = low confidence (the model is uncertain about its prediction)
    /// 
    /// This is especially useful in scenarios where knowing the uncertainty is important,
    /// like in scientific experiments, medical diagnoses, or financial forecasting.
    /// </para>
    /// </remarks>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        var Kus = CalculateKernelVector(_inducingPoints, x);
        var Kss = _kernel.Calculate(x, x);

        var f_mean = Kus.DotProduct(_alpha);

        var v = MatrixSolutionHelper.SolveLinearSystem(_L, Kus, _decompositionType);
        var f_var = _numOps.Subtract(Kss, v.DotProduct(v));

        return (f_mean, f_var);
    }

    /// <summary>
    /// Updates the kernel function used by the model and retrains if data is available.
    /// </summary>
    /// <param name="kernel">The new kernel function to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you change how the model measures similarity between data points.
    /// 
    /// Different kernel functions capture different types of patterns in your data. If you find that
    /// your model isn't performing well, you might want to try a different kernel that better matches
    /// the underlying patterns in your data.
    /// 
    /// For example:
    /// - If your data has seasonal patterns, you might switch to a periodic kernel
    /// - If your data has long-term trends, you might use an RBF kernel with a larger length scale
    /// 
    /// When you update the kernel, the model will automatically retrain itself if you've already
    /// provided training data.
    /// </para>
    /// </remarks>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_y.IsEmpty)
        {
            Fit(_X, _y);
        }
    }

    /// <summary>
    /// Selects a subset of data points to use as inducing points for the sparse Gaussian Process.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a data point and each column is a feature.</param>
    /// <returns>A matrix containing the selected inducing points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inducing points are like "representative samples" of your data.
    /// 
    /// Imagine you have thousands of data points - processing all of them would be very slow.
    /// Instead, we select a smaller set (in this case, up to 100 points) that can represent
    /// the overall patterns in your data.
    /// 
    /// This method uses random sampling to select these points. While there are more sophisticated
    /// ways to select inducing points (like k-means clustering), random selection is simple and
    /// often works well enough in practice.
    /// 
    /// By using these inducing points instead of all data points, the Gaussian Process can make
    /// predictions much faster while still maintaining good accuracy.
    /// </para>
    /// </remarks>
    private Matrix<T> SelectInducingPoints(Matrix<T> X)
    {
        int m = Math.Min(X.Rows, 100); // Number of inducing points, capped at 100 or the number of data points
        var indices = new List<int>();
        var random = RandomHelper.CreateSecureRandom();

        while (indices.Count < m)
        {
            int index = random.Next(0, X.Rows);
            if (!indices.Contains(index))
            {
                indices.Add(index);
            }
        }

        return X.GetRows(indices);
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of data points.
    /// </summary>
    /// <param name="X1">The first set of data points.</param>
    /// <param name="X2">The second set of data points.</param>
    /// <returns>A matrix where each element [i,j] represents the kernel value between the i-th point in X1 and the j-th point in X2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel matrix is a way of measuring how similar each pair of data points is to each other.
    /// 
    /// Think of it as a "similarity table" where each row and column represents a data point,
    /// and each cell shows how similar those two points are according to the kernel function.
    /// 
    /// For example, with an RBF (Gaussian) kernel:
    /// - Points that are close together will have values close to 1
    /// - Points that are far apart will have values close to 0
    /// 
    /// This similarity information is crucial for the Gaussian Process to understand patterns in your data
    /// and make predictions based on those patterns.
    /// 
    /// This method computes this similarity for every possible pair of points between the two sets X1 and X2.
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
    /// <b>For Beginners:</b> This method calculates how similar a new data point is to each of our existing data points.
    /// 
    /// When making a prediction for a new point, we need to know how it relates to our training data.
    /// This method creates a vector of similarity scores between the new point and each of our inducing points.
    /// 
    /// These similarity scores are then used to weight the influence of each inducing point on the final prediction.
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

    /// <summary>
    /// Calculates the kernel values of each data point with itself.
    /// </summary>
    /// <param name="X">A matrix where each row is a data point.</param>
    /// <returns>A vector where each element is the kernel value of the corresponding data point with itself.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the "self-similarity" of each data point.
    /// 
    /// In most kernel functions, when you compare a point to itself, you get the maximum similarity value.
    /// For example, with an RBF kernel, the self-similarity is always 1.
    /// 
    /// These diagonal values are important for several calculations in Gaussian Processes:
    /// 
    /// 1. They're used to calculate the variance (uncertainty) in predictions
    /// 2. They help in constructing the full covariance matrix efficiently
    /// 3. They're needed for the sparse approximation algorithm (FITC) used in this implementation
    /// 
    /// By calculating just the diagonal elements (rather than the full matrix), we save computational
    /// resources when that's all we need.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateKernelDiagonal(Matrix<T> X)
    {
        var diag = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            diag[i] = _kernel.Calculate(X.GetRow(i), X.GetRow(i));
        }

        return diag;
    }
}
