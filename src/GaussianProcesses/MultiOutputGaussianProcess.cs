namespace AiDotNet.GaussianProcesses;

/// <summary>
/// A Gaussian Process model that can predict multiple output values simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Gaussian Process is a flexible machine learning method that can learn patterns from data
/// and provide uncertainty estimates with its predictions. Think of it as drawing a smooth curve through your data points,
/// but also showing how confident it is about different parts of that curve.
/// 
/// This "Multi-Output" version can predict multiple related values at once. For example, if you're predicting
/// the temperature and humidity for weather forecasting, this model can learn how these outputs relate to each other
/// and make better predictions by considering them together.
/// 
/// Unlike simpler models that just give you a single prediction, Gaussian Processes also tell you how confident
/// they are about each prediction (the "variance" or "uncertainty"). This is especially useful when making decisions
/// based on predictions where knowing the confidence level is important.
/// </para>
/// </remarks>
public class MultiOutputGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The kernel function that determines how points in the input space relate to each other.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel function is like the "similarity measure" for your data points.
    /// It determines how much influence nearby points have on each other's predictions.
    /// Different kernel functions create different types of patterns in the predictions.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The input training data matrix.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The output training data matrix (multiple outputs).
    /// </summary>
    private Matrix<T> _Y;

    /// <summary>
    /// The kernel matrix calculated from the training data.
    /// </summary>
    private Matrix<T> _K;

    /// <summary>
    /// The Cholesky decomposition of the kernel matrix, used for efficient calculations.
    /// </summary>
    private Matrix<T> _L;

    /// <summary>
    /// The alpha matrix used for making predictions.
    /// </summary>
    private Matrix<T> _alpha;

    /// <summary>
    /// Operations for the numeric type T.
    /// </summary>
    private INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance of the MultiOutputGaussianProcess with the specified kernel function.
    /// </summary>
    /// <param name="kernel">The kernel function to use for the Gaussian Process.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where you set up your Gaussian Process model by choosing a kernel function.
    /// The kernel function determines what kinds of patterns the model can learn.
    /// 
    /// Common kernel choices include:
    /// - RBF (Radial Basis Function): Good for smooth patterns
    /// - Linear: Good for linear trends
    /// - Periodic: Good for repeating patterns
    /// - Matern: Good for less smooth patterns than RBF
    /// 
    /// Choose a kernel that matches the kind of patterns you expect in your data.
    /// </para>
    /// </remarks>
    public MultiOutputGaussianProcess(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _X = Matrix<T>.Empty();
        _Y = Matrix<T>.Empty();
        _K = Matrix<T>.Empty();
        _L = Matrix<T>.Empty();
        _alpha = Matrix<T>.Empty();
    }

    /// <summary>
    /// This method is not supported for multi-output Gaussian Processes.
    /// </summary>
    /// <param name="X">The input training data.</param>
    /// <param name="y">The output training data.</param>
    /// <exception cref="InvalidOperationException">Always thrown because this method is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is not used for multi-output Gaussian Processes.
    /// Use the FitMultiOutput method instead when you have multiple output values to predict.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        throw new InvalidOperationException("Use FitMultiOutput method for multi-output GP");
    }

    /// <summary>
    /// Trains the Gaussian Process model on the provided multi-output training data.
    /// </summary>
    /// <param name="X">The input training data matrix where each row is a training example and each column is a feature.</param>
    /// <param name="Y">The output training data matrix where each row corresponds to a training example and each column is an output dimension.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where your model "learns" from your training data.
    /// 
    /// - X contains your input features (like temperature, humidity, pressure for weather prediction)
    /// - Y contains your multiple output values (like wind speed, rainfall, cloud cover)
    /// 
    /// The model will learn the relationships between your inputs and outputs, as well as how
    /// the different outputs relate to each other. After fitting, the model is ready to make predictions.
    /// 
    /// Behind the scenes, this method:
    /// 1. Calculates how similar each training point is to every other point (the kernel matrix)
    /// 2. Adds a small value to ensure numerical stability (preventing math errors)
    /// 3. Solves a system of equations to find the best fit for your data
    /// 4. Stores the results for later use when making predictions
    /// </para>
    /// </remarks>
    public void FitMultiOutput(Matrix<T> X, Matrix<T> Y)
    {
        _X = X;
        _Y = Y;

        // Calculate the kernel matrix
        _K = CalculateKernelMatrix(X, X);

        // Add adaptive jitter to diagonal for numerical stability.
        // Kernels like Exponential and Laplacian produce ill-conditioned matrices
        // where Cholesky succeeds but forward/backward substitution produces NaN.
        // We check the condition number via the L diagonal ratio.
        AddJitter(_K);

        // Solve for alpha using Cholesky decomposition
        _alpha = new Matrix<T>(Y.Rows, Y.Columns);
        for (int i = 0; i < Y.Columns; i++)
        {
            var y_col = Y.GetColumn(i);
            var alpha_col = MatrixSolutionHelper.SolveLinearSystem(_K, y_col, MatrixDecompositionType.Cholesky);
            for (int j = 0; j < Y.Rows; j++)
            {
                _alpha[j, i] = alpha_col[j];
            }
        }

        // Store the Cholesky decomposition for later use in predictions
        _L = new CholeskyDecomposition<T>(_K).L;
    }

    /// <summary>
    /// This method is not supported for multi-output Gaussian Processes.
    /// </summary>
    /// <param name="x">The input vector for prediction.</param>
    /// <returns>This method does not return as it throws an exception.</returns>
    /// <exception cref="InvalidOperationException">Always thrown because this method is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is not used for multi-output Gaussian Processes.
    /// Use the PredictMultiOutput method instead when you want to predict multiple output values.
    /// </para>
    /// </remarks>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        throw new InvalidOperationException("Use PredictMultiOutput method for multi-output GP");
    }

    /// <summary>
    /// Makes predictions for a new input point, returning both the predicted means and the covariance matrix.
    /// </summary>
    /// <param name="x">The input vector for which to make predictions.</param>
    /// <returns>A tuple containing the predicted mean values and the covariance matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where you get predictions from your trained model for new data points.
    /// 
    /// For example, if you've trained a weather model on historical data, you can use this method
    /// to predict multiple weather variables (temperature, humidity, etc.) for tomorrow.
    /// 
    /// The method returns two things:
    /// 1. means: The predicted values for each output (your best guess)
    /// 2. covariance: How confident the model is about each prediction and how the predictions relate to each other
    /// 
    /// A large value in the covariance matrix means the model is uncertain about that prediction.
    /// This uncertainty information is one of the main advantages of Gaussian Processes over other methods.
    /// 
    /// You can use this uncertainty to:
    /// - Know when to trust or be cautious about predictions
    /// - Decide if you need more training data in certain areas
    /// - Calculate confidence intervals for your predictions
    /// </para>
    /// </remarks>
    public (Vector<T> means, Matrix<T> covariance) PredictMultiOutput(Vector<T> x)
    {
        var k_star = CalculateKernelVector(_X, x);
        var means = new Vector<T>(_Y.Columns);

        for (int i = 0; i < _Y.Columns; i++)
        {
            means[i] = _numOps.Zero;
            for (int j = 0; j < k_star.Length; j++)
            {
                means[i] = _numOps.Add(means[i], _numOps.Multiply(k_star[j], _alpha[j, i]));
            }
        }

        var v = MatrixSolutionHelper.SolveLinearSystem(_K, k_star, MatrixDecompositionType.Cholesky);
        var variance = _numOps.Subtract(_kernel.Calculate(x, x), k_star.DotProduct(v));
        var covariance = new Matrix<T>(_Y.Columns, _Y.Columns);

        for (int i = 0; i < _Y.Columns; i++)
        {
            covariance[i, i] = variance;
        }

        return (means, covariance);
    }

    /// <summary>
    /// Updates the kernel function used by the Gaussian Process and retrains the model if data is available.
    /// </summary>
    /// <param name="kernel">The new kernel function to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you change the "similarity measure" (kernel function) of your model
    /// after you've created it.
    /// 
    /// You might want to do this if:
    /// - You're experimenting with different kernels to see which works best
    /// - Your data characteristics have changed and you need a different kernel
    /// - You're using an optimization algorithm to find the best kernel parameters
    /// 
    /// If you've already trained the model with data, calling this method will automatically
    /// retrain the model with the new kernel using that same data.
    /// </para>
    /// </remarks>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_Y.IsEmpty)
        {
            FitMultiOutput(_X, _Y);
        }
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of input points.
    /// </summary>
    /// <param name="X1">The first set of input points.</param>
    /// <param name="X2">The second set of input points.</param>
    /// <returns>A matrix containing the kernel values between each pair of points from X1 and X2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how similar each point in the first set is to each point in the second set.
    /// 
    /// Think of it like measuring the "distance" or "similarity" between data points. The kernel function
    /// transforms these distances into similarity scores, where:
    /// - Higher values mean points are more similar
    /// - Lower values mean points are less similar
    /// 
    /// The result is a grid (matrix) of similarity scores. For example, if you have 5 points in X1 and 3 points in X2,
    /// you'll get a 5×3 grid showing how similar each point in X1 is to each point in X2.
    /// 
    /// This similarity information is fundamental to how Gaussian Processes work - they make predictions
    /// based on how similar new points are to the training data points.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Adds adaptive jitter to the diagonal of a kernel matrix for numerical stability.
    /// </summary>
    private void AddJitter(Matrix<T> K)
    {
        double jitterValue = 1e-6;
        const double maxJitter = 1e-1;

        while (jitterValue <= maxJitter)
        {
            var jitter = _numOps.FromDouble(jitterValue);
            for (int i = 0; i < K.Rows; i++)
                K[i, i] = _numOps.Add(K[i, i], jitter);

            try
            {
                var chol = new CholeskyDecomposition<T>(K);

                double minDiag = double.MaxValue;
                double maxDiag = 0;
                bool hasNaN = false;
                for (int i = 0; i < chol.L.Rows; i++)
                {
                    double d = _numOps.ToDouble(chol.L[i, i]);
                    if (double.IsNaN(d) || double.IsInfinity(d) || d <= 0)
                    {
                        hasNaN = true;
                        break;
                    }
                    minDiag = Math.Min(minDiag, d);
                    maxDiag = Math.Max(maxDiag, d);
                }

                if (!hasNaN && minDiag > 0 && maxDiag / minDiag < 1e8)
                    return;

                for (int i = 0; i < K.Rows; i++)
                    K[i, i] = _numOps.Subtract(K[i, i], jitter);
                jitterValue *= 10;
            }
            catch (ArgumentException)
            {
                for (int i = 0; i < K.Rows; i++)
                    K[i, i] = _numOps.Subtract(K[i, i], jitter);
                jitterValue *= 10;
            }
        }

        var maxJitterT = _numOps.FromDouble(maxJitter);
        for (int i = 0; i < K.Rows; i++)
            K[i, i] = _numOps.Add(K[i, i], maxJitterT);
    }

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
    /// Calculates the kernel vector between a set of input points and a single input point.
    /// </summary>
    /// <param name="X">The set of input points.</param>
    /// <param name="x">The single input point.</param>
    /// <returns>A vector containing the kernel values between each point in X and the point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how similar a new data point is to each of your training data points.
    /// 
    /// When making predictions with a Gaussian Process, we need to know how similar the new point
    /// (that we want to predict) is to each of our training points. This method creates that list of similarities.
    /// 
    /// For example, if you have 100 training points and want to predict for a new point:
    /// - This method will return 100 similarity scores
    /// - Higher scores mean the new point is more similar to that training point
    /// - Lower scores mean the new point is less similar to that training point
    /// 
    /// The Gaussian Process uses these similarity scores to make its prediction, giving more weight
    /// to training points that are more similar to the new point. This is why Gaussian Processes
    /// can capture complex patterns - they use this "weighted voting" from all training points.
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
