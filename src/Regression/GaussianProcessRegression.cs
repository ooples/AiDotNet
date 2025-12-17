global using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Implements a Gaussian Process Regression model, which is a non-parametric, probabilistic approach 
/// to regression that provides uncertainty estimates along with predictions.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian Process Regression (GPR) is a flexible, non-parametric approach to regression that models
/// the target function as a sample from a Gaussian process. It provides not only predictions but also
/// uncertainty estimates, making it suitable for applications where quantifying prediction uncertainty
/// is important. The model is defined by a kernel function that determines the covariance between any
/// two points in the input space.
/// </para>
/// <para><b>For Beginners:</b> A Gaussian Process Regression model is like a sophisticated way to draw smooth curves through data points.
/// 
/// Unlike simpler models that assume a specific shape (like a straight line or parabola), Gaussian Process Regression:
/// - Adapts to fit the data without assuming a predefined shape
/// - Provides not just predictions but also how confident it is in each prediction
/// - Works well with small to medium-sized datasets
/// - Can capture complex patterns in the data
/// 
/// You can think of it as drawing a smooth curve through your data points, where the model considers
/// all possible curves that could fit your data and chooses the most likely one based on how similar
/// input points are to each other (defined by a "kernel function").
/// 
/// A unique advantage of this model is that it tells you not just what the prediction is, but also
/// how certain or uncertain that prediction is - like saying "I predict the value is about 42, 
/// and I'm pretty confident it's between 40 and 44."
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GaussianProcessRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// The kernel matrix (also known as the covariance matrix) that represents the similarity between all training points.
    /// </summary>
    private Matrix<T> _kernelMatrix;

    /// <summary>
    /// The vector of coefficients used for making predictions.
    /// </summary>
    private Vector<T> _alpha;

    /// <summary>
    /// Gets the configuration options specific to Gaussian Process Regression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the configuration options that control the Gaussian Process Regression
    /// algorithm, such as kernel parameters, noise level, and optimization settings.
    /// </para>
    /// </remarks>
    private new GaussianProcessRegressionOptions Options { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianProcessRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the Gaussian Process Regression algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Gaussian Process Regression model with the specified options and regularization
    /// strategy. If no options are provided, default values are used. If no regularization is specified, no regularization
    /// is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Gaussian Process Regression model.
    /// 
    /// When creating a model, you can specify:
    /// - Options: Controls settings like kernel parameters and how much to trust the training data
    /// - Regularization: Helps prevent the model from becoming too complex
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a Gaussian Process Regression model with default settings
    /// var gpr = new GaussianProcessRegression&lt;double&gt;();
    /// 
    /// // Create a model with custom options
    /// var options = new GaussianProcessRegressionOptions { 
    ///     NoiseLevel = 0.1,
    ///     OptimizeHyperparameters = true
    /// };
    /// var customGpr = new GaussianProcessRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public GaussianProcessRegression(GaussianProcessRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        Options = options ?? new GaussianProcessRegressionOptions();
        _kernelMatrix = new Matrix<T>(0, 0);
        _alpha = new Vector<T>(0);
    }

    /// <summary>
    /// Optimizes the Gaussian Process model based on the provided training data.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method builds the Gaussian Process Regression model by computing the kernel matrix, optimizing
    /// hyperparameters if requested, and solving for the alpha coefficients that will be used for making
    /// predictions. The kernel matrix represents the similarity or covariance between all pairs of training
    /// points.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// During this process:
    /// 1. The model calculates how similar each data point is to every other data point
    /// 2. It creates a "kernel matrix" that represents these similarities
    /// 3. If requested, it automatically tunes the model parameters to better fit your data
    /// 4. It solves a mathematical equation to find the best coefficients for making predictions
    /// 5. It stores these coefficients and the training data for later use
    /// 
    /// This is a key step that prepares the model to make accurate predictions with appropriate
    /// uncertainty estimates.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        _kernelMatrix = new Matrix<T>(n, n);

        // Compute the kernel matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = KernelFunction(x.GetRow(i), x.GetRow(j));
                _kernelMatrix[i, j] = value;
                _kernelMatrix[j, i] = value;
            }
        }

        // Add noise to the diagonal for numerical stability (vectorized)
        var diagonal = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            diagonal[i] = _kernelMatrix[i, i];
        }

        var noiseVec = new Vector<T>(n);
        T noise = NumOps.FromDouble(Options.NoiseLevel);
        noiseVec.Fill(noise);

        diagonal = (Vector<T>)Engine.Add(diagonal, noiseVec);

        for (int i = 0; i < n; i++)
        {
            _kernelMatrix[i, i] = diagonal[i];
        }

        if (Options.OptimizeHyperparameters)
        {
            OptimizeHyperparameters(x, y);
        }

        // Apply regularization to the kernel matrix
        Matrix<T> regularizedKernelMatrix = Regularization.Regularize(_kernelMatrix);

        // Solve (K + s²I + R)a = y, where R is the regularization term
        _alpha = MatrixSolutionHelper.SolveLinearSystem(regularizedKernelMatrix, y, Options.DecompositionType);

        // Apply regularization to the alpha coefficients
        _alpha = Regularization.Regularize(_alpha);

        // Store x as support vectors for prediction
        SupportVectors = x;
        Alphas = _alpha;
    }

    /// <summary>
    /// Optimizes the hyperparameters of the Gaussian Process model using gradient ascent on the marginal log-likelihood.
    /// </summary>
    /// <param name="x">The feature matrix of training samples.</param>
    /// <param name="y">The target vector of training samples.</param>
    private void OptimizeHyperparameters(Matrix<T> x, Vector<T> y)
    {
        int maxIterations = Options.MaxIterations;
        double tolerance = Options.Tolerance;
        double learningRate = 0.01;

        double lengthScale = Options.LengthScale;
        double signalVariance = Options.SignalVariance;
        double prevLogLikelihood = double.NegativeInfinity;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Compute the kernel matrix with current hyperparameters
            Matrix<T> K = ComputeKernelMatrix(x, lengthScale, signalVariance);

            // Add noise to the diagonal for numerical stability (vectorized)
            int kSize = K.Rows;
            var diagonal = new Vector<T>(kSize);
            for (int i = 0; i < kSize; i++)
            {
                diagonal[i] = K[i, i];
            }

            var noiseVec = new Vector<T>(kSize);
            T noise = NumOps.FromDouble(Options.NoiseLevel);
            noiseVec.Fill(noise);

            diagonal = (Vector<T>)Engine.Add(diagonal, noiseVec);

            for (int i = 0; i < kSize; i++)
            {
                K[i, i] = diagonal[i];
            }

            // Compute the log likelihood
            double logLikelihood = ComputeLogLikelihood(K, y);

            // Check for convergence
            if (Math.Abs(logLikelihood - prevLogLikelihood) < tolerance)
            {
                break;
            }

            // Compute gradients
            (double gradLengthScale, double gradSignalVariance) = ComputeGradients(x, y, K, lengthScale, signalVariance);

            // Update hyperparameters
            lengthScale += learningRate * gradLengthScale;
            signalVariance += learningRate * gradSignalVariance;

            // Ensure hyperparameters remain positive
            lengthScale = Math.Max(lengthScale, 1e-6);
            signalVariance = Math.Max(signalVariance, 1e-6);

            prevLogLikelihood = logLikelihood;
        }

        // Update the options with optimized hyperparameters
        Options.LengthScale = lengthScale;
        Options.SignalVariance = signalVariance;
    }

    /// <summary>
    /// Computes the kernel matrix for a given set of samples using the specified hyperparameters.
    /// </summary>
    /// <param name="x">The feature matrix of samples.</param>
    /// <param name="lengthScale">The length scale hyperparameter of the RBF kernel.</param>
    /// <param name="signalVariance">The signal variance hyperparameter of the RBF kernel.</param>
    /// <returns>The computed kernel matrix.</returns>
    private Matrix<T> ComputeKernelMatrix(Matrix<T> x, double lengthScale, double signalVariance)
    {
        int n = x.Rows;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = RBFKernel(x.GetRow(i), x.GetRow(j), lengthScale, signalVariance);
                K[i, j] = value;
                K[j, i] = value;
            }
        }

        return K;
    }

    /// <summary>
    /// Computes the RBF (Radial Basis Function) kernel value between two feature vectors.
    /// </summary>
    /// <param name="x1">The first feature vector.</param>
    /// <param name="x2">The second feature vector.</param>
    /// <param name="lengthScale">The length scale hyperparameter of the RBF kernel.</param>
    /// <param name="signalVariance">The signal variance hyperparameter of the RBF kernel.</param>
    /// <returns>The computed kernel value.</returns>
    private T RBFKernel(Vector<T> x1, Vector<T> x2, double lengthScale, double signalVariance)
    {
        T squaredDistance = x1.Subtract(x2).PointwiseMultiply(x1.Subtract(x2)).Sum();
        return NumOps.Multiply(NumOps.FromDouble(signalVariance),
            NumOps.Exp(NumOps.Divide(NumOps.Negate(squaredDistance), NumOps.FromDouble(2 * lengthScale * lengthScale))));
    }

    /// <summary>
    /// Computes the log marginal likelihood of the Gaussian Process model.
    /// </summary>
    /// <param name="K">The kernel matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>The computed log marginal likelihood.</returns>
    private double ComputeLogLikelihood(Matrix<T> K, Vector<T> y)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        Vector<T> alpha = choleskyDecomposition.Solve(y);

        double logDeterminant = 0;
        for (int i = 0; i < K.Rows; i++)
        {
            logDeterminant += Math.Log(Convert.ToDouble(K[i, i]));
        }

        return -0.5 * Convert.ToDouble(y.DotProduct(alpha)) - 0.5 * logDeterminant - 0.5 * K.Rows * Math.Log(2 * Math.PI);
    }

    /// <summary>
    /// Computes the gradients of the log marginal likelihood with respect to the hyperparameters.
    /// </summary>
    /// <param name="x">The feature matrix of training samples.</param>
    /// <param name="y">The target vector of training samples.</param>
    /// <param name="K">The kernel matrix.</param>
    /// <param name="lengthScale">The current length scale hyperparameter.</param>
    /// <param name="signalVariance">The current signal variance hyperparameter.</param>
    /// <returns>A tuple containing the gradients with respect to length scale and signal variance.</returns>
    private (double gradLengthScale, double gradSignalVariance) ComputeGradients(Matrix<T> x, Vector<T> y, Matrix<T> K, double lengthScale, double signalVariance)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        Vector<T> alpha = choleskyDecomposition.Solve(y);

        Matrix<T> KInverse = MatrixHelper<T>.InvertUsingDecomposition(choleskyDecomposition);

        Matrix<T> dK_dLengthScale = ComputeKernelMatrixDerivative(x, lengthScale, signalVariance, true);
        Matrix<T> dK_dSignalVariance = ComputeKernelMatrixDerivative(x, lengthScale, signalVariance, false);

        double gradLengthScale = 0.5 * Convert.ToDouble(alpha.DotProduct(dK_dLengthScale.Multiply(alpha))) - 0.5 * Convert.ToDouble(KInverse.ElementWiseMultiplyAndSum(dK_dLengthScale));
        double gradSignalVariance = 0.5 * Convert.ToDouble(alpha.DotProduct(dK_dSignalVariance.Multiply(alpha))) - 0.5 * Convert.ToDouble(KInverse.ElementWiseMultiplyAndSum(dK_dSignalVariance));

        return (gradLengthScale, gradSignalVariance);
    }

    /// <summary>
    /// Computes the derivative of the kernel matrix with respect to the specified hyperparameter.
    /// </summary>
    /// <param name="x">The feature matrix of samples.</param>
    /// <param name="lengthScale">The current length scale hyperparameter.</param>
    /// <param name="signalVariance">The current signal variance hyperparameter.</param>
    /// <param name="isLengthScale">True if computing derivative with respect to length scale, false for signal variance.</param>
    /// <returns>The computed derivative matrix.</returns>
    private Matrix<T> ComputeKernelMatrixDerivative(Matrix<T> x, double lengthScale, double signalVariance, bool isLengthScale)
    {
        int n = x.Rows;
        var dK = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = RBFKernelDerivative(x.GetRow(i), x.GetRow(j), lengthScale, signalVariance, isLengthScale);
                dK[i, j] = value;
                dK[j, i] = value;
            }
        }

        return dK;
    }

    /// <summary>
    /// Computes the derivative of the RBF kernel with respect to the specified hyperparameter.
    /// </summary>
    /// <param name="x1">The first feature vector.</param>
    /// <param name="x2">The second feature vector.</param>
    /// <param name="lengthScale">The current length scale hyperparameter.</param>
    /// <param name="signalVariance">The current signal variance hyperparameter.</param>
    /// <param name="isLengthScale">True if computing derivative with respect to length scale, false for signal variance.</param>
    /// <returns>The computed derivative value.</returns>
    private T RBFKernelDerivative(Vector<T> x1, Vector<T> x2, double lengthScale, double signalVariance, bool isLengthScale)
    {
        T squaredDistance = x1.Subtract(x2).PointwiseMultiply(x1.Subtract(x2)).Sum();
        T kernelValue = RBFKernel(x1, x2, lengthScale, signalVariance);

        if (isLengthScale)
        {
            return NumOps.Multiply(kernelValue, NumOps.Divide(squaredDistance, NumOps.FromDouble(lengthScale * lengthScale * lengthScale)));
        }
        else
        {
            return NumOps.Divide(kernelValue, NumOps.FromDouble(signalVariance));
        }
    }

    /// <summary>
    /// Gets metadata about the Gaussian Process Regression model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and configuration options. This information
    /// can be useful for model management, comparison, and documentation purposes. The metadata includes the noise
    /// level, hyperparameter optimization settings, and current hyperparameter values.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your Gaussian Process Regression model.
    /// 
    /// The metadata includes:
    /// - The type of model (Gaussian Process Regression)
    /// - Noise level: How much random noise is assumed in the data
    /// - Whether hyperparameter optimization was performed
    /// - Length scale: How quickly the correlation between points falls off with distance
    /// - Signal variance: The overall variance of the process
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// - Understanding the model's behavior
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = gpr.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Length scale: {metadata.AdditionalInfo["LengthScale"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NoiseLevel"] = Options.NoiseLevel;
        metadata.AdditionalInfo["OptimizeHyperparameters"] = Options.OptimizeHyperparameters;
        metadata.AdditionalInfo["MaxIterations"] = Options.MaxIterations;
        metadata.AdditionalInfo["Tolerance"] = Options.Tolerance;
        metadata.AdditionalInfo["LengthScale"] = Options.LengthScale;
        metadata.AdditionalInfo["SignalVariance"] = Options.SignalVariance;

        return metadata;
    }

    /// <summary>
    /// Gets the model type of the Gaussian Process Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.GaussianProcessRegression;
    }

    /// <summary>
    /// Creates a new instance of the Gaussian Process Regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="GaussianProcessRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Gaussian Process Regression model that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// The new instance will have the same hyperparameters and options as the original,
    /// but will not share learned data unless explicitly trained with the same dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your model that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar Gaussian Process models
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its hyperparameters
    /// 
    /// Note that while the settings are copied, the training data and learned patterns are not automatically
    /// transferred, so the new instance will need training before it can make predictions.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create and return a new instance with the same configuration
        return new GaussianProcessRegression<T>(Options, Regularization);
    }
}
