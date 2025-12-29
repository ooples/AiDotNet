namespace AiDotNet.Regression;

/// <summary>
/// Implements Bayesian Linear Regression with support for various kernels and uncertainty estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Bayesian Linear Regression extends traditional linear regression by using Bayesian inference to provide
/// a probabilistic model of the regression problem. Instead of point estimates of the model parameters,
/// it computes a full posterior distribution over the parameters, allowing for uncertainty quantification
/// in predictions. The model assumes Gaussian prior distributions on the parameters and Gaussian noise
/// in the observations.
/// </para>
/// <para>
/// This implementation supports various kernel functions for non-linear regression, including:
/// - Linear kernel (standard linear regression)
/// - Radial Basis Function (RBF) kernel
/// - Polynomial kernel
/// - Sigmoid kernel
/// - Laplacian kernel
/// The choice of kernel enables the model to capture different types of relationships between features and targets.
/// </para>
/// <para><b>For Beginners:</b> Bayesian regression is a special type of regression model that not only predicts values
/// but also tells you how confident it is about those predictions.
/// 
/// Think of it this way: If you were to guess someone's weight just by looking at their height, you wouldn't
/// be 100% sure about your guess. You'd have some uncertainty. Bayesian regression captures this uncertainty
/// mathematically.
/// 
/// Key features of Bayesian regression:
/// - It calculates probabilities instead of just point estimates
/// - It can tell you which predictions are more reliable than others
/// - It combines prior knowledge with observed data to make inferences
/// - It can incorporate various "kernels" to model different types of relationships
/// 
/// A "kernel" is like a special lens that transforms how the model sees relationships in your data.
/// For example, some kernels are good at capturing curved relationships, while others might be better
/// for periodic patterns.
/// 
/// Bayesian regression is especially useful when:
/// - You have limited data
/// - You want to know how confident the model is in its predictions
/// - You need to incorporate prior knowledge about the problem
/// </para>
/// </remarks>
public class BayesianRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Options specific to Bayesian regression.
    /// </summary>
    private readonly BayesianRegressionOptions<T> _bayesOptions;

    /// <summary>
    /// The covariance matrix of the posterior distribution over model parameters.
    /// </summary>
    private Matrix<T> _posteriorCovariance;

    /// <summary>
    /// Initializes a new instance of the <see cref="BayesianRegression{T}"/> class with the specified options and regularization.
    /// </summary>
    /// <param name="bayesianOptions">The options for configuring the Bayesian regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Optional regularization to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Bayesian regression model with the specified options and regularization.
    /// The options control parameters such as the prior precision (alpha), noise precision (beta),
    /// kernel type, and kernel-specific parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Bayesian regression model with specific settings.
    /// 
    /// The options parameter controls important settings like:
    /// - Alpha: Controls the strength of the prior belief about parameters (higher = stronger prior)
    /// - Beta: Controls the assumed noise level in the data (higher = less noise)
    /// - KernelType: Specifies what kind of relationship pattern to look for (linear, curved, etc.)
    /// - DecompositionType: Technical setting for how certain matrix operations are performed
    /// 
    /// The regularization parameter helps prevent "overfitting" - a situation where the model works well
    /// on training data but poorly on new data because it's too closely tailored to the specific examples
    /// it was trained on.
    /// 
    /// If you're not sure what values to use, the default options typically provide a good starting point
    /// for many regression problems.
    /// </para>
    /// </remarks>
    public BayesianRegression(BayesianRegressionOptions<T>? bayesianOptions = null,
                              IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(bayesianOptions, regularization)
    {
        _bayesOptions = bayesianOptions ?? new BayesianRegressionOptions<T>();
        _posteriorCovariance = new Matrix<T>(0, 0);
    }

    /// <summary>
    /// Trains the Bayesian regression model on the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to the input samples.</param>
    /// <remarks>
    /// <para>
    /// This method implements Bayesian inference for linear regression. It computes the posterior distribution
    /// over the regression coefficients given the input data, target values, and prior distribution.
    /// The main steps of the algorithm are:
    /// 1. Preprocess the input data (add intercept, apply kernel, regularize)
    /// 2. Compute the prior precision matrix (inverse of prior covariance)
    /// 3. Compute the data likelihood precision matrix
    /// 4. Compute the posterior precision matrix (prior + likelihood)
    /// 5. Invert the posterior precision to get the posterior covariance
    /// 6. Compute the posterior mean (coefficients)
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions based on your training data.
    /// 
    /// Here's what happens during training:
    /// 1. The method first prepares your data:
    ///    - It adds a constant term if you're using an intercept (like y-intercept in a line equation)
    ///    - It applies a kernel transformation if you've selected a non-linear kernel
    ///    - It applies regularization to help prevent overfitting
    /// 
    /// 2. Then it uses Bayesian math to:
    ///    - Calculate how much to trust the prior beliefs (prior precision)
    ///    - Calculate how much to trust the data (design precision)
    ///    - Combine these to get the final model parameters
    ///    - Store information about uncertainty for later use
    /// 
    /// Unlike regular regression that gives single "best" values for each coefficient,
    /// Bayesian regression captures a distribution of likely values, which allows it to
    /// estimate uncertainty in its predictions.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;

        // Add bias term if using intercept
        if (Options.UseIntercept)
        {
            x = x.AddConstantColumn(NumOps.One);
            d++;
        }

        // Apply kernel if specified
        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            x = ApplyKernel(x);
        }

        // Note: Bayesian regression has built-in regularization through the prior precision (alpha).
        // Additional regularization is not applied through data transformation.

        // Compute prior precision (inverse of prior covariance)
        var priorPrecision = Matrix<T>.CreateIdentity(d).Multiply(NumOps.FromDouble(_bayesOptions.Alpha));

        // Compute the design matrix precision
        var noisePrecision = NumOps.FromDouble(_bayesOptions.Beta);
        var designPrecision = x.Transpose().Multiply(x).Multiply(noisePrecision);

        // Compute posterior precision and covariance
        var posteriorPrecision = priorPrecision.Add(designPrecision);

        // Use the factory to create the appropriate decomposition
        var decomposition = MatrixDecompositionFactory.CreateDecomposition(posteriorPrecision, _bayesOptions.DecompositionType);
        _posteriorCovariance = MatrixHelper<T>.InvertUsingDecomposition(decomposition);

        // Compute posterior mean (coefficients)
        var xTy = x.Transpose().Multiply(y).Multiply(noisePrecision);
        var coeffs = _posteriorCovariance.Multiply(xTy);

        if (Options.UseIntercept)
        {
            Intercept = coeffs[0];
            Coefficients = new Vector<T>([.. coeffs.Skip(1)]);
        }
        else
        {
            Coefficients = coeffs;
        }
    }

    /// <summary>
    /// Makes predictions on new data using the trained Bayesian regression model.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample to predict.</param>
    /// <returns>The predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs predictions using the mean of the posterior distribution over the model parameters.
    /// The prediction process consists of the following steps:
    /// 1. Preprocess the input data (add intercept, apply kernel)
    /// 2. Compute the predicted values using the trained model parameters
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the trained model to make predictions on new data.
    /// 
    /// Here's how the prediction works:
    /// 1. The method first prepares your input data:
    ///    - It adds a constant term if you're using an intercept
    ///    - It applies the same kernel transformation used during training (if any)
    /// 
    /// 2. Then it multiplies the prepared input by the learned coefficients to get predictions
    /// 
    /// This method gives you the "expected" or "mean" prediction, without information about uncertainty.
    /// If you want uncertainty estimates as well, use the PredictWithUncertainty method instead.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Options.UseIntercept)
        {
            input = input.AddConstantColumn(NumOps.One);
        }

        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            input = ApplyKernel(input);
        }

        // Create coefficient vector with intercept at position 0 (matching constant column at front)
        // input × coefficients = (N, d+1) × (d+1,) = (N,)
        Vector<T> allCoeffs;
        if (Options.UseIntercept)
        {
            // Prepend intercept to match the constant column at front
            allCoeffs = new Vector<T>(Coefficients.Length + 1);
            allCoeffs[0] = Intercept;
            for (int i = 0; i < Coefficients.Length; i++)
            {
                allCoeffs[i + 1] = Coefficients[i];
            }
        }
        else
        {
            allCoeffs = Coefficients;
        }

        return input.Multiply(allCoeffs);
    }

    /// <summary>
    /// Makes predictions with uncertainty estimates on new data using the trained Bayesian regression model.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample to predict.</param>
    /// <returns>A tuple containing the mean predictions and their variances.</returns>
    /// <remarks>
    /// <para>
    /// This method performs predictions using the full posterior distribution over the model parameters,
    /// providing both the mean prediction and the variance for each prediction. The variance represents
    /// the uncertainty in the prediction and is composed of two terms:
    /// 1. The uncertainty due to the model parameters (epistemic uncertainty)
    /// 2. The irreducible noise in the data (aleatoric uncertainty)
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions AND tells you how confident the model
    /// is about each prediction.
    /// 
    /// For example, if predicting house prices:
    /// - A prediction of "$300,000 ± $10,000" is more confident than
    /// - A prediction of "$300,000 ± $50,000"
    /// 
    /// The method returns two values for each input:
    /// - Mean: The best guess prediction (same as the regular Predict method)
    /// - Variance: A measure of uncertainty or confidence in that prediction
    /// 
    /// This uncertainty comes from two sources:
    /// - Parameter uncertainty: How confident the model is about its learned coefficients
    /// - Noise uncertainty: The inherent randomness in the data that can't be explained
    /// 
    /// Having uncertainty estimates is extremely valuable for decision-making,
    /// risk assessment, and understanding when to trust or question the model's predictions.
    /// </para>
    /// </remarks>
    public (Vector<T> Mean, Vector<T> Variance) PredictWithUncertainty(Matrix<T> input)
    {
        if (Options.UseIntercept)
        {
            input = input.AddConstantColumn(NumOps.One);
        }

        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            input = ApplyKernel(input);
        }

        var mean = Predict(input);
        var variance = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var x = input.GetRow(i);
            var xCov = x.DotProduct(_posteriorCovariance.Multiply(x));
            variance[i] = NumOps.Add(xCov, NumOps.FromDouble(1.0 / _bayesOptions.Beta));
        }

        return (mean, variance);
    }

    /// <summary>
    /// Applies the selected kernel transformation to the input matrix.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>The transformed features matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the kernel transformation specified in the options to the input features matrix.
    /// Kernel transformations enable non-linear regression by implicitly mapping the features to a higher-dimensional space.
    /// The method dispatches to the appropriate specific kernel implementation based on the selected kernel type.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your input data according to the selected kernel.
    /// 
    /// A kernel is like a special lens that transforms how the model sees relationships in your data.
    /// Different kernels are good for different types of patterns:
    /// - Linear: Good for simple straight-line relationships
    /// - RBF (Radial Basis Function): Good for smooth, curved relationships
    /// - Polynomial: Good for relationships with curves and interactions
    /// - Sigmoid: Creates S-shaped patterns
    /// - Laplacian: Similar to RBF but handles outliers better
    /// 
    /// This transformation allows the model to capture complex relationships that couldn't be
    /// represented with a simple linear equation.
    /// </para>
    /// </remarks>
    private Matrix<T> ApplyKernel(Matrix<T> input)
    {
        return _bayesOptions.KernelType switch
        {
            KernelType.RBF => ApplyRBFKernel(input),
            KernelType.Polynomial => ApplyPolynomialKernel(input),
            KernelType.Sigmoid => ApplySigmoidKernel(input),
            KernelType.Linear => input,// Linear kernel (no change)
            KernelType.Laplacian => ApplyLaplacianKernel(input),
            _ => throw new ArgumentException($"Unsupported kernel type: {_bayesOptions.KernelType}"),
        };
    }

    /// <summary>
    /// Applies the Laplacian kernel transformation to the input matrix.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>The kernel matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Laplacian kernel matrix for the input features. The Laplacian kernel is defined as
    /// K(x, y) = exp(-? * |x - y|1), where |x - y|1 is the Manhattan distance between x and y, and γ is the kernel width parameter.
    /// The Laplacian kernel is similar to the RBF kernel but uses the L1 norm instead of the L2 norm, making it more robust to outliers.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data using the Laplacian kernel.
    /// 
    /// The Laplacian kernel works by measuring how similar each data point is to every other data point,
    /// using a measure called the "Manhattan distance" (like walking on a city grid - you can only move
    /// along streets, not diagonally through buildings).
    /// 
    /// This kernel is particularly good at handling outliers (unusual data points that are far from the others)
    /// because it doesn't penalize large distances as severely as some other kernels.
    /// 
    /// The LaplacianGamma parameter controls how quickly similarity decreases with distance:
    /// - Higher values make distant points seem very different
    /// - Lower values make even distant points seem somewhat similar
    /// </para>
    /// </remarks>
    private Matrix<T> ApplyLaplacianKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var output = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.LaplacianGamma); // Kernel width parameter

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++) // We only need to compute half of the matrix due to symmetry
            {
                if (i == j)
                {
                    output[i, j] = NumOps.One; // The kernel of a point with itself is always 1
                }
                else
                {
                    var distance = CalculateManhattanDistance(input.GetRow(i), input.GetRow(j));
                    var kernelValue = NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, distance)));
                    output[i, j] = kernelValue;
                    output[j, i] = kernelValue; // The kernel matrix is symmetric
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Calculates the Manhattan distance between two vectors.
    /// </summary>
    /// <param name="x">The first vector.</param>
    /// <param name="y">The second vector.</param>
    /// <returns>The Manhattan distance between x and y.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Manhattan distance (also known as L1 distance or taxicab distance) between two vectors.
    /// The Manhattan distance is the sum of the absolute differences between corresponding elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the "city block" distance between two points.
    /// 
    /// Imagine a city with a grid layout like Manhattan, New York. To get from one point to another,
    /// you can only travel along the grid (streets), not diagonally through blocks.
    /// 
    /// For example, to get from (1,1) to (4,5):
    /// - You need to go 3 blocks east (from 1 to 4)
    /// - You need to go 4 blocks north (from 1 to 5)
    /// - Total Manhattan distance: 3 + 4 = 7 blocks
    /// 
    /// This is different from the straight-line (Euclidean) distance, which would be shorter
    /// but wouldn't follow the street grid.
    /// </para>
    /// </remarks>
    private T CalculateManhattanDistance(Vector<T> x, Vector<T> y)
    {
        var diff = (Vector<T>)Engine.Subtract(x, y);
        var absDiff = (Vector<T>)Engine.Abs(diff);
        return Engine.Sum(absDiff);
    }

    /// <summary>
    /// Applies the Radial Basis Function (RBF) kernel transformation to the input matrix.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>The kernel matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the RBF kernel matrix for the input features. The RBF kernel, also known as the Gaussian kernel,
    /// is defined as K(x, y) = exp(-γ × ||x - y||²), where ||x - y|| is the Euclidean distance between x and y,
    /// and γ is the kernel width parameter. The RBF kernel is one of the most widely used kernels due to its smooth properties
    /// and ability to capture non-linear relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data using the RBF (Radial Basis Function) kernel.
    /// 
    /// The RBF kernel (also called the Gaussian kernel) works by measuring how similar each data point is
    /// to every other data point, based on their distance from each other. Points that are close together
    /// are considered very similar, while points that are far apart are considered very different.
    /// 
    /// This kernel is particularly good at capturing smooth, curved relationships in your data.
    /// 
    /// The Gamma parameter controls how quickly similarity decreases with distance:
    /// - Higher gamma values mean that only very close points are considered similar
    /// - Lower gamma values mean that even somewhat distant points are considered similar
    /// 
    /// The RBF kernel is often a good default choice when you're not sure which kernel to use.
    /// </para>
    /// </remarks>
    private Matrix<T> ApplyRBFKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var diff = input.GetRow(i).Subtract(input.GetRow(j));
                var squaredDistance = diff.DotProduct(diff);
                var value = NumOps.Exp(NumOps.Multiply(NumOps.Negate(gamma), squaredDistance));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the Polynomial kernel transformation to the input matrix.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>The kernel matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Polynomial kernel matrix for the input features. The Polynomial kernel is defined as
    /// K(x, y) = (? * x²y + coef0)^degree, where x²y is the dot product between x and y, ? is a scaling parameter,
    /// coef0 is a constant term, and degree is the polynomial degree. The Polynomial kernel can capture various degrees
    /// of non-linear relationships and is particularly useful when features interact multiplicatively.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data using the Polynomial kernel.
    /// 
    /// The Polynomial kernel captures interactions between features raised to a certain power (degree).
    /// It's particularly useful when you believe the relationship in your data involves products
    /// of features rather than just their individual effects.
    /// 
    /// For example, in predicting crop yield, the combination of both temperature AND rainfall
    /// might be more important than either factor alone. The Polynomial kernel can capture
    /// these kinds of interactions.
    /// 
    /// Parameters that control this kernel:
    /// - PolynomialDegree: Higher degrees capture more complex interactions but may overfit
    /// - Gamma: Controls the influence of higher vs. lower degree terms
    /// - Coef0: Adds a constant term; higher values make the kernel less sensitive to changes in input
    /// 
    /// A polynomial degree of 1 is equivalent to linear regression, while higher degrees
    /// capture progressively more complex relationships.
    /// </para>
    /// </remarks>
    private Matrix<T> ApplyPolynomialKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);
        var coef0 = NumOps.FromDouble(_bayesOptions.Coef0);
        var degree = _bayesOptions.PolynomialDegree;

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var dot = input.GetRow(i).DotProduct(input.GetRow(j));
                var value = NumOps.Power(NumOps.Add(NumOps.Multiply(gamma, dot), coef0), NumOps.FromDouble(degree));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the Sigmoid kernel transformation to the input matrix.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>The kernel matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Sigmoid kernel matrix for the input features. The Sigmoid kernel is defined as
    /// K(x, y) = tanh(? * x²y + coef0), where x²y is the dot product between x and y, ? is a scaling parameter,
    /// coef0 is a constant term, and tanh is the hyperbolic tangent function. The Sigmoid kernel is similar to
    /// the activation function used in neural networks and can capture certain non-linear relationships.
    /// Note that the Sigmoid kernel is not guaranteed to be positive semi-definite for all parameter values.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data using the Sigmoid kernel.
    /// 
    /// The Sigmoid kernel (also called the Hyperbolic Tangent kernel) creates an S-shaped transformation
    /// of your data, similar to the activation functions used in neural networks. It produces a value
    /// between -1 and 1 for each pair of points.
    /// 
    /// This kernel can capture certain types of non-linear relationships, particularly those with
    /// threshold effects or saturation (where the relationship levels off at certain extremes).
    /// 
    /// Parameters that control this kernel:
    /// - Gamma: Controls the steepness of the S-curve
    /// - Coef0: Shifts the curve horizontally
    /// 
    /// The Sigmoid kernel is less commonly used than RBF or Polynomial kernels in regression,
    /// but can be effective for certain types of data, especially when there are clear
    /// threshold effects in your variables.
    /// </para>
    /// </remarks>
    private Matrix<T> ApplySigmoidKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);
        var coef0 = NumOps.FromDouble(_bayesOptions.Coef0);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var dot = input.GetRow(i).DotProduct(input.GetRow(j));
                var value = MathHelper.Tanh(NumOps.Add(NumOps.Multiply(gamma, dot), coef0));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the model type for serialization purposes.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the type identifier for the Bayesian regression model, which is used
    /// during serialization and deserialization to correctly reconstruct the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies the type of model for saving and loading purposes.
    /// 
    /// When you save a model to a file or database, the system needs to know what kind of model it is
    /// in order to load it correctly later. This method provides that identification.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.BayesianRegression;
    }

    /// <summary>
    /// Creates a new instance of the Bayesian regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="BayesianRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Bayesian regression model that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your model that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar models
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create and return a new instance with the same configuration
        return new BayesianRegression<T>(_bayesOptions, Regularization);
    }
}
