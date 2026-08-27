using AiDotNet.Attributes;
using AiDotNet.Enums;

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
/// <example>
/// <code>
/// // Create a Bayesian regression model with uncertainty estimation
/// var options = new BayesianRegressionOptions&lt;double&gt;();
/// var model = new BayesianRegression&lt;double&gt;(options);
///
/// // Prepare training data: 5 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(5, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10 });
/// var targets = new Vector&lt;double&gt;(new double[] { 2.5, 5.3, 8.1, 10.9, 13.7 });
///
/// // Train the model with Bayesian inference
/// model.Train(features, targets);
///
/// // Predict for a new sample (provides posterior distribution)
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 11, 12 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Pattern Recognition and Machine Learning", "https://www.springer.com/gp/book/9780387310732")]
[CustomSerializationFormat]
public partial class BayesianRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Options specific to Bayesian regression.
    /// </summary>
    private readonly BayesianRegressionOptions<T> _bayesOptions;

    /// <summary>
    /// The covariance matrix of the posterior distribution over model parameters.
    /// </summary>
    [Buffer]
    private Matrix<T> _posteriorCovariance;

    /// <summary>Training features retained as kernel centres for non-linear prediction.</summary>
    [Buffer(Availability = Models.Parameters.ParameterAvailability.Fit)]
    private Matrix<T> _kernelTrainingFeatures = new(0, 0);

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
    /// Bayesian regression computes its posterior analytically, so the optimizer must not inject
    /// random starting parameters.
    /// </summary>
    /// <remarks>
    /// This used to be expressed as <c>ParameterCount =&gt; 0</c>, because a zero count makes the
    /// inherited <c>SupportsParameterInitialization</c> return false. That works, but it overloads a
    /// COUNT to carry a CAPABILITY: the model does have parameters — the base getter returns its
    /// coefficients and intercept — so the count contradicted the vector, and anything pairing the
    /// two by length saw a model with parameters it claimed not to have. Saying "no injection"
    /// directly leaves the count free to tell the truth. <c>LinearDiscriminantAnalysis</c> already
    /// used this form.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

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
        TrainingFeatureCount = d;

        // This method previously fitted ORDINARY LEAST SQUARES and returned immediately. The
        // guard that followed it was written as a condition but is always true for any real
        // problem, so it acted as an unconditional return and left the real estimation below
        // unreachable: callers received a plain linear least-squares fit from a model named for a
        // different algorithm. The real estimation now runs.

        if (_bayesOptions.KernelType == KernelType.Linear)
        {
            _kernelTrainingFeatures = new Matrix<T>(0, 0);
            if (Options.UseIntercept)
            {
                x = x.AddConstantColumn(NumOps.One);
            }
        }
        else
        {
            // Kernel Bayesian regression operates in the n-dimensional dual feature space. Keep
            // an owned copy of the centres so prediction can build K(test, train), not K(test,test).
            _kernelTrainingFeatures = x.Clone();
            x = ApplyKernel(_kernelTrainingFeatures);
            if (Options.UseIntercept)
            {
                x = x.AddConstantColumn(NumOps.One);
            }
        }
        d = x.Columns;

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
        Matrix<T> design = CreatePredictionDesign(input);
        var predictions = design.Multiply(Coefficients);
        for (int i = 0; i < predictions.Length; i++)
            predictions[i] = NumOps.Add(predictions[i], Intercept);
        return predictions;
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
        // Call Predict with original input - it handles its own augmentation
        var mean = Predict(input);

        // Now augment input for variance calculation
        var augmentedInput = CreatePredictionDesign(input);
        if (Options.UseIntercept)
        {
            augmentedInput = augmentedInput.AddConstantColumn(NumOps.One);
        }

        var variance = new Vector<T>(augmentedInput.Rows);

        for (int i = 0; i < augmentedInput.Rows; i++)
        {
            var x = augmentedInput.GetRow(i);
            var xCov = x.DotProduct(_posteriorCovariance.Multiply(x));
            variance[i] = NumOps.Add(xCov, NumOps.FromDouble(1.0 / _bayesOptions.Beta));
        }

        return (mean, variance);
    }

    private Matrix<T> CreatePredictionDesign(Matrix<T> input)
    {
        if (input.Columns != TrainingFeatureCount)
        {
            throw new ArgumentException(
                $"Prediction input has {input.Columns} features; expected {TrainingFeatureCount}.",
                nameof(input));
        }

        if (_bayesOptions.KernelType == KernelType.Linear)
        {
            return input;
        }
        if (_kernelTrainingFeatures.Rows == 0)
        {
            throw new InvalidOperationException(
                "The non-linear Bayesian model has no fitted kernel centres. Train or deserialize it first.");
        }

        return ApplyCrossKernel(input, _kernelTrainingFeatures);
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
        return _bayesOptions.KernelType == KernelType.Linear
            ? input
            : ApplyCrossKernel(input, input);
    }

    /// <summary>Computes K(left, right) for prediction against the fitted kernel centres.</summary>
    private Matrix<T> ApplyCrossKernel(Matrix<T> left, Matrix<T> right)
    {
        if (left.Columns != right.Columns)
        {
            throw new ArgumentException("Kernel operands must have the same feature count.");
        }

        var result = new Matrix<T>(left.Rows, right.Rows);
        for (int i = 0; i < left.Rows; i++)
        {
            Vector<T> leftRow = left.GetRow(i);
            for (int j = 0; j < right.Rows; j++)
            {
                Vector<T> rightRow = right.GetRow(j);
                result[i, j] = _bayesOptions.KernelType switch
                {
                    KernelType.RBF => RbfKernel(leftRow, rightRow),
                    KernelType.Polynomial => PolynomialKernel(leftRow, rightRow),
                    KernelType.Sigmoid => SigmoidKernel(leftRow, rightRow),
                    KernelType.Laplacian => LaplacianKernel(leftRow, rightRow),
                    _ => throw new ArgumentException(
                        $"Unsupported cross-kernel type: {_bayesOptions.KernelType}"),
                };
            }
        }

        return result;
    }

    /// <summary>
    /// Evaluates the Gaussian radial-basis kernel
    /// <c>K(x,y) = exp(-gamma * ||x-y||^2)</c>.
    /// </summary>
    private T RbfKernel(Vector<T> left, Vector<T> right)
    {
        var difference = (Vector<T>)Engine.Subtract(left, right);
        T squaredDistance = difference.DotProduct(difference);
        return NumOps.Exp(NumOps.Negate(
            NumOps.Multiply(NumOps.FromDouble(_bayesOptions.Gamma), squaredDistance)));
    }

    /// <summary>
    /// Evaluates the polynomial kernel
    /// <c>K(x,y) = (gamma * x^T y + coef0)^degree</c>.
    /// </summary>
    private T PolynomialKernel(Vector<T> left, Vector<T> right)
    {
        T scaledDot = NumOps.Multiply(
            NumOps.FromDouble(_bayesOptions.Gamma), left.DotProduct(right));
        return NumOps.Power(
            NumOps.Add(scaledDot, NumOps.FromDouble(_bayesOptions.Coef0)),
            NumOps.FromDouble(_bayesOptions.PolynomialDegree));
    }

    /// <summary>
    /// Evaluates the sigmoid kernel <c>K(x,y) = tanh(gamma * x^T y + coef0)</c>.
    /// </summary>
    /// <remarks>The sigmoid kernel is not positive-semidefinite for every parameter choice.</remarks>
    private T SigmoidKernel(Vector<T> left, Vector<T> right)
    {
        T scaledDot = NumOps.Multiply(
            NumOps.FromDouble(_bayesOptions.Gamma), left.DotProduct(right));
        return MathHelper.Tanh(NumOps.Add(scaledDot, NumOps.FromDouble(_bayesOptions.Coef0)));
    }

    /// <summary>
    /// Evaluates the Laplacian kernel <c>K(x,y) = exp(-gamma * ||x-y||_1)</c>.
    /// </summary>
    private T LaplacianKernel(Vector<T> left, Vector<T> right)
    {
        T distance = CalculateManhattanDistance(left, right);
        return NumOps.Exp(NumOps.Negate(
            NumOps.Multiply(NumOps.FromDouble(_bayesOptions.LaplacianGamma), distance)));
    }

    public override byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);
        WriteMatrix(writer, _posteriorCovariance);
        WriteMatrix(writer, _kernelTrainingFeatures);
        writer.Write(TrainingFeatureCount);
        return stream.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var stream = new MemoryStream(modelData);
        using var reader = new BinaryReader(stream);
        int baseLength = reader.ReadInt32();
        if (baseLength < 0 || baseLength > stream.Length - stream.Position)
            throw new InvalidDataException("Invalid Bayesian regression base-state length.");
        base.Deserialize(reader.ReadBytes(baseLength));
        _posteriorCovariance = ReadMatrix(reader);
        _kernelTrainingFeatures = ReadMatrix(reader);
        TrainingFeatureCount = reader.ReadInt32();
    }

    private void WriteMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int r = 0; r < matrix.Rows; r++)
            for (int c = 0; c < matrix.Columns; c++) writer.Write(NumOps.ToDouble(matrix[r, c]));
    }

    private Matrix<T> ReadMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();
        if (rows < 0 || columns < 0) throw new InvalidDataException("Invalid Bayesian matrix shape.");

        long elementCount;
        long requiredBytes;
        try
        {
            elementCount = checked((long)rows * columns);
            requiredBytes = checked(elementCount * sizeof(double));
        }
        catch (OverflowException exception)
        {
            throw new InvalidDataException("The Bayesian matrix shape is too large.", exception);
        }

        Stream stream = reader.BaseStream;
        if (!stream.CanSeek || requiredBytes > stream.Length - stream.Position)
        {
            throw new InvalidDataException(
                $"The Bayesian matrix payload is truncated: shape {rows}x{columns} requires {requiredBytes} bytes.");
        }

        var matrix = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++) matrix[r, c] = NumOps.FromDouble(reader.ReadDouble());
        return matrix;
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

}
