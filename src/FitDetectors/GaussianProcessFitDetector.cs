namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that uses Gaussian Process regression to analyze model uncertainty and performance.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gaussian Process regression is a probabilistic machine learning technique that 
/// not only makes predictions but also provides uncertainty estimates for those predictions. This detector 
/// uses these uncertainty estimates to assess how well a model fits the data.
/// </para>
/// <para>
/// By analyzing both the accuracy of predictions (RMSE) and the uncertainty in those predictions, this 
/// detector can identify issues like overfitting (high confidence but poor performance) or underfitting 
/// (high uncertainty and poor performance).
/// </para>
/// </remarks>
public class GaussianProcessFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Gaussian Process fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets model performance and 
    /// uncertainty, including thresholds for determining different types of model fit and parameters 
    /// for the Gaussian Process regression.
    /// </remarks>
    private readonly GaussianProcessFitDetectorOptions _options;

    private Vector<T> _meanPrediction;
    private Vector<T> _variancePrediction;
    private T _averageUncertainty;
    private T _rmse;

    /// <summary>
    /// Initializes a new instance of the GaussianProcessFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new Gaussian Process fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Thresholds for determining good fit, overfit, and underfit based on RMSE</description></item>
    /// <item><description>Thresholds for high and low uncertainty</description></item>
    /// <item><description>Parameters for the Gaussian Process regression (length scale, noise variance)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public GaussianProcessFitDetector(GaussianProcessFitDetectorOptions? options = null)
    {
        _options = options ?? new GaussianProcessFitDetectorOptions();
        _meanPrediction = Vector<T>.Empty();
        _variancePrediction = Vector<T>.Empty();
        _averageUncertainty = NumOps.Zero;
        _rmse = NumOps.Zero;
    }

    /// <summary>
    /// Detects the fit type of a model based on Gaussian Process regression analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance and uncertainty using 
    /// Gaussian Process regression to determine if it's underfitting, overfitting, or has a good fit.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model is underfitting, overfitting, has a good fit, or is unstable</description></item>
    /// <item><description>ConfidenceLevel: How confident the detector is in its assessment</description></item>
    /// <item><description>Recommendations: Suggestions for improving the model based on the detected fit type</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    /// <summary>
    /// Determines the fit type based on Gaussian Process regression analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on Gaussian Process regression analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses Gaussian Process regression to analyze your model's 
    /// performance and uncertainty, then determines what type of fit your model has.
    /// </para>
    /// <para>
    /// The method looks at:
    /// <list type="bullet">
    /// <item><description>RMSE (Root Mean Squared Error): How accurate the model's predictions are</description></item>
    /// <item><description>Average uncertainty: How confident the model is in its predictions</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Based on these metrics, it categorizes the model as having:
    /// <list type="bullet">
    /// <item><description>Good Fit: Low RMSE and low uncertainty (accurate predictions with high confidence)</description></item>
    /// <item><description>Overfit: High RMSE but low uncertainty (inaccurate predictions with high confidence)</description></item>
    /// <item><description>Underfit: High RMSE and high uncertainty (inaccurate predictions with low confidence)</description></item>
    /// <item><description>Unstable: Any other pattern that doesn't fit the above categories</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        (_meanPrediction, _variancePrediction) = PerformGaussianProcessRegression(evaluationData);
        _averageUncertainty = _variancePrediction.Average();

        // Calculate RMSE between model's predictions and actual values to measure model fit quality
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        _rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(actual, predicted);

        if (NumOps.LessThan(_rmse, NumOps.FromDouble(_options.GoodFitThreshold)) &&
            NumOps.LessThan(_averageUncertainty, NumOps.FromDouble(_options.LowUncertaintyThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(_rmse, NumOps.FromDouble(_options.OverfitThreshold)) &&
                 NumOps.LessThan(_averageUncertainty, NumOps.FromDouble(_options.LowUncertaintyThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.GreaterThan(_rmse, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.GreaterThan(_averageUncertainty, NumOps.FromDouble(_options.HighUncertaintyThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the Gaussian Process-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on the model's accuracy and uncertainty.
    /// </para>
    /// <para>
    /// The method calculates two factors:
    /// <list type="bullet">
    /// <item><description>Uncertainty factor: Higher when the model's uncertainty is low</description></item>
    /// <item><description>RMSE factor: Higher when the model's error is low</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// These factors are multiplied together to produce a confidence score between 0 and 1, with higher 
    /// values indicating greater confidence in the fit assessment.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Normalize confidence level to [0, 1]
        var uncertaintyFactor = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, _averageUncertainty));
        var rmseFactor = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, _rmse));

        return NumOps.Multiply(uncertaintyFactor, rmseFactor);
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and Gaussian Process analysis.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model based on Gaussian Process analysis.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Good Fit: The model is performing well but may benefit from further validation and exploration</description></item>
    /// <item><description>Overfit: The model is too confident in incorrect predictions and needs to be more flexible</description></item>
    /// <item><description>Underfit: The model is uncertain and inaccurate, suggesting it's too simple</description></item>
    /// <item><description>Unstable: The model's behavior is inconsistent and may have numerical issues</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The recommendations are specific to Gaussian Process models and focus on kernel selection, 
    /// hyperparameter tuning, and data preprocessing.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The Gaussian Process model appears to be well-fitted. Consider:");
                recommendations.Add("1. Validating the model on new, unseen data to ensure generalization.");
                recommendations.Add("2. Exploring different kernel functions to potentially improve performance.");
                recommendations.Add("3. Investigating areas of higher uncertainty in the predictions.");
                break;
            case FitType.Overfit:
                recommendations.Add("The Gaussian Process model may be overfitting. Consider:");
                recommendations.Add("1. Increasing the noise parameter in the kernel to reduce sensitivity to individual data points.");
                recommendations.Add("2. Using a simpler kernel function to reduce model complexity.");
                recommendations.Add("3. Collecting more training data, especially in areas where the model shows high confidence but poor performance.");
                break;
            case FitType.Underfit:
                recommendations.Add("The Gaussian Process model may be underfitting. Consider:");
                recommendations.Add("1. Using a more complex kernel function to capture more intricate patterns in the data.");
                recommendations.Add("2. Reducing the noise parameter in the kernel if you believe your data is relatively noise-free.");
                recommendations.Add("3. Feature engineering or adding more relevant features to provide more information to the model.");
                break;
            case FitType.Unstable:
                recommendations.Add("The Gaussian Process model appears unstable. Consider:");
                recommendations.Add("1. Checking for numerical instabilities in the kernel matrix calculations.");
                recommendations.Add("2. Normalizing or standardizing your input features.");
                recommendations.Add("3. Using a different optimization method for hyperparameter tuning.");
                break;
        }

        return recommendations;
    }

    /// <summary>
    /// Performs Gaussian Process regression on the model's residuals.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A tuple containing the GP mean predictions of residuals and variance predictions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method implements Gaussian Process regression on the model's
    /// residuals (differences between predictions and actual values) to detect patterns that indicate
    /// overfitting, underfitting, or other fit issues.
    /// </para>
    /// <para>
    /// The method performs the following steps:
    /// <list type="number">
    /// <item><description>Calculate residuals from the model's predictions and actual values</description></item>
    /// <item><description>Optimize the hyperparameters (length scale and noise variance) using grid search</description></item>
    /// <item><description>Calculate the kernel matrix using the optimized hyperparameters</description></item>
    /// <item><description>Perform Cholesky decomposition of the kernel matrix for numerical stability</description></item>
    /// <item><description>Calculate the mean predictions and variance predictions</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The result is a tuple containing:
    /// <list type="bullet">
    /// <item><description>Mean predictions: The GP model's expected residual patterns (should be near zero for good fit)</description></item>
    /// <item><description>Variance predictions: The uncertainty associated with each prediction</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private (Vector<T>, Vector<T>) PerformGaussianProcessRegression(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var X = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features);
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);

        // Calculate residuals: the differences between model's predictions and actual values
        // GP regression on residuals helps detect systematic patterns that indicate fit issues
        var y = actual - predicted;

        // Hyperparameter optimization
        var optimizedHyperparameters = OptimizeHyperparameters(X, y);
        _options.LengthScale = optimizedHyperparameters.LengthScale;
        _options.NoiseVariance = optimizedHyperparameters.NoiseVariance;

        // Calculate kernel matrix with optimized hyperparameters
        var K = CalculateKernelMatrix(X, X);

        // Add noise to the diagonal for numerical stability
        for (int i = 0; i < K.Rows; i++)
        {
            K[i, i] = NumOps.Add(K[i, i], NumOps.FromDouble(_options.NoiseVariance));
        }

        // Perform Cholesky decomposition
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);

        // Solve for alpha: K * alpha = y
        var alpha = choleskyDecomposition.Solve(y);

        // Prepare for predictions (in this case, we're predicting on the same points as the training data)
        var XStar = X;
        var KStar = CalculateKernelMatrix(X, XStar);

        // Calculate mean predictions
        var meanPrediction = KStar.Transpose().Multiply(alpha);

        // Calculate variance predictions
        var variancePrediction = new Vector<T>(XStar.Rows);
        for (int i = 0; i < XStar.Rows; i++)
        {
            var kStarStar = CalculateRBFKernel(XStar.GetRow(i), XStar.GetRow(i));
            var v = choleskyDecomposition.Solve(KStar.GetColumn(i));
            variancePrediction[i] = NumOps.Subtract(kStarStar, v.DotProduct(v));
        }

        return (meanPrediction, variancePrediction);
    }

    /// <summary>
    /// Optimizes the hyperparameters for the Gaussian Process model.
    /// </summary>
    /// <param name="X">Feature matrix.</param>
    /// <param name="y">Target vector.</param>
    /// <returns>Optimized length scale and noise variance hyperparameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method finds the best values for the Gaussian Process model's 
    /// hyperparameters using a technique called grid search.
    /// </para>
    /// <para>
    /// The method tests different combinations of:
    /// <list type="bullet">
    /// <item><description>Length scale: Controls how quickly the correlation between points decreases with distance</description></item>
    /// <item><description>Noise variance: Represents the amount of noise in the data</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// For each combination, it calculates the log likelihood (a measure of how well the model fits the data) 
    /// and selects the combination with the highest log likelihood.
    /// </para>
    /// </remarks>
    private (double LengthScale, double NoiseVariance) OptimizeHyperparameters(Matrix<T> X, Vector<T> y)
    {
        // Implement a simple grid search for hyperparameter optimization
        double bestLengthScale = 1.0;
        double bestNoiseVariance = 0.1;
        T bestLogLikelihood = NumOps.MinValue;

        var lengthScales = new[] { 0.1, 0.5, 1.0, 2.0, 5.0 };
        var noiseVariances = new[] { 0.01, 0.1, 0.5, 1.0 };

        foreach (var ls in lengthScales)
        {
            foreach (var nv in noiseVariances)
            {
                _options.LengthScale = ls;
                _options.NoiseVariance = nv;

                var K = CalculateKernelMatrix(X, X);
                for (int i = 0; i < K.Rows; i++)
                {
                    K[i, i] = NumOps.Add(K[i, i], NumOps.FromDouble(nv));
                }

                var logLikelihood = CalculateLogLikelihood(K, y);

                if (NumOps.GreaterThan(logLikelihood, bestLogLikelihood))
                {
                    bestLogLikelihood = logLikelihood;
                    bestLengthScale = ls;
                    bestNoiseVariance = nv;
                }
            }
        }

        return (bestLengthScale, bestNoiseVariance);
    }

    /// <summary>
    /// Calculates the log likelihood of the Gaussian Process model.
    /// </summary>
    /// <param name="K">Kernel matrix.</param>
    /// <param name="y">Target vector.</param>
    /// <returns>The log likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how well the Gaussian Process model with 
    /// specific hyperparameters fits the data.
    /// </para>
    /// <para>
    /// The log likelihood consists of three components:
    /// <list type="bullet">
    /// <item><description>Data fit term: How well the model's predictions match the actual values</description></item>
    /// <item><description>Complexity penalty: Penalizes overly complex models</description></item>
    /// <item><description>Normalization term: A constant term based on the data size</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Higher log likelihood values indicate better model fit. This value is used during hyperparameter 
    /// optimization to select the best hyperparameters.
    /// </para>
    /// </remarks>
    private T CalculateLogLikelihood(Matrix<T> K, Vector<T> y)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        var L = choleskyDecomposition.L;
        var alpha = choleskyDecomposition.Solve(y);

        var dataFit = NumOps.Multiply(NumOps.FromDouble(-0.5), y.DotProduct(alpha));
        var complexity = NumOps.Multiply(NumOps.FromDouble(-1), L.LogDeterminant());
        var normalization = NumOps.Multiply(NumOps.FromDouble(-0.5 * K.Rows), NumOps.Log(NumOps.FromDouble(2 * Math.PI)));

        return NumOps.Add(NumOps.Add(dataFit, complexity), normalization);
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of points.
    /// </summary>
    /// <param name="X1">First set of points.</param>
    /// <param name="X2">Second set of points.</param>
    /// <returns>The kernel matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates the similarity between all pairs of points 
    /// from two sets using the RBF (Radial Basis Function) kernel.
    /// </para>
    /// <para>
    /// The kernel matrix is a key component of Gaussian Process regression, as it encodes the similarity 
    /// or correlation between data points. Points that are close together in feature space will have 
    /// high kernel values, while distant points will have low kernel values.
    /// </para>
    /// <para>
    /// The resulting matrix has dimensions [X1.Rows × X2.Rows], where each element [i,j] represents 
    /// the similarity between point i from X1 and point j from X2.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = CalculateRBFKernel(X1.GetRow(i), X2.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Calculates the RBF (Radial Basis Function) kernel between two points.
    /// </summary>
    /// <param name="x1">First point.</param>
    /// <param name="x2">Second point.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates the similarity between two points using 
    /// the RBF kernel, also known as the Gaussian kernel.
    /// </para>
    /// <para>
    /// The RBF kernel is defined as:
    /// k(x1, x2) = exp(-||x1 - x2||² / (2 * lengthScale²))
    /// </para>
    /// <para>
    /// Where:
    /// <list type="bullet">
    /// <item><description>||x1 - x2||² is the squared Euclidean distance between the points</description></item>
    /// <item><description>lengthScale is a hyperparameter that controls how quickly the similarity decreases with distance</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The kernel value is always between 0 and 1:
    /// <list type="bullet">
    /// <item><description>1: The points are identical</description></item>
    /// <item><description>Close to 1: The points are very similar</description></item>
    /// <item><description>Close to 0: The points are very different</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private T CalculateRBFKernel(Vector<T> x1, Vector<T> x2)
    {
        // RBF Kernel
        var diff = x1 - x2;
        var squaredDist = diff.DotProduct(diff);

        return NumOps.Exp(NumOps.Negate(NumOps.Divide(squaredDist, NumOps.FromDouble(2 * _options.LengthScale * _options.LengthScale))));
    }
}
