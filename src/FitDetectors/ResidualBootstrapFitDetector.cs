namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality using residual bootstrap resampling techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine if your machine learning model is a good fit for your data.
/// It uses a technique called "bootstrap resampling" which creates many simulated datasets by
/// randomly reusing the errors (residuals) from your original model. This helps understand if your
/// model is too complex (overfit), too simple (underfit), or just right (good fit).
/// </para>
/// </remarks>
public class ResidualBootstrapFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the residual bootstrap fit detector.
    /// </summary>
    private readonly ResidualBootstrapFitDetectorOptions _options;

    /// <summary>
    /// Random number generator used for bootstrap resampling.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the ResidualBootstrapFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If null, default settings are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a new detector that will analyze your model's performance.
    /// You can customize how it works by providing options, or just use the default settings.
    /// </para>
    /// </remarks>
    public ResidualBootstrapFitDetector(ResidualBootstrapFitDetectorOptions? options = null)
    {
        _options = options ?? new ResidualBootstrapFitDetectorOptions();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Analyzes model evaluation data to determine the fit type and provide recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model and tells you:
    /// 1. Whether your model is a good fit, overfit, or underfit
    /// 2. How confident the detector is in this assessment
    /// 3. What steps you might take to improve your model
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
    /// Determines the type of fit (overfit, underfit, or good fit) based on bootstrap analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>The determined fit type (Overfit, Underfit, or GoodFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method decides if your model is:
    /// - Overfit: Too complex, memorizing training data rather than learning patterns
    /// - Underfit: Too simple, missing important patterns in the data
    /// - Good fit: Just right, capturing the important patterns without memorizing noise
    /// 
    /// It makes this decision by comparing your model's error to errors from many simulated datasets.
    /// If your model's error is significantly different from what we'd expect by chance,
    /// it suggests there might be a problem with how well your model fits the data.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var bootstrapMSEs = PerformResidualBootstrap(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var meanBootstrapMSE = StatisticsHelper<T>.CalculateMean(bootstrapMSEs);
        var stdDevBootstrapMSE = StatisticsHelper<T>.CalculateStandardDeviation(bootstrapMSEs);

        var zScore = NumOps.Divide(NumOps.Subtract(originalMSE, meanBootstrapMSE), stdDevBootstrapMSE);

        if (NumOps.GreaterThan(zScore, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(zScore, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    /// <summary>
    /// Calculates how confident the detector is in its assessment of the model's fit type.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sure the detector is about its assessment of your model.
    /// Think of it like a percentage of certainty.
    /// 
    /// The confidence is calculated by comparing your model's error to the errors from many simulated datasets.
    /// If your model's error is very similar to what we'd expect by chance, the detector is more confident
    /// that your model has a good fit. If the error is unusual (much higher or lower than expected),
    /// the detector is less confident in its assessment.
    /// 
    /// The final score is between 0 and 1, where 1 means completely confident and 0 means not confident at all.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var bootstrapMSEs = PerformResidualBootstrap(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var meanBootstrapMSE = StatisticsHelper<T>.CalculateMean(bootstrapMSEs);
        var stdDevBootstrapMSE = StatisticsHelper<T>.CalculateStandardDeviation(bootstrapMSEs);

        var zScore = NumOps.Abs(NumOps.Divide(NumOps.Subtract(originalMSE, meanBootstrapMSE), stdDevBootstrapMSE));

        return NumOps.Subtract(NumOps.One, NumOps.Divide(zScore, NumOps.FromDouble(3.0))); // Normalize to [0, 1]
    }

    /// <summary>
    /// Generates practical recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type (Overfit, Underfit, or GoodFit).</param>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on
    /// whether it's overfit, underfit, or a good fit.
    /// 
    /// - For overfit models: Suggestions to make the model simpler or add more training data
    /// - For underfit models: Suggestions to make the model more complex or improve features
    /// - For good fit models: Suggestions to maintain performance and possibly enhance it further
    /// 
    /// These recommendations are starting points that you can try to improve your model's performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization to reduce model complexity.");
                recommendations.Add("Try reducing the number of features or using feature selection techniques.");
                recommendations.Add("Increase the size of the training dataset if possible.");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider increasing model complexity by adding more features or interactions.");
                recommendations.Add("Reduce regularization if it's being used.");
                recommendations.Add("Ensure that all relevant features are included in the model.");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit based on residual bootstrap analysis.");
                recommendations.Add("Continue monitoring performance on new, unseen data.");
                recommendations.Add("Consider ensemble methods or advanced techniques to potentially improve performance further.");
                break;
        }

        return recommendations;
    }

    /// <summary>
    /// Performs residual bootstrap resampling to generate multiple simulated datasets and calculate their error metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and predictions.</param>
    /// <returns>A vector of Mean Squared Error (MSE) values from bootstrap samples.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the sample size is too small for reliable bootstrap analysis.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates many "what-if" scenarios to test how reliable your model is.
    /// 
    /// Here's how it works in simple terms:
    /// 1. It calculates the errors (residuals) between what your model predicted and the actual values
    /// 2. It creates many new fake datasets by randomly reusing these errors
    /// 3. For each fake dataset, it calculates how much error the model would have
    /// 4. It returns all these error measurements so we can analyze if your model is behaving as expected
    /// 
    /// This helps us understand if your model's performance is stable or if it might be influenced too much
    /// by specific data points in your original dataset.
    /// </para>
    /// <para>
    /// Technical details:
    /// - Residual bootstrap resampling preserves the structure of the original data while simulating variation
    /// - Each bootstrap sample combines the original predictions with randomly resampled residuals
    /// - The resulting MSE distribution helps assess model stability and fit quality
    /// </para>
    /// </remarks>
    private Vector<T> PerformResidualBootstrap(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Extract actual values and model predictions from the evaluation data
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        var sampleSize = actual.Length;

        // Check if we have enough data for reliable bootstrap analysis
        if (sampleSize < _options.MinSampleSize)
        {
            throw new InvalidOperationException($"Sample size ({sampleSize}) is too small for residual bootstrap. Minimum required: {_options.MinSampleSize}");
        }

        // Calculate residuals (differences between actual and predicted values)
        var residuals = new Vector<T>(sampleSize);
        for (int i = 0; i < sampleSize; i++)
        {
            residuals[i] = NumOps.Subtract(actual[i], predicted[i]);
        }

        // Create a vector to store MSE values from each bootstrap sample
        var bootstrapMSEs = new Vector<T>(_options.NumBootstrapSamples);

        // Generate multiple bootstrap samples and calculate their MSEs
        for (int i = 0; i < _options.NumBootstrapSamples; i++)
        {
            var bootstrapSample = new Vector<T>(sampleSize);
            var bootstrapPredicted = new Vector<T>(sampleSize);

            // Create a bootstrap sample by adding randomly selected residuals to predictions
            for (int j = 0; j < sampleSize; j++)
            {
                int randomIndex = _random.Next(sampleSize);
                bootstrapSample[j] = NumOps.Add(predicted[j], residuals[randomIndex]);
                bootstrapPredicted[j] = predicted[j];
            }

            // Calculate and store the MSE for this bootstrap sample
            bootstrapMSEs[i] = StatisticsHelper<T>.CalculateMeanSquaredError(bootstrapSample, bootstrapPredicted);
        }

        return bootstrapMSEs;
    }
}
