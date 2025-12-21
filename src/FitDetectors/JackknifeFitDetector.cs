namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit using the jackknife resampling technique.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine if your machine learning model is a good fit for your data
/// using a technique called "jackknife resampling". This is like testing your model multiple times, 
/// each time leaving out one data point, to see how stable your model's performance is. 
/// 
/// If your model performs very differently when certain data points are removed, it might be overfitting
/// (memorizing the data rather than learning patterns). If it performs consistently regardless of which
/// points are removed, it's likely a more robust model.
/// </para>
/// </remarks>
public class JackknifeFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the jackknife fit detector.
    /// </summary>
    private readonly JackknifeFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the JackknifeFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If null, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new jackknife fit detector. You can provide
    /// custom settings through the options parameter, or leave it empty to use the default settings.
    /// </para>
    /// </remarks>
    public JackknifeFitDetector(JackknifeFitDetectorOptions? options = null)
    {
        _options = options ?? new JackknifeFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes a model's performance data to determine the type of fit and provide recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to check if your model is a good fit.
    /// You provide your model's performance data, and it returns:
    /// 1. What type of fit your model has (good fit, overfit, or underfit)
    /// 2. How confident the detector is in its assessment
    /// 3. Specific recommendations to improve your model
    /// 
    /// This method coordinates the entire analysis process by calling other specialized methods.
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
    /// Determines whether the model is overfitting, underfitting, or has a good fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The detected fit type (GoodFit, Overfit, or Underfit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method figures out if your model is:
    /// - Overfitting: Your model is too complex and has memorized the training data instead of learning general patterns
    /// - Underfitting: Your model is too simple and isn't capturing important patterns in the data
    /// - Good fit: Your model has the right balance of complexity
    /// 
    /// It works by comparing how your model performs on the original data versus how it performs
    /// when using the jackknife technique (testing with different subsets of data). The difference
    /// between these performances helps determine the fit type.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var jackknifeMSE = PerformJackknifeResampling(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var relativeDifference = NumOps.Divide(NumOps.Subtract(jackknifeMSE, originalMSE), originalMSE);

        if (NumOps.GreaterThan(relativeDifference, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(relativeDifference, NumOps.Negate(NumOps.FromDouble(_options.UnderfitThreshold))))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    /// <summary>
    /// Calculates how confident the detector is in its assessment of the model fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sure the detector is about its assessment of your model.
    /// 
    /// The confidence is based on how much the model's performance changes when using jackknife resampling
    /// compared to the original performance. If there's a big difference, the confidence will be lower.
    /// If there's little difference, the confidence will be higher.
    /// 
    /// The result is a number between 0 and 1:
    /// - Values close to 1 mean high confidence (you can trust the assessment)
    /// - Values close to 0 mean low confidence (the assessment might not be reliable)
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var jackknifeMSE = PerformJackknifeResampling(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var relativeDifference = NumOps.Abs(NumOps.Divide(NumOps.Subtract(jackknifeMSE, originalMSE), originalMSE));

        return NumOps.Subtract(NumOps.One, NumOps.LessThan(relativeDifference, NumOps.One) ? relativeDifference : NumOps.One);
    }

    /// <summary>
    /// Generates specific recommendations based on the detected fit type of the model.
    /// </summary>
    /// <param name="fitType">The type of fit detected (GoodFit, Overfit, or Underfit).</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of string recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of practical suggestions to help you improve your model
    /// based on what type of fit was detected. Think of it as personalized advice for your specific situation:
    /// 
    /// - If your model is "Overfitting", it suggests ways to make your model more generalized (less memorizing, more learning).
    /// - If your model is "Underfitting", it suggests ways to make your model more powerful to capture patterns better.
    /// - If your model has a "Good Fit", it confirms you're on the right track and suggests ways to maintain or slightly improve performance.
    /// 
    /// Terminology explained:
    /// - Regularization: A technique that prevents models from becoming too complex by adding penalties for complexity
    /// - Feature selection: The process of choosing which input variables (features) to include in your model
    /// - Ensemble methods: Combining multiple models together to improve performance and stability
    /// - Epochs: Complete passes through the training dataset during model training
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization or reducing model complexity.");
                recommendations.Add("Try collecting more training data if possible.");
                recommendations.Add("Implement feature selection to reduce the number of input variables.");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider increasing model complexity or reducing regularization.");
                recommendations.Add("Explore additional relevant features that could improve model performance.");
                recommendations.Add("Increase the number of training iterations or epochs if applicable.");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit based on jackknife resampling.");
                recommendations.Add("Continue monitoring performance on new, unseen data.");
                recommendations.Add("Consider ensemble methods to potentially improve performance further.");
                break;
        }

        return recommendations;
    }

    /// <summary>
    /// Performs jackknife resampling to evaluate model stability and calculate the average Mean Squared Error (MSE).
    /// </summary>
    /// <param name="evaluationData">Data containing the model's actual and predicted values.</param>
    /// <returns>The average Mean Squared Error (MSE) across all jackknife samples.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the sample size is too small for jackknife resampling.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements the "jackknife" technique, which is a way to test how stable your model is.
    /// 
    /// Here's how it works in simple terms:
    /// 1. It takes your original data (actual values and what your model predicted)
    /// 2. It creates multiple new datasets, each one missing a different data point
    /// 3. For each of these datasets, it calculates how much error the model has
    /// 4. Finally, it averages all these errors to give you a single number
    /// 
    /// This helps us understand if your model's performance depends too much on specific data points.
    /// If removing certain points drastically changes the error, your model might be "memorizing" 
    /// those points instead of learning general patterns.
    /// 
    /// Terminology explained:
    /// - MSE (Mean Squared Error): A way to measure prediction error by averaging the squared differences 
    ///   between predicted and actual values. Lower values mean better predictions.
    /// - Resampling: Creating new datasets from your original data by selecting different subsets of points.
    /// - Jackknife: A specific resampling technique where you leave out one data point at a time.
    /// </para>
    /// </remarks>
    private T PerformJackknifeResampling(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        var sampleSize = actual.Length;

        if (sampleSize < _options.MinSampleSize)
        {
            throw new InvalidOperationException($"Sample size ({sampleSize}) is too small for jackknife resampling. Minimum required: {_options.MinSampleSize}");
        }

        var jackknifeMSEs = new Vector<T>(Math.Min(sampleSize, _options.MaxIterations));

        for (int i = 0; i < Math.Min(sampleSize, _options.MaxIterations); i++)
        {
            var jackknifeSample = new Vector<T>(sampleSize - 1);
            var jackknifePredicted = new Vector<T>(sampleSize - 1);
            int index = 0;

            for (int j = 0; j < sampleSize; j++)
            {
                if (j != i)
                {
                    jackknifeSample[index] = actual[j];
                    jackknifePredicted[index] = predicted[j];
                    index++;
                }
            }

            var mse = StatisticsHelper<T>.CalculateMeanSquaredError(jackknifeSample, jackknifePredicted);
            jackknifeMSEs[i] = mse;
        }

        return StatisticsHelper<T>.CalculateMean(jackknifeMSEs);
    }
}
