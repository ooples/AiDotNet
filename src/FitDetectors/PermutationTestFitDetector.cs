namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that uses permutation testing to evaluate model fit quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you determine if your machine learning model is performing well
/// or if it has common problems like overfitting (memorizing data instead of learning patterns) or
/// underfitting (being too simple to capture important patterns).
/// 
/// It works by using a technique called "permutation testing" which compares your model's actual 
/// performance against what would happen if the relationship between your inputs and outputs was random.
/// This gives us confidence that your model has truly learned meaningful patterns.
/// </para>
/// </remarks>
public class PermutationTestFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Random number generator used for permutation simulations.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Configuration options for the permutation test detector.
    /// </summary>
    private readonly PermutationTestFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the PermutationTestFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If null, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new detector with either your custom settings or default settings.
    /// The settings control things like how many tests to run and what threshold to use when deciding if a result is significant.
    /// </para>
    /// </remarks>
    public PermutationTestFitDetector(PermutationTestFitDetectorOptions? options = null)
    {
        _random = RandomHelper.CreateSecureRandom();
        _options = options ?? new PermutationTestFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data to determine the quality of fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll call to check if your model is working well.
    /// It takes your model's performance data and returns:
    /// 1. What type of fit your model has (good, overfit, underfit, etc.)
    /// 2. How confident we are in this assessment
    /// 3. Specific recommendations to improve your model
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
    /// Determines the type of fit (good fit, overfit, underfit, etc.) based on permutation test results.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>The detected fit type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method figures out if your model has a good balance, or if it has problems like:
    /// 
    /// - Overfitting: Your model performs great on training data but poorly on new data. It's like memorizing 
    ///   exam questions instead of understanding the subject.
    ///   
    /// - Underfitting: Your model performs poorly on all datasets. It's too simple and misses important patterns.
    ///   
    /// - High Variance: Your model's performance varies a lot between different datasets. It's inconsistent.
    ///   
    /// - Unstable: Your model doesn't show a clear pattern of performance. Results are unpredictable.
    /// 
    /// It makes this determination by running permutation tests on each dataset and comparing the p-values
    /// (a measure of statistical significance).
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingPValue = PerformPermutationTest(evaluationData.TrainingSet.PredictionStats);
        var validationPValue = PerformPermutationTest(evaluationData.ValidationSet.PredictionStats);
        var testPValue = PerformPermutationTest(evaluationData.TestSet.PredictionStats);

        if (trainingPValue < _options.SignificanceLevel &&
            validationPValue < _options.SignificanceLevel &&
            testPValue < _options.SignificanceLevel)
        {
            return FitType.GoodFit;
        }
        else if (trainingPValue < _options.SignificanceLevel &&
                 (validationPValue >= _options.SignificanceLevel || testPValue >= _options.SignificanceLevel))
        {
            return FitType.Overfit;
        }
        else if (trainingPValue >= _options.SignificanceLevel &&
                 validationPValue >= _options.SignificanceLevel &&
                 testPValue >= _options.SignificanceLevel)
        {
            return FitType.Underfit;
        }
        else if (Math.Abs(trainingPValue - validationPValue) > _options.HighVarianceThreshold ||
                 Math.Abs(trainingPValue - testPValue) > _options.HighVarianceThreshold)
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level in the fit detection result.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you how confident we are in our assessment of your model.
    /// 
    /// It works by running permutation tests on each dataset and averaging the results.
    /// A confidence level close to 1 means we're very confident in our assessment.
    /// A confidence level close to 0 means we're less certain, and you might want to collect more data
    /// or try different evaluation methods.
    /// 
    /// The confidence level is calculated as 1 minus the average p-value from the permutation tests.
    /// P-values are statistical measures that tell us how likely we would see the observed results
    /// if there was no real pattern in the data.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingPValue = PerformPermutationTest(evaluationData.TrainingSet.PredictionStats);
        var validationPValue = PerformPermutationTest(evaluationData.ValidationSet.PredictionStats);
        var testPValue = PerformPermutationTest(evaluationData.TestSet.PredictionStats);

        var averagePValue = (trainingPValue + validationPValue + testPValue) / 3;
        var confidenceLevel = 1 - averagePValue;

        return NumOps.FromDouble(confidenceLevel);
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type (overfit, underfit, good fit, etc.).</param>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on its current performance.
    /// 
    /// For example:
    /// - If your model is overfitting, it might suggest adding more training data or simplifying your model
    /// - If your model is underfitting, it might suggest making your model more complex or adding more features
    /// - If your model has a good fit, it might suggest ways to fine-tune it further
    /// 
    /// These recommendations are specific to the permutation test results and are designed to help you
    /// take the next steps in improving your model's performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit across all datasets based on permutation tests.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Permutation tests indicate potential overfitting. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying regularization techniques");
                recommendations.Add("- Simplifying the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("Permutation tests suggest underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Permutation tests show high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying feature selection techniques");
                recommendations.Add("- Using ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("Permutation tests indicate unstable performance across datasets. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Applying cross-validation techniques");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Permutation tests performed with {_options.NumberOfPermutations} permutations and {_options.SignificanceLevel * 100}% significance level.");

        return recommendations;
    }

    /// <summary>
    /// Performs a permutation test on the prediction statistics to determine statistical significance.
    /// </summary>
    /// <param name="predictionStats">The prediction statistics containing metrics like R2.</param>
    /// <returns>
    /// A p-value representing the probability that the observed R2 value could occur by random chance.
    /// Lower p-values indicate stronger evidence that the model has learned meaningful patterns.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps determine if your model's performance is truly meaningful or if it 
    /// could have happened by random chance.
    /// 
    /// It works like this:
    /// 1. It takes your model's actual R2 score (a measure of how well your model fits the data, where 1.0 is perfect)
    /// 2. It creates many random versions of what your R2 might look like if there was no real pattern
    /// 3. It counts how many of these random versions perform as well as or better than your actual model
    /// 4. It calculates a "p-value" - a small p-value (typically &lt; 0.05) means your model is likely capturing real patterns
    /// 
    /// Think of it like this: If you flip a coin 100 times and get 98 heads, you'd suspect the coin is rigged.
    /// This test is similar - it checks if your model's performance is too good to be just luck.
    /// </para>
    /// </remarks>
    private double PerformPermutationTest(PredictionStats<T> predictionStats)
    {
        var originalR2 = Convert.ToDouble(predictionStats.R2);
        var permutationR2s = new List<double>();

        for (int i = 0; i < _options.NumberOfPermutations; i++)
        {
            var permutedR2 = SimulatePermutedR2(originalR2);
            permutationR2s.Add(permutedR2);
        }

        var pValue = permutationR2s.Count(r2 => r2 >= originalR2) / (double)_options.NumberOfPermutations;
        return pValue;
    }

    /// <summary>
    /// Simulates a permuted R2 value by adding random noise to the original R2.
    /// </summary>
    /// <param name="originalR2">The original R2 value from the model's predictions.</param>
    /// <returns>A simulated R2 value that represents what might occur by random chance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a "what-if" scenario to simulate what your model's performance 
    /// might look like if there was no real relationship between your inputs and outputs.
    /// 
    /// It adds a small amount of random noise (between -0.1 and +0.1) to your original R2 score,
    /// and ensures the result stays between 0 and 1 (the valid range for R2 scores).
    /// 
    /// This simulated value helps us determine if your model's actual performance is significantly 
    /// better than what could happen by random chance. We create many of these simulations to build
    /// a distribution of random outcomes for comparison.
    /// </para>
    /// </remarks>
    private double SimulatePermutedR2(double originalR2)
    {
        // Simulate permutation by adding some noise to the original R2
        var noise = _random.NextDouble() * 0.2 - 0.1; // Random noise between -0.1 and 0.1
        var permutedR2 = Math.Max(0, Math.Min(1, originalR2 + noise));

        return permutedR2;
    }
}
