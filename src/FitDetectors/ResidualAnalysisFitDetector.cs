namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality by analyzing the residuals (errors) of the model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you understand how well your model is performing by looking at 
/// the "residuals" - the differences between what your model predicted and the actual values.
/// 
/// Think of residuals like the errors your model makes. By analyzing these errors in different ways,
/// we can tell if your model:
/// - Is generally accurate (good fit)
/// - Consistently makes errors in the same direction (bias)
/// - Makes wildly different errors each time (high variance)
/// - Works well on training data but poorly on new data (overfitting)
/// - Doesn't capture the complexity of your data (underfitting)
/// 
/// This detector examines these patterns to give you a clear picture of your model's performance
/// and how to improve it.
/// </para>
/// </remarks>
public class ResidualAnalysisFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the residual analysis detector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how strict or lenient the detector is when evaluating
    /// your model. They include thresholds for acceptable error rates and other statistical measures.
    /// </para>
    /// </remarks>
    private readonly ResidualAnalysisFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the ResidualAnalysisFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options for the detector. If null, default options will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector. You can provide custom options
    /// to control how strict the detector should be, or leave it as null to use the default settings.
    /// </para>
    /// </remarks>
    public ResidualAnalysisFitDetector(ResidualAnalysisFitDetectorOptions? options = null)
    {
        _options = options ?? new ResidualAnalysisFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes the model's performance data and determines the quality of fit based on residual analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model's errors (residuals) and tells you
    /// how well it's performing. It:
    /// 
    /// 1. Determines what type of fit your model has (good, overfit, underfit, etc.)
    /// 2. Calculates how confident it is in this assessment
    /// 3. Provides specific recommendations to help you improve your model
    /// 
    /// The result contains all this information in an organized way that you can use to understand
    /// and improve your model.
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
    /// Determines the type of fit by analyzing residual patterns and statistical measures.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A FitType enum value indicating the quality of the model fit.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines your model's errors in detail to determine if your model has:
    /// 
    /// - Good Fit: Your model makes small, random errors with no clear pattern
    /// - Unstable: Your model's errors show patterns that suggest problems (like autocorrelation)
    /// - Underfit: Your model makes large errors because it's too simple for your data
    /// - High Variance: Your model's errors vary widely, suggesting it's too sensitive to small changes
    /// - High Bias: Your model consistently makes errors in the same direction
    /// - Overfit: Your model works well on training data but poorly on new data
    /// 
    /// The method uses several statistical tests to make this determination:
    /// 
    /// - Durbin-Watson statistic: Checks if errors are related to each other (autocorrelation)
    /// - MAPE (Mean Absolute Percentage Error): Measures overall error size
    /// - Mean and standard deviation of residuals: Examines error patterns
    /// - R-squared differences: Compares model performance across different datasets
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Check for autocorrelation using Durbin-Watson statistic
        if (NumOps.LessThan(evaluationData.TestSet.ErrorStats.DurbinWatsonStatistic, NumOps.FromDouble(1.5)) ||
            NumOps.GreaterThan(evaluationData.TestSet.ErrorStats.DurbinWatsonStatistic, NumOps.FromDouble(2.5)))
        {
            return FitType.Unstable;
        }

        // Check MAPE for overall fit
        if (NumOps.GreaterThan(evaluationData.TestSet.ErrorStats.MAPE, NumOps.FromDouble(_options.MapeThreshold)))
        {
            return FitType.Underfit;
        }

        // Analyze residuals across datasets
        var meanThreshold = NumOps.FromDouble(_options.MeanThreshold);
        var stdThreshold = NumOps.FromDouble(_options.StdThreshold);

        var trainingResidualMean = evaluationData.TrainingSet.ErrorStats.MeanBiasError;
        var validationResidualMean = evaluationData.ValidationSet.ErrorStats.MeanBiasError;
        var testResidualMean = evaluationData.TestSet.ErrorStats.MeanBiasError;

        var trainingResidualStd = evaluationData.TrainingSet.ErrorStats.PopulationStandardError;
        var validationResidualStd = evaluationData.ValidationSet.ErrorStats.PopulationStandardError;
        var testResidualStd = evaluationData.TestSet.ErrorStats.PopulationStandardError;

        if (NumOps.LessThan(NumOps.Abs(trainingResidualMean), meanThreshold) &&
            NumOps.LessThan(NumOps.Abs(validationResidualMean), meanThreshold) &&
            NumOps.LessThan(NumOps.Abs(testResidualMean), meanThreshold))
        {
            if (NumOps.LessThan(trainingResidualStd, stdThreshold) &&
                NumOps.LessThan(validationResidualStd, stdThreshold) &&
                NumOps.LessThan(testResidualStd, stdThreshold))
            {
                return FitType.GoodFit;
            }
            else
            {
                return FitType.HighVariance;
            }
        }
        else if (NumOps.GreaterThan(NumOps.Abs(trainingResidualMean), meanThreshold) &&
                 NumOps.GreaterThan(NumOps.Abs(validationResidualMean), meanThreshold) &&
                 NumOps.GreaterThan(NumOps.Abs(testResidualMean), meanThreshold))
        {
            return FitType.HighBias;
        }
        else if (NumOps.LessThan(NumOps.Abs(trainingResidualMean), meanThreshold) &&
                 NumOps.GreaterThan(NumOps.Abs(validationResidualMean), meanThreshold))
        {
            return FitType.Overfit;
        }

        // Check for significant differences in R-squared values
        var r2Threshold = NumOps.FromDouble(_options.R2Threshold);
        if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2)), r2Threshold) ||
            NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.TestSet.PredictionStats.R2)), r2Threshold))
        {
            return FitType.Unstable;
        }

        return FitType.GoodFit;
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
    /// The confidence is calculated by:
    /// 
    /// 1. Looking at how consistent the errors are in each dataset (training, validation, and test)
    ///    - More consistent errors = higher confidence
    ///    - Wildly varying errors = lower confidence
    /// 
    /// 2. Considering the R-squared values across all datasets
    ///    - R-squared is a measure of how well your model explains the data (from 0 to 1)
    ///    - Higher R-squared values = higher confidence
    ///    - Lower R-squared values = lower confidence
    /// 
    /// The final confidence score combines these factors to give you a single number between 0 and 1.
    /// A score closer to 1 means the detector is very confident in its assessment of your model,
    /// while a score closer to 0 means there's more uncertainty.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Helper function to safely calculate confidence, avoiding division by zero
        T SafeConfidence(T populationStdError, T meanBiasError)
        {
            // Avoid division by zero - if mean bias error is close to zero, use R-squared only
            if (NumOps.LessThanOrEquals(NumOps.Abs(meanBiasError), NumOps.FromDouble(1e-10)))
            {
                return NumOps.FromDouble(0.5);  // Neutral confidence when bias is near zero
            }
            var ratio = NumOps.Divide(populationStdError, meanBiasError);
            var conf = NumOps.Subtract(NumOps.One, NumOps.Abs(ratio));
            // Clamp to [0, 1]
            if (NumOps.LessThan(conf, NumOps.Zero)) conf = NumOps.Zero;
            if (NumOps.GreaterThan(conf, NumOps.One)) conf = NumOps.One;
            return conf;
        }

        var trainingConfidence = SafeConfidence(evaluationData.TrainingSet.ErrorStats.PopulationStandardError, evaluationData.TrainingSet.ErrorStats.MeanBiasError);
        var validationConfidence = SafeConfidence(evaluationData.ValidationSet.ErrorStats.PopulationStandardError, evaluationData.ValidationSet.ErrorStats.MeanBiasError);
        var testConfidence = SafeConfidence(evaluationData.TestSet.ErrorStats.PopulationStandardError, evaluationData.TestSet.ErrorStats.MeanBiasError);

        var averageConfidence = NumOps.Divide(NumOps.Add(NumOps.Add(trainingConfidence, validationConfidence), testConfidence), NumOps.FromDouble(3));

        // Adjust confidence based on R-squared values (clamp R2 values to [0, 1])
        var r2Training = evaluationData.TrainingSet.PredictionStats.R2;
        if (NumOps.LessThan(r2Training, NumOps.Zero)) r2Training = NumOps.Zero;
        if (NumOps.GreaterThan(r2Training, NumOps.One)) r2Training = NumOps.One;

        var r2Validation = evaluationData.ValidationSet.PredictionStats.R2;
        if (NumOps.LessThan(r2Validation, NumOps.Zero)) r2Validation = NumOps.Zero;
        if (NumOps.GreaterThan(r2Validation, NumOps.One)) r2Validation = NumOps.One;

        var r2Test = evaluationData.TestSet.PredictionStats.R2;
        if (NumOps.LessThan(r2Test, NumOps.Zero)) r2Test = NumOps.Zero;
        if (NumOps.GreaterThan(r2Test, NumOps.One)) r2Test = NumOps.One;
        var r2Adjustment = NumOps.Divide(NumOps.Add(NumOps.Add(r2Training, r2Validation), r2Test), NumOps.FromDouble(3));

        var result = NumOps.Multiply(averageConfidence, r2Adjustment);

        // Final clamp to ensure [0, 1]
        if (NumOps.LessThan(result, NumOps.Zero)) result = NumOps.Zero;
        if (NumOps.GreaterThan(result, NumOps.One)) result = NumOps.One;

        return result;
    }
}
