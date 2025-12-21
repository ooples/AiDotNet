namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that analyzes model performance across training, validation, and test datasets to assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cross-validation is a technique for evaluating how well a model will generalize 
/// to independent data by comparing its performance on different subsets of the data. This detector 
/// analyzes the model's performance metrics across training, validation, and test datasets to determine 
/// if it's underfitting, overfitting, or has a good fit.
/// </para>
/// <para>
/// By comparing metrics like R² (coefficient of determination) across different datasets, the detector 
/// can identify issues like overfitting (performing much better on training than validation data) or 
/// underfitting (performing poorly across all datasets).
/// </para>
/// </remarks>
public class CrossValidationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the cross-validation fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets performance metrics 
    /// across different datasets, including thresholds for determining different types of model fit.
    /// </remarks>
    private readonly CrossValidationFitDetectorOptions _options;

    /// <summary>
    /// Threshold for determining if a model is overfitting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the maximum acceptable difference between training and validation 
    /// performance before a model is considered to be overfitting.
    /// </remarks>
    private readonly T _overfitThreshold;

    /// <summary>
    /// Threshold for determining if a model is underfitting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the minimum acceptable performance across all datasets before 
    /// a model is considered to be underfitting.
    /// </remarks>
    private readonly T _underfitThreshold;

    /// <summary>
    /// Threshold for determining if a model has a good fit.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the minimum acceptable performance across all datasets for 
    /// a model to be considered as having a good fit.
    /// </remarks>
    private readonly T _goodFitThreshold;

    /// <summary>
    /// Initializes a new instance of the CrossValidationFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new cross-validation fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Overfit threshold: Maximum acceptable difference between training and validation performance</description></item>
    /// <item><description>Underfit threshold: Minimum acceptable performance across all datasets</description></item>
    /// <item><description>Good fit threshold: Minimum performance for a model to be considered as having a good fit</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public CrossValidationFitDetector(CrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new();
        _overfitThreshold = NumOps.FromDouble(_options.OverfitThreshold);
        _underfitThreshold = NumOps.FromDouble(_options.UnderfitThreshold);
        _goodFitThreshold = NumOps.FromDouble(_options.GoodFitThreshold);
    }

    /// <summary>
    /// Detects the fit type of a model based on cross-validation analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values for training, validation, and test datasets.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance across training, validation, 
    /// and test datasets to determine if it's underfitting, overfitting, or has a good fit.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model is underfitting, overfitting, has a good fit, has high variance, or is unstable</description></item>
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
    /// Determines the fit type based on cross-validation analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values for training, validation, and test datasets.</param>
    /// <returns>The detected fit type based on cross-validation analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method compares the model's performance (R² values) across training, 
    /// validation, and test datasets to determine what type of fit your model has.
    /// </para>
    /// <para>
    /// The method looks at:
    /// <list type="bullet">
    /// <item><description>R² values for each dataset (training, validation, test)</description></item>
    /// <item><description>Differences between training and validation R² values</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Based on these metrics, it categorizes the model as having:
    /// <list type="bullet">
    /// <item><description>Good Fit: High R² values across all datasets</description></item>
    /// <item><description>Overfit: Much higher R² on training than validation</description></item>
    /// <item><description>Underfit: Low R² values across all datasets</description></item>
    /// <item><description>High Variance: Large differences between datasets but not clearly overfitting</description></item>
    /// <item><description>Unstable: Inconsistent performance that doesn't fit other categories</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingR2 = evaluationData.TrainingSet.PredictionStats.R2;
        var validationR2 = evaluationData.ValidationSet.PredictionStats.R2;
        var testR2 = evaluationData.TestSet.PredictionStats.R2;

        var r2Difference = NumOps.Abs(NumOps.Subtract(trainingR2, validationR2));

        if (NumOps.GreaterThan(trainingR2, _goodFitThreshold) &&
            NumOps.GreaterThan(validationR2, _goodFitThreshold) &&
            NumOps.GreaterThan(testR2, _goodFitThreshold))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(r2Difference, _overfitThreshold) &&
                 NumOps.GreaterThan(trainingR2, validationR2))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(trainingR2, _underfitThreshold) &&
                 NumOps.LessThan(validationR2, _underfitThreshold) &&
                 NumOps.LessThan(testR2, _underfitThreshold))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(r2Difference, _overfitThreshold))
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the cross-validation-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values for training, validation, and test datasets.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on the consistency of performance metrics across 
    /// different datasets.
    /// </para>
    /// <para>
    /// The method calculates:
    /// <list type="bullet">
    /// <item><description>R² consistency: Average R² across all datasets</description></item>
    /// <item><description>MSE consistency: Average Mean Squared Error across all datasets</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// These are combined to produce a confidence score between 0 and 1, with higher values indicating 
    /// greater confidence in the fit assessment. Models with consistent performance across datasets 
    /// will have higher confidence scores.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var r2Consistency = NumOps.Divide(
            NumOps.Add(NumOps.Add(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2), evaluationData.TestSet.PredictionStats.R2),
            NumOps.FromDouble(3));

        var mseConsistency = NumOps.Divide(
            NumOps.Add(NumOps.Add(evaluationData.TrainingSet.ErrorStats.MSE, evaluationData.ValidationSet.ErrorStats.MSE), evaluationData.TestSet.ErrorStats.MSE),
            NumOps.FromDouble(3));

        var confidenceLevel = NumOps.Multiply(r2Consistency, NumOps.Subtract(NumOps.One, mseConsistency));
        var lessThan = NumOps.LessThan(NumOps.One, confidenceLevel) ? NumOps.One : confidenceLevel;
        return NumOps.GreaterThan(NumOps.Zero, lessThan) ? NumOps.Zero : lessThan;
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and cross-validation analysis.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values for training, validation, and test datasets.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model based on cross-validation analysis.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Good Fit: The model is performing well and may be ready for deployment</description></item>
    /// <item><description>Overfitting: The model is too complex and needs to be simplified</description></item>
    /// <item><description>Underfitting: The model is too simple and needs more complexity</description></item>
    /// <item><description>High Variance: The model's performance varies too much across different data</description></item>
    /// <item><description>Unstable: The model's performance is inconsistent and needs further investigation</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The method also provides specific recommendations based on the R² values of each dataset, 
    /// highlighting areas where the model's performance could be improved.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit across all datasets. Consider deploying the model.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model shows signs of overfitting. Consider the following:");
                recommendations.Add("- Increase the amount of training data");
                recommendations.Add("- Apply regularization techniques");
                recommendations.Add("- Simplify the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("The model shows signs of underfitting. Consider the following:");
                recommendations.Add("- Increase model complexity");
                recommendations.Add("- Add more relevant features");
                recommendations.Add("- Reduce regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("The model shows high variance. Consider the following:");
                recommendations.Add("- Increase the amount of training data");
                recommendations.Add("- Apply feature selection techniques");
                recommendations.Add("- Use ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("The model performance is unstable across datasets. Consider the following:");
                recommendations.Add("- Investigate data quality and consistency");
                recommendations.Add("- Apply cross-validation techniques");
                recommendations.Add("- Use more robust feature selection methods");
                break;
        }

        if (NumOps.LessThan(evaluationData.TrainingSet.PredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Training R2 ({evaluationData.TrainingSet.PredictionStats.R2}) is below the good fit threshold. Consider improving model performance on training data.");
        }

        if (NumOps.LessThan(evaluationData.ValidationSet.PredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Validation R2 ({evaluationData.ValidationSet.PredictionStats.R2}) is below the good fit threshold. Focus on improving model generalization.");
        }

        if (NumOps.LessThan(evaluationData.TestSet.PredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Test R2 ({evaluationData.TestSet.PredictionStats.R2}) is below the good fit threshold. Evaluate model performance on unseen data.");
        }

        return recommendations;
    }
}
