namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that analyzes confusion matrix metrics to assess classification model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A confusion matrix is a table that summarizes the performance of a classification 
/// model by showing the counts of true positives, false positives, true negatives, and false negatives. 
/// This detector uses metrics derived from the confusion matrix to evaluate how well a model is performing.
/// </para>
/// <para>
/// Unlike other fit detectors that focus on underfitting and overfitting, this detector primarily assesses 
/// the overall quality of the model's predictions and can identify issues like class imbalance that might 
/// affect performance.
/// </para>
/// </remarks>
public class ConfusionMatrixFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the confusion matrix fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector analyzes confusion matrix metrics, 
    /// including which metric to use as the primary evaluation criterion and thresholds for determining 
    /// different levels of model fit.
    /// </remarks>
    private readonly ConfusionMatrixFitDetectorOptions _options;

    private ConfusionMatrix<T> _confusionMatrix;

    /// <summary>
    /// Initializes a new instance of the ConfusionMatrixFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new confusion matrix fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Primary metric (often F1 score)</description></item>
    /// <item><description>Thresholds for determining good, moderate, and poor fit</description></item>
    /// <item><description>Confidence threshold for converting probabilities to binary predictions</description></item>
    /// <item><description>Threshold for detecting class imbalance</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public ConfusionMatrixFitDetector(ConfusionMatrixFitDetectorOptions options)
    {
        _options = options ?? new ConfusionMatrixFitDetectorOptions();
        _confusionMatrix = new ConfusionMatrix<T>(NumOps.Zero, NumOps.Zero, NumOps.Zero, NumOps.Zero);
    }

    /// <summary>
    /// Detects the fit type of a model based on confusion matrix analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance using confusion matrix metrics 
    /// to determine if it has a good fit, moderate fit, or poor fit.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model has a good fit, moderate fit, or poor fit</description></item>
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
    /// Determines the fit type based on confusion matrix metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on confusion matrix analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a confusion matrix from your model's predictions and 
    /// actual values, then computes a primary metric (like accuracy, precision, recall, or F1 score) to 
    /// determine the quality of the model's fit.
    /// </para>
    /// <para>
    /// Based on the value of the primary metric compared to predefined thresholds, the model is categorized as having:
    /// <list type="bullet">
    /// <item><description>Good Fit: High metric value, indicating strong performance</description></item>
    /// <item><description>Moderate Fit: Medium metric value, indicating acceptable but improvable performance</description></item>
    /// <item><description>Poor Fit: Low metric value, indicating suboptimal performance</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        _confusionMatrix = CalculateConfusionMatrix(actual, predicted);
        var metric = CalculatePrimaryMetric(_confusionMatrix);

        if (NumOps.GreaterThanOrEquals(metric, NumOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThanOrEquals(metric, NumOps.FromDouble(_options.ModerateFitThreshold)))
        {
            return FitType.Moderate;
        }
        else
        {
            return FitType.PoorFit;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the confusion matrix-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on how far the primary metric is from the thresholds 
    /// used to determine fit type.
    /// </para>
    /// <para>
    /// A metric value that is close to a threshold will result in lower confidence, while a value that 
    /// is far from any threshold will result in higher confidence. The confidence level is normalized 
    /// to a value between 0 and 1, with higher values indicating greater confidence.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var metric = CalculatePrimaryMetric(_confusionMatrix);

        // Normalize the metric to a 0-1 range
        var normalizedMetric = NumOps.Divide(
            NumOps.Subtract(metric, NumOps.FromDouble(_options.ModerateFitThreshold)),
            NumOps.FromDouble(_options.GoodFitThreshold - _options.ModerateFitThreshold)
        );

        var lessThan = NumOps.LessThan(normalizedMetric, NumOps.One) ? normalizedMetric : NumOps.One;
        return NumOps.GreaterThan(lessThan, NumOps.Zero) ? lessThan : NumOps.Zero;
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and confusion matrix analysis.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for improving your model based 
    /// on the detected fit type and additional analysis of the confusion matrix.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Good Fit: The model is performing well and may only need fine-tuning</description></item>
    /// <item><description>Moderate Fit: The model is performing acceptably but could be improved</description></item>
    /// <item><description>Poor Fit: The model is not performing well and needs significant improvements</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The method also checks for class imbalance, which occurs when one class is much more common than 
    /// the other in your dataset. If detected, it provides specific recommendations for addressing this issue.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance based on the confusion matrix analysis.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model shows moderate performance. There's room for improvement.");
                recommendations.Add("Consider adjusting the classification threshold to optimize the trade-off between different types of errors.");
                recommendations.Add("Analyze feature importance and consider feature engineering or selection.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model's performance is suboptimal based on the confusion matrix analysis.");
                recommendations.Add("Review your feature set and consider adding more relevant features.");
                recommendations.Add("Experiment with different algorithms or ensemble methods.");
                break;
        }

        if (IsClassImbalanced(_confusionMatrix))
        {
            recommendations.Add("The dataset appears to be imbalanced. Consider using techniques like oversampling, undersampling, or SMOTE to address this issue.");
        }

        return recommendations;
    }

    /// <summary>
    /// Calculates a confusion matrix from actual and predicted values.
    /// </summary>
    /// <param name="actual">Vector of actual values.</param>
    /// <param name="predicted">Vector of predicted probabilities.</param>
    /// <returns>A confusion matrix containing counts of true positives, false positives, true negatives, and false negatives.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method converts predicted probabilities to binary predictions 
    /// using a confidence threshold, then compares these predictions to the actual values to create a 
    /// confusion matrix.
    /// </para>
    /// <para>
    /// A confusion matrix contains four values:
    /// <list type="bullet">
    /// <item><description>True Positives (TP): Cases correctly predicted as positive</description></item>
    /// <item><description>False Positives (FP): Cases incorrectly predicted as positive (Type I error)</description></item>
    /// <item><description>True Negatives (TN): Cases correctly predicted as negative</description></item>
    /// <item><description>False Negatives (FN): Cases incorrectly predicted as negative (Type II error)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private ConfusionMatrix<T> CalculateConfusionMatrix(Vector<T> actual, Vector<T> predicted)
    {
        return StatisticsHelper<T>.CalculateConfusionMatrix(actual, predicted, NumOps.FromDouble(_options.ConfidenceThreshold));
    }

    /// <summary>
    /// Calculates the primary metric from a confusion matrix.
    /// </summary>
    /// <param name="confusionMatrix">The confusion matrix to analyze.</param>
    /// <returns>The value of the primary metric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method computes the primary evaluation metric specified in the 
    /// options from the confusion matrix. Common metrics include:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><description>Accuracy: (TP + TN) / (TP + FP + TN + FN) - Overall correctness</description></item>
    /// <item><description>Precision: TP / (TP + FP) - Exactness (how many selected items are relevant)</description></item>
    /// <item><description>Recall: TP / (TP + FN) - Completeness (how many relevant items are selected)</description></item>
    /// <item><description>F1 Score: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Different metrics are appropriate for different problems. For example, in medical diagnosis, 
    /// recall might be more important than precision because missing a disease (false negative) could 
    /// be more harmful than a false alarm (false positive).
    /// </para>
    /// </remarks>
    private T CalculatePrimaryMetric(ConfusionMatrix<T> confusionMatrix)
    {
        return _options.PrimaryMetric switch
        {
            MetricType.Accuracy => confusionMatrix.Accuracy,
            MetricType.Precision => confusionMatrix.Precision,
            MetricType.Recall => confusionMatrix.Recall,
            MetricType.F1Score => confusionMatrix.F1Score,
            _ => throw new ArgumentException("Unsupported primary metric type."),
        };
    }

    /// <summary>
    /// Determines if the dataset has class imbalance.
    /// </summary>
    /// <param name="confusionMatrix">The confusion matrix to analyze.</param>
    /// <returns>True if class imbalance is detected, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method checks if one class is much more common than the other 
    /// in your dataset, which is known as class imbalance.
    /// </para>
    /// <para>
    /// Class imbalance can cause problems for many machine learning algorithms because they may become 
    /// biased toward the majority class. The method calculates the ratio of positive and negative samples 
    /// and compares them to a threshold to detect imbalance.
    /// </para>
    /// <para>
    /// For example, if your dataset has 95% negative samples and only 5% positive samples, it would be 
    /// considered imbalanced, and special techniques might be needed to address this issue.
    /// </para>
    /// </remarks>
    private bool IsClassImbalanced(ConfusionMatrix<T> confusionMatrix)
    {
        T totalSamples = NumOps.Add(NumOps.Add(confusionMatrix.TruePositives, confusionMatrix.FalsePositives),
                                     NumOps.Add(confusionMatrix.TrueNegatives, confusionMatrix.FalseNegatives));
        T positiveRatio = NumOps.Divide(NumOps.Add(confusionMatrix.TruePositives, confusionMatrix.FalseNegatives), totalSamples);
        T negativeRatio = NumOps.Divide(NumOps.Add(confusionMatrix.TrueNegatives, confusionMatrix.FalsePositives), totalSamples);

        return NumOps.LessThan(positiveRatio, NumOps.FromDouble(_options.ClassImbalanceThreshold)) ||
               NumOps.LessThan(negativeRatio, NumOps.FromDouble(_options.ClassImbalanceThreshold));
    }
}
