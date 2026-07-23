namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that analyzes feature importances and correlations to assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Feature importance measures how much each input variable (feature) contributes 
/// to a model's predictions. This detector uses permutation importance, which works by randomly shuffling 
/// each feature and measuring how much the model's performance degrades as a result.
/// </para>
/// <para>
/// By analyzing both feature importances and correlations between features, this detector can identify 
/// issues like overfitting (relying too heavily on specific features) or underfitting (not effectively 
/// using the available features).
/// </para>
/// </remarks>
public class FeatureImportanceFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the feature importance fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets feature importances 
    /// and correlations, including thresholds for determining different types of model fit.
    /// </remarks>
    private readonly FeatureImportanceFitDetectorOptions _options;

    private Matrix<T> _featureCorrelations;

    /// <summary>
    /// Initializes a new instance of the FeatureImportanceFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new feature importance fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Thresholds for high and low feature importance</description></item>
    /// <item><description>Thresholds for high and low variance in feature importances</description></item>
    /// <item><description>Threshold for determining feature correlation</description></item>
    /// <item><description>Random seed for reproducible permutation importance calculations</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public FeatureImportanceFitDetector(FeatureImportanceFitDetectorOptions? options = null)
    {
        _options = options ?? new FeatureImportanceFitDetectorOptions();
        _featureCorrelations = Matrix<T>.Empty();
    }

    /// <summary>
    /// Detects the fit type of a model based on feature importance analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's feature importances and feature correlations 
    /// to determine if it's underfitting, overfitting, or has a good fit.
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
    /// Determines the fit type based on feature importance analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on feature importance analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates feature importances and correlations, then analyzes 
    /// these metrics to determine what type of fit your model has.
    /// </para>
    /// <para>
    /// The method looks at:
    /// <list type="bullet">
    /// <item><description>Average feature importance: How much features contribute to predictions on average</description></item>
    /// <item><description>Standard deviation of feature importances: How much variation there is in feature contributions</description></item>
    /// <item><description>Feature correlations: How related different features are to each other</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Based on these metrics, it categorizes the model as having:
    /// <list type="bullet">
    /// <item><description>Good Fit: High average importance with low variance (features contribute consistently)</description></item>
    /// <item><description>Overfit: High average importance with high variance (model relies too heavily on specific features)</description></item>
    /// <item><description>Underfit: Low average importance or mostly uncorrelated features (model doesn't effectively use features)</description></item>
    /// <item><description>Unstable: Any other pattern that doesn't fit the above categories</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var featureImportances = CalculateFeatureImportances(evaluationData);
        var averageImportance = featureImportances.Average();
        var importanceStdDev = StatisticsHelper<T>.CalculateStandardDeviation(featureImportances);
        _featureCorrelations = CalculateFeatureCorrelations(ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features));

        if (NumOps.GreaterThan(averageImportance, NumOps.FromDouble(_options.HighImportanceThreshold)) &&
            NumOps.LessThan(importanceStdDev, NumOps.FromDouble(_options.LowVarianceThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(averageImportance, NumOps.FromDouble(_options.HighImportanceThreshold)) &&
                 NumOps.GreaterThan(importanceStdDev, NumOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(averageImportance, NumOps.FromDouble(_options.LowImportanceThreshold)) ||
                 AreFeaturesMostlyUncorrelated(_featureCorrelations))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the feature importance-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on feature importances and correlations.
    /// </para>
    /// <para>
    /// The method calculates three factors:
    /// <list type="bullet">
    /// <item><description>Importance factor: How high the average feature importance is relative to the threshold</description></item>
    /// <item><description>Variance factor: How low the standard deviation of feature importances is</description></item>
    /// <item><description>Correlation factor: How uncorrelated the features are (1 minus average absolute correlation)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// These factors are multiplied together to produce a confidence score between 0 and 1, with higher 
    /// values indicating greater confidence in the fit assessment.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var featureImportances = CalculateFeatureImportances(evaluationData);
        var averageImportance = featureImportances.Average();
        var importanceStdDev = StatisticsHelper<T>.CalculateStandardDeviation(featureImportances);
        var importanceFactor = NumOps.Divide(averageImportance, NumOps.FromDouble(_options.HighImportanceThreshold));
        var varianceFactor = NumOps.Divide(NumOps.FromDouble(_options.LowVarianceThreshold), NumOps.Add(NumOps.One, importanceStdDev));
        var correlationFactor = NumOps.Subtract(NumOps.FromDouble(1), AverageAbsoluteCorrelation(_featureCorrelations));

        return NumOps.Multiply(NumOps.Multiply(importanceFactor, varianceFactor), correlationFactor);
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and feature importance analysis.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model based on feature importance analysis.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Good Fit: The model is using features effectively and may only need fine-tuning</description></item>
    /// <item><description>Overfit: The model is relying too heavily on specific features and needs to be more balanced</description></item>
    /// <item><description>Underfit: The model is not effectively using the available features and needs to be more complex</description></item>
    /// <item><description>Unstable: The model's use of features is inconsistent and needs further investigation</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The recommendations also include information about the top 3 most important features to help you 
    /// understand which variables are driving your model's predictions.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();
        var featureImportances = CalculateFeatureImportances(evaluationData);

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model appears to be well-fitted based on feature importances.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Implement regularization techniques.");
                recommendations.Add("2. Reduce model complexity or use simpler models.");
                recommendations.Add("3. Collect more training data if possible.");
                break;
            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity or use more sophisticated models.");
                recommendations.Add("2. Feature engineering to create more informative features.");
                recommendations.Add("3. Collect more relevant features if possible.");
                break;
            case FitType.Unstable:
                recommendations.Add("The model fit is unstable. Consider the following:");
                recommendations.Add("1. Analyze feature correlations and remove highly correlated features.");
                recommendations.Add("2. Use feature selection techniques to identify the most relevant features.");
                recommendations.Add("3. Implement cross-validation to ensure model stability.");
                break;
        }

        recommendations.Add("Top 3 most important features:");
        var topFeatures = featureImportances
            .Select((importance, index) => new { Importance = importance, Index = index })
            .OrderByDescending(x => x.Importance)
            .Take(3);

        foreach (var feature in topFeatures)
        {
            recommendations.Add($"- Feature {feature.Index}: Importance = {feature.Importance:F4}");
        }

        return recommendations;
    }

    /// <summary>
    /// Calculates feature importances using permutation importance.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A vector of importance values for each feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how important each feature is to your model's 
    /// predictions using a technique called permutation importance.
    /// </para>
    /// <para>
    /// The process works as follows:
    /// <list type="number">
    /// <item><description>Calculate the model's baseline error on the original data</description></item>
    /// <item><description>For each feature:
    ///   <list type="bullet">
    ///     <item><description>Randomly shuffle (permute) the values of that feature</description></item>
    ///     <item><description>Recalculate the model's predictions with the shuffled feature</description></item>
    ///     <item><description>Calculate the new error with the shuffled feature</description></item>
    ///     <item><description>The importance is the difference between the new error and the baseline error</description></item>
    ///   </list>
    /// </description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Features that cause a large increase in error when shuffled are considered more important, as 
    /// this indicates the model relies heavily on that feature for accurate predictions.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateFeatureImportances(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Convert generic types to Vector<T> and Matrix<T> for calculations
        var actualVector = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predictedVector = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        var featuresMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features);

        // Check if model is null
        if (evaluationData.ModelStats.Model == null)
        {
            throw new AiDotNetException("Model is null. Cannot calculate feature importances without a valid model.");
        }

        var baselineError = CalculateError(actualVector, predictedVector);
        var featureImportances = new Vector<T>(featuresMatrix.Columns);

        for (int i = 0; i < featuresMatrix.Columns; i++)
        {
            var permutedFeature = PermuteFeature(featuresMatrix.GetColumn(i));
            var permutedMatrix = featuresMatrix.Clone();
            permutedMatrix.SetColumn(i, permutedFeature);

            // Convert back to TInput for prediction
            var permutedOutputs = evaluationData.ModelStats.Model.Predict((TInput)(object)permutedMatrix);
            var permutedPredictedVector = ConversionsHelper.ConvertToVector<T, TOutput>(permutedOutputs);

            var permutedError = CalculateError(actualVector, permutedPredictedVector);

            featureImportances[i] = NumOps.Subtract(permutedError, baselineError);
        }

        return featureImportances;
    }

    /// <summary>
    /// Calculates the error between actual and predicted values.
    /// </summary>
    /// <param name="actual">Vector of actual values.</param>
    /// <param name="predicted">Vector of predicted values.</param>
    /// <returns>The mean squared error between actual and predicted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how far off the model's predictions are 
    /// from the actual values using mean squared error (MSE).
    /// </para>
    /// <para>
    /// Mean squared error is calculated by:
    /// <list type="number">
    /// <item><description>Taking the difference between each predicted value and the corresponding actual value</description></item>
    /// <item><description>Squaring each difference</description></item>
    /// <item><description>Calculating the average of all squared differences</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Lower MSE values indicate better model performance (predictions closer to actual values).
    /// </para>
    /// </remarks>
    private T CalculateError(Vector<T> actual, Vector<T> predicted)
    {
        return StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);
    }

    /// <summary>
    /// Randomly shuffles (permutes) the values in a feature vector.
    /// </summary>
    /// <param name="feature">The feature vector to permute.</param>
    /// <returns>A new vector with the same values in random order.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method takes a feature vector and randomly rearranges its values.
    /// </para>
    /// <para>
    /// The method uses the Fisher-Yates shuffle algorithm, which:
    /// <list type="number">
    /// <item><description>Starts from the last element</description></item>
    /// <item><description>Swaps it with a randomly selected element from the whole array (including itself)</description></item>
    /// <item><description>Moves to the previous element and repeats until the entire array is processed</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// This shuffling breaks any relationship between the feature and the target variable, which is 
    /// essential for calculating permutation importance.
    /// </para>
    /// </remarks>
    private Vector<T> PermuteFeature(Vector<T> feature)
    {
        var permutedFeature = feature.Clone();
        int n = permutedFeature.Length;

        for (int i = n - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            T temp = permutedFeature[i];
            permutedFeature[i] = permutedFeature[j];
            permutedFeature[j] = temp;
        }

        return permutedFeature;
    }

    /// <summary>
    /// Calculates the correlation matrix for all features.
    /// </summary>
    /// <param name="features">Matrix of feature values.</param>
    /// <returns>A matrix of correlation coefficients between all pairs of features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how related each pair of features is to each other 
    /// using Pearson correlation.
    /// </para>
    /// <para>
    /// Pearson correlation measures the linear relationship between two variables, with values ranging from -1 to 1:
    /// <list type="bullet">
    /// <item><description>1: Perfect positive correlation (as one variable increases, the other increases proportionally)</description></item>
    /// <item><description>0: No correlation (variables are independent)</description></item>
    /// <item><description>-1: Perfect negative correlation (as one variable increases, the other decreases proportionally)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The resulting matrix is symmetric, with the correlation of each feature with itself (diagonal elements) being 1.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateFeatureCorrelations(Matrix<T> features)
    {
        int numFeatures = features.Columns;
        var correlations = new Matrix<T>(numFeatures, numFeatures);

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i; j < numFeatures; j++)
            {
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(features.GetColumn(i), features.GetColumn(j));
                correlations[i, j] = correlation;
                correlations[j, i] = correlation;
            }
        }

        return correlations;
    }

    /// <summary>
    /// Determines if most features are uncorrelated with each other.
    /// </summary>
    /// <param name="correlations">Matrix of correlation coefficients between all pairs of features.</param>
    /// <returns>True if most feature pairs have correlation below the threshold, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method checks if your features are mostly independent of each other.
    /// </para>
    /// <para>
    /// The method counts how many pairs of features have correlation coefficients below the threshold 
    /// (indicating weak or no relationship), then calculates what percentage of all possible pairs 
    /// this represents.
    /// </para>
    /// <para>
    /// If this percentage exceeds the uncorrelated ratio threshold (e.g., 80%), the features are 
    /// considered mostly uncorrelated, which can be a sign of underfitting in some cases.
    /// </para>
    /// </remarks>
    private bool AreFeaturesMostlyUncorrelated(Matrix<T> correlations)
    {
        int numFeatures = correlations.Rows;
        int uncorrelatedCount = 0;

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                if (NumOps.LessThan(NumOps.Abs(correlations[i, j]), NumOps.FromDouble(_options.CorrelationThreshold)))
                {
                    uncorrelatedCount++;
                }
            }
        }

        int totalPairs = (numFeatures * (numFeatures - 1)) / 2;
        return (double)uncorrelatedCount / totalPairs > _options.UncorrelatedRatioThreshold;
    }

    /// <summary>
    /// Calculates the average absolute correlation across all feature pairs.
    /// </summary>
    /// <param name="correlations">Matrix of correlation coefficients between all pairs of features.</param>
    /// <returns>The average of the absolute values of all correlation coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how strongly related your features are to each other 
    /// on average, ignoring whether the relationships are positive or negative.
    /// </para>
    /// <para>
    /// The method:
    /// <list type="number">
    /// <item><description>Takes the absolute value of each correlation coefficient (making negative correlations positive)</description></item>
    /// <item><description>Sums these absolute values for all pairs of different features</description></item>
    /// <item><description>Divides by the number of pairs to get the average</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// A high average absolute correlation indicates that many features contain redundant information, 
    /// which can affect model performance and interpretation.
    /// </para>
    /// </remarks>
    private T AverageAbsoluteCorrelation(Matrix<T> correlations)
    {
        int numFeatures = correlations.Rows;
        T sum = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                sum = NumOps.Add(sum, NumOps.Abs(correlations[i, j]));
                count++;
            }
        }

        return NumOps.Divide(sum, NumOps.FromDouble(count));
    }
}
