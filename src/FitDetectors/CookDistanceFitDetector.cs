namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that uses Cook's distance to identify influential data points and assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cook's distance is a statistical measure that identifies influential data points 
/// in a regression analysis. An influential point is one that, if removed, would significantly change the 
/// model's parameters or predictions.
/// </para>
/// <para>
/// This detector analyzes the distribution of Cook's distances across all data points to determine if 
/// the model is overfitting (too sensitive to individual points) or underfitting (not capturing important 
/// patterns in the data).
/// </para>
/// </remarks>
public class CookDistanceFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Cook's distance fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets Cook's distances, 
    /// including thresholds for determining influential points and different types of model fit.
    /// </remarks>
    private readonly CookDistanceFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the CookDistanceFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new Cook's distance fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Threshold for considering a point influential (often 4/n where n is sample size)</description></item>
    /// <item><description>Thresholds for determining overfitting and underfitting based on the ratio of influential points</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public CookDistanceFitDetector(CookDistanceFitDetectorOptions? options = null)
    {
        _options = options ?? new CookDistanceFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model based on Cook's distance analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, recommendations, and additional information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates Cook's distance for each data point, then analyzes 
    /// the distribution of these distances to determine if the model is underfitting, overfitting, or has 
    /// a good fit.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model is underfitting, overfitting, or has a good fit</description></item>
    /// <item><description>ConfidenceLevel: How confident the detector is in its assessment</description></item>
    /// <item><description>Recommendations: Suggestions for improving the model based on the detected fit type</description></item>
    /// <item><description>AdditionalInfo: Contains the calculated Cook's distances for further analysis</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var fitType = DetermineFitType(cookDistances);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CookDistances", cookDistances }
            }
        };
    }

    /// <summary>
    /// Determines the fit type based on Cook's distance analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on Cook's distance analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates Cook's distances for all data points, then calls 
    /// the overloaded DetermineFitType method to analyze these distances.
    /// </para>
    /// <para>
    /// This is an implementation of the abstract method from the base class that adapts the evaluation 
    /// data to the specific Cook's distance analysis approach.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        return DetermineFitType(cookDistances);
    }

    /// <summary>
    /// Determines the fit type based on a vector of Cook's distances.
    /// </summary>
    /// <param name="cookDistances">Vector of Cook's distances for all data points.</param>
    /// <returns>The detected fit type based on Cook's distance analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method analyzes the distribution of Cook's distances to 
    /// determine what type of fit your model has.
    /// </para>
    /// <para>
    /// The method counts how many data points have Cook's distances above the influential threshold, 
    /// then calculates the ratio of influential points to total points. Based on this ratio:
    /// <list type="bullet">
    /// <item><description>High ratio of influential points suggests overfitting (model is too sensitive to individual points)</description></item>
    /// <item><description>Low ratio of influential points might suggest underfitting (model is not capturing important patterns)</description></item>
    /// <item><description>Moderate ratio of influential points suggests a good fit</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private FitType DetermineFitType(Vector<T> cookDistances)
    {
        var influentialPointsCount = cookDistances.Count(d => NumOps.GreaterThan(d, NumOps.FromDouble(_options.InfluentialThreshold)));
        var influentialRatio = NumOps.Divide(NumOps.FromDouble(influentialPointsCount), NumOps.FromDouble(cookDistances.Length));

        if (NumOps.GreaterThan(influentialRatio, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(influentialRatio, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the Cook's distance-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on the ratio of influential points to total points.
    /// </para>
    /// <para>
    /// A lower ratio of influential points generally indicates higher confidence in the fit assessment. 
    /// The confidence level is calculated as 1 minus the influential ratio, resulting in a value 
    /// between 0 and 1, where higher values indicate greater confidence.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var influentialPointsCount = cookDistances.Count(d => NumOps.GreaterThan(d, NumOps.FromDouble(_options.InfluentialThreshold)));
        var influentialRatio = NumOps.Divide(NumOps.FromDouble(influentialPointsCount), NumOps.FromDouble(cookDistances.Length));

        // Normalize confidence level to be between 0 and 1
        return NumOps.Subtract(NumOps.One, influentialRatio);
    }

    /// <summary>
    /// Calculates Cook's distances for all data points.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A vector of Cook's distances for all data points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates Cook's distance for each data point in your dataset. 
    /// Cook's distance measures how much the model's predictions would change if a particular data point 
    /// were removed from the dataset.
    /// </para>
    /// <para>
    /// The calculation involves:
    /// <list type="bullet">
    /// <item><description>Computing the hat matrix (H), which represents the influence of each observed response on each fitted value</description></item>
    /// <item><description>Calculating residuals (differences between actual and predicted values)</description></item>
    /// <item><description>Computing the mean squared error (MSE) of the residuals</description></item>
    /// <item><description>Calculating Cook's distance for each point using a formula that combines these elements</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Higher Cook's distances indicate more influential points. Points with Cook's distances above a 
    /// certain threshold (often 4/n where n is the sample size) are considered highly influential.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateCookDistances(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var X = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features);
        var y = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);

        if (evaluationData.ModelStats.Model == null)
        {
            throw new InvalidOperationException("Model is null. CookDistanceFitDetector requires Model to be set on ModelStats. Cannot calculate Cook's distances without a valid model.");
        }

        var yPredicted = ConversionsHelper.ConvertToVector<T, TOutput>(
            evaluationData.ModelStats.Model.Predict(evaluationData.ModelStats.Features));

        var residuals = y.Subtract(yPredicted);

        var n = X.Rows;
        var p = X.Columns;
        var hatMatrix = X.Multiply(X.Transpose().Multiply(X).Inverse()).Multiply(X.Transpose());
        var mse = NumOps.Divide(residuals.DotProduct(residuals), NumOps.FromDouble(n - p));

        var cookDistances = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            var hii = hatMatrix[i, i];
            var ri = residuals[i];
            var di = NumOps.Divide(NumOps.Multiply(ri, ri), NumOps.Multiply(NumOps.FromDouble(p), mse));
            di = NumOps.Multiply(di, NumOps.Divide(hii, NumOps.Multiply(NumOps.Subtract(NumOps.One, hii), NumOps.Subtract(NumOps.One, hii))));
            cookDistances[i] = di;
        }

        return cookDistances;
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and Cook's distance analysis.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model based on Cook's distance analysis.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Overfitting: The model is too sensitive to individual points and needs to be simplified</description></item>
    /// <item><description>Underfitting: The model is not capturing important patterns and needs more complexity</description></item>
    /// <item><description>Good Fit: The model is appropriate but might benefit from validation and monitoring</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The recommendations also include information about the top 5 most influential points (based on 
    /// Cook's distance) to help you identify specific data points that might be affecting your model.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Investigate and potentially remove highly influential points.");
                recommendations.Add("2. Increase regularization strength.");
                recommendations.Add("3. Simplify the model by reducing the number of features or model complexity.");
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity or add more relevant features.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Investigate if important predictors are missing from the model.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Validate the model on new, unseen data.");
                recommendations.Add("2. Monitor model performance over time for potential drift.");
                recommendations.Add("3. Consider fine-tuning hyperparameters for potential improvements.");
                break;
        }

        recommendations.Add("Top 5 most influential points (index: Cook's Distance):");
        var sortedIndices = cookDistances.Select((value, index) => new { Value = value, Index = index })
                                         .OrderByDescending(x => x.Value)
                                         .Take(5)
                                         .ToList();
        foreach (var item in sortedIndices)
        {
            recommendations.Add($"   - {item.Index}: {item.Value}");
        }

        return recommendations;
    }
}
