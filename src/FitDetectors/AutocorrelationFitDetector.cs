namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that analyzes autocorrelation in model residuals to assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Autocorrelation refers to the correlation of a time series with its own past values. 
/// In the context of model fitting, autocorrelation in residuals (prediction errors) can indicate that the 
/// model is missing important patterns in the data.
/// </para>
/// <para>
/// This detector uses the Durbin-Watson statistic to measure autocorrelation in model residuals. The 
/// Durbin-Watson statistic ranges from 0 to 4, with a value of 2 indicating no autocorrelation, values 
/// less than 2 indicating positive autocorrelation, and values greater than 2 indicating negative 
/// autocorrelation.
/// </para>
/// </remarks>
public class AutocorrelationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the autocorrelation fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets the Durbin-Watson 
    /// statistic, including thresholds for determining different types of autocorrelation.
    /// </remarks>
    private readonly AutocorrelationFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the AutocorrelationFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new autocorrelation fit detector with either 
    /// custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically use standard thresholds for interpreting the Durbin-Watson statistic, 
    /// such as considering values below 1.5 as indicating strong positive autocorrelation and values above 
    /// 2.5 as indicating strong negative autocorrelation.
    /// </para>
    /// </remarks>
    public AutocorrelationFitDetector(AutocorrelationFitDetectorOptions? options = null)
    {
        _options = options ?? new AutocorrelationFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model based on autocorrelation in residuals.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes the pattern of errors your model makes to determine 
    /// if there's autocorrelation, which can indicate that your model is missing important patterns in the data.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: The type of autocorrelation detected (if any)</description></item>
    /// <item><description>ConfidenceLevel: How confident the detector is in its assessment</description></item>
    /// <item><description>Recommendations: Suggestions for improving the model based on the detected autocorrelation</description></item>
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
    /// Determines the fit type based on the Durbin-Watson statistic of model residuals.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on autocorrelation analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the Durbin-Watson statistic from your model's errors 
    /// and interprets it to determine what type of autocorrelation (if any) is present.
    /// </para>
    /// <para>
    /// The Durbin-Watson statistic ranges from 0 to 4:
    /// <list type="bullet">
    /// <item><description>Values near 2 (typically 1.5-2.5) indicate no autocorrelation</description></item>
    /// <item><description>Values below 1.5 suggest positive autocorrelation (errors tend to be followed by errors of the same sign)</description></item>
    /// <item><description>Values above 2.5 suggest negative autocorrelation (errors tend to be followed by errors of the opposite sign)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var durbinWatsonStat = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestSet.ErrorStats.ErrorList);

        if (NumOps.LessThan(durbinWatsonStat, NumOps.FromDouble(_options.StrongPositiveAutocorrelationThreshold)))
        {
            return FitType.StrongPositiveAutocorrelation;
        }
        else if (NumOps.GreaterThan(durbinWatsonStat, NumOps.FromDouble(_options.StrongNegativeAutocorrelationThreshold)))
        {
            return FitType.StrongNegativeAutocorrelation;
        }
        else if (NumOps.GreaterThanOrEquals(durbinWatsonStat, NumOps.FromDouble(_options.NoAutocorrelationLowerBound)) &&
                 NumOps.LessThanOrEquals(durbinWatsonStat, NumOps.FromDouble(_options.NoAutocorrelationUpperBound)))
        {
            return FitType.NoAutocorrelation;
        }
        else
        {
            return FitType.WeakAutocorrelation;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the autocorrelation detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of autocorrelation. The confidence is based on how far the Durbin-Watson statistic is from 
    /// the ideal value of 2.0 (which indicates no autocorrelation).
    /// </para>
    /// <para>
    /// A value closer to 1 indicates high confidence, while a value closer to 0 indicates low confidence.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var durbinWatsonStat = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestSet.ErrorStats.ErrorList);
        var idealDW = NumOps.FromDouble(2.0);
        var maxDeviation = NumOps.FromDouble(2.0); // Maximum possible deviation from ideal (0 or 4)

        var confidenceLevel = NumOps.Subtract(NumOps.One,
            NumOps.Divide(NumOps.Abs(NumOps.Subtract(durbinWatsonStat, idealDW)), maxDeviation));

        return confidenceLevel;
    }

    /// <summary>
    /// Generates recommendations based on the detected autocorrelation type.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of autocorrelation detected in your model's residuals.
    /// </para>
    /// <para>
    /// Different types of autocorrelation require different approaches:
    /// <list type="bullet">
    /// <item><description>Positive autocorrelation often indicates missing time-dependent variables or trends</description></item>
    /// <item><description>Negative autocorrelation might indicate over-differencing or alternating patterns</description></item>
    /// <item><description>Weak autocorrelation might require further investigation to determine its nature</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.StrongPositiveAutocorrelation:
                recommendations.Add("Strong positive autocorrelation detected. Consider:");
                recommendations.Add("- Adding lagged variables to your model");
                recommendations.Add("- Using time series specific models (e.g., ARIMA, SARIMA)");
                recommendations.Add("- Differencing the data if it's non-stationary");
                break;
            case FitType.StrongNegativeAutocorrelation:
                recommendations.Add("Strong negative autocorrelation detected. Consider:");
                recommendations.Add("- Investigating for potential over-differencing");
                recommendations.Add("- Checking for alternating patterns in your data");
                recommendations.Add("- Using moving average terms in your model");
                break;
            case FitType.WeakAutocorrelation:
                recommendations.Add("Weak autocorrelation detected. Consider:");
                recommendations.Add("- Investigating the nature of the autocorrelation (seasonal, trend, etc.)");
                recommendations.Add("- Adding relevant features that might explain the weak autocorrelation");
                recommendations.Add("- Using more robust error terms (e.g., HAC standard errors)");
                break;
            case FitType.NoAutocorrelation:
                recommendations.Add("No significant autocorrelation detected.");
                recommendations.Add("The model appears to capture the time-dependent patterns well.");
                break;
        }

        recommendations.Add($"Durbin-Watson statistic: {StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestSet.ErrorStats.ErrorList):F4}");

        return recommendations;
    }
}
