namespace AiDotNet.FitDetectors;

public class AutocorrelationFitDetector<T> : FitDetectorBase<T>
{
    private readonly AutocorrelationFitDetectorOptions _options;

    public AutocorrelationFitDetector(AutocorrelationFitDetectorOptions? options = null)
    {
        _options = options ?? new AutocorrelationFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
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

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var durbinWatsonStat = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestErrorStats.ErrorList);

        if (_numOps.LessThan(durbinWatsonStat, _numOps.FromDouble(_options.StrongPositiveAutocorrelationThreshold)))
        {
            return FitType.StrongPositiveAutocorrelation;
        }
        else if (_numOps.GreaterThan(durbinWatsonStat, _numOps.FromDouble(_options.StrongNegativeAutocorrelationThreshold)))
        {
            return FitType.StrongNegativeAutocorrelation;
        }
        else if (_numOps.GreaterThanOrEquals(durbinWatsonStat, _numOps.FromDouble(_options.NoAutocorrelationLowerBound)) &&
                 _numOps.LessThanOrEquals(durbinWatsonStat, _numOps.FromDouble(_options.NoAutocorrelationUpperBound)))
        {
            return FitType.NoAutocorrelation;
        }
        else
        {
            return FitType.WeakAutocorrelation;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var durbinWatsonStat = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestErrorStats.ErrorList);
        var idealDW = _numOps.FromDouble(2.0);
        var maxDeviation = _numOps.FromDouble(2.0); // Maximum possible deviation from ideal (0 or 4)

        var confidenceLevel = _numOps.Subtract(_numOps.One, 
            _numOps.Divide(_numOps.Abs(_numOps.Subtract(durbinWatsonStat, idealDW)), maxDeviation));

        return confidenceLevel;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
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

        recommendations.Add($"Durbin-Watson statistic: {StatisticsHelper<T>.CalculateDurbinWatsonStatistic(evaluationData.TestErrorStats.ErrorList):F4}");

        return recommendations;
    }
}