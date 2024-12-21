namespace AiDotNet.FitDetectors;

public class BayesianFitDetector<T> : FitDetectorBase<T>
{
    private readonly BayesianFitDetectorOptions _options;

    public BayesianFitDetector(BayesianFitDetectorOptions? options = null)
    {
        _options = options ?? new BayesianFitDetectorOptions();
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
        var dic = StatisticsHelper<T>.CalculateDIC(evaluationData.ModelStats);
        var waic = StatisticsHelper<T>.CalculateWAIC(evaluationData.ModelStats);
        var loo = StatisticsHelper<T>.CalculateLOO(evaluationData.ModelStats);

        if (_numOps.LessThan(dic, _numOps.FromDouble(_options.GoodFitThreshold)) &&
            _numOps.LessThan(waic, _numOps.FromDouble(_options.GoodFitThreshold)) &&
            _numOps.LessThan(loo, _numOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (_numOps.GreaterThan(dic, _numOps.FromDouble(_options.OverfitThreshold)) ||
                 _numOps.GreaterThan(waic, _numOps.FromDouble(_options.OverfitThreshold)) ||
                 _numOps.GreaterThan(loo, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(dic, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(waic, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(loo, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var posteriorPredictiveCheck = StatisticsHelper<T>.CalculatePosteriorPredictiveCheck(evaluationData.ModelStats);
        var bayes_factor = StatisticsHelper<T>.CalculateBayesFactor(evaluationData.ModelStats);

        var confidenceScore = _numOps.Multiply(posteriorPredictiveCheck, bayes_factor);
        return _numOps.GreaterThan(confidenceScore, _numOps.One) ? _numOps.One : confidenceScore;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Model appears to be overfitting. Consider:");
                recommendations.Add("- Using more informative priors");
                recommendations.Add("- Simplifying the model structure");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying Bayesian regularization techniques");
                break;
            case FitType.Underfit:
                recommendations.Add("Model appears to be underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Using less restrictive priors");
                recommendations.Add("- Adding more relevant features or interactions");
                recommendations.Add("- Exploring non-linear relationships in the data");
                break;
            case FitType.GoodFit:
                recommendations.Add("Model shows good fit. Consider:");
                recommendations.Add("- Fine-tuning hyperparameters for potential improvements");
                recommendations.Add("- Conducting sensitivity analysis on priors");
                recommendations.Add("- Exploring model averaging or ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("Model performance is unstable. Consider:");
                recommendations.Add("- Checking for multimodality in the posterior distribution");
                recommendations.Add("- Investigating potential issues with MCMC convergence");
                recommendations.Add("- Using alternative MCMC samplers or increasing the number of iterations");
                recommendations.Add("- Considering hierarchical models to account for group-level variations");
                break;
        }

        recommendations.Add($"DIC: {StatisticsHelper<T>.CalculateDIC(evaluationData.ModelStats):F4}");
        recommendations.Add($"WAIC: {StatisticsHelper<T>.CalculateWAIC(evaluationData.ModelStats):F4}");
        recommendations.Add($"LOO: {StatisticsHelper<T>.CalculateLOO(evaluationData.ModelStats):F4}");
        recommendations.Add($"Posterior Predictive Check: {StatisticsHelper<T>.CalculatePosteriorPredictiveCheck(evaluationData.ModelStats):F4}");
        recommendations.Add($"Bayes Factor: {StatisticsHelper<T>.CalculateBayesFactor(evaluationData.ModelStats):F4}");

        return recommendations;
    }
}