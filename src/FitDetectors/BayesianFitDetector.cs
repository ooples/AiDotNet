namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that uses Bayesian model comparison metrics to assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Bayesian statistics provides a framework for model evaluation that considers 
/// both how well a model fits the data and its complexity. This detector uses several Bayesian metrics 
/// to determine if a model is underfitting, overfitting, or has a good fit.
/// </para>
/// <para>
/// Unlike traditional methods that only look at prediction errors, Bayesian methods also consider 
/// the model's complexity and uncertainty, providing a more comprehensive assessment of model fit.
/// </para>
/// </remarks>
public class BayesianFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Bayesian fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector interprets various Bayesian metrics, 
    /// including thresholds for determining different types of model fit.
    /// </remarks>
    private readonly BayesianFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the BayesianFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new Bayesian fit detector with either custom options 
    /// or default settings.
    /// </para>
    /// <para>
    /// The default settings typically use standard thresholds for interpreting Bayesian metrics like DIC 
    /// (Deviance Information Criterion), WAIC (Widely Applicable Information Criterion), and LOO 
    /// (Leave-One-Out cross-validation).
    /// </para>
    /// </remarks>
    public BayesianFitDetector(BayesianFitDetectorOptions? options = null)
    {
        _options = options ?? new BayesianFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model based on Bayesian model comparison metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model using several Bayesian metrics to determine 
    /// if it's underfitting, overfitting, or has a good fit.
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
    /// Determines the fit type based on Bayesian model comparison metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on Bayesian analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates several Bayesian metrics (DIC, WAIC, and LOO) and 
    /// uses them to determine what type of fit your model has.
    /// </para>
    /// <para>
    /// These metrics balance how well the model fits the data against its complexity:
    /// <list type="bullet">
    /// <item><description>DIC (Deviance Information Criterion): A measure of model fit that penalizes complexity</description></item>
    /// <item><description>WAIC (Widely Applicable Information Criterion): A more general version of DIC</description></item>
    /// <item><description>LOO (Leave-One-Out cross-validation): Estimates out-of-sample prediction accuracy</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Lower values of these metrics generally indicate better models, but very low values might suggest underfitting, 
    /// while inconsistent values across metrics might indicate instability.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var dic = StatisticsHelper<T>.CalculateDIC(evaluationData.ModelStats);
        var waic = StatisticsHelper<T>.CalculateWAIC(evaluationData.ModelStats);
        var loo = StatisticsHelper<T>.CalculateLOO(evaluationData.ModelStats);

        if (NumOps.LessThan(dic, NumOps.FromDouble(_options.GoodFitThreshold)) &&
            NumOps.LessThan(waic, NumOps.FromDouble(_options.GoodFitThreshold)) &&
            NumOps.LessThan(loo, NumOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(dic, NumOps.FromDouble(_options.OverfitThreshold)) ||
                 NumOps.GreaterThan(waic, NumOps.FromDouble(_options.OverfitThreshold)) ||
                 NumOps.GreaterThan(loo, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(dic, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(waic, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(loo, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the Bayesian fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on two Bayesian metrics:
    /// </para>
    /// <para>
    /// 1. Posterior Predictive Check: Measures how well the model's predictions match the observed data
    /// </para>
    /// <para>
    /// 2. Bayes Factor: Compares the evidence for the model against a simpler alternative
    /// </para>
    /// <para>
    /// These metrics are combined to produce a confidence score between 0 and 1, with higher values 
    /// indicating greater confidence in the fit assessment.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var posteriorPredictiveCheck = StatisticsHelper<T>.CalculatePosteriorPredictiveCheck(evaluationData.ModelStats);
        var bayes_factor = StatisticsHelper<T>.CalculateBayesFactor(evaluationData.ModelStats);

        var confidenceScore = NumOps.Multiply(posteriorPredictiveCheck, bayes_factor);
        return NumOps.GreaterThan(confidenceScore, NumOps.One) ? NumOps.One : confidenceScore;
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Overfitting: The model is too complex and needs to be simplified</description></item>
    /// <item><description>Underfitting: The model is too simple and needs more complexity</description></item>
    /// <item><description>Good Fit: The model is appropriate but might benefit from fine-tuning</description></item>
    /// <item><description>Unstable: The model's performance is inconsistent and needs further investigation</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The recommendations also include the values of various Bayesian metrics to help you understand 
    /// the basis for the assessment.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
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
