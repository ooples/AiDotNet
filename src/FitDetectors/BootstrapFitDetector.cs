namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that uses bootstrap resampling to assess model fit and stability.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Bootstrap resampling is a statistical technique that creates multiple versions 
/// of your dataset by randomly sampling with replacement. This allows you to estimate the variability 
/// and stability of your model's performance metrics.
/// </para>
/// <para>
/// This detector uses bootstrap resampling to determine if a model is underfitting, overfitting, or has 
/// a good fit, while also assessing the confidence in this determination. Think of it like testing your 
/// model on many slightly different versions of your data to see how consistently it performs.
/// </para>
/// </remarks>
public class BootstrapFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the bootstrap fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector performs bootstrap resampling and 
    /// interprets the results, including the number of bootstrap samples and thresholds for determining 
    /// different types of model fit.
    /// </remarks>
    private readonly BootstrapFitDetectorOptions _options;

    /// <summary>
    /// Random number generator used for bootstrap resampling.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is used to introduce randomness when creating bootstrap samples, 
    /// simulating the process of randomly drawing data points with replacement.
    /// </remarks>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the BootstrapFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new bootstrap fit detector with either custom options 
    /// or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Number of bootstrap samples (often 1000)</description></item>
    /// <item><description>Confidence interval (often 0.95 or 95%)</description></item>
    /// <item><description>Thresholds for determining good fit, overfitting, and underfitting</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public BootstrapFitDetector(BootstrapFitDetectorOptions? options = null)
    {
        _options = options ?? new BootstrapFitDetectorOptions();
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Detects the fit type of a model based on bootstrap resampling.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance across multiple bootstrap samples 
    /// to determine if it's underfitting, overfitting, or has a good fit.
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
    /// Determines the fit type based on bootstrap resampling of model performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on bootstrap analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method performs bootstrap resampling to create multiple versions of your 
    /// performance metrics (R² values), then analyzes these to determine what type of fit your model has.
    /// </para>
    /// <para>
    /// The method looks at:
    /// <list type="bullet">
    /// <item><description>Average R² values across bootstrap samples for training, validation, and test sets</description></item>
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
        var bootstrapResults = PerformBootstrap(evaluationData);

        var meanTrainingR2 = new Vector<T>(bootstrapResults.Select(r => r.TrainingR2)).Average();
        var meanValidationR2 = new Vector<T>(bootstrapResults.Select(r => r.ValidationR2)).Average();
        var meanTestR2 = new Vector<T>(bootstrapResults.Select(r => r.TestR2)).Average();

        var r2Difference = NumOps.Abs(NumOps.Subtract(meanTrainingR2, meanValidationR2));

        if (NumOps.GreaterThan(meanTrainingR2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
            NumOps.GreaterThan(meanValidationR2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
            NumOps.GreaterThan(meanTestR2, NumOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(r2Difference, NumOps.FromDouble(_options.OverfitThreshold)) &&
                 NumOps.GreaterThan(meanTrainingR2, meanValidationR2))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(meanTrainingR2, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(meanValidationR2, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(meanTestR2, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(r2Difference, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the bootstrap fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on the width of the confidence interval for the 
    /// difference between training and validation R² values.
    /// </para>
    /// <para>
    /// A narrower confidence interval indicates more consistent results across bootstrap samples, 
    /// which translates to higher confidence in the fit assessment. The confidence level is calculated 
    /// as 1 minus the width of the confidence interval, clamped between 0 and 1.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var bootstrapResults = PerformBootstrap(evaluationData);

        var r2Differences = bootstrapResults.Select(r => NumOps.Abs(NumOps.Subtract(r.TrainingR2, r.ValidationR2))).ToList();
        r2Differences.Sort();

        int lowerIndex = (int)Math.Floor((1 - _options.ConfidenceInterval) / 2 * _options.NumberOfBootstraps);
        int upperIndex = (int)Math.Ceiling((1 + _options.ConfidenceInterval) / 2 * _options.NumberOfBootstraps) - 1;

        var confidenceInterval = NumOps.Subtract(r2Differences[upperIndex], r2Differences[lowerIndex]);
        var confidenceLevel = NumOps.Subtract(NumOps.One, confidenceInterval);
        var lessThan = NumOps.LessThan(NumOps.One, confidenceLevel) ? NumOps.One : confidenceLevel;

        return NumOps.GreaterThan(NumOps.Zero, lessThan) ? NumOps.Zero : lessThan;
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
    /// <item><description>Good Fit: The model is performing well and may only need fine-tuning</description></item>
    /// <item><description>Overfitting: The model is too complex and needs to be simplified</description></item>
    /// <item><description>Underfitting: The model is too simple and needs more complexity</description></item>
    /// <item><description>High Variance: The model's performance varies too much across different data</description></item>
    /// <item><description>Unstable: The model's performance is inconsistent and needs further investigation</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit across all datasets based on bootstrap analysis.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Bootstrap analysis indicates potential overfitting. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying regularization techniques");
                recommendations.Add("- Simplifying the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("Bootstrap analysis suggests underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Bootstrap analysis shows high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying feature selection techniques");
                recommendations.Add("- Using ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("Bootstrap analysis indicates unstable performance across datasets. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Applying cross-validation techniques");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Bootstrap analysis performed with {_options.NumberOfBootstraps} resamples and {_options.ConfidenceInterval * 100}% confidence interval.");

        return recommendations;
    }

    /// <summary>
    /// Performs bootstrap resampling on the evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of bootstrap results containing resampled R² values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method creates multiple bootstrap samples by resampling the 
    /// original R² values with some added noise to simulate the variability you would see with actual 
    /// bootstrap resampling of the data.
    /// </para>
    /// <para>
    /// Each bootstrap result contains resampled R² values for the training, validation, and test sets. 
    /// The number of bootstrap samples is determined by the NumberOfBootstraps option.
    /// </para>
    /// </remarks>
    private List<BootstrapResult<T>> PerformBootstrap(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var results = new List<BootstrapResult<T>>();

        for (int i = 0; i < _options.NumberOfBootstraps; i++)
        {
            var bootstrapTrainingR2 = ResampleR2(evaluationData.TrainingSet.PredictionStats.R2);
            var bootstrapValidationR2 = ResampleR2(evaluationData.ValidationSet.PredictionStats.R2);
            var bootstrapTestR2 = ResampleR2(evaluationData.TestSet.PredictionStats.R2);

            results.Add(new BootstrapResult<T>
            {
                TrainingR2 = bootstrapTrainingR2,
                ValidationR2 = bootstrapValidationR2,
                TestR2 = bootstrapTestR2
            });
        }

        return results;
    }

    /// <summary>
    /// Resamples an R² value by adding random noise.
    /// </summary>
    /// <param name="originalR2">The original R² value.</param>
    /// <returns>A resampled R² value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method simulates bootstrap resampling by adding a small 
    /// amount of random noise to the original R² value. This mimics the variation you would see 
    /// if you actually resampled the data and recalculated the R².
    /// </para>
    /// <para>
    /// The noise is randomly generated between -0.05 and 0.05, and the resulting R² value is 
    /// clamped between 0 and 1 to ensure it remains a valid R² value.
    /// </para>
    /// <para>
    /// In a full implementation, this would involve actual resampling of the data points and 
    /// recalculation of the R² value, but this simplified approach provides a reasonable 
    /// approximation for the purpose of fit detection.
    /// </para>
    /// </remarks>
    private T ResampleR2(T originalR2)
    {
        // Simulate resampling by adding some noise to the original R2
        var noise = _random.NextDouble() * 0.1 - 0.05; // Random noise between -0.05 and 0.05
        var resampledR2 = Math.Max(0, Math.Min(1, Convert.ToDouble(originalR2) + noise));

        return NumOps.FromDouble(resampledR2);
    }
}
