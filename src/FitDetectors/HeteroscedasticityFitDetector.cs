namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates whether a model's errors have consistent variance across all predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Heteroscedasticity is a statistical term that means "uneven spread" of errors. 
/// In a good model, the errors (differences between predictions and actual values) should be 
/// roughly the same size regardless of what you're predicting. If errors get much larger or smaller 
/// for certain predictions (like having more accurate predictions for small values but less accurate 
/// for large values), that's called heteroscedasticity, and it can make your model less reliable.
/// 
/// This detector helps you identify if your model has this problem and suggests ways to fix it.
/// </para>
/// </remarks>
public class HeteroscedasticityFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options that control how the detector evaluates heteroscedasticity.
    /// </summary>
    private readonly HeteroscedasticityFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="HeteroscedasticityFitDetector{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you create this detector, you can customize how it works by providing options.
    /// If you don't provide any options, it will use reasonable default settings.
    /// </para>
    /// </remarks>
    public HeteroscedasticityFitDetector(HeteroscedasticityFitDetectorOptions? options = null)
    {
        _options = options ?? new HeteroscedasticityFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data to determine if the model has consistent error variance.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines your model's predictions and actual values to see if the errors 
    /// are consistent across all predictions. It runs statistical tests (Breusch-Pagan and White tests) 
    /// to check for heteroscedasticity. The result tells you if your model has consistent errors (good fit), 
    /// somewhat inconsistent errors (moderate fit), or very inconsistent errors (unstable fit), along with 
    /// specific recommendations to improve your model.
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
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "BreuschPaganTestStatistic", Convert.ToDouble(CalculateBreuschPaganTestStatistic(evaluationData)) },
                { "WhiteTestStatistic", Convert.ToDouble(CalculateWhiteTestStatistic(evaluationData)) }
            }
        };
    }

    /// <summary>
    /// Determines the type of fit based on statistical tests for heteroscedasticity.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A classification of the model fit quality (GoodFit, Moderate, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs two statistical tests (Breusch-Pagan and White tests) to check if your 
    /// model's errors are consistent. If both tests show consistent errors, it returns "GoodFit". If both tests 
    /// show very inconsistent errors, it returns "Unstable". If the results are somewhere in between, it returns "Moderate".
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        if (NumOps.GreaterThan(breuschPaganTestStatistic, NumOps.FromDouble(_options.HeteroscedasticityThreshold)) ||
            NumOps.GreaterThan(whiteTestStatistic, NumOps.FromDouble(_options.HeteroscedasticityThreshold)))
        {
            return FitType.Unstable;
        }
        else if (NumOps.LessThan(breuschPaganTestStatistic, NumOps.FromDouble(_options.HomoscedasticityThreshold)) &&
                 NumOps.LessThan(whiteTestStatistic, NumOps.FromDouble(_options.HomoscedasticityThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Moderate;
        }
    }

    /// <summary>
    /// Calculates how confident the detector is in its assessment of the model fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how sure the detector is about its assessment of your model. 
    /// It looks at the results of the statistical tests and calculates a confidence score between 0 and 1. 
    /// A higher score (closer to 1) means the detector is very confident in its assessment, while a lower 
    /// score means it's less certain.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        var maxTestStatistic = NumOps.GreaterThan(breuschPaganTestStatistic, whiteTestStatistic) ? breuschPaganTestStatistic : whiteTestStatistic;

        // Ensure the test statistic is non-negative
        if (NumOps.LessThan(maxTestStatistic, NumOps.Zero))
        {
            maxTestStatistic = NumOps.Zero;
        }

        var normalizedTestStatistic = NumOps.Divide(maxTestStatistic, NumOps.FromDouble(_options.HeteroscedasticityThreshold));

        // Clamp normalized test statistic to [0, 1]
        if (NumOps.LessThan(normalizedTestStatistic, NumOps.Zero))
        {
            normalizedTestStatistic = NumOps.Zero;
        }
        else if (NumOps.GreaterThan(normalizedTestStatistic, NumOps.One))
        {
            normalizedTestStatistic = NumOps.One;
        }

        // Invert the normalized test statistic to get a confidence level (higher test statistic = lower confidence)
        return NumOps.Subtract(NumOps.One, normalizedTestStatistic);
    }

    /// <summary>
    /// Calculates the Breusch-Pagan test statistic to detect heteroscedasticity.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The Breusch-Pagan test statistic value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Breusch-Pagan test is a statistical test that checks if the errors in your model 
    /// have consistent variance. This method calculates a test statistic - a single number that summarizes 
    /// the test result. Higher values suggest your model has inconsistent errors (heteroscedasticity), 
    /// which is generally not desirable.
    /// 
    /// The test works by checking if the squared errors from your model can be predicted using your input features.
    /// If they can, it suggests the error size depends on the input values, indicating heteroscedasticity.
    /// </para>
    /// </remarks>
    private T CalculateBreuschPaganTestStatistic(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var X = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features);
        var y = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);

        Vector<T> yPredicted;
        if (evaluationData.ModelStats.Model == null)
        {
            yPredicted = new Vector<T>(y.Length);
        }
        else
        {
            var predictions = evaluationData.ModelStats.Model.Predict(evaluationData.ModelStats.Features);
            yPredicted = ConversionsHelper.ConvertToVector<T, TOutput>(predictions);
        }

        var residuals = y.Subtract(yPredicted);
        var squaredResiduals = residuals.Select(r => NumOps.Multiply(r, r));
        var meanSquaredResidual = NumOps.Divide(squaredResiduals.Sum(), NumOps.FromDouble(squaredResiduals.Length));

        var scaledResiduals = squaredResiduals.Divide(meanSquaredResidual);
        var auxiliaryRegression = new SimpleRegression<T>();
        auxiliaryRegression.Train(X, scaledResiduals);

        // Calculate R-squared using PredictionStats
        var predictionStatsInputs = new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(scaledResiduals),
            Predicted = auxiliaryRegression.Predict(X),
            NumberOfParameters = X.Columns,
        };
        var predictionStats = new PredictionStats<T>(predictionStatsInputs);

        return NumOps.Multiply(NumOps.FromDouble(X.Rows), predictionStats.R2);
    }

    /// <summary>
    /// Calculates the White test statistic to detect heteroscedasticity.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The White test statistic value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The White test is another statistical test (similar to Breusch-Pagan) that checks 
    /// if your model's errors have consistent variance. It's more general than the Breusch-Pagan test 
    /// because it doesn't assume the errors follow a normal distribution.
    /// 
    /// This test works by creating an "augmented" version of your input data that includes the original 
    /// features, their squares, and their cross-products (interactions between features). Then it checks 
    /// if these augmented features can predict the squared errors from your model. If they can, it suggests 
    /// the error size depends on the input values, indicating heteroscedasticity.
    /// 
    /// Higher values of the test statistic suggest your model has inconsistent errors, which is generally 
    /// not desirable for reliable predictions.
    /// </para>
    /// </remarks>
    private T CalculateWhiteTestStatistic(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var X = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features);
        var y = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);

        Vector<T> yPredicted;
        if (evaluationData.ModelStats.Model == null)
        {
            yPredicted = new Vector<T>(y.Length);
        }
        else
        {
            var predictions = evaluationData.ModelStats.Model.Predict(evaluationData.ModelStats.Features);
            yPredicted = ConversionsHelper.ConvertToVector<T, TOutput>(predictions);
        }

        var residuals = y.Subtract(yPredicted);
        var squaredResiduals = residuals.Select(r => NumOps.Multiply(r, r));

        // Create augmented X matrix with squared terms and cross products
        var augmentedX = new Matrix<T>(X.Rows, X.Columns * (X.Columns + 3) / 2 + 1);
        int column = 0;
        for (int i = 0; i < X.Columns; i++)
        {
            augmentedX.SetColumn(column++, X.GetColumn(i));
            augmentedX.SetColumn(column++, X.GetColumn(i).Select(x => NumOps.Multiply(x, x)));
            for (int j = i + 1; j < X.Columns; j++)
            {
                augmentedX.SetColumn(column++, new Vector<T>(X.GetColumn(i).Zip(X.GetColumn(j), (a, b) => NumOps.Multiply(a, b))));
            }
        }
        augmentedX.SetColumn(column, Vector<T>.CreateDefault(X.Rows, NumOps.One));

        // Use MultipleRegression since the augmented matrix has multiple columns
        // (original features + squared terms + cross-products + constant)
        var auxiliaryRegression = new MultipleRegression<T>();
        auxiliaryRegression.Train(augmentedX, new Vector<T>(squaredResiduals));
        var predictionStatsInputs = new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(squaredResiduals),
            Predicted = auxiliaryRegression.Predict(augmentedX),
            NumberOfParameters = augmentedX.Columns,
        };

        var predictionStats = new PredictionStats<T>(predictionStatsInputs);

        return NumOps.Multiply(NumOps.FromDouble(X.Rows), predictionStats.R2);
    }

    /// <summary>
    /// Generates specific recommendations based on the detected fit type of the model.
    /// </summary>
    /// <param name="fitType">The classification of model fit quality (GoodFit, Moderate, or Unstable).</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of recommendations for improving or maintaining model quality.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of practical suggestions based on how well your model's 
    /// errors are distributed. If your model has inconsistent errors (heteroscedasticity), it will suggest 
    /// specific techniques to fix the problem, like transforming your data or using different regression methods.
    /// 
    /// The recommendations are tailored to three scenarios:
    /// - Unstable fit: Your model has significant heteroscedasticity problems that need addressing
    /// - Moderate fit: Your model has some heteroscedasticity that might benefit from investigation
    /// - Good fit: Your model has consistent errors (homoscedasticity), which is desirable
    /// 
    /// The method also includes the actual test statistics so you can see the numerical evidence behind 
    /// the recommendations.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Unstable:
                recommendations.Add("The model shows signs of heteroscedasticity. Consider the following:");
                recommendations.Add("1. Transform the dependent variable (e.g., log transformation).");
                recommendations.Add("2. Use weighted least squares regression.");
                recommendations.Add("3. Consider using robust standard errors.");
                recommendations.Add("4. Investigate if important predictors are missing from the model.");
                break;

            case FitType.Moderate:
                recommendations.Add("The model shows some signs of heteroscedasticity. Consider the following:");
                recommendations.Add("1. Investigate potential causes of heteroscedasticity in your data.");
                recommendations.Add("2. Consider mild transformations of variables.");
                recommendations.Add("3. Use diagnostic plots to visualize the residuals.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have homoscedastic residuals. Consider the following:");
                recommendations.Add("1. Validate the model on new, unseen data.");
                recommendations.Add("2. Monitor model performance over time for potential changes in residual patterns.");
                recommendations.Add("3. Consider other aspects of model fit, such as linearity and normality of residuals.");
                break;
        }

        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        recommendations.Add($"Breusch-Pagan test statistic: {breuschPaganTestStatistic}");
        recommendations.Add($"White test statistic: {whiteTestStatistic}");

        return recommendations;
    }
}
