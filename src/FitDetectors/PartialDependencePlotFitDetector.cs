namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that uses Partial Dependence Plots to analyze model fit and detect overfitting or underfitting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Partial Dependence Plot (PDP) helps you understand how each feature in your data
/// affects your model's predictions. It shows the relationship between a feature and the predicted outcome
/// while accounting for the effects of all other features. This detector uses these plots to determine if
/// your model is learning appropriate patterns from your data.
/// </para>
/// </remarks>
public class PartialDependencePlotFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Partial Dependence Plot fit detector.
    /// </summary>
    private readonly PartialDependencePlotFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="PartialDependencePlotFitDetector{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the detector. If null, default options will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new PDP fit detector. You can customize how it works by providing
    /// options, or just use the default settings which work well for most cases.
    /// </para>
    /// </remarks>
    public PartialDependencePlotFitDetector(PartialDependencePlotFitDetectorOptions? options = null)
    {
        _options = options ?? new PartialDependencePlotFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes a model's fit using Partial Dependence Plots.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines your model to determine if it's learning the right patterns from your data.
    /// It creates special plots (PDPs) that show how each feature affects your model's predictions, then uses these
    /// plots to decide if your model is:
    /// - Overfitting: Your model is "memorizing" the training data instead of learning general patterns
    /// - Underfitting: Your model is too simple and isn't capturing important patterns in the data
    /// - Good fit: Your model has found the right balance
    /// 
    /// The method returns detailed results including specific recommendations to improve your model.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        var fitType = DetermineFitType(pdpResults);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, pdpResults);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PartialDependencePlots", pdpResults }
            }
        };
    }

    /// <summary>
    /// Determines the type of fit (overfit, underfit, or good fit) based on model evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <returns>The detected fit type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance to determine if it has learned
    /// the right patterns from your data. It calculates PDPs and uses them to classify your model's fit.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        return DetermineFitType(pdpResults);
    }

    /// <summary>
    /// Determines the type of fit based on the calculated partial dependence plots.
    /// </summary>
    /// <param name="pdpResults">Dictionary mapping feature names to their partial dependence plots.</param>
    /// <returns>The detected fit type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at how "wiggly" or "smooth" the relationship is between each feature
    /// and your model's predictions. If these relationships are too complex (very wiggly), your model might be
    /// overfitting. If they're too simple (very flat), your model might be underfitting.
    /// 
    /// The method calculates a "nonlinearity score" for each feature and compares the average score
    /// against thresholds to determine if your model is overfitting, underfitting, or has a good fit.
    /// </para>
    /// </remarks>
    private FitType DetermineFitType(Dictionary<string, Vector<T>> pdpResults)
    {
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sumNonlinearity = nonlinearityScores.Aggregate(NumOps.Zero, (acc, score) => NumOps.Add(acc, score));
        var averageNonlinearity = NumOps.Divide(sumNonlinearity, NumOps.FromDouble(nonlinearityScores.Count));

        if (NumOps.GreaterThan(averageNonlinearity, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(averageNonlinearity, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <returns>A value between 0 and 1 representing the confidence level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident we are in our assessment of your model's fit.
    /// A value closer to 1 means we're very confident in our assessment, while a value closer to 0
    /// means we're less certain. The confidence is based on how far the nonlinearity scores are from
    /// the threshold for overfitting.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sumNonlinearity = nonlinearityScores.Aggregate(NumOps.Zero, (acc, score) => NumOps.Add(acc, score));
        var averageNonlinearity = NumOps.Divide(sumNonlinearity, NumOps.FromDouble(nonlinearityScores.Count));

        // Normalize confidence level to be between 0 and 1
        return NumOps.Subtract(NumOps.One,
            NumOps.Divide(
                averageNonlinearity,
                NumOps.FromDouble(_options.OverfitThreshold)
            )
        );
    }

    /// <summary>
    /// Calculates partial dependence plots for all features in the model.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <returns>A dictionary mapping feature names to their partial dependence plots.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a special plot for each feature in your data. Each plot shows
    /// how that feature affects your model's predictions when all other features are held constant.
    /// 
    /// For example, if you have a feature "age" in your data, the partial dependence plot would show
    /// how predictions change as age increases or decreases, while accounting for all other features.
    /// These plots help us understand which features have complex relationships with the target variable.
    /// </para>
    /// </remarks>
    private Dictionary<string, Vector<T>> CalculatePartialDependencePlots(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var pdpResults = new Dictionary<string, Vector<T>>();
        var features = evaluationData.ModelStats.FeatureNames;

        foreach (var feature in features)
        {
            var featureValues = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.FeatureValues[feature]);
            var pdp = CalculatePartialDependencePlot(evaluationData, feature, featureValues);
            pdpResults[feature] = pdp;
        }

        return pdpResults;
    }

    /// <summary>
    /// Calculates a partial dependence plot for a specific feature.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <param name="feature">The name of the feature to analyze.</param>
    /// <param name="featureValues">The values of the feature in the dataset.</param>
    /// <returns>A vector representing the partial dependence plot for the feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a plot showing how a single feature affects your model's predictions.
    /// It works by:
    /// 1. Finding all unique values of the feature in your dataset
    /// 2. For each unique value:
    ///    - Creating a modified version of your dataset where this feature has the same value for all rows
    ///    - Making predictions with your model on this modified dataset
    ///    - Calculating the average prediction
    /// 3. The resulting plot shows how predictions change as the feature value changes
    /// 
    /// This helps us understand if your model has learned a simple relationship (like a straight line)
    /// or a complex relationship (with many ups and downs) for each feature.
    /// </para>
    /// </remarks>
    private Vector<T> CalculatePartialDependencePlot(ModelEvaluationData<T, TInput, TOutput> evaluationData, string feature, Vector<T> featureValues)
    {
        var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();
        var pdp = new Vector<T>(uniqueValues.Count);

        for (int i = 0; i < uniqueValues.Count; i++)
        {
            var value = uniqueValues[i];
            var modifiedMatrix = CreateModifiedDataset(evaluationData, feature, value);

            Vector<T> predictions;
            if (evaluationData.ModelStats.Model == null)
            {
                predictions = Vector<T>.Empty();
            }
            else
            {
                // Convert the matrix to the appropriate input type
                var modelPredictions = evaluationData.ModelStats.Model.Predict((TInput)(object)modifiedMatrix);
                predictions = ConversionsHelper.ConvertToVector<T, TOutput>(modelPredictions);
            }

            pdp[i] = predictions.Average();
        }

        return pdp;
    }

    /// <summary>
    /// Creates a modified dataset where a specific feature has the same value across all samples.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <param name="feature">The name of the feature to modify.</param>
    /// <param name="value">The value to set for the specified feature across all samples.</param>
    /// <returns>A modified copy of the feature matrix with the specified feature set to the same value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a "what-if" version of your dataset. It takes your original data
    /// and changes just one feature (column) so that it has the same value for every row. This helps us
    /// isolate the effect of that specific feature on your model's predictions.
    /// 
    /// For example, if you have a feature "age" and you want to see how your model would predict if everyone
    /// had the same age (say, 30), this method creates that modified dataset for you.
    /// </para>
    /// </remarks>
    private Matrix<T> CreateModifiedDataset(ModelEvaluationData<T, TInput, TOutput> evaluationData, string feature, T value)
    {
        var modifiedData = ConversionsHelper.ConvertToMatrix<T, TInput>(evaluationData.ModelStats.Features).Clone();
        var featureIndex = evaluationData.ModelStats.FeatureNames.IndexOf(feature);

        for (int i = 0; i < modifiedData.Rows; i++)
        {
            modifiedData[i, featureIndex] = value;
        }

        return modifiedData;
    }

    /// <summary>
    /// Calculates nonlinearity scores for each feature based on their partial dependence plots.
    /// </summary>
    /// <param name="pdpResults">Dictionary mapping feature names to their partial dependence plots.</param>
    /// <returns>A list of nonlinearity scores corresponding to each feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how "wiggly" or complex the relationship is between each feature
    /// and your model's predictions. A higher score means the relationship is more complex (has more ups and downs),
    /// while a lower score means the relationship is simpler (more like a straight line).
    /// 
    /// These scores help us identify which features might be causing your model to overfit (learn patterns
    /// that are too complex) or underfit (miss important patterns in the data).
    /// </para>
    /// </remarks>
    private List<T> CalculateNonlinearityScores(Dictionary<string, Vector<T>> pdpResults)
    {
        var nonlinearityScores = new List<T>();

        foreach (var pdp in pdpResults.Values)
        {
            var nonlinearity = CalculateNonlinearity(pdp);
            nonlinearityScores.Add(nonlinearity);
        }

        return nonlinearityScores;
    }

    /// <summary>
    /// Calculates the nonlinearity of a partial dependence plot.
    /// </summary>
    /// <param name="pdp">The partial dependence plot vector to analyze.</param>
    /// <returns>A measure of nonlinearity (variability) in the plot.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how "wiggly" a single feature's relationship is with your model's
    /// predictions. It works by:
    /// 1. Finding the difference between adjacent points in the plot
    /// 2. Taking the absolute value of these differences (making them all positive)
    /// 3. Calculating the standard deviation (a measure of how spread out these differences are)
    /// 
    /// A higher standard deviation means the relationship has more dramatic changes (very wiggly),
    /// which could indicate overfitting. A lower standard deviation means the relationship is smoother,
    /// which could indicate a simpler relationship or potentially underfitting.
    /// </para>
    /// </remarks>
    private T CalculateNonlinearity(Vector<T> pdp)
    {
        var differences = new Vector<T>(pdp.Length - 1);
        for (int i = 0; i < pdp.Length - 1; i++)
        {
            differences[i] = NumOps.Abs(NumOps.Subtract(pdp[i + 1], pdp[i]));
        }

        return StatisticsHelper<T>.CalculateStandardDeviation(differences);
    }

    /// <summary>
    /// Generates recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type (overfit, underfit, or good fit).</param>
    /// <param name="evaluationData">Data containing model performance metrics and feature information.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of practical suggestions to improve your model based on
    /// whether it's overfitting (too complex), underfitting (too simple), or has a good fit.
    /// It first calculates partial dependence plots to understand how each feature affects your model,
    /// then uses this information to provide targeted recommendations.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        return GenerateRecommendations(fitType, pdpResults);
    }

    /// <summary>
    /// Generates detailed recommendations based on the fit type and partial dependence plots.
    /// </summary>
    /// <param name="fitType">The detected fit type (overfit, underfit, or good fit).</param>
    /// <param name="pdpResults">Dictionary mapping feature names to their partial dependence plots.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides specific advice to improve your model based on its current performance.
    /// The recommendations are tailored to whether your model is:
    /// 
    /// - Overfitting: Your model is too complex and has "memorized" the training data instead of learning general patterns.
    ///   Recommendations focus on simplifying your model and identifying which features might be causing the problem.
    ///   
    /// - Underfitting: Your model is too simple and isn't capturing important patterns in your data.
    ///   Recommendations focus on making your model more complex and suggesting features that might benefit from
    ///   nonlinear transformations (like squaring a value or taking its logarithm).
    ///   
    /// - Good Fit: Your model has found a good balance between simplicity and complexity.
    ///   Recommendations focus on fine-tuning and validating your model.
    ///   
    /// The method also identifies the top 5 most "nonlinear" features (those with the most complex relationships
    /// with your target variable), which can help you understand which features have the biggest impact on your model.
    /// </para>
    /// </remarks>
    private List<string> GenerateRecommendations(FitType fitType, Dictionary<string, Vector<T>> pdpResults)
    {
        var recommendations = new List<string>();
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sortedFeatures = pdpResults.Keys.OrderByDescending(f => nonlinearityScores[pdpResults.Keys.ToList().IndexOf(f)]).ToList();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Increase regularization strength.");
                recommendations.Add("2. Reduce model complexity.");
                recommendations.Add("3. Gather more training data.");
                recommendations.Add("4. Consider simplifying or removing highly nonlinear features:");
                for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
                {
                    recommendations.Add($"   - {sortedFeatures[i]}");
                }
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Add more relevant features or engineer new features.");
                recommendations.Add("4. Consider adding nonlinear transformations for these features:");
                for (int i = sortedFeatures.Count - 1; i >= Math.Max(0, sortedFeatures.Count - 5); i--)
                {
                    recommendations.Add($"   - {sortedFeatures[i]}");
                }
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Fine-tune hyperparameters for potential improvements.");
                recommendations.Add("2. Validate the model on new, unseen data.");
                recommendations.Add("3. Monitor model performance over time for potential drift.");
                break;
        }

        recommendations.Add("Top 5 most nonlinear features:");
        for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
        {
            recommendations.Add($"   - {sortedFeatures[i]}");
        }

        return recommendations;
    }
}
