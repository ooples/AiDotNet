namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality using Shapley values to determine feature importance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you understand which features (input variables) in your model are most 
/// important for making predictions. It uses something called "Shapley values" from game theory to 
/// fairly distribute the "credit" for predictions among all your features.
/// 
/// Think of it like figuring out which players on a sports team contributed most to winning a game.
/// Shapley values help determine if your model is:
/// - Overfitting: Relying too much on just a few features (like a team depending only on one star player)
/// - Underfitting: Not using features effectively (like a team not using any player's strengths)
/// - Good fit: Using features in a balanced way (like a well-coordinated team)
/// 
/// This detector will give you recommendations on how to improve your model based on this analysis.
/// </para>
/// </remarks>
public class ShapleyValueFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Shapley value fit detector.
    /// </summary>
    private readonly ShapleyValueFitDetectorOptions _options;

    /// <summary>
    /// Random number generator used for Monte Carlo sampling.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the ShapleyValueFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options for the detector. If null, default options will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new Shapley value detector.
    /// You can customize how it works by providing options, or just use the default settings.
    /// </para>
    /// </remarks>
    public ShapleyValueFitDetector(ShapleyValueFitDetectorOptions options)
    {
        _options = options ?? new ShapleyValueFitDetectorOptions();
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Analyzes the model's fit using Shapley values and returns detailed results.
    /// </summary>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model and tells you if it's a good fit.
    /// It calculates how important each feature is, determines if your model is overfitting or underfitting,
    /// and gives you specific recommendations on how to improve it.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        var fitType = DetermineFitType(shapleyValues);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, shapleyValues);
        var shapleyValuesStrings = shapleyValues.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value
        );

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ShapleyValues", shapleyValuesStrings }
            }
        };
    }

    /// <summary>
    /// Determines the fit type of the model based on evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <returns>The determined fit type (Overfit, Underfit, or GoodFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at your model's performance and decides if it's:
    /// - Overfitting: Your model is "memorizing" the training data rather than learning general patterns
    /// - Underfitting: Your model is too simple to capture the patterns in your data
    /// - Good fit: Your model has found the right balance
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        return DetermineFitType(shapleyValues);
    }

    /// <summary>
    /// Determines the fit type based on calculated Shapley values.
    /// </summary>
    /// <param name="shapleyValues">Dictionary mapping feature names to their Shapley values.</param>
    /// <returns>The determined fit type (Overfit, Underfit, or GoodFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes how many features are actually important to your model.
    /// - If only a few features matter a lot, it might be overfitting (focusing too much on specific patterns)
    /// - If too many features matter equally, it might be underfitting (not finding specific patterns)
    /// - If there's a good balance of important features, it's likely a good fit
    /// </para>
    /// </remarks>
    private FitType DetermineFitType(Dictionary<string, T> shapleyValues)
    {
        var sortedValues = shapleyValues.OrderByDescending(kv => kv.Value).ToList();
        var totalImportance = sortedValues.Aggregate(NumOps.Zero, (acc, kv) => NumOps.Add(acc, kv.Value));
        var cumulativeImportance = NumOps.Zero;
        var featureCount = 0;

        foreach (var kv in sortedValues)
        {
            cumulativeImportance = NumOps.Add(cumulativeImportance, kv.Value);
            featureCount++;

            if (NumOps.GreaterThanOrEquals(
                NumOps.Divide(cumulativeImportance, totalImportance),
                NumOps.FromDouble(_options.ImportanceThreshold)))
            {
                break;
            }
        }

        var importantFeatureRatio = NumOps.Divide(NumOps.FromDouble(featureCount), NumOps.FromDouble(shapleyValues.Count));

        if (NumOps.LessThanOrEquals(importantFeatureRatio, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.GreaterThanOrEquals(importantFeatureRatio, NumOps.FromDouble(_options.UnderfitThreshold)))
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
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <returns>A value representing the confidence level of the fit detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you how confident the detector is about its assessment.
    /// A higher value means more confidence in the fit type determination.
    /// 
    /// The confidence is based on how clearly the feature importance is distributed:
    /// - If there's a clear distinction between important and unimportant features, confidence is high
    /// - If feature importance is more evenly distributed, confidence is lower
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        var sortedValues = shapleyValues.Values.OrderByDescending(v => v).ToList();
        var totalImportance = sortedValues.Aggregate(NumOps.Zero, (acc, v) => NumOps.Add(acc, v));
        var cumulativeImportance = NumOps.Zero;
        var featureCount = 0;

        foreach (var value in sortedValues)
        {
            cumulativeImportance = NumOps.Add(cumulativeImportance, value);
            featureCount++;

            if (NumOps.GreaterThanOrEquals(
                NumOps.Divide(cumulativeImportance, totalImportance),
                NumOps.FromDouble(_options.ImportanceThreshold)))
            {
                break;
            }
        }

        return NumOps.Subtract(
            NumOps.One,
            NumOps.Divide(
                NumOps.FromDouble(featureCount),
                NumOps.FromDouble(shapleyValues.Count)
            )
        );
    }

    /// <summary>
    /// Calculates Shapley values for each feature in the model.
    /// </summary>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <param name="features">List of feature names to calculate Shapley values for.</param>
    /// <returns>Dictionary mapping feature names to their calculated Shapley values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how much each feature contributes to your model's predictions.
    /// 
    /// Shapley values come from game theory and provide a fair way to distribute "credit" among features.
    /// The process works by:
    /// 1. Randomly ordering features many times
    /// 2. For each feature, measuring how much the model's performance changes when that feature is added
    /// 3. Averaging these measurements to get a fair assessment of each feature's importance
    /// 
    /// Higher Shapley values mean more important features.
    /// </para>
    /// </remarks>
    private Dictionary<string, T> CalculateShapleyValues(ModelEvaluationData<T, TInput, TOutput> evaluationData, List<string> features)
    {
        var shapleyValues = new Dictionary<string, T>();
        var n = features.Count;

        foreach (var feature in features)
        {
            T shapleyValue = NumOps.Zero;

            for (int i = 0; i < _options.MonteCarloSamples; i++)
            {
                var permutation = features.OrderBy(x => _random.Next()).ToList();
                var index = permutation.IndexOf(feature);
                var withFeature = new HashSet<string>(permutation.Take(index + 1));
                var withoutFeature = new HashSet<string>(permutation.Take(index));

                var marginalContribution = NumOps.Subtract(
                    CalculatePerformance(evaluationData, withFeature),
                    CalculatePerformance(evaluationData, withoutFeature));

                shapleyValue = NumOps.Add(shapleyValue, marginalContribution);
            }

            shapleyValues[feature] = NumOps.Divide(shapleyValue, NumOps.FromDouble(_options.MonteCarloSamples));
        }

        return shapleyValues;
    }

    /// <summary>
    /// Retrieves the list of feature names from the evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <returns>A list of feature names used in the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply extracts the names of all the input variables (features) 
    /// that your model uses to make predictions. Think of it as getting a list of all the 
    /// different pieces of information your model considers when making a decision.
    /// </para>
    /// </remarks>
    private List<string> GetFeatures(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        return evaluationData.ModelStats.FeatureNames;
    }

    /// <summary>
    /// Calculates the performance of the model using only a specific subset of features.
    /// </summary>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <param name="features">The set of feature names to include in the calculation.</param>
    /// <returns>A performance metric (R² score) for the model using only the specified features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how well your model performs when using only certain features.
    /// 
    /// Imagine you have a recipe (your model) that uses 10 ingredients (features). This method lets you 
    /// test how good the recipe would be if you only used, say, 5 specific ingredients. It helps 
    /// determine which ingredients are most important for making the dish taste good.
    /// 
    /// The R² score returned is a measure of how well your model fits the data - higher values 
    /// (closer to 1.0) mean better performance.
    /// </para>
    /// </remarks>
    private T CalculatePerformance(ModelEvaluationData<T, TInput, TOutput> evaluationData, HashSet<string> features)
    {
        var subsetFeatures = evaluationData.ModelStats.FeatureValues
            .Where(kv => features.Contains(kv.Key))
            .ToDictionary(
                kv => kv.Key,
                kv => ConversionsHelper.ConvertToVector<T, TOutput>(kv.Value)
            );

        var featureMatrix = CreateFeatures(subsetFeatures);

        Vector<T> predictions;
        if (evaluationData.ModelStats.Model == null)
        {
            predictions = Vector<T>.Empty();
        }
        else
        {
            // Convert the matrix to the appropriate input type
            var modelPredictions = evaluationData.ModelStats.Model.Predict((TInput)(object)featureMatrix);
            predictions = ConversionsHelper.ConvertToVector<T, TOutput>(modelPredictions);
        }

        return StatisticsHelper<T>.CalculateR2(
            ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual),
            predictions
        );
    }

    /// <summary>
    /// Creates a matrix from feature vectors for model prediction.
    /// </summary>
    /// <param name="features">Dictionary mapping feature names to their value vectors.</param>
    /// <returns>A matrix where each column represents a feature and each row represents a data point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method organizes your data in a format that the model can use for making predictions.
    /// 
    /// Think of a spreadsheet: each row represents one data point (like a person or a house), and 
    /// each column represents one feature (like age, height, or number of bedrooms). This method 
    /// takes your individual feature columns and arranges them into this spreadsheet-like structure 
    /// (a matrix) so your model can process all the data together.
    /// </para>
    /// </remarks>
    private Matrix<T> CreateFeatures(Dictionary<string, Vector<T>> features)
    {
        int rowCount = features.First().Value.Length;
        int colCount = features.Count;
        var matrix = new Matrix<T>(rowCount, colCount);

        int colIndex = 0;
        foreach (var feature in features.Values)
        {
            for (int i = 0; i < rowCount; i++)
            {
                matrix[i, colIndex] = feature[i];
            }
            colIndex++;
        }

        return matrix;
    }

    /// <summary>
    /// Generates recommendations for improving the model based on its fit type.
    /// </summary>
    /// <param name="fitType">The determined fit type (Overfit, Underfit, or GoodFit).</param>
    /// <param name="evaluationData">Data containing the model and its performance metrics.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of suggestions to help you improve your model.
    /// 
    /// It first calculates how important each feature is (using Shapley values), then passes this 
    /// information along with the fit type to another method that creates specific recommendations.
    /// 
    /// This is like a doctor first running tests, then using the test results to recommend treatment.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        return GenerateRecommendations(fitType, shapleyValues);
    }

    /// <summary>
    /// Generates specific recommendations based on the fit type and feature importance.
    /// </summary>
    /// <param name="fitType">The determined fit type (Overfit, Underfit, or GoodFit).</param>
    /// <param name="shapleyValues">Dictionary mapping feature names to their importance values.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates specific advice for improving your model based on:
    /// 1. Whether your model is overfitting, underfitting, or has a good fit
    /// 2. Which features are most and least important to your model
    /// 
    /// For example:
    /// - If your model is overfitting, it might suggest removing less important features
    /// - If your model is underfitting, it might suggest adding more features
    /// - If your model has a good fit, it might suggest ways to maintain or slightly improve it
    /// 
    /// It also tells you which features are most important, helping you understand what your model 
    /// is actually learning from your data.
    /// </para>
    /// </remarks>
    private List<string> GenerateRecommendations(FitType fitType, Dictionary<string, T> shapleyValues)
    {
        var recommendations = new List<string>();
        var sortedFeatures = shapleyValues.OrderByDescending(kv => kv.Value).ToList();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Increase regularization strength.");
                recommendations.Add("2. Reduce model complexity.");
                recommendations.Add("3. Gather more training data.");
                recommendations.Add("4. Consider removing less important features:");
                for (int i = sortedFeatures.Count - 1; i >= Math.Max(0, sortedFeatures.Count - 5); i--)
                {
                    recommendations.Add($"   - {sortedFeatures[i].Key}");
                }
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Add more relevant features.");
                recommendations.Add("4. Engineer new features based on domain knowledge.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Fine-tune hyperparameters for potential improvements.");
                recommendations.Add("2. Validate the model on new, unseen data.");
                recommendations.Add("3. Monitor model performance over time for potential drift.");
                break;
        }

        recommendations.Add("Top 5 most important features:");
        for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
        {
            recommendations.Add($"   - {sortedFeatures[i].Key}");
        }

        return recommendations;
    }
}
