namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for working with model types and their metadata.
/// </summary>
/// <remarks>
/// <para>
/// This static class contains methods to access and utilize the metadata attached to model types,
/// such as determining a model's category, retrieving valid metrics, or checking
/// if a particular metric is appropriate for a given model.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class provides tools to help you work with model types and
/// understand their capabilities. You can use these methods to determine what a model
/// is good for, what metrics should be used to evaluate it, and how it relates to
/// other models in the system.
/// </para>
/// </remarks>
public static class ModelTypeHelper
{
    /// <summary>
    /// Gets the category to which a model type belongs.
    /// </summary>
    /// <param name="modelType">The model type to check.</param>
    /// <returns>The category of the specified model type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you what broad category a model belongs to,
    /// such as Regression, Classification, or Neural Network. This can help you understand
    /// what kinds of problems the model is designed to solve.
    /// </para>
    /// </remarks>
    public static ModelCategory GetCategory(ModelType modelType)
    {
        var fieldInfo = typeof(ModelType).GetField(modelType.ToString());
        if (fieldInfo == null)
            return ModelCategory.None;

        var attributes = fieldInfo.GetCustomAttributes(typeof(ModelInfoAttribute), false);
        if (attributes.Length == 0)
            return ModelCategory.None;

        return ((ModelInfoAttribute)attributes[0]).Category;
    }

    /// <summary>
    /// Checks if a model type supports a specific metric group.
    /// </summary>
    /// <param name="modelType">The model type to check.</param>
    /// <param name="metricGroup">The metric group to check for support.</param>
    /// <returns>True if the model supports the specified metric group; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps you determine whether a particular type of
    /// evaluation metric makes sense for a given model. For example, it would tell you
    /// that using classification metrics (like accuracy) doesn't make sense for a 
    /// regression model that predicts continuous values.
    /// </para>
    /// </remarks>
    public static bool SupportsMetricGroup(ModelType modelType, MetricGroups metricGroup)
    {
        var fieldInfo = typeof(ModelType).GetField(modelType.ToString());
        if (fieldInfo == null)
            return false;

        var attributes = fieldInfo.GetCustomAttributes(typeof(ModelInfoAttribute), false);
        if (attributes.Length == 0)
            return false;

        var validGroups = ((ModelInfoAttribute)attributes[0]).ValidMetrics;
        return validGroups.Contains(metricGroup);
    }

    /// <summary>
    /// Gets all metric groups supported by a model type.
    /// </summary>
    /// <param name="modelType">The model type to check.</param>
    /// <returns>An array of supported metric groups for the specified model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method returns all the different types of metrics
    /// that make sense for evaluating a particular model. It helps you understand
    /// the different ways you can measure how well your model is performing.
    /// </para>
    /// </remarks>
    public static MetricGroups[] GetSupportedMetricGroups(ModelType modelType)
    {
        var fieldInfo = typeof(ModelType).GetField(modelType.ToString());
        if (fieldInfo == null)
            return [];

        var attributes = fieldInfo.GetCustomAttributes(typeof(ModelInfoAttribute), false);
        if (attributes.Length == 0)
            return [];

        return ((ModelInfoAttribute)attributes[0]).ValidMetrics;
    }

    /// <summary>
    /// Gets the description of a model type.
    /// </summary>
    /// <param name="modelType">The model type to describe.</param>
    /// <returns>A descriptive string explaining the model type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides a brief explanation of what a particular
    /// model does, helping you understand its purpose and typical use cases without
    /// having to look up detailed documentation.
    /// </para>
    /// </remarks>
    public static string GetDescription(ModelType modelType)
    {
        var fieldInfo = typeof(ModelType).GetField(modelType.ToString());
        if (fieldInfo == null)
            return string.Empty;

        var attributes = fieldInfo.GetCustomAttributes(typeof(ModelInfoAttribute), false);
        if (attributes.Length == 0)
            return string.Empty;

        return ((ModelInfoAttribute)attributes[0]).Description;
    }

    /// <summary>
    /// Checks if a specific metric type is valid for a given model type.
    /// </summary>
    /// <param name="modelType">The model type to check.</param>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric type is valid for the model type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you whether a specific performance metric
    /// (like accuracy or mean squared error) makes sense for a particular model. It helps
    /// prevent the use of inappropriate metrics that could lead to misleading evaluation results.
    /// </para>
    /// </remarks>
    public static bool IsValidMetric(ModelType modelType, MetricType metricType)
    {
        // Get the metric groups for this metric type
        var metricGroups = GetMetricGroups(metricType);

        // Get the supported metric groups for this model type
        var supportedGroups = GetSupportedMetricGroups(modelType);

        // Check if any of the metric's groups are supported by the model
        return metricGroups.Any(group => supportedGroups.Contains(group));
    }

    /// <summary>
    /// Gets all model types that belong to a specific category.
    /// </summary>
    /// <param name="category">The category to filter by.</param>
    /// <returns>An array of model types belonging to the specified category.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lists all the models that fall under a particular
    /// category, such as all regression models or all neural network models. It's helpful
    /// when you know what type of problem you're solving but aren't sure which specific
    /// model to choose.
    /// </para>
    /// </remarks>
    public static ModelType[] GetModelsByCategory(ModelCategory category)
    {
        var allModelTypes = Enum.GetValues(typeof(ModelType)).Cast<ModelType>();
        return [.. allModelTypes.Where(mt => GetCategory(mt) == category)];
    }

    /// <summary>
    /// Determines which metric groups a specific metric type belongs to.
    /// </summary>
    /// <param name="metricType">The metric type to categorize.</param>
    /// <returns>A list of metric groups to which the metric type belongs.</returns>
    /// <remarks>
    /// <para>
    /// This method maps each individual metric to the appropriate groups based on where
    /// the metric is commonly applied. Many metrics naturally belong to multiple groups,
    /// such as MAE which is relevant for both regression and time series analysis.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Some metrics can be used for different types of models. 
    /// For example, Root Mean Squared Error (RMSE) can be used both for regular regression
    /// models and for time series forecasting. This method returns all the different
    /// model types where each metric makes sense to use.
    /// </para>
    /// </remarks>
    public static MetricGroups[] GetMetricGroups(MetricType metricType)
    {
        return metricType switch
        {
            // Regression metrics (many also apply to time series)
            MetricType.R2 => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.AdjustedR2 => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.ExplainedVarianceScore => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MeanPredictionError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MedianPredictionError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MAE => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MSE => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.RMSE => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MAPE => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MeanBiasError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MedianAbsoluteError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MeanSquaredLogError => new[] { MetricGroups.Regression },
            MetricType.SMAPE => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.MaxError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries, MetricGroups.General },
            MetricType.PredictionIntervalCoverage => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.SampleStandardError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.PopulationStandardError => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.AIC => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.BIC => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.AICAlt => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },
            MetricType.RSS => new[] { MetricGroups.Regression, MetricGroups.TimeSeries },

            // Binary classification metrics
            MetricType.Precision => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.Recall => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.F1Score => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.F2Score => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.FBetaScore => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.GMean => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.AUCROC => new[] { MetricGroups.BinaryClassification },
            MetricType.AUCPR => new[] { MetricGroups.BinaryClassification },
            MetricType.AveragePrecision => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.CalibrationError => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.BrierScore => new[] { MetricGroups.BinaryClassification, MetricGroups.NeuralNetwork },

            // Multiclass classification metrics
            MetricType.Accuracy => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.CrossEntropyLoss => new[] { MetricGroups.MulticlassClassification, MetricGroups.NeuralNetwork },
            MetricType.CohenKappa => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.MutualInformation => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.Clustering },
            MetricType.NormalizedMutualInformation => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.Clustering },

            // Time series specific metrics
            MetricType.TheilUStatistic => new[] { MetricGroups.TimeSeries },
            MetricType.DurbinWatsonStatistic => new[] { MetricGroups.TimeSeries },
            MetricType.AutoCorrelationFunction => new[] { MetricGroups.TimeSeries },
            MetricType.PartialAutoCorrelationFunction => new[] { MetricGroups.TimeSeries },
            MetricType.DynamicTimeWarping => new[] { MetricGroups.TimeSeries },

            // Neural network / deep learning metrics
            MetricType.Perplexity => new[] { MetricGroups.NeuralNetwork }, // For language models
            MetricType.KLDivergence => new[] { MetricGroups.NeuralNetwork, MetricGroups.General },
            MetricType.LogLikelihood => new[] { MetricGroups.NeuralNetwork, MetricGroups.General },

            // Clustering metrics
            MetricType.SilhouetteScore => new[] { MetricGroups.Clustering },
            MetricType.VariationOfInformation => new[] { MetricGroups.Clustering },
            MetricType.CalinskiHarabaszIndex => new[] { MetricGroups.Clustering },
            MetricType.DaviesBouldinIndex => new[] { MetricGroups.Clustering },

            // Information Retrieval / Ranking metrics
            MetricType.MeanAveragePrecision => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.NormalizedDiscountedCumulativeGain => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
            MetricType.MeanReciprocalRank => new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },

            // String distance metrics
            MetricType.LevenshteinDistance => new[] { MetricGroups.General },

            // General descriptive statistics
            MetricType.Mean => new[] { MetricGroups.General },
            MetricType.Median => new[] { MetricGroups.General },
            MetricType.Mode => new[] { MetricGroups.General },
            MetricType.Variance => new[] { MetricGroups.General },
            MetricType.StandardDeviation => new[] { MetricGroups.General },
            MetricType.Range => new[] { MetricGroups.General },
            MetricType.InterquartileRange => new[] { MetricGroups.General },
            MetricType.Min => new[] { MetricGroups.General },
            MetricType.Max => new[] { MetricGroups.General },
            MetricType.N => new[] { MetricGroups.General },
            MetricType.FirstQuartile => new[] { MetricGroups.General },
            MetricType.ThirdQuartile => new[] { MetricGroups.General },
            MetricType.MAD => new[] { MetricGroups.General },
            MetricType.Skewness => new[] { MetricGroups.General },
            MetricType.Kurtosis => new[] { MetricGroups.General },

            // Correlation and similarity metrics
            MetricType.PearsonCorrelation => new[] { MetricGroups.Regression, MetricGroups.TimeSeries, MetricGroups.General },
            MetricType.SpearmanCorrelation => new[] { MetricGroups.Regression, MetricGroups.TimeSeries, MetricGroups.General },
            MetricType.KendallTau => new[] { MetricGroups.Regression, MetricGroups.TimeSeries, MetricGroups.General },
            MetricType.CosineSimilarity => new[] { MetricGroups.General },
            MetricType.JaccardSimilarity => new[] { MetricGroups.General, MetricGroups.Clustering },

            // Distance metrics
            MetricType.EuclideanDistance => new[] { MetricGroups.General, MetricGroups.Clustering },
            MetricType.ManhattanDistance => new[] { MetricGroups.General, MetricGroups.Clustering },
            MetricType.HammingDistance => new[] { MetricGroups.General },
            MetricType.MahalanobisDistance => new[] { MetricGroups.General, MetricGroups.Clustering },

            // Advanced statistical metrics
            MetricType.Likelihood => new[] { MetricGroups.General },
            MetricType.BhattacharyyaDistance => new[] { MetricGroups.General },
            MetricType.ConditionNumber => new[] { MetricGroups.General },
            MetricType.LogPointwisePredictiveDensity => new[] { MetricGroups.General },
            MetricType.ObservedTestStatistic => new[] { MetricGroups.General },
            MetricType.MarginalLikelihood => new[] { MetricGroups.General },
            MetricType.ReferenceModelMarginalLikelihood => new[] { MetricGroups.General },
            MetricType.EffectiveNumberOfParameters => new[] { MetricGroups.General },

            // Model structure metrics
            MetricType.CorrelationMatrix => new[] { MetricGroups.General },
            MetricType.CovarianceMatrix => new[] { MetricGroups.General },
            MetricType.VIF => new[] { MetricGroups.General },

            // Distribution metrics
            MetricType.BestDistributionFit => new[] { MetricGroups.General },
            MetricType.BestCorrelationType => new[] { MetricGroups.General },

            // Cross-validation metrics
            MetricType.LeaveOneOutPredictiveDensities => new[] { MetricGroups.General },
            MetricType.PosteriorPredictiveSamples => new[] { MetricGroups.General },
            MetricType.LearningCurve => new[] { MetricGroups.General },

            // Default case for any metrics not explicitly categorized
            _ => new[] { MetricGroups.General }
        };
    }
}