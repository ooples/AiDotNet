namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that selects features based on importance weights from a trained model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is an "embedded" feature selection method that extracts feature
/// importance scores from a trained model and selects features based on those scores.
/// </para>
/// <para>
/// Think of it like asking an expert (the trained model) which features they think are most important,
/// and then keeping only those important features. Different types of models calculate importance
/// in different ways:
/// </para>
/// <para>
/// - Tree-based models (Decision Trees, Random Forests): Use how often a feature is used to split
///   the data and how much it reduces error when used
/// - Linear models with L1 regularization (Lasso): Use the absolute values of coefficients
/// - Neural networks: Can use various methods like gradient-based importance
/// </para>
/// <para>
/// This approach is efficient because you only need to train the model once, then use its knowledge
/// to select features. This is faster than wrapper methods (like SequentialFeatureSelector) which
/// require training many models.
/// </para>
/// </remarks>
public class SelectFromModel<T, TInput> : FeatureSelectorBase<T, TInput>
{

    /// <summary>
    /// The trained model from which to extract feature importances.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the model that has learned which features are important.
    /// It must be already trained on your data before using this selector.
    /// </remarks>
    private readonly IFeatureImportance<T> _model;

    /// <summary>
    /// The threshold strategy to use for selecting features.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how to decide which features are "important enough"
    /// to keep. Options include keeping features above the mean or median importance.
    /// </remarks>
    private readonly ImportanceThresholdStrategy? _thresholdStrategy;

    /// <summary>
    /// A specific threshold value for feature importance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If specified, only features with importance above this value are kept.
    /// This gives you direct control over the cutoff point.
    /// </remarks>
    private readonly T _customThreshold;

    /// <summary>
    /// The maximum number of features to select.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If specified, keeps at most this many features (the top K by importance),
    /// regardless of their importance scores.
    /// </remarks>
    private readonly int? _maxFeatures;

    /// <summary>
    /// The total number of features in the original dataset.
    /// </summary>
    private int _totalFeatures;

    /// <summary>
    /// Initializes a new instance of the SelectFromModel class using a threshold strategy.
    /// </summary>
    /// <param name="model">A trained model that provides feature importance scores.</param>
    /// <param name="thresholdStrategy">The strategy for determining the importance threshold (default: Mean).</param>
    /// <param name="maxFeatures">Optional maximum number of features to select.</param>
    /// <param name="higherDimensionStrategy">Strategy for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a selector that uses a threshold strategy
    /// (Mean or Median) to decide which features to keep.
    /// </para>
    /// <para>
    /// The model parameter must be a trained model that implements IFeatureImportance. Common models include:
    /// - Decision Trees
    /// - Random Forests
    /// - Gradient Boosting models
    /// - Lasso (L1-regularized linear regression)
    /// </para>
    /// <para>
    /// The thresholdStrategy determines the cutoff:
    /// - Mean: Keep features with importance above the average
    /// - Median: Keep the top 50% of features
    /// </para>
    /// <para>
    /// If maxFeatures is specified, the selector will keep at most that many features, even if
    /// more features exceed the threshold.
    /// </para>
    /// </remarks>
    public SelectFromModel(
        IFeatureImportance<T> model,
        ImportanceThresholdStrategy thresholdStrategy = ImportanceThresholdStrategy.Mean,
        int? maxFeatures = null,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _thresholdStrategy = thresholdStrategy;
        _customThreshold = NumOps.Zero;
        _maxFeatures = maxFeatures;
    }

    /// <summary>
    /// Initializes a new instance of the SelectFromModel class using a custom threshold value.
    /// </summary>
    /// <param name="model">A trained model that provides feature importance scores.</param>
    /// <param name="threshold">The minimum importance value for a feature to be selected.</param>
    /// <param name="maxFeatures">Optional maximum number of features to select.</param>
    /// <param name="higherDimensionStrategy">Strategy for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a selector that uses a specific threshold value
    /// to decide which features to keep. Only features with importance >= threshold are selected.
    /// </para>
    /// <para>
    /// This gives you precise control over the cutoff point. For example, if you set threshold = 0.01,
    /// only features with importance of at least 0.01 will be kept.
    /// </para>
    /// </remarks>
    public SelectFromModel(
        IFeatureImportance<T> model,
        T threshold,
        int? maxFeatures = null,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _thresholdStrategy = null;
        _customThreshold = threshold;
        _maxFeatures = maxFeatures;
    }

    /// <summary>
    /// Initializes a new instance of the SelectFromModel class that selects top K features.
    /// </summary>
    /// <param name="model">A trained model that provides feature importance scores.</param>
    /// <param name="k">The number of top features to select.</param>
    /// <param name="higherDimensionStrategy">Strategy for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a selector that keeps exactly K features -
    /// the top K features with the highest importance scores.
    /// </para>
    /// <para>
    /// This is useful when you know exactly how many features you want, regardless of their
    /// actual importance values.
    /// </para>
    /// </remarks>
    public SelectFromModel(
        IFeatureImportance<T> model,
        int k,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _thresholdStrategy = null;
        _customThreshold = NumOps.Zero;
        _maxFeatures = k;
    }

    /// <summary>
    /// Determines which features to select based on model importance scores.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <returns>A list of indices of the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Extracts feature importance scores from the trained model
    /// 2. Determines the threshold based on the specified strategy or custom value
    /// 3. Selects features that meet the importance criteria
    /// 4. Optionally limits to the top K features if maxFeatures is specified
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        _totalFeatures = numFeatures;

        // Get feature importance scores from the model
        var importanceDict = _model.GetFeatureImportance();

        // Convert to a list of (index, importance) pairs
        var featureImportances = ParseFeatureImportances(importanceDict, numFeatures);

        // Sort by importance descending
        var sortedFeatures = featureImportances
            .OrderByDescending(fi => fi.importance, Comparer<T>.Create((a, b) =>
            {
                if (NumOps.GreaterThan(a, b)) return 1;
                if (NumOps.LessThan(a, b)) return -1;
                return 0;
            }))
            .ToList();

        List<int> selectedFeatures;

        // Check if we're in "top-K only" mode (no threshold strategy or custom threshold, just maxFeatures)
        // This is when constructor with k parameter was used
        if (_thresholdStrategy == null && NumOps.Equals(_customThreshold, NumOps.Zero) && _maxFeatures.HasValue)
        {
            // Top-K mode: select top K features by importance, no threshold filtering
            selectedFeatures = sortedFeatures
                .Take(_maxFeatures.Value)
                .Select(fi => fi.index)
                .ToList();
        }
        else
        {
            // Threshold-based mode: determine threshold and filter
            T threshold;
            if (_thresholdStrategy == null && !NumOps.Equals(_customThreshold, NumOps.Zero))
            {
                // Use custom threshold
                threshold = _customThreshold;
            }
            else if (_thresholdStrategy == ImportanceThresholdStrategy.Median)
            {
                // Use median
                threshold = CalculateMedian(featureImportances.Select(fi => fi.importance).ToList());
            }
            else // Default to Mean
            {
                // Use mean
                var importances = featureImportances.Select(fi => fi.importance).ToList();
                threshold = importances.Count > 0
                    ? StatisticsHelper<T>.CalculateMean(importances)
                    : NumOps.Zero;
            }

            // Select features above threshold
            var tolerance = GetThresholdTolerance();
            var effectiveThreshold = NumOps.Subtract(threshold, tolerance);
            selectedFeatures = sortedFeatures
                .Where(fi => NumOps.GreaterThanOrEquals(fi.importance, effectiveThreshold))
                .Select(fi => fi.index)
                .ToList();

            // Apply max features limit if specified
            if (_maxFeatures.HasValue && selectedFeatures.Count > _maxFeatures.Value)
            {
                selectedFeatures = selectedFeatures.Take(_maxFeatures.Value).ToList();
            }
        }

        // Ensure at least one feature is selected
        if (selectedFeatures.Count == 0 && sortedFeatures.Count > 0)
        {
            selectedFeatures.Add(sortedFeatures[0].index);
        }

        return selectedFeatures.OrderBy(x => x).ToList();
    }

    private T GetThresholdTolerance()
    {
        if (typeof(T) == typeof(float))
        {
            return NumOps.FromDouble(1e-6);
        }

        if (typeof(T) == typeof(double))
        {
            return NumOps.FromDouble(1e-12);
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Parses the feature importance dictionary from the model into a list of (index, importance) pairs.
    /// </summary>
    /// <param name="importanceDict">The dictionary of feature names to importance scores.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <returns>A list of (index, importance) tuples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Models return feature importances with feature names as keys.
    /// This method converts those names to numeric indices so we can select the right columns
    /// from the data.
    /// </para>
    /// <para>
    /// Feature names are typically in the format "Feature_0", "Feature_1", etc., where the
    /// number indicates the column index.
    /// </para>
    /// </remarks>
    private List<(int index, T importance)> ParseFeatureImportances(
        Dictionary<string, T> importanceDict,
        int numFeatures)
    {
        var result = new List<(int index, T importance)>();

        // Try to parse feature names as "Feature_N" format
        foreach (var kvp in importanceDict)
        {
            if (TryParseFeatureIndex(kvp.Key, out int index) && index >= 0 && index < numFeatures)
            {
                result.Add((index, kvp.Value));
            }
        }

        // If parsing failed, assume dictionary keys are ordered 0, 1, 2, ...
        if (result.Count == 0)
        {
            int index = 0;
            foreach (var kvp in importanceDict.OrderBy(x => x.Key))
            {
                if (index < numFeatures)
                {
                    result.Add((index, kvp.Value));
                    index++;
                }
            }
        }

        // Fill in any missing features with zero importance
        var existingIndices = new HashSet<int>(result.Select(r => r.index));
        for (int i = 0; i < numFeatures; i++)
        {
            if (!existingIndices.Contains(i))
            {
                result.Add((i, NumOps.Zero));
            }
        }

        return result;
    }

    /// <summary>
    /// Attempts to parse a feature index from a feature name.
    /// </summary>
    /// <param name="featureName">The feature name (e.g., "Feature_5").</param>
    /// <param name="index">The parsed index.</param>
    /// <returns>True if parsing succeeded, false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts the number from feature names like "Feature_5",
    /// "feature5", "col_5", etc.
    /// </remarks>
    private bool TryParseFeatureIndex(string featureName, out int index)
    {
        index = -1;

        if (string.IsNullOrEmpty(featureName))
        {
            return false;
        }

        // Try common patterns: "Feature_N", "feature_N", "FeatureN", "col_N", "x_N", etc.
        var patterns = new[]
        {
            @"Feature_(\d+)",
            @"feature_(\d+)",
            @"Feature(\d+)",
            @"feature(\d+)",
            @"col_(\d+)",
            @"x_(\d+)",
            @"X_(\d+)",
            @"f_(\d+)",
            @"^(\d+)$" // Just a number
        };

        foreach (var pattern in patterns)
        {
            var match = RegexHelper.Match(featureName, pattern, System.Text.RegularExpressions.RegexOptions.None);
            if (match.Success && int.TryParse(match.Groups[1].Value, out index))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Calculates the median of a list of values.
    /// </summary>
    /// <param name="values">The values to calculate the median of.</param>
    /// <returns>The median value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median is the middle value when all values are sorted.
    /// If there's an even number of values, it's the average of the two middle values.
    /// </para>
    /// <para>
    /// For example:
    /// - For [1, 2, 3, 4, 5], the median is 3 (the middle value)
    /// - For [1, 2, 3, 4], the median is 2.5 (average of 2 and 3)
    /// </para>
    /// </remarks>
    private T CalculateMedian(List<T> values)
    {
        if (values.Count == 0)
        {
            return NumOps.Zero;
        }

        var sorted = values.OrderBy(x => x, Comparer<T>.Create((a, b) =>
        {
            if (NumOps.GreaterThan(a, b)) return 1;
            if (NumOps.LessThan(a, b)) return -1;
            return 0;
        })).ToList();
        int mid = sorted.Count / 2;

        // Even number of elements: average the two middle values
        // Odd number of elements: return the middle value
        return (sorted.Count % 2 == 0)
            ? NumOps.Divide(
                NumOps.Add(sorted[mid - 1], sorted[mid]),
                NumOps.FromDouble(2.0))
            : sorted[mid];
    }
}



