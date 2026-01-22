namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that ranks features using univariate statistical tests and selects the top K features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This feature selector evaluates each feature independently by measuring
/// its statistical relationship with the target variable. It then keeps only the features with
/// the strongest relationships.
/// </para>
/// <para>
/// Think of it like choosing team members by testing each person individually at a specific skill,
/// then picking the top performers. This selector doesn't consider how features work together - it
/// evaluates each one on its own merits.
/// </para>
/// <para>
/// The selector supports three types of statistical tests:
/// - Chi-Squared: Best for categorical features
/// - ANOVA F-Value: Best for continuous features with categorical targets
/// - Mutual Information: Works well for any type of feature
/// </para>
/// </remarks>
public class UnivariateFeatureSelector<T, TInput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// The target variable used to evaluate feature relevance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is what you're trying to predict. Each feature is evaluated
    /// based on how well it relates to this target variable.
    /// </remarks>
    private readonly Vector<T> _target;

    /// <summary>
    /// The scoring function used to evaluate features.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines which statistical test is used to measure how
    /// important each feature is for predicting the target.
    /// </remarks>
    private readonly UnivariateScoringFunction _scoringFunction;

    /// <summary>
    /// The number of top features to select.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After scoring all features, this selector will keep only this many
    /// of the highest-scoring features. If not specified, it keeps 50% of the features.
    /// </remarks>
    private readonly int? _k;

    /// <summary>
    /// Initializes a new instance of the UnivariateFeatureSelector class.
    /// </summary>
    /// <param name="target">The target variable for evaluating feature relevance.</param>
    /// <param name="scoringFunction">The scoring function to use (default: FValue for ANOVA F-test).</param>
    /// <param name="k">The number of top features to select. If null, selects 50% of features.</param>
    /// <param name="higherDimensionStrategy">Strategy for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new univariate feature selector with your chosen settings.
    /// </para>
    /// <para>
    /// The target parameter should be a vector containing the values you're trying to predict (like house prices
    /// or classification labels). Each element corresponds to one sample in your dataset.
    /// </para>
    /// <para>
    /// The scoringFunction determines which statistical test to use:
    /// - ChiSquared: Use when both features and target are categorical
    /// - FValue: Use when features are continuous and target is categorical (default)
    /// - MutualInformation: Use for any combination of feature and target types
    /// </para>
    /// <para>
    /// The k parameter controls how many features to keep. For example, if you have 100 features and set k=10,
    /// only the 10 highest-scoring features will be selected. If you don't specify k, it defaults to keeping
    /// half of your features.
    /// </para>
    /// </remarks>
    public UnivariateFeatureSelector(
        Vector<T> target,
        UnivariateScoringFunction scoringFunction = UnivariateScoringFunction.FValue,
        int? k = null,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _target = target ?? throw new ArgumentNullException(nameof(target));
        _scoringFunction = scoringFunction;

        if (k.HasValue && k.Value <= 0)
        {
            throw new ArgumentException("The number of features to select (k) must be greater than 0.", nameof(k));
        }

        _k = k;
    }

    /// <summary>
    /// Determines which features to select based on univariate statistical scores.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <returns>A list of indices of the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Calculates a statistical score for each feature measuring its relationship with the target
    /// 2. Ranks features by their scores (higher scores = stronger relationship)
    /// 3. Selects the top K features with the highest scores
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // Validate target length matches number of samples
        if (_target.Length != numSamples)
        {
            throw new ArgumentException(
                $"Target length ({_target.Length}) must match the number of samples ({numSamples}).");
        }

        // Determine how many features to select
        int numToSelect = _k ?? Math.Max(1, numFeatures / 2);
        numToSelect = Math.Min(numToSelect, numFeatures);

        // Calculate scores for each feature
        var featureScores = new List<(int index, T score)>();

        for (int i = 0; i < numFeatures; i++)
        {
            var featureVector = ExtractFeatureVector(allFeatures, i, numSamples);
            T score = CalculateScore(featureVector);
            featureScores.Add((i, score));
        }

        // Sort by score descending and select top K
        var selectedIndices = featureScores
            .OrderByDescending(fs => fs.score, Comparer<T>.Create((a, b) =>
            {
                if (NumOps.GreaterThan(a, b)) return 1;
                if (NumOps.LessThan(a, b)) return -1;
                return 0;
            }))
            .Take(numToSelect)
            .Select(fs => fs.index)
            .OrderBy(idx => idx) // Sort indices for consistent output
            .ToList();

        // Safety check: ensure at least one feature is selected
        if (selectedIndices.Count == 0 && featureScores.Count > 0)
        {
            selectedIndices.Add(featureScores[0].index);
        }

        return selectedIndices;
    }

    /// <summary>
    /// Calculates the score for a feature using the specified scoring function.
    /// </summary>
    /// <param name="featureVector">The feature values across all samples.</param>
    /// <returns>The score measuring the feature's relevance to the target.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies the chosen statistical test to measure how
    /// strongly this feature is related to what you're trying to predict.
    /// </para>
    /// <para>
    /// Higher scores indicate stronger relationships, meaning the feature is more useful for
    /// making predictions.
    /// </para>
    /// </remarks>
    private T CalculateScore(Vector<T> featureVector)
    {
        return _scoringFunction switch
        {
            UnivariateScoringFunction.ChiSquared => CalculateChiSquaredScore(featureVector),
            UnivariateScoringFunction.FValue => CalculateFValueScore(featureVector),
            UnivariateScoringFunction.MutualInformation => CalculateMutualInformationScore(featureVector),
            _ => throw new ArgumentException($"Unsupported scoring function: {_scoringFunction}")
        };
    }

    /// <summary>
    /// Calculates the Chi-Squared score for a feature.
    /// </summary>
    /// <param name="featureVector">The feature values across all samples.</param>
    /// <returns>The Chi-Squared statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Chi-Squared test measures how much the distribution of feature
    /// values differs across different target classes. Higher values indicate stronger dependence.
    /// </para>
    /// </remarks>
    private T CalculateChiSquaredScore(Vector<T> featureVector)
    {
        try
        {
            var result = StatisticsHelper<T>.ChiSquareTest(featureVector, _target);
            return result.ChiSquareStatistic;
        }
        catch (ArgumentException)
        {
            // Return zero score if test fails due to invalid arguments (e.g., insufficient variation)
            return NumOps.Zero;
        }
        catch (InvalidOperationException)
        {
            // Return zero score if test fails due to invalid operation (e.g., not enough data)
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Calculates the ANOVA F-value score for a feature.
    /// </summary>
    /// <param name="featureVector">The feature values across all samples.</param>
    /// <returns>The F-statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ANOVA F-value measures whether the means of the feature differ
    /// significantly across different target classes. Higher F-values indicate the feature is better
    /// at distinguishing between classes.
    /// </para>
    /// <para>
    /// For example, if you're predicting whether an email is spam or not, a high F-value for a feature
    /// like "number of exclamation marks" would mean that spam and non-spam emails have significantly
    /// different numbers of exclamation marks.
    /// </para>
    /// </remarks>
    private T CalculateFValueScore(Vector<T> featureVector)
    {
        try
        {
            // Get unique classes in target
            var uniqueClasses = _target.Distinct().ToList();

            if (uniqueClasses.Count < 2)
            {
                // Need at least 2 classes for ANOVA
                return NumOps.Zero;
            }

            // Group feature values by class
            var groups = new List<List<T>>();
            foreach (var targetClass in uniqueClasses)
            {
                var group = new List<T>();
                for (int i = 0; i < _target.Length; i++)
                {
                    if (NumOps.Equals(_target[i], targetClass))
                    {
                        group.Add(featureVector[i]);
                    }
                }
                if (group.Count > 0)
                {
                    groups.Add(group);
                }
            }

            // Calculate ANOVA F-statistic
            return CalculateAnovaFStatistic(groups);
        }
        catch (ArgumentException)
        {
            // Return zero score if calculation fails due to invalid arguments
            return NumOps.Zero;
        }
        catch (InvalidOperationException)
        {
            // Return zero score if calculation fails due to invalid operation
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Calculates the Mutual Information score for a feature.
    /// </summary>
    /// <param name="featureVector">The feature values across all samples.</param>
    /// <returns>The mutual information score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mutual Information measures how much knowing the feature value reduces
    /// uncertainty about the target. Higher values mean the feature provides more information.
    /// </para>
    /// </remarks>
    private T CalculateMutualInformationScore(Vector<T> featureVector)
    {
        try
        {
            return StatisticsHelper<T>.CalculateMutualInformation(featureVector, _target);
        }
        catch (InvalidOperationException ex)
        {
            // Log the exception for debugging purposes
            Console.WriteLine($"Mutual information calculation failed: {ex.Message}");
            return NumOps.Zero;
        }
        catch (ArgumentException ex)
        {
            // Log the exception for debugging purposes
            Console.WriteLine($"Mutual information calculation failed: {ex.Message}");
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Calculates the ANOVA F-statistic for multiple groups.
    /// </summary>
    /// <param name="groups">List of groups, where each group contains feature values for one class.</param>
    /// <returns>The F-statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This calculates the F-statistic by comparing:
    /// - Between-group variance: How different are the means of different groups?
    /// - Within-group variance: How spread out are the values within each group?
    /// </para>
    /// <para>
    /// F-statistic = Between-group variance / Within-group variance
    /// </para>
    /// <para>
    /// A higher F-statistic means the groups have more different means relative to the spread
    /// within each group, indicating the feature is good at distinguishing between classes.
    /// </para>
    /// </remarks>
    private T CalculateAnovaFStatistic(List<List<T>> groups)
    {
        if (groups.Count < 2)
        {
            return NumOps.Zero;
        }

        // Calculate overall mean
        var allValues = groups.SelectMany(g => g).ToList();
        int totalCount = allValues.Count;

        if (totalCount == 0)
        {
            return NumOps.Zero;
        }

        T overallMean = StatisticsHelper<T>.CalculateMean(allValues);

        // Calculate between-group sum of squares (SSB)
        T ssb = NumOps.Zero;
        foreach (var group in groups.Where(g => g.Count > 0))
        {
            T groupMean = StatisticsHelper<T>.CalculateMean(group);
            T diff = NumOps.Subtract(groupMean, overallMean);
            T squaredDiff = NumOps.Multiply(diff, diff);
            T weightedSquaredDiff = NumOps.Multiply(NumOps.FromDouble(group.Count), squaredDiff);
            ssb = NumOps.Add(ssb, weightedSquaredDiff);
        }

        // Calculate within-group sum of squares (SSW)
        T ssw = NumOps.Zero;
        foreach (var group in groups.Where(g => g.Count > 0))
        {
            T groupMean = StatisticsHelper<T>.CalculateMean(group);
            foreach (var value in group)
            {
                T diff = NumOps.Subtract(value, groupMean);
                ssw = NumOps.Add(ssw, NumOps.Multiply(diff, diff));
            }
        }

        // Calculate degrees of freedom
        int dfBetween = groups.Count - 1;
        int dfWithin = totalCount - groups.Count;

        if (dfBetween <= 0 || dfWithin <= 0)
        {
            return NumOps.Zero;
        }

        // Calculate mean squares
        T msb = NumOps.Divide(ssb, NumOps.FromDouble(dfBetween));
        T msw = NumOps.Divide(ssw, NumOps.FromDouble(dfWithin));

        // Calculate F-statistic
        if (NumOps.Equals(msw, NumOps.Zero))
        {
            // Avoid division by zero
            return NumOps.Zero;
        }

        T fStatistic = NumOps.Divide(msb, msw);
        return fStatistic;
    }
}
