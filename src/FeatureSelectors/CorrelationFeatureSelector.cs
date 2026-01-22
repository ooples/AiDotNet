namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that chooses features based on their Pearson correlation with each other.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps select the most important features from your dataset by 
/// identifying and removing features that are highly correlated with each other. Highly correlated 
/// features often provide redundant information, so keeping just one of them can simplify your model 
/// without losing predictive power.
/// </para>
/// <para>
/// Think of it like removing duplicate information. If two features tell you almost the same thing 
/// (they're highly correlated), you only need to keep one of them to make good predictions.
/// </para>
/// </remarks>
public class CorrelationFeatureSelector<T, TInput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// The correlation threshold above which features are considered highly correlated.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This value determines how similar two features need to be before one is
    /// removed. A higher threshold means features need to be more similar to be considered redundant.
    /// The default value is 0.5.
    /// </remarks>
    private readonly T _threshold;

    /// <summary>
    /// Initializes a new instance of the CorrelationFeatureSelector class.
    /// </summary>
    /// <param name="threshold">Optional correlation threshold value. If not provided, a default value of 0.5 is used.</param>
    /// <param name="higherDimensionStrategy">Strategy to use for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new feature selector with either a custom threshold
    /// or the default threshold of 0.5.
    /// </para>
    /// <para>
    /// The threshold value should be between 0 and 1, where:
    /// <list type="bullet">
    /// <item><description>0 means no correlation (features are completely unrelated)</description></item>
    /// <item><description>1 means perfect correlation (features are identical)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// A common threshold value is 0.5, which means features with a correlation above 0.5 (or below -0.5)
    /// are considered highly correlated.
    /// </para>
    /// </remarks>
    public CorrelationFeatureSelector(double threshold = 0.5, FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _threshold = NumOps.FromDouble(threshold);
    }

    /// <summary>
    /// Determines which features to select by eliminating highly correlated ones.
    /// </summary>
    /// <param name="allFeatures">The input data containing all available features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features in the dataset.</param>
    /// <returns>A list of indices representing the selected features.</returns>
    /// <remarks>
    /// <para>
    /// This method uses a correlation-based approach to identify and select features that are
    /// relatively independent from each other. It processes features sequentially, adding
    /// each feature to the selected set only if it has a low correlation with all features
    /// already selected. This helps reduce redundancy and multicollinearity in the feature set.
    /// </para>
    /// <para><b>For Beginners:</b> This method removes duplicate or redundant information from your data.
    ///
    /// Imagine you're collecting data about houses and include both:
    /// - Square footage of the house
    /// - Number of rooms
    /// - Price
    ///
    /// Square footage and number of rooms are often highly correlated (bigger houses tend to have
    /// more rooms). This method would detect this relationship and might keep only one of these
    /// features, reducing redundancy while preserving the most important information.
    ///
    /// By eliminating redundant features:
    /// - Your model trains faster
    /// - You reduce the risk of overfitting (the model becoming too focused on specific patterns)
    /// - The model becomes easier to interpret and explain
    ///
    /// The _threshold setting controls how strict this filtering is - higher values allow more
    /// features to be included, while lower values result in more features being removed.
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // Track which features to keep
        var selectedFeatureIndices = new List<int>();

        // Process each feature to determine if it should be kept
        // Note: The first feature is always kept since it has nothing to correlate with
        for (int i = 0; i < numFeatures; i++)
        {
            bool isIndependent = true;
            var featureI = ExtractFeatureVector(allFeatures, i, numSamples);

            // Check correlation with already selected features
            foreach (int j in selectedFeatureIndices)
            {
                var featureJ = ExtractFeatureVector(allFeatures, j, numSamples);
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(featureI, featureJ);

                if (NumOps.GreaterThan(NumOps.Abs(correlation), _threshold))
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
            {
                selectedFeatureIndices.Add(i);
            }
        }

        // Safety check: ensure at least one feature is selected (first feature as fallback)
        if (selectedFeatureIndices.Count == 0 && numFeatures > 0)
        {
            selectedFeatureIndices.Add(0);
        }

        return selectedFeatureIndices;
    }
}
