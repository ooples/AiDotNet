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
public class CorrelationFeatureSelector<T, TInput> : IFeatureSelector<T, TInput>
{
    /// <summary>
    /// The correlation threshold above which features are considered highly correlated.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This value determines how similar two features need to be before one is 
    /// removed. A higher threshold means features need to be more similar to be considered redundant.
    /// The default value is 0.5.
    /// </remarks>
    private readonly T _threshold = default!;
    
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// The strategy to use for extracting features from higher-dimensional tensors.
    /// </summary>
    private readonly FeatureExtractionStrategy _higherDimensionStrategy = FeatureExtractionStrategy.Mean;

    /// <summary>
    /// Weights to apply when using the WeightedSum feature extraction strategy.
    /// </summary>
    private readonly Dictionary<int, T> _dimensionWeights = [];

    /// <summary>
    /// Initializes a new instance of the CorrelationFeatureSelector class.
    /// </summary>
    /// <param name="threshold">Optional correlation threshold value. If not provided, a default value of 0.5 is used.</param>
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
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _higherDimensionStrategy = higherDimensionStrategy;
        _threshold = _numOps.FromDouble(threshold);
    }

    /// <summary>
    /// Selects independent features by eliminating highly correlated ones.
    /// </summary>
    /// <param name="allFeatures">The input data containing all available features.</param>
    /// <returns>A filtered dataset containing only the selected independent features.</returns>
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
    public TInput SelectFeatures(TInput allFeatures)
    {
        // Get dimensions using the helper methods
        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeatures);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeatures);
    
        // Track which features to keep
        var selectedFeatureIndices = new List<int>();
    
        // Process each feature to determine if it should be kept
        for (int i = 0; i < numFeatures; i++)
        {
            bool isIndependent = true;
            var featureI = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(allFeatures, i, numSamples, _higherDimensionStrategy, _dimensionWeights);
        
            // Check correlation with already selected features
            foreach (int j in selectedFeatureIndices)
            {
                var featureJ = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(allFeatures, j, numSamples, _higherDimensionStrategy, _dimensionWeights);
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(featureI, featureJ);
            
                if (_numOps.GreaterThan(_numOps.Abs(correlation), _threshold))
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
    
        // Create result with only the selected features
        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, selectedFeatureIndices);
    }
}