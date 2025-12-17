namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that uses Recursive Feature Elimination (RFE) to select the most important features.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <typeparam name="TOutput">The output data type expected by the model.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Recursive Feature Elimination (RFE) is a feature selection method that works by 
/// recursively removing the least important features until the desired number of features is reached.
/// </para>
/// <para>
/// Think of it like a talent competition where the weakest performer gets eliminated in each round until 
/// only the best performers remain. RFE uses a machine learning model to rank features by importance and 
/// then iteratively removes the least important ones.
/// </para>
/// </remarks>
public class RecursiveFeatureElimination<T, TInput, TOutput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// The number of features to select.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how many features will remain after the elimination process.
    /// By default, it's set to 50% of the original features.
    /// </remarks>
    private readonly int _numFeaturesToSelect;

    /// <summary>
    /// The full model used to evaluate feature importance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model helps determine which features are most important by examining
    /// the coefficients (weights) it assigns to each feature during training.
    /// </remarks>
    private readonly IFullModel<T, TInput, TOutput> _model;

    /// <summary>
    /// Function that creates a dummy target of the appropriate TOutput type for training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This function helps create placeholder output values when we need
    /// to train the model just to see which features are important, not to make actual predictions.
    /// </remarks>
    private readonly Func<int, TOutput> _createDummyTarget;

    /// <summary>
    /// Initializes a new instance of the RecursiveFeatureElimination class.
    /// </summary>
    /// <param name="model">The model used to evaluate feature importance.</param>
    /// <param name="createDummyTarget">Function to create a dummy target for training.</param>
    /// <param name="numFeaturesToSelect">Optional number of features to select. If not provided, defaults to 50% of the original features.</param>
    /// <param name="higherDimensionStrategy">Strategy to use for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new feature selector that will use the specified
    /// model to determine which features are most important.
    /// </para>
    /// <para>
    /// The model is used to assign importance scores to features based on their coefficients. Features
    /// with larger coefficient magnitudes (absolute values) are considered more important.
    /// </para>
    /// <para>
    /// If you don't specify how many features to select, it will default to keeping half of the original features.
    /// </para>
    /// <para>
    /// The createDummyTarget function is needed to create appropriate placeholder outputs for training the model.
    /// </para>
    /// </remarks>
    public RecursiveFeatureElimination(
        IFullModel<T, TInput, TOutput> model,
        Func<int, TOutput> createDummyTarget,
        int? numFeaturesToSelect = null,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _model = model;
        _numFeaturesToSelect = numFeaturesToSelect ?? 0; // Will be set in SelectFeatureIndices if still 0
        _createDummyTarget = createDummyTarget;
    }

    /// <summary>
    /// Determines which features to select using Recursive Feature Elimination.
    /// </summary>
    /// <param name="allFeatures">The input data containing all potential features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features in the dataset.</param>
    /// <returns>A list of indices representing the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements the Recursive Feature Elimination algorithm, which works as follows:
    /// </para>
    /// <para>
    /// 1. Start with all features
    /// </para>
    /// <para>
    /// 2. Train the model and rank features by importance (based on coefficient magnitudes)
    /// </para>
    /// <para>
    /// 3. Remove the least important feature
    /// </para>
    /// <para>
    /// 4. Repeat steps 2-3 until the desired number of features remains
    /// </para>
    /// <para>
    /// This approach helps identify the most important features for prediction while considering how
    /// features interact with each other, which is more sophisticated than simply looking at each feature
    /// in isolation.
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // If numFeaturesToSelect wasn't specified in constructor, default to 50% of features
        int actualNumFeaturesToSelect = _numFeaturesToSelect > 0
            ? _numFeaturesToSelect
            : Math.Max(1, (int)(numFeatures * 0.5));

        actualNumFeaturesToSelect = Math.Min(actualNumFeaturesToSelect, numFeatures);

        var remainingFeatureIndices = Enumerable.Range(0, numFeatures).ToList();

        // Create a dummy target for training using the provided function
        TOutput dummyTarget = _createDummyTarget(numSamples);

        // Eliminate features one by one until we reach the desired number
        while (remainingFeatureIndices.Count > actualNumFeaturesToSelect)
        {
            // Create a subset of features to evaluate
            TInput featureSubset = FeatureSelectorHelper<T, TInput>.CreateFeatureSubset(
                allFeatures,
                remainingFeatureIndices);

            // Train model to get feature importances
            _model.Train(featureSubset, dummyTarget);

            // Get feature importances from model coefficients
            var parameters = _model.GetParameters();
            var featureImportances = parameters
                .Select((c, i) => (NumOps.Abs(c), i))
                .ToList();

            // Sort by importance (descending)
            featureImportances.Sort((a, b) =>
                NumOps.GreaterThan(b.Item1, a.Item1) ? -1 :
                (NumOps.Equals(b.Item1, a.Item1) ? 0 : 1));

            // Get the least important feature's index in the current subset
            var leastImportantFeatureIndex = featureImportances.Last().i;

            // Remove the least important feature from the remaining set
            remainingFeatureIndices.RemoveAt(leastImportantFeatureIndex);
        }

        // Return the remaining most important features, sorted for consistent output
        return remainingFeatureIndices.OrderBy(x => x).ToList();
    }
}
