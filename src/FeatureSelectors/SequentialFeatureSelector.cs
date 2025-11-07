namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that uses sequential feature selection (forward or backward) to identify optimal feature subsets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <typeparam name="TOutput">The output data type expected by the model.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sequential feature selection is a "wrapper" method that uses a machine learning
/// model to evaluate different combinations of features. Unlike univariate methods that evaluate each feature
/// independently, this approach considers how features work together.
/// </para>
/// <para>
/// Think of it like assembling a sports team. Instead of just looking at individual player stats (univariate),
/// you actually have the team play games with different combinations of players to see which combination
/// performs best together.
/// </para>
/// <para>
/// The selector can work in two directions:
/// - Forward Selection: Start with no features, add one at a time (like building a team from scratch)
/// - Backward Elimination: Start with all features, remove one at a time (like cutting players from a roster)
/// </para>
/// <para>
/// This method is more accurate than univariate selection because it captures feature interactions, but it's
/// also more computationally expensive since it requires training many models.
/// </para>
/// </remarks>
public class SequentialFeatureSelector<T, TInput, TOutput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// The model used to evaluate feature subsets.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the machine learning model that will be trained with different
    /// feature combinations to see which combination performs best. The model is cloned for each
    /// evaluation to ensure fair comparisons.
    /// </remarks>
    private readonly IFullModel<T, TInput, TOutput> _baseModel;

    /// <summary>
    /// The target variable for training and evaluation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is what you're trying to predict. The model is trained on different
    /// feature subsets, and the predictions are compared against this target to measure performance.
    /// </remarks>
    private readonly TOutput _target;

    /// <summary>
    /// The direction of sequential selection (forward or backward).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines whether we start with no features and add them (forward)
    /// or start with all features and remove them (backward).
    /// </remarks>
    private readonly SequentialFeatureSelectionDirection _direction;

    /// <summary>
    /// The number of features to select.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the target number of features to keep. If not specified,
    /// it defaults to 50% of the original features.
    /// </remarks>
    private readonly int? _numFeaturesToSelect;

    /// <summary>
    /// The scoring function used to evaluate model performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function measures how well the model performs with a given set of features.
    /// Higher scores indicate better performance.
    /// </para>
    /// <para>
    /// Common scoring functions include:
    /// - Accuracy: Percentage of correct predictions (for classification)
    /// - R²: How well the model fits the data (for regression)
    /// - F1 Score: Balance between precision and recall (for classification)
    /// </para>
    /// </remarks>
    private readonly Func<TOutput, TOutput, T> _scoringFunction;

    /// <summary>
    /// Initializes a new instance of the SequentialFeatureSelector class.
    /// </summary>
    /// <param name="model">The model to use for evaluating feature subsets. Must support cloning.</param>
    /// <param name="target">The target variable for training and evaluation.</param>
    /// <param name="scoringFunction">A function that takes (predictions, actual) and returns a performance score (higher is better).</param>
    /// <param name="direction">The direction of selection (default: Forward).</param>
    /// <param name="numFeaturesToSelect">The number of features to select. If null, selects 50% of features.</param>
    /// <param name="higherDimensionStrategy">Strategy for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up a sequential feature selector with your chosen settings.
    /// </para>
    /// <para>
    /// The model parameter should be a machine learning model that can be trained and used for predictions.
    /// The selector will create copies of this model to test different feature combinations.
    /// </para>
    /// <para>
    /// The target parameter contains the correct answers (what you're trying to predict) for your training data.
    /// </para>
    /// <para>
    /// The scoringFunction should return higher values for better performance. For example:
    /// <code>
    /// // For classification (accuracy):
    /// (predictions, actual) => CalculateAccuracy(predictions, actual)
    ///
    /// // For regression (negative MSE, so higher is better):
    /// (predictions, actual) => -CalculateMeanSquaredError(predictions, actual)
    /// </code>
    /// </para>
    /// <para>
    /// The direction parameter controls whether to start with no features (Forward) or all features (Backward).
    /// Forward selection is typically faster when you want few features; backward elimination is better when
    /// you want to keep most features.
    /// </para>
    /// </remarks>
    public SequentialFeatureSelector(
        IFullModel<T, TInput, TOutput> model,
        TOutput target,
        Func<TOutput, TOutput, T> scoringFunction,
        SequentialFeatureSelectionDirection direction = SequentialFeatureSelectionDirection.Forward,
        int? numFeaturesToSelect = null,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
        : base(higherDimensionStrategy)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        _target = target ?? throw new ArgumentNullException(nameof(target));
        _scoringFunction = scoringFunction ?? throw new ArgumentNullException(nameof(scoringFunction));
        _direction = direction;
        _numFeaturesToSelect = numFeaturesToSelect;
    }

    /// <summary>
    /// Determines which features to select using sequential selection.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <returns>A list of indices of the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method performs the sequential selection process:
    /// </para>
    /// <para>
    /// For Forward Selection:
    /// 1. Start with an empty set of selected features
    /// 2. For each iteration, try adding each remaining feature one at a time
    /// 3. Keep the feature that gives the best performance improvement
    /// 4. Repeat until desired number of features is reached
    /// </para>
    /// <para>
    /// For Backward Elimination:
    /// 1. Start with all features selected
    /// 2. For each iteration, try removing each feature one at a time
    /// 3. Remove the feature whose removal least hurts performance
    /// 4. Repeat until desired number of features is reached
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // Determine target number of features
        int targetCount = _numFeaturesToSelect ?? (numFeatures / 2);
        targetCount = Math.Min(targetCount, numFeatures);
        targetCount = Math.Max(1, targetCount);

        return _direction switch
        {
            SequentialFeatureSelectionDirection.Forward => ForwardSelection(allFeatures, numFeatures, targetCount),
            SequentialFeatureSelectionDirection.Backward => BackwardElimination(allFeatures, numFeatures, targetCount),
            _ => throw new ArgumentException($"Unsupported direction: {_direction}")
        };
    }

    /// <summary>
    /// Performs forward feature selection.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <param name="targetCount">The target number of features to select.</param>
    /// <returns>A list of selected feature indices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forward selection starts with no features and adds them one by one.
    /// At each step, it tries adding each remaining feature, trains a model, evaluates performance,
    /// and keeps the feature that gives the best improvement.
    /// </para>
    /// <para>
    /// This is like building a recipe by adding ingredients one at a time, always choosing the
    /// ingredient that makes the dish taste better at each step.
    /// </para>
    /// </remarks>
    private List<int> ForwardSelection(TInput allFeatures, int numFeatures, int targetCount)
    {
        var selectedFeatures = new List<int>();
        var remainingFeatures = Enumerable.Range(0, numFeatures).ToList();

        while (selectedFeatures.Count < targetCount && remainingFeatures.Count > 0)
        {
            T bestScore = NumOps.FromDouble(double.NegativeInfinity);
            int bestFeature = -1;

            // Try adding each remaining feature
            foreach (int candidateFeature in remainingFeatures)
            {
                var candidateSet = new List<int>(selectedFeatures) { candidateFeature };
                T score = EvaluateFeatureSubset(allFeatures, candidateSet);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestFeature = candidateFeature;
                }
            }

            // Add the best feature
            if (bestFeature >= 0)
            {
                selectedFeatures.Add(bestFeature);
                remainingFeatures.Remove(bestFeature);
            }
            else
            {
                // No improvement possible, stop early
                break;
            }
        }

        return selectedFeatures.OrderBy(x => x).ToList();
    }

    /// <summary>
    /// Performs backward feature elimination.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numFeatures">The total number of features.</param>
    /// <param name="targetCount">The target number of features to keep.</param>
    /// <returns>A list of selected feature indices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backward elimination starts with all features and removes them one by one.
    /// At each step, it tries removing each feature, trains a model, evaluates performance, and removes
    /// the feature whose absence hurts performance the least.
    /// </para>
    /// <para>
    /// This is like simplifying a recipe by removing ingredients one at a time, always removing the
    /// ingredient that you'll miss the least.
    /// </para>
    /// </remarks>
    private List<int> BackwardElimination(TInput allFeatures, int numFeatures, int targetCount)
    {
        var selectedFeatures = Enumerable.Range(0, numFeatures).ToList();

        while (selectedFeatures.Count > targetCount)
        {
            T bestScore = NumOps.FromDouble(double.NegativeInfinity);
            int worstFeature = -1;

            // Try removing each feature
            foreach (int candidateFeature in selectedFeatures)
            {
                var candidateSet = selectedFeatures.Where(f => f != candidateFeature).ToList();
                T score = EvaluateFeatureSubset(allFeatures, candidateSet);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    worstFeature = candidateFeature;
                }
            }

            // Remove the worst feature
            if (worstFeature >= 0)
            {
                selectedFeatures.Remove(worstFeature);
            }
            else
            {
                // No feature can be removed without significant loss, stop early
                break;
            }
        }

        return selectedFeatures.OrderBy(x => x).ToList();
    }

    /// <summary>
    /// Evaluates a subset of features by training a model and measuring performance.
    /// </summary>
    /// <param name="allFeatures">The complete feature matrix.</param>
    /// <param name="featureIndices">The indices of features to include in this evaluation.</param>
    /// <returns>The performance score (higher is better).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Creates a new dataset with only the selected features
    /// 2. Creates a fresh copy of the model
    /// 3. Trains the model on the reduced dataset
    /// 4. Makes predictions
    /// 5. Calculates and returns a performance score
    /// </para>
    /// <para>
    /// This is repeated many times during selection to compare different feature combinations.
    /// </para>
    /// </remarks>
    private T EvaluateFeatureSubset(TInput allFeatures, List<int> featureIndices)
    {
        try
        {
            // Handle edge case: no features selected
            if (featureIndices.Count == 0)
            {
                return NumOps.FromDouble(double.NegativeInfinity);
            }

            // Create subset with selected features
            var featureSubset = FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, featureIndices);

            // Clone the model for this evaluation
            var model = _baseModel.Clone();

            // Train the model
            model.Train(featureSubset, _target);

            // Make predictions
            var predictions = model.Predict(featureSubset);

            // Calculate score
            return _scoringFunction(predictions, _target);
        }
        catch
        {
            // Return worst possible score if evaluation fails
            return NumOps.FromDouble(double.NegativeInfinity);
        }
    }
}
