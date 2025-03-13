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
public class CorrelationFeatureSelector<T> : IFeatureSelector<T>
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
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

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
    public CorrelationFeatureSelector(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? GetDefaultThreshold();
    }

    /// <summary>
    /// Gets the default correlation threshold value.
    /// </summary>
    /// <returns>The default threshold value of 0.5.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method simply returns the default threshold value of 0.5 
    /// when no custom threshold is provided.
    /// </remarks>
    private T GetDefaultThreshold()
    {
        return _numOps.FromDouble(0.5);
    }

    /// <summary>
    /// Selects features from the input matrix based on correlation analysis.
    /// </summary>
    /// <param name="allFeaturesMatrix">The matrix containing all potential features.</param>
    /// <returns>A matrix containing only the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes all the features in your dataset and selects a subset 
    /// of them that aren't highly correlated with each other. It works by:
    /// </para>
    /// <para>
    /// 1. Starting with an empty set of selected features
    /// </para>
    /// <para>
    /// 2. For each feature in the original dataset:
    ///    - Checking if it's highly correlated with any already-selected feature
    ///    - If not, adding it to the selected features
    /// </para>
    /// <para>
    /// This approach ensures that each selected feature provides unique information that isn't 
    /// already captured by other selected features.
    /// </para>
    /// </remarks>
    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        var selectedFeatures = new List<Vector<T>>();
        var numFeatures = allFeaturesMatrix.Columns;

        for (int i = 0; i < numFeatures; i++)
        {
            bool isIndependent = true;
            var featureI = allFeaturesMatrix.GetColumn(i);

            for (int j = 0; j < selectedFeatures.Count; j++)
            {
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(featureI, selectedFeatures[j]);
                if (_numOps.GreaterThan(_numOps.Abs(correlation), _threshold))
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
            {
                selectedFeatures.Add(featureI);
            }
        }

        return new Matrix<T>(selectedFeatures);
    }
}