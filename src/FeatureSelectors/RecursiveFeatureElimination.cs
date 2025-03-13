namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that uses Recursive Feature Elimination (RFE) to select the most important features.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
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
public class RecursiveFeatureElimination<T> : IFeatureSelector<T>
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
    /// The regression model used to evaluate feature importance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model helps determine which features are most important by examining 
    /// the coefficients (weights) it assigns to each feature during training.
    /// </remarks>
    private readonly IRegression<T> _model;
    
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the RecursiveFeatureElimination class.
    /// </summary>
    /// <param name="model">The regression model used to evaluate feature importance.</param>
    /// <param name="numFeaturesToSelect">Optional number of features to select. If not provided, defaults to 50% of the original features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new feature selector that will use the specified 
    /// regression model to determine which features are most important.
    /// </para>
    /// <para>
    /// The model is used to assign importance scores to features based on their coefficients. Features 
    /// with larger coefficient magnitudes (absolute values) are considered more important.
    /// </para>
    /// <para>
    /// If you don't specify how many features to select, it will default to keeping half of the original features.
    /// </para>
    /// </remarks>
    public RecursiveFeatureElimination(IRegression<T> model, int? numFeaturesToSelect = null)
    {
        _model = model;
        _numOps = MathHelper.GetNumericOperations<T>();
        _numFeaturesToSelect = numFeaturesToSelect ?? GetDefaultNumFeatures();
    }

    /// <summary>
    /// Gets the default number of features to select.
    /// </summary>
    /// <returns>50% of the original number of features, with a minimum of 1.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method calculates the default number of features to keep 
    /// when no specific number is provided. It uses 50% of the original features as a reasonable default.
    /// </remarks>
    private int GetDefaultNumFeatures()
    {
        return Math.Max(1, (int)(_model.Coefficients.Length * 0.5)); // Default to 50% of features
    }

    /// <summary>
    /// Selects features from the input matrix using Recursive Feature Elimination.
    /// </summary>
    /// <param name="allFeaturesMatrix">The matrix containing all potential features.</param>
    /// <returns>A matrix containing only the selected features.</returns>
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
    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        var numFeatures = allFeaturesMatrix.Columns;
        var featureIndices = Enumerable.Range(0, numFeatures).ToList();
        var selectedFeatures = new List<int>();

        while (selectedFeatures.Count < _numFeaturesToSelect && featureIndices.Count > 0)
        {
            var subMatrix = new Matrix<T>(allFeaturesMatrix.Rows, featureIndices.Count);
            for (int i = 0; i < featureIndices.Count; i++)
            {
                subMatrix.SetColumn(i, allFeaturesMatrix.GetColumn(featureIndices[i]));
            }

            var dummyTarget = new Vector<T>(allFeaturesMatrix.Rows);
            _model.Train(subMatrix, dummyTarget);

            var featureImportances = _model.Coefficients.Select((c, i) => (_numOps.Abs(c), i)).ToList();
            featureImportances.Sort((a, b) => _numOps.GreaterThan(b.Item1, a.Item1) ? -1 : (_numOps.Equals(b.Item1, a.Item1) ? 0 : 1));

            var leastImportantFeatureIndex = featureImportances.Last().i;
            selectedFeatures.Insert(0, featureIndices[leastImportantFeatureIndex]);
            featureIndices.RemoveAt(leastImportantFeatureIndex);
        }

        var result = new Matrix<T>(allFeaturesMatrix.Rows, _numFeaturesToSelect);
        for (int i = 0; i < _numFeaturesToSelect; i++)
        {
            result.SetColumn(i, allFeaturesMatrix.GetColumn(selectedFeatures[i]));
        }

        return result;
    }
}