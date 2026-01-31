using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Feature selector that removes highly correlated features to reduce redundancy.
/// </summary>
/// <remarks>
/// <para>
/// CorrelationSelector identifies and removes features that are highly correlated with other
/// features. When features are strongly correlated, they provide redundant information, so
/// keeping just one of them can simplify your model without losing predictive power.
/// </para>
/// <para><b>For Beginners:</b> This selector removes duplicate or redundant information from your data.
///
/// Imagine you're collecting data about houses and include both:
/// - Square footage of the house
/// - Number of rooms
/// - Price
///
/// Square footage and number of rooms are often highly correlated (bigger houses tend to have
/// more rooms). This selector would detect this relationship and might keep only one of these
/// features, reducing redundancy while preserving the most important information.
///
/// By eliminating redundant features:
/// - Your model trains faster
/// - You reduce the risk of overfitting
/// - The model becomes easier to interpret and explain
///
/// The threshold setting controls how strict this filtering is:
/// - Higher values (e.g., 0.9) allow more features to be included
/// - Lower values (e.g., 0.5) result in more features being removed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class CorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    // Fitted parameters
    private int[]? _selectedFeatures;
    private int _nInputFeatures;
    private double[,]? _correlationMatrix;

    /// <summary>
    /// Gets the correlation threshold above which features are considered highly correlated.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This value determines how similar two features need to be before
    /// one is removed. Values range from 0 to 1:
    /// - 0: No correlation (features are completely unrelated)
    /// - 1: Perfect correlation (features are identical)
    ///
    /// A common threshold is 0.5, meaning features with absolute correlation above 0.5 are
    /// considered highly correlated.
    /// </para>
    /// </remarks>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the indices of selected features after fitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After fitting, this property tells you which original column
    /// indices were kept. For example, if you started with 10 features and this returns [0, 2, 5, 8],
    /// it means features 0, 2, 5, and 8 were kept and the others were removed due to high correlation.
    /// </para>
    /// </remarks>
    public int[]? SelectedFeatures => _selectedFeatures;

    /// <summary>
    /// Gets the number of selected features after fitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many features remain after removing
    /// correlated ones. If you started with 20 features and this returns 8, it means
    /// 12 features were removed due to high correlation.
    /// </para>
    /// </remarks>
    public int SelectedCount => _selectedFeatures?.Length ?? 0;

    /// <summary>
    /// Gets the correlation matrix computed during fitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a square matrix where the value at position [i, j]
    /// represents the correlation between feature i and feature j. Values range from -1 to 1:
    /// - +1: Perfect positive correlation
    /// - 0: No correlation
    /// - -1: Perfect negative correlation
    /// </para>
    /// </remarks>
    public double[,]? CorrelationMatrix => _correlationMatrix;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature selection removes columns entirely, so we cannot
    /// recover the removed data. Unlike scaling which can be reversed, feature selection
    /// is a one-way transformation.
    /// </para>
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="CorrelationSelector{T}"/>.
    /// </summary>
    /// <param name="threshold">Correlation threshold above which features are considered highly correlated.
    /// Features with absolute correlation above this with an already-selected feature will be removed.
    /// Defaults to 0.5.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new correlation-based feature selector.
    ///
    /// The threshold parameter controls how aggressive the filtering is:
    /// - threshold = 0.9: Only removes nearly identical features
    /// - threshold = 0.5: Removes moderately correlated features (default)
    /// - threshold = 0.2: Aggressively removes even weakly correlated features
    ///
    /// The columnIndices parameter lets you apply selection only to specific columns.
    /// </para>
    /// </remarks>
    public CorrelationSelector(
        double threshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the correlation matrix and determines which features to keep.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method analyzes your data to find correlated features:
    ///
    /// 1. First, it calculates the Pearson correlation between every pair of features
    /// 2. Then, it processes features in order:
    ///    - The first feature is always kept
    ///    - For each subsequent feature, check if it's highly correlated with any already-selected feature
    ///    - If not highly correlated with any selected feature, add it to the selected set
    ///
    /// This ensures we keep the first feature from each group of correlated features.
    /// </para>
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        // Compute full correlation matrix for analysis
        _correlationMatrix = ComputeCorrelationMatrix(data);

        // Select features using greedy forward selection
        var selectedList = new List<int>();

        for (int i = 0; i < _nInputFeatures; i++)
        {
            if (!processSet.Contains(i))
            {
                // Pass-through columns are always included
                selectedList.Add(i);
                continue;
            }

            bool isIndependent = true;

            // Check correlation with already selected features
            foreach (int j in selectedList)
            {
                if (!processSet.Contains(j))
                {
                    continue; // Skip pass-through columns in correlation check
                }

                double correlation = Math.Abs(_correlationMatrix[i, j]);
                if (correlation > _threshold)
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
            {
                selectedList.Add(i);
            }
        }

        // Safety check: ensure at least one feature is selected
        if (selectedList.Count == 0 && _nInputFeatures > 0)
        {
            selectedList.Add(0);
        }

        _selectedFeatures = selectedList.ToArray();
    }

    /// <summary>
    /// Computes the Pearson correlation matrix for all features.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <returns>A square matrix of correlations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how related each pair of features is.
    /// The result is a symmetric matrix where:
    /// - Diagonal values are always 1.0 (each feature is perfectly correlated with itself)
    /// - Off-diagonal values range from -1 to +1
    /// </para>
    /// </remarks>
    private double[,] ComputeCorrelationMatrix(Matrix<T> data)
    {
        int n = data.Columns;
        var correlations = new double[n, n];

        // Pre-compute column vectors for efficiency
        var columns = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            columns[i] = data.GetColumn(i);
        }

        // Compute correlations
        for (int i = 0; i < n; i++)
        {
            correlations[i, i] = 1.0; // Self-correlation is always 1

            for (int j = i + 1; j < n; j++)
            {
                double corr = NumOps.ToDouble(
                    StatisticsHelper<T>.CalculatePearsonCorrelation(columns[i], columns[j]));

                // Handle NaN (constant features have undefined correlation)
                if (double.IsNaN(corr))
                {
                    corr = 0.0;
                }

                correlations[i, j] = corr;
                correlations[j, i] = corr; // Symmetric
            }
        }

        return correlations;
    }

    /// <summary>
    /// Removes features that are highly correlated with other selected features.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with correlated features removed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the feature selection learned during fitting.
    /// It keeps only the columns that were determined to be independent (not highly correlated).
    ///
    /// For example, if you have 10 columns and the fitting determined that columns [0, 2, 5, 8]
    /// are independent, this method returns a matrix with only those 4 columns.
    /// </para>
    /// </remarks>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        int numRows = data.Rows;
        int numOutputCols = _selectedFeatures.Length;
        var result = new T[numRows, numOutputCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numOutputCols; j++)
            {
                int sourceCol = _selectedFeatures[j];
                result[i, j] = data[i, sourceCol];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for feature selection.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because feature selection cannot be reversed.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Once features are removed, we cannot get them back because the
    /// data is gone. Unlike scaling (where we can multiply/divide to reverse), feature selection
    /// permanently removes columns from the data.
    /// </para>
    /// </remarks>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "CorrelationSelector does not support inverse transformation. " +
            "Feature selection removes data that cannot be recovered.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    /// <returns>Array where true indicates the feature is selected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns an array of true/false values, one for each original feature.
    /// True means the feature was kept; false means it was removed due to high correlation.
    ///
    /// For example, if GetSupportMask() returns [true, false, true, false, true], it means
    /// features 0, 2, and 4 were kept, while features 1 and 3 were removed.
    /// </para>
    /// </remarks>
    public bool[] GetSupportMask()
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        var selectedSet = new HashSet<int>(_selectedFeatures);

        for (int i = 0; i < _nInputFeatures; i++)
        {
            mask[i] = selectedSet.Contains(i);
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">Optional names of the input features.</param>
    /// <returns>Names of the selected features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns the names of features that were kept.
    /// If you provided names like ["age", "height", "weight", "income"] and features 0 and 2
    /// were selected, this returns ["age", "weight"].
    /// </para>
    /// </remarks>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedFeatures is null)
        {
            return Array.Empty<string>();
        }

        var names = new string[_selectedFeatures.Length];
        for (int i = 0; i < _selectedFeatures.Length; i++)
        {
            int col = _selectedFeatures[i];
            names[i] = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";
        }

        return names;
    }

    /// <summary>
    /// Gets the correlations between a specific feature and all selected features.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>Dictionary mapping selected feature indices to their correlation with the specified feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is useful for understanding why a feature was kept or removed.
    /// It shows how correlated a specific feature is with each of the selected features.
    ///
    /// For example, if feature 5 was removed and you call this method with index 5, you might see
    /// that it has a high correlation (0.95) with feature 2, which explains why it was removed.
    /// </para>
    /// </remarks>
    public Dictionary<int, double> GetCorrelationsWithSelected(int featureIndex)
    {
        if (_selectedFeatures is null || _correlationMatrix is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        if (featureIndex < 0 || featureIndex >= _nInputFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        }

        var correlations = new Dictionary<int, double>();
        foreach (int selectedIdx in _selectedFeatures)
        {
            correlations[selectedIdx] = _correlationMatrix[featureIndex, selectedIdx];
        }

        return correlations;
    }
}
