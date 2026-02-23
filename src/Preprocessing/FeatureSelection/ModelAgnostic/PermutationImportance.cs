using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// Permutation Importance for model-agnostic feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Permutation Importance measures feature importance by randomly shuffling each feature
/// and observing how much model performance degrades. Features that cause large
/// performance drops when shuffled are important.
/// </para>
/// <para><b>For Beginners:</b> This method works with any model. It asks: "What happens
/// if I scramble this feature's values?" If the model gets much worse, that feature
/// was important. If the model doesn't care, the feature wasn't useful. It's like
/// testing each ingredient by removing it from a recipe.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PermutationImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scorer;
    private readonly int? _randomState;

    private double[]? _importanceScores;
    private double[]? _importanceStd;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NPermutations => _nPermutations;
    public double[]? ImportanceScores => _importanceScores;
    public double[]? ImportanceStd => _importanceStd;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PermutationImportance(
        int nFeaturesToSelect = 10,
        int nPermutations = 10,
        Func<Matrix<T>, Vector<T>, double>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPermutations < 1)
            throw new ArgumentException("Number of permutations must be at least 1.", nameof(nPermutations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PermutationImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scorer = _scorer ?? DefaultScorer;

        // Compute baseline score
        double baselineScore = scorer(data, target);

        _importanceScores = new double[p];
        _importanceStd = new double[p];

        // For each feature, permute and measure performance drop
        for (int j = 0; j < p; j++)
        {
            var permScores = new double[_nPermutations];

            for (int perm = 0; perm < _nPermutations; perm++)
            {
                // Create permuted data
                var permutedData = new T[n, p];
                for (int i = 0; i < n; i++)
                    for (int f = 0; f < p; f++)
                        permutedData[i, f] = data[i, f];

                // Shuffle column j
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                    permutedData[i, j] = data[shuffledIndices[i], j];

                // Score with permuted feature
                double permScore = scorer(new Matrix<T>(permutedData), target);
                permScores[perm] = baselineScore - permScore; // Importance = drop in score
            }

            // Mean importance
            _importanceScores[j] = permScores.Average();

            // Standard deviation
            double mean = _importanceScores[j];
            double sumSq = 0;
            foreach (double s in permScores)
                sumSq += (s - mean) * (s - mean);
            _importanceStd[j] = Math.Sqrt(sumSq / _nPermutations);
        }

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _importanceScores
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        // Use R-squared as default score
        int n = data.Rows;
        int p = data.Columns;

        // Simple linear regression prediction using all features
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        // Compute feature means
        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;
        }

        // Simple prediction using correlation-weighted sum
        double ssTotal = 0;
        double ssResidual = 0;

        for (int i = 0; i < n; i++)
        {
            double yTrue = NumOps.ToDouble(target[i]);
            double yPred = yMean;

            // Weighted prediction
            for (int j = 0; j < p; j++)
            {
                double xVal = NumOps.ToDouble(data[i, j]);
                yPred += 0.1 * (xVal - xMeans[j]); // Simple linear approximation
            }

            ssTotal += (yTrue - yMean) * (yTrue - yMean);
            ssResidual += (yTrue - yPred) * (yTrue - yPred);
        }

        return ssTotal > 0 ? 1 - (ssResidual / ssTotal) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PermutationImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
