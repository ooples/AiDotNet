using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Interaction;

/// <summary>
/// Interaction Strength based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their interaction strength with other features
/// in predicting the target.
/// </para>
/// <para><b>For Beginners:</b> Some features are more useful when combined with
/// others (like age and income together predicting spending). This selector
/// finds features that interact strongly with others.
/// </para>
/// </remarks>
public class InteractionStrengthSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _topInteractions;

    private double[]? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int TopInteractions => _topInteractions;
    public double[]? InteractionScores => _interactionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InteractionStrengthSelector(
        int nFeaturesToSelect = 10,
        int topInteractions = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _topInteractions = topInteractions;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InteractionStrengthSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Standardize features
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];

        _interactionScores = new double[p];

        // For each feature, measure interaction strength with top correlated features
        for (int j1 = 0; j1 < p; j1++)
        {
            var interactionStrengths = new List<double>();

            for (int j2 = 0; j2 < p; j2++)
            {
                if (j1 == j2) continue;

                // Create interaction term
                var interaction = new double[n];
                for (int i = 0; i < n; i++)
                    interaction[i] = X[i, j1] * X[i, j2];

                // Correlation of interaction with target
                double interactionMean = interaction.Average();
                double yMean = y.Average();

                double numerator = 0, interactionSumSq = 0, ySumSq = 0;
                for (int i = 0; i < n; i++)
                {
                    numerator += (interaction[i] - interactionMean) * (y[i] - yMean);
                    interactionSumSq += (interaction[i] - interactionMean) * (interaction[i] - interactionMean);
                    ySumSq += (y[i] - yMean) * (y[i] - yMean);
                }

                double denominator = Math.Sqrt(interactionSumSq * ySumSq);
                double interactionCorr = denominator > 1e-10 ? Math.Abs(numerator / denominator) : 0;

                // Individual correlations
                double corr1 = ComputeCorrelation(X, j1, y, n);
                double corr2 = ComputeCorrelation(X, j2, y, n);

                // Interaction strength = how much the interaction adds beyond individuals
                double interactionBoost = interactionCorr - Math.Max(corr1, corr2);
                if (interactionBoost > 0)
                    interactionStrengths.Add(interactionBoost);
            }

            // Feature score = sum of top interaction strengths
            _interactionScores[j1] = interactionStrengths
                .OrderByDescending(s => s)
                .Take(_topInteractions)
                .Sum();
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _interactionScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeCorrelation(double[,] X, int j, double[] y, int n)
    {
        double xMean = 0, yMean = y.Average();
        for (int i = 0; i < n; i++)
            xMean += X[i, j];
        xMean /= n;

        double numerator = 0, xSumSq = 0, ySumSq = 0;
        for (int i = 0; i < n; i++)
        {
            numerator += (X[i, j] - xMean) * (y[i] - yMean);
            xSumSq += (X[i, j] - xMean) * (X[i, j] - xMean);
            ySumSq += (y[i] - yMean) * (y[i] - yMean);
        }

        double denominator = Math.Sqrt(xSumSq * ySumSq);
        return denominator > 1e-10 ? Math.Abs(numerator / denominator) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionStrengthSelector has not been fitted.");

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
        throw new NotSupportedException("InteractionStrengthSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionStrengthSelector has not been fitted.");

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
