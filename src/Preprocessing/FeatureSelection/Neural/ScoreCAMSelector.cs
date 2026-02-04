using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Score-CAM based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Score-CAM, a gradient-free class activation mapping
/// method that uses forward passing scores instead of gradients.
/// </para>
/// <para><b>For Beginners:</b> Unlike GradCAM which uses gradients, Score-CAM
/// measures feature importance by masking each feature and observing how the
/// prediction score changes. Features that cause larger score drops when
/// masked are considered more important. This is more stable than gradient methods.
/// </para>
/// </remarks>
public class ScoreCAMSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _scoreWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ScoreWeights => _scoreWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ScoreCAMSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ScoreCAMSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _scoreWeights = new double[p];

        // Compute baseline prediction score (using correlation as proxy)
        double baselineScore = ComputePredictionScore(X, y, Enumerable.Range(0, p).ToArray());

        // For each feature, compute activation map score
        for (int j = 0; j < p; j++)
        {
            // Normalize feature to [0, 1] as activation map
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double min = col.Min();
            double max = col.Max();
            double range = max - min;

            var activationMap = new double[n];
            if (range > 1e-10)
            {
                for (int i = 0; i < n; i++)
                    activationMap[i] = (col[i] - min) / range;
            }

            // Create masked input (element-wise multiplication with activation)
            var maskedX = new double[n, p];
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < p; k++)
                    maskedX[i, k] = X[i, k] * activationMap[i];
            }

            // Compute score with this activation map
            double maskedScore = ComputePredictionScore(maskedX, y, Enumerable.Range(0, p).ToArray());

            // Score-CAM weight: how much does this activation map contribute?
            // Using softmax-normalized increase in confidence
            _scoreWeights[j] = Math.Max(0, maskedScore);
        }

        // Softmax normalization
        double maxScore = _scoreWeights.Max();
        double expSum = _scoreWeights.Sum(s => Math.Exp(s - maxScore));
        for (int j = 0; j < p; j++)
            _scoreWeights[j] = Math.Exp(_scoreWeights[j] - maxScore) / expSum;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _scoreWeights[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePredictionScore(double[,] X, double[] y, int[] features)
    {
        int n = X.GetLength(0);

        // Use R² as prediction score proxy
        double meanY = y.Average();
        double ssTotal = y.Sum(v => (v - meanY) * (v - meanY));

        if (ssTotal < 1e-10) return 0;

        // Simple linear combination
        var pred = new double[n];
        foreach (int j in features)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double meanX = col.Average();
            double cov = 0, varX = 0;
            for (int i = 0; i < n; i++)
            {
                cov += (col[i] - meanX) * (y[i] - meanY);
                varX += (col[i] - meanX) * (col[i] - meanX);
            }

            double beta = varX > 1e-10 ? cov / varX : 0;
            for (int i = 0; i < n; i++)
                pred[i] += beta * (col[i] - meanX);
        }

        for (int i = 0; i < n; i++)
            pred[i] += meanY;

        double ssResid = 0;
        for (int i = 0; i < n; i++)
            ssResid += (y[i] - pred[i]) * (y[i] - pred[i]);

        return 1 - ssResid / ssTotal;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ScoreCAMSelector has not been fitted.");

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
        throw new NotSupportedException("ScoreCAMSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ScoreCAMSelector has not been fitted.");

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
