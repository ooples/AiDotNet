using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Image;

/// <summary>
/// Saliency-based feature selection for image features.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on gradient-based saliency, identifying features that
/// contribute most to model predictions. Originally designed for neural network
/// interpretability but adapted here for general feature selection.
/// </para>
/// <para><b>For Beginners:</b> Saliency measures how much a small change in each
/// feature would affect the prediction. Features with high saliency are "sensitive"
/// to changes, suggesting they're important for the model's decisions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SaliencySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _epsilon;

    private double[]? _saliencyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SaliencyScores => _saliencyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SaliencySelector(
        int nFeaturesToSelect = 100,
        double epsilon = 1e-5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epsilon = epsilon;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SaliencySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute saliency using numerical gradients
        _saliencyScores = new double[p];

        // First, fit a simple linear model
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var coefficients = new double[p];
        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
            }

            if (ssXX > 1e-10)
                coefficients[j] = ssXY / ssXX;
        }

        // Compute saliency as average absolute gradient
        for (int j = 0; j < p; j++)
        {
            double totalSaliency = 0;
            for (int i = 0; i < n; i++)
            {
                double x = NumOps.ToDouble(data[i, j]);
                // Saliency = |dLoss/dInput| ≈ |coefficient * x * sign(residual)|
                double pred = coefficients[j] * x;
                double residual = NumOps.ToDouble(target[i]) - pred;
                totalSaliency += Math.Abs(coefficients[j]) * Math.Abs(x);
            }
            _saliencyScores[j] = totalSaliency / n;
        }

        // Select top features by saliency
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _saliencyScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SaliencySelector has not been fitted.");

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
        throw new NotSupportedException("SaliencySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SaliencySelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
