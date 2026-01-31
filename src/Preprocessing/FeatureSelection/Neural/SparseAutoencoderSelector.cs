using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Sparse Autoencoder-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a sparse autoencoder with sparsity constraints on the hidden layer
/// to learn which input features are essential for reconstruction.
/// </para>
/// <para><b>For Beginners:</b> A sparse autoencoder forces most of its internal
/// neurons to be "off" most of the time. This makes it focus on the most
/// important input features. Features that activate the sparse neurons are
/// the ones we select.
/// </para>
/// </remarks>
public class SparseAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _sparsityTarget;
    private readonly double _sparsityWeight;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SparseAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 20,
        int nEpochs = 100,
        double sparsityTarget = 0.05,
        double sparsityWeight = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _sparsityTarget = sparsityTarget;
        _sparsityWeight = sparsityWeight;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Normalize
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        int h = Math.Min(_hiddenSize, p);
        var rand = RandomHelper.CreateSecureRandom();

        var W1 = new double[p, h];
        var W2 = new double[h, p];
        double scale = Math.Sqrt(2.0 / (p + h));
        for (int i = 0; i < p; i++)
            for (int j = 0; j < h; j++)
                W1[i, j] = (rand.NextDouble() * 2 - 1) * scale;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < p; j++)
                W2[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        var rhoHat = new double[h];
        double lr = 0.01;

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            Array.Clear(rhoHat, 0, h);

            for (int i = 0; i < n; i++)
            {
                // Forward
                var hidden = new double[h];
                for (int j = 0; j < h; j++)
                {
                    for (int k = 0; k < p; k++)
                        hidden[j] += X[i, k] * W1[k, j];
                    hidden[j] = Sigmoid(hidden[j]);
                    rhoHat[j] += hidden[j];
                }

                var output = new double[p];
                for (int j = 0; j < p; j++)
                    for (int k = 0; k < h; k++)
                        output[j] += hidden[k] * W2[k, j];

                // Backward
                var outputError = new double[p];
                for (int j = 0; j < p; j++)
                    outputError[j] = output[j] - X[i, j];

                var hiddenError = new double[h];
                for (int j = 0; j < h; j++)
                {
                    for (int k = 0; k < p; k++)
                        hiddenError[j] += outputError[k] * W2[j, k];
                    hiddenError[j] *= hidden[j] * (1 - hidden[j]);
                }

                // Update
                for (int j = 0; j < h; j++)
                    for (int k = 0; k < p; k++)
                        W2[j, k] -= lr * outputError[k] * hidden[j];

                for (int j = 0; j < p; j++)
                    for (int k = 0; k < h; k++)
                        W1[j, k] -= lr * hiddenError[k] * X[i, j];
            }

            // Sparsity penalty
            for (int j = 0; j < h; j++)
            {
                rhoHat[j] /= n;
                double sparsityGrad = _sparsityWeight * (-_sparsityTarget / (rhoHat[j] + 1e-10) +
                                                         (1 - _sparsityTarget) / (1 - rhoHat[j] + 1e-10));
                for (int k = 0; k < p; k++)
                    W1[k, j] -= lr * sparsityGrad * 0.01;
            }
        }

        _featureImportances = new double[p];
        for (int j = 0; j < p; j++)
            for (int k = 0; k < h; k++)
                _featureImportances[j] += Math.Abs(W1[j, k]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Max(-500, Math.Min(500, x))));

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseAutoencoderSelector has not been fitted.");

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
        throw new NotSupportedException("SparseAutoencoderSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseAutoencoderSelector has not been fitted.");

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
