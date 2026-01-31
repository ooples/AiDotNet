using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Autoencoder based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their reconstruction importance in a simple autoencoder,
/// measuring how much each feature contributes to the learned representation.
/// </para>
/// <para><b>For Beginners:</b> An autoencoder compresses data and reconstructs it.
/// Features that are harder to reconstruct accurately are often more important
/// because they carry unique information not captured by other features.
/// </para>
/// </remarks>
public class AutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private double[]? _reconstructionImportance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int HiddenSize => _hiddenSize;
    public int Epochs => _epochs;
    public double LearningRate => _learningRate;
    public double[]? ReconstructionImportance => _reconstructionImportance;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 10,
        int epochs = 100,
        double learningRate = 0.01,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _epochs = epochs;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Standardize data
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

        int h = Math.Min(_hiddenSize, p);

        // Initialize weights (Xavier initialization)
        var W1 = new double[p, h];
        var W2 = new double[h, p];
        var b1 = new double[h];
        var b2 = new double[p];

        double scale1 = Math.Sqrt(2.0 / (p + h));
        double scale2 = Math.Sqrt(2.0 / (h + p));

        for (int j = 0; j < p; j++)
            for (int k = 0; k < h; k++)
                W1[j, k] = (rand.NextDouble() - 0.5) * 2 * scale1;

        for (int k = 0; k < h; k++)
            for (int j = 0; j < p; j++)
                W2[k, j] = (rand.NextDouble() - 0.5) * 2 * scale2;

        // Train autoencoder
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass
                var hidden = new double[h];
                for (int k = 0; k < h; k++)
                {
                    hidden[k] = b1[k];
                    for (int j = 0; j < p; j++)
                        hidden[k] += X[i, j] * W1[j, k];
                    hidden[k] = Math.Tanh(hidden[k]); // Activation
                }

                var output = new double[p];
                for (int j = 0; j < p; j++)
                {
                    output[j] = b2[j];
                    for (int k = 0; k < h; k++)
                        output[j] += hidden[k] * W2[k, j];
                }

                // Backward pass
                var dOutput = new double[p];
                for (int j = 0; j < p; j++)
                    dOutput[j] = output[j] - X[i, j];

                var dHidden = new double[h];
                for (int k = 0; k < h; k++)
                {
                    double sum = 0;
                    for (int j = 0; j < p; j++)
                        sum += dOutput[j] * W2[k, j];
                    dHidden[k] = sum * (1 - hidden[k] * hidden[k]); // tanh derivative
                }

                // Update weights
                for (int k = 0; k < h; k++)
                    for (int j = 0; j < p; j++)
                        W2[k, j] -= _learningRate * dOutput[j] * hidden[k];

                for (int j = 0; j < p; j++)
                    b2[j] -= _learningRate * dOutput[j];

                for (int j = 0; j < p; j++)
                    for (int k = 0; k < h; k++)
                        W1[j, k] -= _learningRate * dHidden[k] * X[i, j];

                for (int k = 0; k < h; k++)
                    b1[k] -= _learningRate * dHidden[k];
            }
        }

        // Compute reconstruction error per feature
        _reconstructionImportance = new double[p];
        for (int i = 0; i < n; i++)
        {
            var hidden = new double[h];
            for (int k = 0; k < h; k++)
            {
                hidden[k] = b1[k];
                for (int j = 0; j < p; j++)
                    hidden[k] += X[i, j] * W1[j, k];
                hidden[k] = Math.Tanh(hidden[k]);
            }

            var output = new double[p];
            for (int j = 0; j < p; j++)
            {
                output[j] = b2[j];
                for (int k = 0; k < h; k++)
                    output[j] += hidden[k] * W2[k, j];
            }

            for (int j = 0; j < p; j++)
                _reconstructionImportance[j] += Math.Abs(X[i, j] - output[j]);
        }

        for (int j = 0; j < p; j++)
            _reconstructionImportance[j] /= n;

        // Features with higher reconstruction error are potentially more important
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _reconstructionImportance[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutoencoderSelector has not been fitted.");

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
        throw new NotSupportedException("AutoencoderSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutoencoderSelector has not been fitted.");

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
