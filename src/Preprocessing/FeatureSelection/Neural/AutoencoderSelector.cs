using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Autoencoder-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a neural autoencoder to learn which features are essential for
/// reconstructing the data. Features with high contribution to the
/// bottleneck representation are selected.
/// </para>
/// <para><b>For Beginners:</b> An autoencoder is like a compression algorithm
/// that learns to squeeze data through a small "bottleneck" and then expand
/// it back. Features that are most important for this compression/reconstruction
/// are the ones we keep - they carry the most essential information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _bottleneckSize;
    private readonly int _nEpochs;
    private readonly double _learningRate;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double[,]? _encoderWeights;
    private double[,]? _decoderWeights;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AutoencoderSelector(
        int nFeaturesToSelect = 10,
        int bottleneckSize = 5,
        int nEpochs = 100,
        double learningRate = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _bottleneckSize = bottleneckSize;
        _nEpochs = nEpochs;
        _learningRate = learningRate;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert and normalize data
        var X = new double[n, p];
        var means = new double[p];
        var stds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;

            for (int i = 0; i < n; i++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - means[j]) / stds[j];
        }

        int bottleneck = Math.Min(_bottleneckSize, p - 1);
        var rand = RandomHelper.CreateSecureRandom();

        // Initialize weights
        _encoderWeights = new double[p, bottleneck];
        _decoderWeights = new double[bottleneck, p];
        double scale = Math.Sqrt(2.0 / (p + bottleneck));

        for (int i = 0; i < p; i++)
            for (int j = 0; j < bottleneck; j++)
                _encoderWeights[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        for (int i = 0; i < bottleneck; i++)
            for (int j = 0; j < p; j++)
                _decoderWeights[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        // Train autoencoder
        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass
                var encoded = new double[bottleneck];
                for (int j = 0; j < bottleneck; j++)
                {
                    for (int k = 0; k < p; k++)
                        encoded[j] += X[i, k] * _encoderWeights[k, j];
                    encoded[j] = Math.Tanh(encoded[j]);
                }

                var decoded = new double[p];
                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < bottleneck; k++)
                        decoded[j] += encoded[k] * _decoderWeights[k, j];
                }

                // Backward pass
                var outputError = new double[p];
                for (int j = 0; j < p; j++)
                    outputError[j] = decoded[j] - X[i, j];

                var hiddenError = new double[bottleneck];
                for (int j = 0; j < bottleneck; j++)
                {
                    for (int k = 0; k < p; k++)
                        hiddenError[j] += outputError[k] * _decoderWeights[j, k];
                    hiddenError[j] *= (1 - encoded[j] * encoded[j]); // tanh derivative
                }

                // Update weights
                for (int j = 0; j < bottleneck; j++)
                    for (int k = 0; k < p; k++)
                        _decoderWeights[j, k] -= _learningRate * outputError[k] * encoded[j];

                for (int j = 0; j < p; j++)
                    for (int k = 0; k < bottleneck; k++)
                        _encoderWeights[j, k] -= _learningRate * hiddenError[k] * X[i, j];
            }
        }

        // Compute feature importance from encoder weights
        _featureImportances = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < bottleneck; k++)
                _featureImportances[j] += Math.Abs(_encoderWeights[j, k]);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
