using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Variational Autoencoder-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a Variational Autoencoder (VAE) to learn a probabilistic latent space,
/// then selects features based on their contribution to the learned distribution.
/// </para>
/// <para><b>For Beginners:</b> A VAE learns not just a compressed representation
/// but a probability distribution over possible representations. Features that
/// contribute most to this learned distribution are the important ones.
/// </para>
/// </remarks>
public class VariationalAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _latentDim;
    private readonly int _nEpochs;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VariationalAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int latentDim = 5,
        int nEpochs = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _latentDim = latentDim;
        _nEpochs = nEpochs;
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

        // Normalize to [0, 1]
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++) { min = Math.Min(min, X[i, j]); max = Math.Max(max, X[i, j]); }
            double range = max - min + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - min) / range;
        }

        int z = Math.Min(_latentDim, p - 1);
        var rand = RandomHelper.CreateSecureRandom();
        double scale = 0.1;

        var WMu = new double[p, z];
        var WLogVar = new double[p, z];
        var WDec = new double[z, p];

        for (int i = 0; i < p; i++)
            for (int j = 0; j < z; j++)
            {
                WMu[i, j] = (rand.NextDouble() * 2 - 1) * scale;
                WLogVar[i, j] = (rand.NextDouble() * 2 - 1) * scale;
            }
        for (int i = 0; i < z; i++)
            for (int j = 0; j < p; j++)
                WDec[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        double lr = 0.001;

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Encode
                var mu = new double[z];
                var logVar = new double[z];
                for (int j = 0; j < z; j++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        mu[j] += X[i, k] * WMu[k, j];
                        logVar[j] += X[i, k] * WLogVar[k, j];
                    }
                    logVar[j] = Math.Max(-10, Math.Min(10, logVar[j]));
                }

                // Reparameterize
                var latent = new double[z];
                for (int j = 0; j < z; j++)
                {
                    double eps = SampleNormal(rand);
                    latent[j] = mu[j] + Math.Exp(0.5 * logVar[j]) * eps;
                }

                // Decode
                var output = new double[p];
                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < z; k++)
                        output[j] += latent[k] * WDec[k, j];
                    output[j] = Sigmoid(output[j]);
                }

                // Reconstruction loss gradient
                var reconGrad = new double[p];
                for (int j = 0; j < p; j++)
                    reconGrad[j] = output[j] - X[i, j];

                // KL divergence gradient
                var klGradMu = new double[z];
                var klGradLogVar = new double[z];
                for (int j = 0; j < z; j++)
                {
                    klGradMu[j] = mu[j];
                    klGradLogVar[j] = 0.5 * (Math.Exp(logVar[j]) - 1);
                }

                // Backprop through decoder
                var latentGrad = new double[z];
                for (int j = 0; j < z; j++)
                    for (int k = 0; k < p; k++)
                        latentGrad[j] += reconGrad[k] * WDec[j, k];

                // Update decoder
                for (int j = 0; j < z; j++)
                    for (int k = 0; k < p; k++)
                        WDec[j, k] -= lr * reconGrad[k] * latent[j];

                // Update encoder
                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < z; k++)
                    {
                        WMu[j, k] -= lr * (latentGrad[k] + 0.01 * klGradMu[k]) * X[i, j];
                        WLogVar[j, k] -= lr * (latentGrad[k] * 0.5 * Math.Exp(0.5 * logVar[k]) + 0.01 * klGradLogVar[k]) * X[i, j];
                    }
                }
            }
        }

        _featureImportances = new double[p];
        for (int j = 0; j < p; j++)
            for (int k = 0; k < z; k++)
                _featureImportances[j] += Math.Abs(WMu[j, k]) + Math.Abs(WLogVar[j, k]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Max(-500, Math.Min(500, x))));
    private double SampleNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VariationalAutoencoderSelector has not been fitted.");

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
        throw new NotSupportedException("VariationalAutoencoderSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VariationalAutoencoderSelector has not been fitted.");

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
