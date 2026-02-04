using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.SemiSupervised;

/// <summary>
/// Semi-Supervised Feature Selection using Laplacian regularization.
/// </summary>
/// <remarks>
/// <para>
/// Combines supervised information from labeled samples with the manifold
/// structure captured from all samples (labeled and unlabeled) using the
/// graph Laplacian.
/// </para>
/// <para><b>For Beginners:</b> When you have few labeled samples but many
/// unlabeled ones, this method uses both. It builds a neighborhood graph
/// from all data to understand the data's structure, while using labels
/// where available.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SemiSupervisedLaplacian<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _alpha;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SemiSupervisedLaplacian(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double alpha = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (alpha < 0 || alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SemiSupervisedLaplacian requires partial labels. Use Fit with labeled mask.");
    }

    public void Fit(Matrix<T> data, Vector<T> target, bool[] isLabeled)
    {
        if (data.Rows != target.Length || data.Rows != isLabeled.Length)
            throw new ArgumentException("Data rows, target length, and label mask must match.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabeled = isLabeled.Count(l => l);

        // Build k-NN graph
        var W = BuildKNNGraph(data, n, p);

        // Compute graph Laplacian
        var D = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                D[i] += W[i, j];

        // Compute feature scores combining supervised and unsupervised
        _featureScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract feature values
            var fValues = new double[n];
            for (int i = 0; i < n; i++)
                fValues[i] = NumOps.ToDouble(data[i, j]);

            // Unsupervised: Laplacian score
            double laplacianScore = ComputeLaplacianScore(fValues, W, D, n);

            // Supervised: correlation with labeled targets
            double supervisedScore = 0;
            if (nLabeled > 0)
            {
                var labeledFeatures = new List<double>();
                var labeledTargets = new List<double>();
                for (int i = 0; i < n; i++)
                {
                    if (isLabeled[i])
                    {
                        labeledFeatures.Add(fValues[i]);
                        labeledTargets.Add(NumOps.ToDouble(target[i]));
                    }
                }

                supervisedScore = Math.Abs(ComputeCorrelation(
                    labeledFeatures.ToArray(),
                    labeledTargets.ToArray(),
                    nLabeled));
            }

            // Combine scores
            _featureScores[j] = _alpha * supervisedScore + (1 - _alpha) * (1 - laplacianScore);
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] BuildKNNGraph(Matrix<T> data, int n, int p)
    {
        var W = new double[n, n];

        // Compute pairwise distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = NumOps.ToDouble(data[i, k]) - NumOps.ToDouble(data[j, k]);
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        // Build k-NN adjacency
        for (int i = 0; i < n; i++)
        {
            var neighbors = Enumerable.Range(0, n)
                .Where(j => j != i)
                .OrderBy(j => distances[i, j])
                .Take(_nNeighbors)
                .ToList();

            foreach (int j in neighbors)
            {
                double sigma = distances[i, neighbors.Last()] + 1e-10;
                W[i, j] = Math.Exp(-distances[i, j] * distances[i, j] / (2 * sigma * sigma));
                W[j, i] = W[i, j];  // Symmetric
            }
        }

        return W;
    }

    private double ComputeLaplacianScore(double[] f, double[,] W, double[] D, int n)
    {
        // Center feature
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += D[i] * f[i];
        double totalD = D.Sum();
        double fBar = sum / (totalD + 1e-10);

        var fCentered = new double[n];
        for (int i = 0; i < n; i++)
            fCentered[i] = f[i] - fBar;

        // Compute f^T * L * f / f^T * D * f
        double fDf = 0;
        for (int i = 0; i < n; i++)
            fDf += fCentered[i] * D[i] * fCentered[i];

        if (fDf < 1e-10) return 0;

        double fLf = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (W[i, j] > 0)
                    fLf += W[i, j] * (fCentered[i] - fCentered[j]) * (fCentered[i] - fCentered[j]);
            }
        }
        fLf /= 2;  // Each pair counted twice

        return fLf / fDf;
    }

    private double ComputeCorrelation(double[] x, double[] y, int n)
    {
        double xMean = x.Average();
        double yMean = y.Average();

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - xMean;
            double dy = y[i] - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10) return 0;
        return ssXY / Math.Sqrt(ssXX * ssYY);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target, bool[] isLabeled)
    {
        Fit(data, target, isLabeled);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SemiSupervisedLaplacian has not been fitted.");

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
        throw new NotSupportedException("SemiSupervisedLaplacian does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SemiSupervisedLaplacian has not been fitted.");

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
