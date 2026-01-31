using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Significance Analysis of Microarrays (SAM) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SAM uses a modified t-statistic with a fudge factor to identify significantly
/// differentially expressed genes. It controls the false discovery rate (FDR)
/// using permutation-based estimation.
/// </para>
/// <para><b>For Beginners:</b> SAM was designed for gene expression data where we
/// want to find genes that behave differently between conditions (e.g., healthy vs
/// disease). It uses permutation testing to estimate how many "significant" genes
/// we'd find just by chance, helping control false positives.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SAM<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly double _fdrThreshold;
    private readonly double _s0Percentile;
    private readonly int? _randomState;

    private double[]? _dStatistics;
    private double[]? _fdr;
    private double[]? _qValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DStatistics => _dStatistics;
    public double[]? FDR => _fdr;
    public double[]? QValues => _qValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SAM(
        int nFeaturesToSelect = 100,
        int nPermutations = 100,
        double fdrThreshold = 0.05,
        double s0Percentile = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPermutations < 10)
            throw new ArgumentException("Number of permutations must be at least 10.", nameof(nPermutations));
        if (fdrThreshold <= 0 || fdrThreshold >= 1)
            throw new ArgumentException("FDR threshold must be between 0 and 1.", nameof(fdrThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _fdrThreshold = fdrThreshold;
        _s0Percentile = s0Percentile;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SAM requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Identify two-class labels
        var class0 = new List<int>();
        var class1 = new List<int>();
        double threshold = 0.5;

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < threshold)
                class0.Add(i);
            else
                class1.Add(i);
        }

        int n0 = class0.Count;
        int n1 = class1.Count;

        if (n0 < 2 || n1 < 2)
            throw new ArgumentException("SAM requires at least 2 samples in each class.");

        // Calculate s0 (fudge factor) from data
        var standardErrors = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;
            mean1 /= n1;

            double ss0 = 0, ss1 = 0;
            foreach (int i in class0)
                ss0 += Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2);
            foreach (int i in class1)
                ss1 += Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2);

            double pooledVar = (ss0 + ss1) / (n0 + n1 - 2);
            standardErrors[j] = Math.Sqrt(pooledVar * (1.0 / n0 + 1.0 / n1));
        }

        // s0 is the percentile of standard errors
        var sortedSE = standardErrors.OrderBy(x => x).ToArray();
        int s0Index = (int)(_s0Percentile * (p - 1));
        double s0 = sortedSE[s0Index];

        // Calculate d-statistics
        _dStatistics = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;
            mean1 /= n1;

            _dStatistics[j] = (mean1 - mean0) / (standardErrors[j] + s0);
        }

        // Permutation test
        var permutedD = new double[_nPermutations, p];
        var indices = Enumerable.Range(0, n).ToArray();

        for (int perm = 0; perm < _nPermutations; perm++)
        {
            // Shuffle indices
            for (int i = n - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            // Create permuted classes
            var permClass0 = indices.Take(n0).ToList();
            var permClass1 = indices.Skip(n0).ToList();

            for (int j = 0; j < p; j++)
            {
                double mean0 = 0, mean1 = 0;
                foreach (int i in permClass0)
                    mean0 += NumOps.ToDouble(data[i, j]);
                foreach (int i in permClass1)
                    mean1 += NumOps.ToDouble(data[i, j]);
                mean0 /= n0;
                mean1 /= n1;

                permutedD[perm, j] = (mean1 - mean0) / (standardErrors[j] + s0);
            }
        }

        // Calculate q-values (FDR-adjusted p-values)
        _qValues = new double[p];
        _fdr = new double[p];

        for (int j = 0; j < p; j++)
        {
            double dObs = Math.Abs(_dStatistics[j]);
            int falsePositives = 0;
            int truePositives = 0;

            // Count how many permuted d-values exceed observed
            for (int perm = 0; perm < _nPermutations; perm++)
            {
                for (int k = 0; k < p; k++)
                {
                    if (Math.Abs(permutedD[perm, k]) >= dObs)
                        falsePositives++;
                }
            }

            // Count observed d-values exceeding threshold
            for (int k = 0; k < p; k++)
            {
                if (Math.Abs(_dStatistics[k]) >= dObs)
                    truePositives++;
            }

            double expectedFP = (double)falsePositives / _nPermutations;
            _fdr[j] = truePositives > 0 ? expectedFP / truePositives : 1.0;
            _fdr[j] = Math.Min(1.0, _fdr[j]);
            _qValues[j] = _fdr[j];
        }

        // Select significant features
        var significant = new List<(int Index, double AbsD)>();
        for (int j = 0; j < p; j++)
        {
            if (_qValues[j] <= _fdrThreshold)
                significant.Add((j, Math.Abs(_dStatistics[j])));
        }

        if (significant.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = significant
                .OrderByDescending(x => x.AbsD)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Fall back to top by |d|
            _selectedIndices = _dStatistics
                .Select((d, idx) => (D: Math.Abs(d), Index: idx))
                .OrderByDescending(x => x.D)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("SAM has not been fitted.");

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
        throw new NotSupportedException("SAM does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SAM has not been fitted.");

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
