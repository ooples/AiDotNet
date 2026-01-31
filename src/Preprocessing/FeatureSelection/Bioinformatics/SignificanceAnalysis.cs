using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Significance Analysis of Microarrays (SAM) for bioinformatics feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SAM uses a modified t-statistic that includes a small constant (fudge factor) in
/// the denominator to avoid false positives from genes with very low variance. It
/// estimates false discovery rate (FDR) using permutation testing.
/// </para>
/// <para><b>For Beginners:</b> When analyzing gene expression data, some genes have
/// very small changes but even smaller variance, making them appear significant by
/// chance. SAM adds a small buffer to prevent these false alarms. It also uses
/// shuffling to estimate how many of our "discoveries" might be false.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SignificanceAnalysis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly double _fdrThreshold;
    private readonly int? _randomState;

    private double[]? _samScores;
    private double[]? _qValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NPermutations => _nPermutations;
    public double FDRThreshold => _fdrThreshold;
    public double[]? SAMScores => _samScores;
    public double[]? QValues => _qValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SignificanceAnalysis(
        int nFeaturesToSelect = 10,
        int nPermutations = 100,
        double fdrThreshold = 0.05,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _fdrThreshold = fdrThreshold;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SignificanceAnalysis requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Separate samples by class
        var class0Idx = new List<int>();
        var class1Idx = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0Idx.Add(i);
            else
                class1Idx.Add(i);
        }

        int n0 = class0Idx.Count;
        int n1 = class1Idx.Count;

        if (n0 < 2 || n1 < 2)
        {
            // Not enough samples in each class
            _samScores = new double[p];
            _qValues = new double[p];
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // Compute SAM statistics
        _samScores = new double[p];

        // Estimate s0 (fudge factor) using median standard deviation
        var pooledStds = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0Idx)
                mean0 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;

            foreach (int i in class1Idx)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean1 /= n1;

            double ss0 = 0, ss1 = 0;
            foreach (int i in class0Idx)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean0;
                ss0 += diff * diff;
            }
            foreach (int i in class1Idx)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean1;
                ss1 += diff * diff;
            }

            double pooledVar = (ss0 + ss1) / (n0 + n1 - 2);
            pooledStds[j] = Math.Sqrt(pooledVar * (1.0 / n0 + 1.0 / n1));
        }

        double s0 = pooledStds.OrderBy(x => x).Skip(p / 2).First(); // Median

        // Compute d_i (SAM statistic)
        for (int j = 0; j < p; j++)
        {
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0Idx)
                mean0 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;

            foreach (int i in class1Idx)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean1 /= n1;

            _samScores[j] = (mean1 - mean0) / (pooledStds[j] + s0);
        }

        // Permutation testing for FDR estimation
        var permutedScores = new List<double>[p];
        for (int j = 0; j < p; j++)
            permutedScores[j] = [];

        var allIndices = Enumerable.Range(0, n).ToArray();

        for (int perm = 0; perm < _nPermutations; perm++)
        {
            // Shuffle labels
            var shuffled = allIndices.OrderBy(_ => random.Next()).ToArray();
            var permClass0 = shuffled.Take(n0).ToList();
            var permClass1 = shuffled.Skip(n0).ToList();

            for (int j = 0; j < p; j++)
            {
                double mean0 = 0, mean1 = 0;
                foreach (int i in permClass0)
                    mean0 += NumOps.ToDouble(data[i, j]);
                mean0 /= n0;

                foreach (int i in permClass1)
                    mean1 += NumOps.ToDouble(data[i, j]);
                mean1 /= n1;

                double permScore = (mean1 - mean0) / (pooledStds[j] + s0);
                permutedScores[j].Add(Math.Abs(permScore));
            }
        }

        // Estimate q-values (FDR)
        _qValues = new double[p];
        for (int j = 0; j < p; j++)
        {
            double absScore = Math.Abs(_samScores[j]);
            int falsePosCount = permutedScores[j].Count(ps => ps >= absScore);
            _qValues[j] = (double)falsePosCount / _nPermutations;
        }

        // Select features
        var candidates = Enumerable.Range(0, p)
            .Where(j => _qValues[j] <= _fdrThreshold)
            .OrderByDescending(j => Math.Abs(_samScores[j]))
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        if (numToSelect == 0)
        {
            // No features pass FDR, select top by score
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => Math.Abs(_samScores[j]))
                .Take(Math.Min(_nFeaturesToSelect, p))
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = candidates
                .Take(numToSelect)
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
            throw new InvalidOperationException("SignificanceAnalysis has not been fitted.");

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
        throw new NotSupportedException("SignificanceAnalysis does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignificanceAnalysis has not been fitted.");

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
