using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// MMPC (Max-Min Parents and Children) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using the MMPC algorithm, which identifies the Markov
/// blanket of the target - the minimal set of features needed for prediction.
/// </para>
/// <para><b>For Beginners:</b> MMPC finds features that are direct causes or
/// effects of the target (parents/children in a causal graph). It uses conditional
/// independence tests to identify these relationships. The result is a minimal
/// set of truly relevant features, removing spurious correlations.
/// </para>
/// <para>
/// <b>Note:</b> The alpha parameter in this implementation is used as a minimum association
/// threshold (correlation magnitude), not as a p-value significance level as in traditional MMPC.
/// Features with association below this threshold are considered conditionally independent.
/// </para>
/// </remarks>
public class MMPCSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minAssociationThreshold;

    private double[]? _associationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinAssociationThreshold => _minAssociationThreshold;
    public double[]? AssociationScores => _associationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MMPCSelector(
        int nFeaturesToSelect = 10,
        double minAssociationThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minAssociationThreshold <= 0 || minAssociationThreshold >= 1)
            throw new ArgumentException("Minimum association threshold must be between 0 and 1.", nameof(minAssociationThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minAssociationThreshold = minAssociationThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MMPCSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _associationScores = new double[p];

        // Phase 1: Forward - find candidate parents/children
        var candidates = new HashSet<int>();
        var candidateScores = new Dictionary<int, double>();

        // Initial pass: compute marginal association with target
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double assoc = ComputeAssociation(col, y);
            _associationScores[j] = assoc;
            candidateScores[j] = assoc;
        }

        // Greedy forward selection based on min-max criterion
        while (candidates.Count < Math.Min(_nFeaturesToSelect, p))
        {
            int bestFeature = -1;
            double bestMinAssoc = double.MinValue;

            foreach (var j in Enumerable.Range(0, p).Where(j => !candidates.Contains(j)))
            {
                // Min association conditioned on current candidates
                double minAssoc = candidateScores[j];

                foreach (var s in candidates)
                {
                    // Compute partial correlation (conditional association)
                    var colJ = new double[n];
                    var colS = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        colJ[i] = X[i, j];
                        colS[i] = X[i, s];
                    }

                    double condAssoc = ComputePartialAssociation(colJ, y, colS);
                    minAssoc = Math.Min(minAssoc, condAssoc);
                }

                if (minAssoc > bestMinAssoc)
                {
                    bestMinAssoc = minAssoc;
                    bestFeature = j;
                }
            }

            if (bestFeature < 0 || bestMinAssoc < _minAssociationThreshold)
                break;

            candidates.Add(bestFeature);
        }

        // Phase 2: Backward - remove false positives
        var toRemove = new HashSet<int>();
        foreach (var j in candidates)
        {
            var others = candidates.Where(k => k != j).ToList();
            if (others.Count == 0) continue;

            var colJ = new double[n];
            for (int i = 0; i < n; i++)
                colJ[i] = X[i, j];

            // Check if j is conditionally independent of y given others
            bool isIndependent = true;
            foreach (var s in others)
            {
                var colS = new double[n];
                for (int i = 0; i < n; i++)
                    colS[i] = X[i, s];

                double condAssoc = ComputePartialAssociation(colJ, y, colS);
                if (condAssoc >= _minAssociationThreshold)
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
                toRemove.Add(j);
        }

        foreach (var j in toRemove)
            candidates.Remove(j);

        // If we have fewer than requested, add more by score
        if (candidates.Count < _nFeaturesToSelect)
        {
            var remaining = Enumerable.Range(0, p)
                .Where(j => !candidates.Contains(j))
                .OrderByDescending(j => _associationScores[j])
                .Take(_nFeaturesToSelect - candidates.Count);
            foreach (var j in remaining)
                candidates.Add(j);
        }

        _selectedIndices = candidates.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double ComputeAssociation(double[] x, double[] y)
    {
        int n = x.Length;
        double meanX = x.Average();
        double meanY = y.Average();

        double cov = 0, varX = 0, varY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        double denom = Math.Sqrt(varX * varY);
        return denom > 1e-10 ? Math.Abs(cov / denom) : 0;
    }

    private double ComputePartialAssociation(double[] x, double[] y, double[] z)
    {
        int n = x.Length;

        // Regress x on z
        double meanZ = z.Average();
        double meanX = x.Average();
        double covXZ = 0, varZ = 0;
        for (int i = 0; i < n; i++)
        {
            covXZ += (x[i] - meanX) * (z[i] - meanZ);
            varZ += (z[i] - meanZ) * (z[i] - meanZ);
        }
        double slopeX = varZ > 1e-10 ? covXZ / varZ : 0;
        var residX = new double[n];
        for (int i = 0; i < n; i++)
            residX[i] = x[i] - meanX - slopeX * (z[i] - meanZ);

        // Regress y on z
        double meanY = y.Average();
        double covYZ = 0;
        for (int i = 0; i < n; i++)
            covYZ += (y[i] - meanY) * (z[i] - meanZ);
        double slopeY = varZ > 1e-10 ? covYZ / varZ : 0;
        var residY = new double[n];
        for (int i = 0; i < n; i++)
            residY[i] = y[i] - meanY - slopeY * (z[i] - meanZ);

        return ComputeAssociation(residX, residY);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

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
        throw new NotSupportedException("MMPCSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
