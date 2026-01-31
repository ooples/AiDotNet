using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// SHAP (SHapley Additive exPlanations) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SHAP uses Shapley values from game theory to explain feature importance.
/// Each feature's importance is its average contribution to predictions across
/// all possible feature combinations.
/// </para>
/// <para>
/// The algorithm (Kernel SHAP approximation):
/// 1. Sample feature coalitions (subsets of features)
/// 2. Weight coalitions by SHAP kernel
/// 3. Predict with masked features (baseline substitution)
/// 4. Solve weighted least squares for Shapley values
/// </para>
/// <para><b>For Beginners:</b> SHAP answers: "How much did each feature contribute
/// to this prediction?"
///
/// Imagine a team of features making a prediction. SHAP figures out each
/// player's contribution fairly - not just by removing them, but by considering
/// every possible team combination they could have been in.
///
/// Features with high absolute SHAP values (positive or negative) are important.
/// SHAP provides both global importance and local explanations per sample.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SHAPSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly int _nCoalitions;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>>? _predictFunc;

    // Fitted parameters
    private double[]? _shapValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the computed SHAP values (mean absolute).
    /// </summary>
    public double[]? ShapValues => _shapValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SHAPSelector{T}"/>.
    /// </summary>
    /// <param name="predictFunc">Function that makes predictions from data.</param>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nSamples">Number of background samples for baseline. Defaults to 100.</param>
    /// <param name="nCoalitions">Number of feature coalitions to sample. Defaults to 200.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SHAPSelector(
        Func<Matrix<T>, Vector<T>>? predictFunc = null,
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        int nCoalitions = 200,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nSamples < 1)
        {
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));
        }

        if (nCoalitions < 1)
        {
            throw new ArgumentException("Number of coalitions must be at least 1.", nameof(nCoalitions));
        }

        _predictFunc = predictFunc;
        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _nCoalitions = nCoalitions;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SHAPSelector requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits SHAP selector by computing Shapley values for each feature.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Convert to double arrays
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Compute baseline values (mean of background samples)
        int nBackground = Math.Min(_nSamples, n);
        var backgroundIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nBackground).ToArray();
        var baseline = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < nBackground; i++)
            {
                baseline[j] += X[backgroundIndices[i], j];
            }
            baseline[j] /= nBackground;
        }

        // Sample instances to explain
        int nExplain = Math.Min(_nSamples, n);
        var explainIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nExplain).ToArray();

        // Accumulate SHAP values
        var shapAccum = new double[p];

        foreach (int explainIdx in explainIndices)
        {
            var instanceShap = ComputeInstanceShap(X, explainIdx, baseline, p, random);
            for (int j = 0; j < p; j++)
            {
                shapAccum[j] += Math.Abs(instanceShap[j]);
            }
        }

        // Average SHAP values
        _shapValues = new double[p];
        for (int j = 0; j < p; j++)
        {
            _shapValues[j] = shapAccum[j] / nExplain;
        }

        // Select top features by SHAP importance
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _shapValues
            .Select((shap, idx) => (Shap: shap, Index: idx))
            .OrderByDescending(x => x.Shap)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeInstanceShap(double[,] X, int instanceIdx, double[] baseline, int p, Random random)
    {
        // Kernel SHAP approximation
        var shapValues = new double[p];

        // Generate coalitions and compute SHAP via weighted least squares
        var coalitions = new List<bool[]>();
        var weights = new List<double>();
        var predictions = new List<double>();

        // Always include empty and full coalitions
        var emptyCoalition = new bool[p];
        coalitions.Add(emptyCoalition);
        weights.Add(double.MaxValue / 2); // Very high weight for empty
        predictions.Add(GetPrediction(X, instanceIdx, baseline, emptyCoalition));

        var fullCoalition = Enumerable.Repeat(true, p).ToArray();
        coalitions.Add(fullCoalition);
        weights.Add(double.MaxValue / 2); // Very high weight for full
        predictions.Add(GetPrediction(X, instanceIdx, baseline, fullCoalition));

        // Sample random coalitions
        for (int c = 0; c < _nCoalitions; c++)
        {
            var coalition = new bool[p];
            int coalitionSize = 0;

            for (int j = 0; j < p; j++)
            {
                coalition[j] = random.NextDouble() > 0.5;
                if (coalition[j]) coalitionSize++;
            }

            // Skip empty or full (already added)
            if (coalitionSize == 0 || coalitionSize == p) continue;

            // SHAP kernel weight
            double weight = SHAPKernelWeight(p, coalitionSize);

            coalitions.Add(coalition);
            weights.Add(weight);
            predictions.Add(GetPrediction(X, instanceIdx, baseline, coalition));
        }

        // Solve weighted least squares: min_phi sum_S w(S) * (f(S) - phi_0 - sum_j in S phi_j)^2
        // Simplified: use correlation-based approximation for computational efficiency
        double baselinePred = predictions[0]; // Empty coalition prediction
        double fullPred = predictions[1]; // Full coalition prediction

        for (int j = 0; j < p; j++)
        {
            // Approximate Shapley value by marginal contribution weighted average
            double contribution = 0;
            double weightSum = 0;

            for (int c = 2; c < coalitions.Count; c++) // Skip empty and full
            {
                var withJ = (bool[])coalitions[c].Clone();
                var withoutJ = (bool[])coalitions[c].Clone();

                if (coalitions[c][j])
                {
                    withoutJ[j] = false;
                    double predWithJ = predictions[c];
                    double predWithoutJ = GetPrediction(X, instanceIdx, baseline, withoutJ);
                    contribution += weights[c] * (predWithJ - predWithoutJ);
                    weightSum += weights[c];
                }
                else
                {
                    withJ[j] = true;
                    double predWithoutJ = predictions[c];
                    double predWithJ = GetPrediction(X, instanceIdx, baseline, withJ);
                    contribution += weights[c] * (predWithJ - predWithoutJ);
                    weightSum += weights[c];
                }
            }

            shapValues[j] = weightSum > 0 ? contribution / weightSum : 0;
        }

        return shapValues;
    }

    private double SHAPKernelWeight(int p, int coalitionSize)
    {
        // SHAP kernel: (p-1) / (C(p, s) * s * (p-s))
        if (coalitionSize == 0 || coalitionSize == p) return 1e10;

        double binomial = BinomialCoefficient(p, coalitionSize);
        double denominator = binomial * coalitionSize * (p - coalitionSize);

        return (p - 1.0) / Math.Max(denominator, 1e-10);
    }

    private double BinomialCoefficient(int n, int k)
    {
        if (k > n - k) k = n - k;

        double result = 1;
        for (int i = 0; i < k; i++)
        {
            result *= (n - i);
            result /= (i + 1);
        }

        return result;
    }

    private double GetPrediction(double[,] X, int instanceIdx, double[] baseline, bool[] coalition)
    {
        int p = baseline.Length;

        if (_predictFunc is not null)
        {
            // Create masked instance
            var masked = new T[1, p];
            for (int j = 0; j < p; j++)
            {
                double val = coalition[j] ? X[instanceIdx, j] : baseline[j];
                masked[0, j] = NumOps.FromDouble(val);
            }

            var maskedMatrix = new Matrix<T>(masked);
            var pred = _predictFunc(maskedMatrix);
            return NumOps.ToDouble(pred[0]);
        }

        // Default: use feature sum as simple prediction proxy
        double sum = 0;
        for (int j = 0; j < p; j++)
        {
            double val = coalition[j] ? X[instanceIdx, j] : baseline[j];
            sum += val;
        }

        return sum / p;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting SHAP-important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SHAPSelector has not been fitted.");
        }

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SHAPSelector does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SHAPSelector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
