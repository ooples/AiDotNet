using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// LIME (Local Interpretable Model-agnostic Explanations) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// LIME explains individual predictions by approximating the model locally with
/// an interpretable linear model. Feature importance is derived from the linear
/// coefficients of these local explanations.
/// </para>
/// <para>
/// The algorithm:
/// 1. For each instance to explain:
///    a. Generate perturbed samples around the instance
///    b. Weight samples by proximity to original instance
///    c. Fit weighted linear regression on perturbed samples
///    d. Extract feature importance from coefficients
/// 2. Aggregate importance across all explained instances
/// </para>
/// <para><b>For Beginners:</b> LIME answers: "For this specific prediction,
/// which features mattered most?"
///
/// It works by creating slightly modified versions of your data point,
/// seeing how predictions change, and fitting a simple linear model to
/// understand which features drive the prediction locally.
///
/// Unlike global methods, LIME provides instance-specific explanations.
/// For feature selection, we average importance across many instances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LIMESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamplesToExplain;
    private readonly int _nPerturbations;
    private readonly double _kernelWidth;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>>? _predictFunc;

    // Fitted parameters
    private double[]? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the computed LIME importance values (mean absolute coefficients).
    /// </summary>
    public double[]? Importances => _importances;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LIMESelector{T}"/>.
    /// </summary>
    /// <param name="predictFunc">Function that makes predictions from data.</param>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nSamplesToExplain">Number of instances to explain. Defaults to 100.</param>
    /// <param name="nPerturbations">Number of perturbations per instance. Defaults to 500.</param>
    /// <param name="kernelWidth">Width of the exponential kernel. Defaults to 0.75.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public LIMESelector(
        Func<Matrix<T>, Vector<T>>? predictFunc = null,
        int nFeaturesToSelect = 10,
        int nSamplesToExplain = 100,
        int nPerturbations = 500,
        double kernelWidth = 0.75,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nSamplesToExplain < 1)
        {
            throw new ArgumentException("Number of samples to explain must be at least 1.", nameof(nSamplesToExplain));
        }

        if (nPerturbations < 1)
        {
            throw new ArgumentException("Number of perturbations must be at least 1.", nameof(nPerturbations));
        }

        if (kernelWidth <= 0)
        {
            throw new ArgumentException("Kernel width must be positive.", nameof(kernelWidth));
        }

        _predictFunc = predictFunc;
        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamplesToExplain = nSamplesToExplain;
        _nPerturbations = nPerturbations;
        _kernelWidth = kernelWidth;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LIMESelector requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits LIME selector by computing local explanations and aggregating importance.
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

        // Compute feature means and stds for perturbation
        var means = new double[p];
        var stds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += X[i, j];
            }
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(X[i, j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        // Sample instances to explain
        int nExplain = Math.Min(_nSamplesToExplain, n);
        var explainIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nExplain).ToArray();

        // Accumulate importance
        var importanceAccum = new double[p];

        foreach (int explainIdx in explainIndices)
        {
            var instanceImportance = ExplainInstance(X, explainIdx, means, stds, p, random, data);
            for (int j = 0; j < p; j++)
            {
                importanceAccum[j] += Math.Abs(instanceImportance[j]);
            }
        }

        // Average importance
        _importances = new double[p];
        for (int j = 0; j < p; j++)
        {
            _importances[j] = importanceAccum[j] / nExplain;
        }

        // Select top features by importance
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _importances
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ExplainInstance(double[,] X, int instanceIdx, double[] means, double[] stds, int p, Random random, Matrix<T> originalData)
    {
        int n = X.GetLength(0);

        // Generate perturbed samples
        var perturbedX = new double[_nPerturbations, p];
        var binaryMasks = new bool[_nPerturbations, p];
        var predictions = new double[_nPerturbations];
        var weights = new double[_nPerturbations];

        for (int s = 0; s < _nPerturbations; s++)
        {
            // Random binary mask (which features to keep original)
            int nFeaturesOn = 0;
            for (int j = 0; j < p; j++)
            {
                binaryMasks[s, j] = random.NextDouble() > 0.5;
                if (binaryMasks[s, j])
                {
                    perturbedX[s, j] = X[instanceIdx, j];
                    nFeaturesOn++;
                }
                else
                {
                    // Sample from data distribution
                    perturbedX[s, j] = means[j] + stds[j] * (random.NextDouble() * 2 - 1);
                }
            }

            // Compute distance (Hamming distance on binary mask)
            double distance = 0;
            for (int j = 0; j < p; j++)
            {
                if (!binaryMasks[s, j]) distance += 1;
            }
            distance /= p;

            // Exponential kernel weight
            weights[s] = Math.Exp(-distance * distance / (_kernelWidth * _kernelWidth));

            // Get prediction
            predictions[s] = GetPrediction(perturbedX, s, p, originalData);
        }

        // Fit weighted linear regression
        // Using normal equations with weights: (X'WX)^-1 X'Wy
        var coefficients = FitWeightedLinearRegression(binaryMasks, predictions, weights, p);

        return coefficients;
    }

    private double[] FitWeightedLinearRegression(bool[,] X, double[] y, double[] w, int p)
    {
        int n = y.Length;

        // Convert boolean mask to double (0/1)
        var Xd = new double[n, p + 1]; // +1 for intercept
        for (int i = 0; i < n; i++)
        {
            Xd[i, 0] = 1; // Intercept
            for (int j = 0; j < p; j++)
            {
                Xd[i, j + 1] = X[i, j] ? 1.0 : 0.0;
            }
        }

        // Compute X'WX
        var XtWX = new double[p + 1, p + 1];
        for (int i = 0; i < p + 1; i++)
        {
            for (int j = 0; j < p + 1; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    XtWX[i, j] += Xd[k, i] * w[k] * Xd[k, j];
                }
            }
        }

        // Compute X'Wy
        var XtWy = new double[p + 1];
        for (int i = 0; i < p + 1; i++)
        {
            for (int k = 0; k < n; k++)
            {
                XtWy[i] += Xd[k, i] * w[k] * y[k];
            }
        }

        // Add regularization
        for (int i = 0; i < p + 1; i++)
        {
            XtWX[i, i] += 1e-6;
        }

        // Solve system (simple Gaussian elimination)
        var coeffs = SolveLinearSystem(XtWX, XtWy, p + 1);

        // Return feature coefficients (skip intercept)
        var result = new double[p];
        Array.Copy(coeffs, 1, result, 0, p);

        return result;
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Gaussian elimination with partial pivoting
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = 0; j <= n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            // Eliminate
            if (Math.Abs(augmented[col, col]) > 1e-10)
            {
                for (int row = col + 1; row < n; row++)
                {
                    double factor = augmented[row, col] / augmented[col, col];
                    for (int j = col; j <= n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= augmented[i, j] * x[j];
            }
            if (Math.Abs(augmented[i, i]) > 1e-10)
            {
                x[i] /= augmented[i, i];
            }
        }

        return x;
    }

    private double GetPrediction(double[,] perturbedX, int sampleIdx, int p, Matrix<T> originalData)
    {
        if (_predictFunc is not null)
        {
            var sample = new T[1, p];
            for (int j = 0; j < p; j++)
            {
                sample[0, j] = NumOps.FromDouble(perturbedX[sampleIdx, j]);
            }

            var sampleMatrix = new Matrix<T>(sample);
            var pred = _predictFunc(sampleMatrix);
            return NumOps.ToDouble(pred[0]);
        }

        // Default: use feature sum as simple prediction proxy
        double sum = 0;
        for (int j = 0; j < p; j++)
        {
            sum += perturbedX[sampleIdx, j];
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
    /// Transforms the data by selecting LIME-important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("LIMESelector has not been fitted.");
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
        throw new NotSupportedException("LIMESelector does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("LIMESelector has not been fitted.");
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
