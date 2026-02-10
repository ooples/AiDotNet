using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Transforms tabular data using Variational Gaussian Mixture (VGM) mode-specific normalization
/// for continuous columns and one-hot encoding for categorical columns. Used by CTGAN and TVAE.
/// </summary>
/// <remarks>
/// <para>
/// The transformation follows the CTGAN paper (Xu et al., NeurIPS 2019):
/// - <b>Continuous columns</b>: A Gaussian mixture model (GMM) is fitted per column.
///   Each value is then represented as (normalized_value, one-hot_mode_indicator),
///   where the normalized value is relative to the selected mode's mean and std.
/// - <b>Categorical columns</b>: One-hot encoded into binary indicator vectors.
/// - <b>Inverse transform</b>: Reconstructs original-scale values from the transformed representation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-world data often has columns with very different distributions.
/// A "price" column might have values clustered around $10 and $100 (two modes), while an
/// "age" column might be normally distributed. Simple min-max or z-score normalization doesn't
/// handle multi-modal (multiple-peak) distributions well.
///
/// VGM normalization solves this by:
/// 1. Fitting a Gaussian mixture (like fitting multiple bell curves) to each continuous column
/// 2. For each value, finding which bell curve (mode) it belongs to
/// 3. Normalizing relative to that mode (so both "cheap" and "expensive" items get reasonable values)
/// 4. Adding a one-hot indicator showing which mode was chosen
///
/// This helps the generator learn multi-modal distributions much more effectively.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabularDataTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _vgmModes;
    private readonly Random _random;

    // Per-column VGM parameters (for continuous columns)
    private readonly List<VGMColumnInfo> _continuousColumnInfos = new();

    // Per-column category info (for categorical columns)
    private readonly List<CategoricalColumnInfo> _categoricalColumnInfos = new();

    // Mapping from original column index to transform info
    private readonly Dictionary<int, ColumnTransformInfo> _columnTransforms = new();

    // Total width of the transformed representation
    private int _transformedWidth;

    // Column metadata reference
    private IReadOnlyList<ColumnMetadata> _columns = Array.Empty<ColumnMetadata>();

    /// <summary>
    /// Gets the width of the transformed data representation.
    /// </summary>
    public int TransformedWidth => _transformedWidth;

    /// <summary>
    /// Gets the column metadata this transformer was fitted on.
    /// </summary>
    public IReadOnlyList<ColumnMetadata> Columns => _columns;

    /// <summary>
    /// Gets whether this transformer has been fitted.
    /// </summary>
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new <see cref="TabularDataTransformer{T}"/>.
    /// </summary>
    /// <param name="vgmModes">Number of Gaussian mixture components per continuous column.</param>
    /// <param name="random">Random number generator for initialization.</param>
    public TabularDataTransformer(int vgmModes = 10, Random? random = null)
    {
        _vgmModes = Math.Max(1, vgmModes);
        _random = random ?? RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the transformer to the data by learning VGM parameters for continuous columns
    /// and category mappings for categorical columns.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Column metadata describing each column.</param>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _columns = columns;
        _continuousColumnInfos.Clear();
        _categoricalColumnInfos.Clear();
        _columnTransforms.Clear();
        _transformedWidth = 0;

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col];

            if (meta.IsNumerical)
            {
                var info = FitContinuousColumn(data, col);
                int contIdx = _continuousColumnInfos.Count;
                _continuousColumnInfos.Add(info);

                // Transformed width: 1 (normalized value) + numActiveModes (mode one-hot)
                int width = 1 + info.NumActiveModes;
                _columnTransforms[col] = new ColumnTransformInfo(
                    isContinuous: true,
                    index: contIdx,
                    startOffset: _transformedWidth,
                    width: width);
                _transformedWidth += width;
            }
            else
            {
                var info = FitCategoricalColumn(meta);
                int catIdx = _categoricalColumnInfos.Count;
                _categoricalColumnInfos.Add(info);

                int width = info.NumCategories;
                _columnTransforms[col] = new ColumnTransformInfo(
                    isContinuous: false,
                    index: catIdx,
                    startOffset: _transformedWidth,
                    width: width);
                _transformedWidth += width;
            }
        }

        IsFitted = true;
    }

    /// <summary>
    /// Transforms the raw data matrix into the VGM-normalized + one-hot representation.
    /// </summary>
    /// <param name="data">The raw data matrix [numSamples, numColumns].</param>
    /// <returns>Transformed matrix [numSamples, transformedWidth].</returns>
    public Matrix<T> Transform(Matrix<T> data)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException("Transformer must be fitted before transforming data.");
        }

        var result = new Matrix<T>(data.Rows, _transformedWidth);

        for (int row = 0; row < data.Rows; row++)
        {
            for (int col = 0; col < _columns.Count; col++)
            {
                var transform = _columnTransforms[col];
                if (transform.IsContinuous)
                {
                    TransformContinuousValue(data, row, col, transform, result);
                }
                else
                {
                    TransformCategoricalValue(data, row, col, transform, result);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Inverse-transforms generated data back to the original column space.
    /// </summary>
    /// <param name="transformed">Transformed data matrix [numSamples, transformedWidth].</param>
    /// <returns>Reconstructed data in original column space [numSamples, numColumns].</returns>
    public Matrix<T> InverseTransform(Matrix<T> transformed)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException("Transformer must be fitted before inverse transforming.");
        }

        var result = new Matrix<T>(transformed.Rows, _columns.Count);

        for (int row = 0; row < transformed.Rows; row++)
        {
            for (int col = 0; col < _columns.Count; col++)
            {
                var transform = _columnTransforms[col];
                if (transform.IsContinuous)
                {
                    result[row, col] = InverseTransformContinuousValue(transformed, row, transform);
                }
                else
                {
                    result[row, col] = InverseTransformCategoricalValue(transformed, row, transform);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the transform info for a specific original column index.
    /// </summary>
    /// <param name="colIndex">The original column index.</param>
    /// <returns>The transform info for that column.</returns>
    public ColumnTransformInfo GetTransformInfo(int colIndex)
    {
        return _columnTransforms[colIndex];
    }

    #region VGM Fitting

    private VGMColumnInfo FitContinuousColumn(Matrix<T> data, int colIndex)
    {
        int n = data.Rows;
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(data[i, colIndex]);
        }

        // Fit Gaussian mixture using simplified EM algorithm
        int k = Math.Min(_vgmModes, Math.Max(1, n / 5));
        var means = new double[k];
        var stds = new double[k];
        var weights = new double[k];

        // Initialize: spread modes evenly across the data range
        Array.Sort(values);
        double valMin = values[0];
        double valMax = values[n - 1];
        double range = valMax - valMin;
        if (range < 1e-10) range = 1.0;

        for (int m = 0; m < k; m++)
        {
            means[m] = valMin + range * (m + 0.5) / k;
            stds[m] = range / k;
            weights[m] = 1.0 / k;
        }

        // EM iterations
        int maxIter = 25;
        var responsibilities = new double[n, k];

        for (int iter = 0; iter < maxIter; iter++)
        {
            // E-step: compute responsibilities
            for (int i = 0; i < n; i++)
            {
                double totalResp = 0;
                for (int m = 0; m < k; m++)
                {
                    double diff = values[i] - means[m];
                    double s = Math.Max(stds[m], 1e-10);
                    double logProb = -0.5 * (diff * diff) / (s * s) - Math.Log(s) + Math.Log(Math.Max(weights[m], 1e-10));
                    responsibilities[i, m] = Math.Exp(logProb);
                    totalResp += responsibilities[i, m];
                }

                if (totalResp > 1e-10)
                {
                    for (int m = 0; m < k; m++)
                    {
                        responsibilities[i, m] /= totalResp;
                    }
                }
                else
                {
                    // Assign to nearest mode
                    int nearest = 0;
                    double minDist = double.MaxValue;
                    for (int m = 0; m < k; m++)
                    {
                        double dist = Math.Abs(values[i] - means[m]);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            nearest = m;
                        }
                    }
                    responsibilities[i, nearest] = 1.0;
                }
            }

            // M-step: update parameters
            for (int m = 0; m < k; m++)
            {
                double sumResp = 0;
                double sumVal = 0;

                for (int i = 0; i < n; i++)
                {
                    sumResp += responsibilities[i, m];
                    sumVal += responsibilities[i, m] * values[i];
                }

                if (sumResp > 1e-10)
                {
                    means[m] = sumVal / sumResp;

                    double sumSqDiff = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double diff = values[i] - means[m];
                        sumSqDiff += responsibilities[i, m] * diff * diff;
                    }

                    stds[m] = Math.Sqrt(sumSqDiff / sumResp);
                    if (stds[m] < 1e-10) stds[m] = 1e-10;
                    weights[m] = sumResp / n;
                }
                else
                {
                    weights[m] = 0;
                }
            }

            // Normalize weights
            double totalWeight = 0;
            for (int m = 0; m < k; m++) totalWeight += weights[m];
            if (totalWeight > 1e-10)
            {
                for (int m = 0; m < k; m++) weights[m] /= totalWeight;
            }
        }

        // Keep only active modes (weight > threshold)
        double threshold = 0.01;
        var activeModes = new List<int>();
        for (int m = 0; m < k; m++)
        {
            if (weights[m] > threshold) activeModes.Add(m);
        }

        if (activeModes.Count == 0)
        {
            activeModes.Add(0); // Keep at least one mode
        }

        // Build compact arrays of active modes
        var activeMeans = new double[activeModes.Count];
        var activeStds = new double[activeModes.Count];
        var activeWeights = new double[activeModes.Count];

        for (int i = 0; i < activeModes.Count; i++)
        {
            int m = activeModes[i];
            activeMeans[i] = means[m];
            activeStds[i] = stds[m];
            activeWeights[i] = weights[m];
        }

        return new VGMColumnInfo(activeMeans, activeStds, activeWeights);
    }

    private static CategoricalColumnInfo FitCategoricalColumn(ColumnMetadata meta)
    {
        var categoryToIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < meta.Categories.Count; i++)
        {
            categoryToIndex[meta.Categories[i]] = i;
        }

        return new CategoricalColumnInfo(meta.Categories, categoryToIndex);
    }

    #endregion

    #region Forward Transform

    private void TransformContinuousValue(Matrix<T> data, int row, int col,
        ColumnTransformInfo transform, Matrix<T> result)
    {
        var info = _continuousColumnInfos[transform.Index];
        double value = NumOps.ToDouble(data[row, col]);

        // Find the most likely mode for this value
        int bestMode = 0;
        double bestProb = double.MinValue;

        for (int m = 0; m < info.NumActiveModes; m++)
        {
            double diff = value - info.Means[m];
            double s = info.Stds[m];
            double logProb = -0.5 * (diff * diff) / (s * s) - Math.Log(s) + Math.Log(info.Weights[m]);
            if (logProb > bestProb)
            {
                bestProb = logProb;
                bestMode = m;
            }
        }

        // Normalize value relative to selected mode: (value - mean) / (4 * std)
        // Clipped to [-1, 1] as in CTGAN paper
        double normalized = (value - info.Means[bestMode]) / (4.0 * info.Stds[bestMode]);
        normalized = Math.Max(-0.99, Math.Min(0.99, normalized));

        int offset = transform.StartOffset;
        result[row, offset] = NumOps.FromDouble(normalized);

        // One-hot mode indicator
        for (int m = 0; m < info.NumActiveModes; m++)
        {
            result[row, offset + 1 + m] = m == bestMode ? NumOps.One : NumOps.Zero;
        }
    }

    private void TransformCategoricalValue(Matrix<T> data, int row, int col,
        ColumnTransformInfo transform, Matrix<T> result)
    {
        var info = _categoricalColumnInfos[transform.Index];
        double value = NumOps.ToDouble(data[row, col]);
        string key = value.ToString(CultureInfo.InvariantCulture);

        int offset = transform.StartOffset;

        // One-hot encode
        if (info.CategoryToIndex.TryGetValue(key, out int catIdx))
        {
            for (int c = 0; c < info.NumCategories; c++)
            {
                result[row, offset + c] = c == catIdx ? NumOps.One : NumOps.Zero;
            }
        }
        else
        {
            // Unknown category: all zeros
            for (int c = 0; c < info.NumCategories; c++)
            {
                result[row, offset + c] = NumOps.Zero;
            }
        }
    }

    #endregion

    #region Inverse Transform

    private T InverseTransformContinuousValue(Matrix<T> transformed, int row, ColumnTransformInfo transform)
    {
        var info = _continuousColumnInfos[transform.Index];
        int offset = transform.StartOffset;

        double normalizedValue = NumOps.ToDouble(transformed[row, offset]);

        // Find the selected mode from the one-hot indicator (argmax)
        int selectedMode = 0;
        double maxModeVal = double.MinValue;
        for (int m = 0; m < info.NumActiveModes; m++)
        {
            double modeVal = NumOps.ToDouble(transformed[row, offset + 1 + m]);
            if (modeVal > maxModeVal)
            {
                maxModeVal = modeVal;
                selectedMode = m;
            }
        }

        // Inverse: value = normalized * 4 * std + mean
        double reconstructed = normalizedValue * 4.0 * info.Stds[selectedMode] + info.Means[selectedMode];

        return NumOps.FromDouble(reconstructed);
    }

    private T InverseTransformCategoricalValue(Matrix<T> transformed, int row, ColumnTransformInfo transform)
    {
        var info = _categoricalColumnInfos[transform.Index];
        int offset = transform.StartOffset;

        // Find argmax of the one-hot vector
        int selectedCat = 0;
        double maxVal = double.MinValue;
        for (int c = 0; c < info.NumCategories; c++)
        {
            double val = NumOps.ToDouble(transformed[row, offset + c]);
            if (val > maxVal)
            {
                maxVal = val;
                selectedCat = c;
            }
        }

        // Return the category index as the value
        return NumOps.FromDouble(selectedCat);
    }

    #endregion

    #region Internal Types

    /// <summary>
    /// Stores VGM parameters for a single continuous column.
    /// </summary>
    internal sealed class VGMColumnInfo
    {
        public double[] Means { get; }
        public double[] Stds { get; }
        public double[] Weights { get; }
        public int NumActiveModes => Means.Length;

        public VGMColumnInfo(double[] means, double[] stds, double[] weights)
        {
            Means = means;
            Stds = stds;
            Weights = weights;
        }
    }

    /// <summary>
    /// Stores category mapping for a single categorical column.
    /// </summary>
    internal sealed class CategoricalColumnInfo
    {
        public IReadOnlyList<string> Categories { get; }
        public Dictionary<string, int> CategoryToIndex { get; }
        public int NumCategories => Categories.Count;

        public CategoricalColumnInfo(IReadOnlyList<string> categories, Dictionary<string, int> categoryToIndex)
        {
            Categories = categories;
            CategoryToIndex = categoryToIndex;
        }
    }

    #endregion
}

/// <summary>
/// Describes how a single original column maps into the transformed representation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When the transformer processes your data, each column gets
/// expanded into a different number of features:
/// - A continuous column becomes 1 (normalized value) + K (mode indicators) features
/// - A categorical column becomes N (one-hot categories) features
///
/// This info tracks where each original column's features start and how wide they are.
/// </para>
/// </remarks>
public sealed class ColumnTransformInfo
{
    /// <summary>
    /// Whether this column is continuous (true) or categorical (false).
    /// </summary>
    public bool IsContinuous { get; }

    /// <summary>
    /// Index into the continuous or categorical info arrays.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Starting offset in the transformed data vector.
    /// </summary>
    public int StartOffset { get; }

    /// <summary>
    /// Number of features this column occupies in the transformed representation.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Initializes a new <see cref="ColumnTransformInfo"/>.
    /// </summary>
    public ColumnTransformInfo(bool isContinuous, int index, int startOffset, int width)
    {
        IsContinuous = isContinuous;
        Index = index;
        StartOffset = startOffset;
        Width = width;
    }
}
