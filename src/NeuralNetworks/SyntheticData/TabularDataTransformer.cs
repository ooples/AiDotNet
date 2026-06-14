using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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

    /// <summary>
    /// Serializes the fitted transformer state (column metadata, per-column transform layout, and
    /// the fitted VGM / categorical parameters) so a saved or cloned generator can inverse-transform
    /// generated samples back to the original column space without re-fitting.
    /// </summary>
    /// <remarks>
    /// The VGM mode count and RNG are intentionally not persisted: they only influence
    /// <see cref="Fit"/>, never <see cref="Transform"/> / <see cref="InverseTransform"/>, and the
    /// owning generator reconstructs the transformer with the configured mode count before loading.
    /// </remarks>
    public void Serialize(System.IO.BinaryWriter writer)
    {
        writer.Write(IsFitted);
        if (!IsFitted)
        {
            return;
        }

        writer.Write(_transformedWidth);

        writer.Write(_columns.Count);
        foreach (var column in _columns)
        {
            column.Serialize(writer);
        }

        // Column-transform layout. Keys are the contiguous original-column indices 0..count-1.
        writer.Write(_columnTransforms.Count);
        for (int col = 0; col < _columns.Count; col++)
        {
            var transform = _columnTransforms[col];
            writer.Write(col);
            writer.Write(transform.IsContinuous);
            writer.Write(transform.Index);
            writer.Write(transform.StartOffset);
            writer.Write(transform.Width);
        }

        writer.Write(_continuousColumnInfos.Count);
        foreach (var info in _continuousColumnInfos)
        {
            WriteDoubleArray(writer, info.Means);
            WriteDoubleArray(writer, info.Stds);
            WriteDoubleArray(writer, info.Weights);
        }

        writer.Write(_categoricalColumnInfos.Count);
        foreach (var info in _categoricalColumnInfos)
        {
            writer.Write(info.Categories.Count);
            foreach (var category in info.Categories)
            {
                writer.Write(category);
            }
        }
    }

    /// <summary>
    /// Restores transformer state previously written by <see cref="Serialize"/>.
    /// </summary>
    public void Deserialize(System.IO.BinaryReader reader)
    {
        _continuousColumnInfos.Clear();
        _categoricalColumnInfos.Clear();
        _columnTransforms.Clear();

        IsFitted = reader.ReadBoolean();
        if (!IsFitted)
        {
            _transformedWidth = 0;
            _columns = Array.Empty<ColumnMetadata>();
            return;
        }

        _transformedWidth = reader.ReadInt32();

        int columnCount = reader.ReadInt32();
        var columns = new List<ColumnMetadata>(columnCount);
        for (int i = 0; i < columnCount; i++)
        {
            columns.Add(ColumnMetadata.Deserialize(reader));
        }
        _columns = columns;

        int transformCount = reader.ReadInt32();
        for (int i = 0; i < transformCount; i++)
        {
            int key = reader.ReadInt32();
            bool isContinuous = reader.ReadBoolean();
            int index = reader.ReadInt32();
            int startOffset = reader.ReadInt32();
            int width = reader.ReadInt32();
            _columnTransforms[key] = new ColumnTransformInfo(isContinuous, index, startOffset, width);
        }

        int continuousCount = reader.ReadInt32();
        for (int i = 0; i < continuousCount; i++)
        {
            double[] means = ReadDoubleArray(reader);
            double[] stds = ReadDoubleArray(reader);
            double[] weights = ReadDoubleArray(reader);
            _continuousColumnInfos.Add(new VGMColumnInfo(means, stds, weights));
        }

        int categoricalCount = reader.ReadInt32();
        for (int i = 0; i < categoricalCount; i++)
        {
            int numCategories = reader.ReadInt32();
            var categories = new string[numCategories];
            var categoryToIndex = new Dictionary<string, int>();
            for (int c = 0; c < numCategories; c++)
            {
                categories[c] = reader.ReadString();
                categoryToIndex[categories[c]] = c;
            }
            _categoricalColumnInfos.Add(new CategoricalColumnInfo(categories, categoryToIndex));
        }
    }

    private static void WriteDoubleArray(System.IO.BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);
        foreach (double value in values)
        {
            writer.Write(value);
        }
    }

    private static double[] ReadDoubleArray(System.IO.BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var values = new double[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadDouble();
        }

        return values;
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

        // Variational Bayesian Gaussian Mixture (Bishop PRML §10.2) — this is the
        // "V" in VGM. The CTGAN paper (Xu et al. 2019 §4.2) and the official
        // implementation fit a *Bayesian* GMM whose Dirichlet weight-concentration
        // prior drives unused components' weights toward zero, so the number of
        // active modes is discovered from the data rather than fixed. Plain EM
        // (the previous code) keeps all K components alive and relies on a post-hoc
        // 0.01 threshold, which mis-estimates mode structure on real multimodal
        // columns. Here we run mean-field VB with a symmetric Dirichlet prior on the
        // weights and a Normal-Gamma conjugate prior on each component's
        // (mean, precision), then keep the components whose posterior weight is
        // non-negligible.
        int k = Math.Min(_vgmModes, Math.Max(1, n / 5));

        Array.Sort(values);
        double valMin = values[0];
        double valMax = values[n - 1];
        double range = valMax - valMin;
        if (range < 1e-10) range = 1.0;

        double dataMean = 0;
        for (int i = 0; i < n; i++) dataMean += values[i];
        dataMean /= n;
        double dataVar = 0;
        for (int i = 0; i < n; i++) { double d = values[i] - dataMean; dataVar += d * d; }
        dataVar = Math.Max(dataVar / Math.Max(1, n), 1e-6);

        // Prior hyperparameters. A small Dirichlet concentration (alpha0 << 1) gives
        // the sparse, automatic-pruning behaviour of a Dirichlet-process mixture —
        // the regime the official CTGAN BayesianGaussianMixture runs in.
        double alpha0 = 1e-3;                 // symmetric Dirichlet weight prior
        double beta0 = 1.0;                   // mean prior strength
        double m0 = dataMean;                 // mean prior location
        double a0 = 1.0;                      // Gamma (precision) prior shape
        double b0 = dataVar;                  // Gamma (precision) prior rate

        // Posterior parameters, initialized by spreading component means across the
        // data range (k-means-style seeding) with broad precision.
        var alpha = new double[k];
        var betap = new double[k];
        var mp = new double[k];
        var ap = new double[k];
        var bp = new double[k];
        for (int m = 0; m < k; m++)
        {
            alpha[m] = alpha0 + (double)n / k;
            betap[m] = beta0 + (double)n / k;
            mp[m] = valMin + range * (m + 0.5) / k;
            ap[m] = a0 + 0.5 * n / k;
            bp[m] = b0 + 0.5 * dataVar * n / k;
        }

        var r = new double[n, k];
        const int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++)
        {
            // --- Variational E-step: responsibilities from expected log-likelihoods.
            double alphaSum = 0;
            for (int m = 0; m < k; m++) alphaSum += alpha[m];
            double psiAlphaSum = Digamma(alphaSum);

            for (int i = 0; i < n; i++)
            {
                double maxLog = double.NegativeInfinity;
                for (int m = 0; m < k; m++)
                {
                    double eLnPi = Digamma(alpha[m]) - psiAlphaSum;           // E[ln π_k]
                    double eLnTau = Digamma(ap[m]) - Math.Log(bp[m]);          // E[ln τ_k]
                    double eTau = ap[m] / bp[m];                              // E[τ_k]
                    double diff = values[i] - mp[m];
                    // E[τ_k (x-μ_k)^2] = 1/β_k + E[τ_k](x-m_k)^2
                    double eTauSq = 1.0 / betap[m] + eTau * diff * diff;
                    double logRho = eLnPi + 0.5 * eLnTau - 0.5 * eTauSq - 0.5 * Math.Log(2 * Math.PI);
                    r[i, m] = logRho;
                    if (logRho > maxLog) maxLog = logRho;
                }
                // log-sum-exp normalize
                double sum = 0;
                for (int m = 0; m < k; m++) { r[i, m] = Math.Exp(r[i, m] - maxLog); sum += r[i, m]; }
                if (sum <= 1e-300) { for (int m = 0; m < k; m++) r[i, m] = 1.0 / k; }
                else { for (int m = 0; m < k; m++) r[i, m] /= sum; }
            }

            // --- Variational M-step: update posteriors from sufficient statistics.
            for (int m = 0; m < k; m++)
            {
                double Nk = 0, xbar = 0;
                for (int i = 0; i < n; i++) { Nk += r[i, m]; xbar += r[i, m] * values[i]; }
                if (Nk > 1e-12) xbar /= Nk;
                double sk = 0;
                for (int i = 0; i < n; i++) { double d = values[i] - xbar; sk += r[i, m] * d * d; }
                if (Nk > 1e-12) sk /= Nk;

                alpha[m] = alpha0 + Nk;
                betap[m] = beta0 + Nk;
                mp[m] = (beta0 * m0 + Nk * xbar) / betap[m];
                ap[m] = a0 + 0.5 * Nk;
                bp[m] = b0 + 0.5 * (Nk * sk + beta0 * Nk * (xbar - m0) * (xbar - m0) / betap[m]);
                if (bp[m] < 1e-10) bp[m] = 1e-10;
            }
        }

        // Posterior point estimates. Component weight = E[π_k] = α_k / Σα; mode is
        // kept when its effective count clears a small floor (the Dirichlet prior has
        // already shrunk genuinely-empty components to ≈ alpha0/Σα).
        double totalAlpha = 0;
        for (int m = 0; m < k; m++) totalAlpha += alpha[m];

        var activeMeans = new List<double>();
        var activeStds = new List<double>();
        var activeWeights = new List<double>();
        for (int m = 0; m < k; m++)
        {
            double w = alpha[m] / totalAlpha;
            double effectiveCount = alpha[m] - alpha0;     // ≈ N_k assigned to this mode
            if (effectiveCount < 1.0 || w < 1e-3) continue; // pruned by the variational prior
            activeMeans.Add(mp[m]);
            activeStds.Add(Math.Sqrt(Math.Max(bp[m] / ap[m], 1e-10))); // E[1/τ] ≈ b_k/a_k
            activeWeights.Add(w);
        }

        if (activeWeights.Count == 0)
        {
            // Degenerate column (e.g. constant): one mode at the data mean.
            activeMeans.Add(dataMean);
            activeStds.Add(Math.Sqrt(dataVar));
            activeWeights.Add(1.0);
        }

        // Renormalize the kept weights so they sum to 1.
        double wsum = 0;
        for (int i = 0; i < activeWeights.Count; i++) wsum += activeWeights[i];
        for (int i = 0; i < activeWeights.Count; i++) activeWeights[i] /= wsum;

        return new VGMColumnInfo(activeMeans.ToArray(), activeStds.ToArray(), activeWeights.ToArray());
    }

    /// <summary>
    /// Digamma function ψ(x) = d/dx ln Γ(x), used by the variational GMM E-step.
    /// Uses the asymptotic series with a recurrence to push the argument above 6.
    /// </summary>
    private static double Digamma(double x)
    {
        double result = 0;
        while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
        double inv = 1.0 / x;
        double inv2 = inv * inv;
        // ln(x) - 1/(2x) - series in 1/x^2 (Bernoulli)
        result += Math.Log(x) - 0.5 * inv
                  - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
        return result;
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

        // Mode assignment by SAMPLING from the posterior responsibilities, per
        // Xu et al. 2019 §4.2 and the official CTGAN DataTransformer
        // (np.random.choice with p = predict_proba). Argmax (the previous code)
        // collapses every value onto its single most-likely mode, erasing the
        // multimodal structure the one-hot β is meant to expose to the generator.
        int selMode = SampleMode(info, value);

        // Normalize value relative to selected mode: (value - mean) / (4 * std),
        // clipped to (-1, 1) as in the paper (4σ covers ~99.99% of a Gaussian).
        double normalized = (value - info.Means[selMode]) / (4.0 * info.Stds[selMode]);
        normalized = Math.Max(-0.99, Math.Min(0.99, normalized));

        int offset = transform.StartOffset;
        result[row, offset] = NumOps.FromDouble(normalized);

        // One-hot mode indicator (β).
        for (int m = 0; m < info.NumActiveModes; m++)
        {
            result[row, offset + 1 + m] = m == selMode ? NumOps.One : NumOps.Zero;
        }
    }

    /// <summary>
    /// Samples a mode index for <paramref name="value"/> proportional to the
    /// component responsibilities ρ_m ∝ w_m · N(value; μ_m, σ_m), matching the
    /// paper's probabilistic mode assignment. Uses the transformer's seeded RNG.
    /// </summary>
    private int SampleMode(VGMColumnInfo info, double value)
    {
        int k = info.NumActiveModes;
        if (k == 1) return 0;

        Span<double> probs = k <= 64 ? stackalloc double[k] : new double[k];
        double maxLog = double.NegativeInfinity;
        for (int m = 0; m < k; m++)
        {
            double diff = value - info.Means[m];
            double s = Math.Max(info.Stds[m], 1e-10);
            double logp = -0.5 * (diff * diff) / (s * s) - Math.Log(s) + Math.Log(Math.Max(info.Weights[m], 1e-12));
            probs[m] = logp;
            if (logp > maxLog) maxLog = logp;
        }
        double sum = 0;
        for (int m = 0; m < k; m++) { probs[m] = Math.Exp(probs[m] - maxLog); sum += probs[m]; }

        double u = _random.NextDouble() * sum;
        double acc = 0;
        for (int m = 0; m < k; m++)
        {
            acc += probs[m];
            if (u <= acc) return m;
        }
        return k - 1;
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
