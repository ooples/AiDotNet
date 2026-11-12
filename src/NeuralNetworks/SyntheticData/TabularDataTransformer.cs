using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
public class TabularDataTransformer<T> : IModelSerializer
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

        int rows = data.Rows;
        if (rows == 0)
        {
            return new Matrix<T>(0, _transformedWidth);
        }

        // Each column becomes one [rows, width] block, laid side by side in column order. Continuous
        // columns are encoded whole-column by the engine; categorical cells keep their per-cell
        // index-or-name lookup (a name match needs the cell's text form).
        var blocks = new List<Tensor<T>>(_columns.Count);
        for (int col = 0; col < _columns.Count; col++)
        {
            var transform = _columnTransforms[col];
            if (transform.Width == 0)
            {
                continue;
            }

            blocks.Add(transform.IsContinuous
                ? TransformContinuousColumn(data, col, transform)
                : TransformCategoricalColumn(data, col, transform));
        }

        if (blocks.Count == 0)
        {
            return new Matrix<T>(rows, _transformedWidth);
        }

        var engine = AiDotNetEngine.Current;
        return (blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 1)).ToMatrix();
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

        int rows = transformed.Rows;
        if (rows == 0 || _columns.Count == 0)
        {
            return new Matrix<T>(rows, _columns.Count);
        }

        var source = Tensor<T>.FromMatrix(transformed);
        var columns = new Tensor<T>[_columns.Count];
        for (int col = 0; col < _columns.Count; col++)
        {
            var transform = _columnTransforms[col];
            columns[col] = transform.IsContinuous
                ? InverseTransformContinuousColumn(source, transform)
                : InverseTransformCategoricalColumn(source, transform);
        }

        var engine = AiDotNetEngine.Current;
        return (columns.Length == 1 ? columns[0] : engine.TensorConcatenate(columns, 1)).ToMatrix();
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
    /// Gets the fitted VGM parameters of the continuous column at <paramref name="continuousIndex"/>
    /// (its <see cref="ColumnTransformInfo.Index"/>).
    /// </summary>
    internal VGMColumnInfo GetContinuousColumnInfo(int continuousIndex)
    {
        return _continuousColumnInfos[continuousIndex];
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
            // Raw T elements (count, element size, bytes): the fitted parameters round-trip exactly.
            AuxLayerSerialization.WriteElements(writer, info.Means.ToVector());
            AuxLayerSerialization.WriteElements(writer, info.Stds.ToVector());
            AuxLayerSerialization.WriteElements(writer, info.Weights.ToVector());
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
            var means = Tensor<T>.FromVector(AuxLayerSerialization.ReadElements<T>(reader));
            var stds = Tensor<T>.FromVector(AuxLayerSerialization.ReadElements<T>(reader));
            var weights = Tensor<T>.FromVector(AuxLayerSerialization.ReadElements<T>(reader));
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

    /// <inheritdoc />
    public byte[] Serialize()
    {
        using var stream = new System.IO.MemoryStream();
        using (var writer = new System.IO.BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            Serialize(writer);
        }

        return stream.ToArray();
    }

    /// <inheritdoc />
    public void Deserialize(byte[] data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));

        using var stream = new System.IO.MemoryStream(data, writable: false);
        using var reader = new System.IO.BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: false);
        Deserialize(reader);
    }

    /// <inheritdoc />
    public void SaveModel(string filePath)
    {
        if (filePath is null) throw new ArgumentNullException(nameof(filePath));
        System.IO.File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc />
    public void LoadModel(string filePath)
    {
        if (filePath is null) throw new ArgumentNullException(nameof(filePath));
        Deserialize(System.IO.File.ReadAllBytes(filePath));
    }

    #region VGM Fitting

    private VGMColumnInfo FitContinuousColumn(Matrix<T> data, int colIndex)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Rows;
        var x = Tensor<T>.FromVector(data.GetColumn(colIndex));

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
        // non-negligible. Every step runs on whole [n, k] tensors in T.
        int k = Math.Min(_vgmModes, Math.Max(1, n / 5));
        int[] grid = { n, k };

        // The fit runs on the column divided by its largest magnitude and the fitted means and
        // stds are scaled back at the end. Squaring raw values overflows narrow types (float data
        // at ±1e20 squares to +inf, which turned the variance into inf and every responsibility
        // into NaN); in scaled units every value is within [-1, 1], so its square cannot overflow.
        // The Normal-Gamma posterior is scale-equivariant (b0 is the data variance), and every floor
        // below stays in ORIGINAL data units (a length floor is divided by scale, a variance floor by
        // scale twice; the dimensionless mass and weight floors are unchanged), so the fitted modes
        // match an unscaled fit exactly up to rounding.
        T scale = MathHelper.Max(
            NumOps.Abs(engine.TensorMinValue(x)), NumOps.Abs(engine.TensorMaxValue(x)));
        if (!NumOps.GreaterThan(scale, NumOps.Zero)) scale = NumOps.One;
        x = engine.TensorDivideScalar(x, scale);

        // Divided by scale twice rather than by scale squared, which itself overflows float at 1e20.
        T VarianceInScaledUnits(double floor) => NumOps.Divide(NumOps.Divide(NumOps.FromDouble(floor), scale), scale);

        T valMin = engine.TensorMinValue(x);
        T valMax = engine.TensorMaxValue(x);
        T range = NumOps.Subtract(valMax, valMin);
        if (NumOps.LessThan(range, NumOps.Divide(NumOps.FromDouble(1e-10), scale)))
            range = NumOps.Divide(NumOps.One, scale);

        T dataMean = engine.TensorMean(x);
        var centered = engine.TensorSubtractScalar(x, dataMean);
        T dataVar = MathHelper.Max(
            engine.TensorMean(engine.TensorMultiply(centered, centered)), VarianceInScaledUnits(1e-6));

        // Prior hyperparameters. A small Dirichlet concentration (alpha0 << 1) gives
        // the sparse, automatic-pruning behaviour of a Dirichlet-process mixture —
        // the regime the official CTGAN BayesianGaussianMixture runs in.
        T alpha0 = NumOps.FromDouble(1e-3);   // symmetric Dirichlet weight prior
        T beta0 = NumOps.One;                 // mean prior strength
        T m0 = dataMean;                      // mean prior location
        T a0 = NumOps.One;                    // Gamma (precision) prior shape
        T b0 = dataVar;                       // Gamma (precision) prior rate
        T beta0M0 = NumOps.Multiply(beta0, m0);
        T half = NumOps.FromDouble(0.5);
        T negHalf = NumOps.FromDouble(-0.5);
        T halfLogTwoPi = NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI));
        T massFloor = NumOps.FromDouble(1e-12);
        T rateFloor = VarianceInScaledUnits(1e-10);   // floors b_k and b_k/a_k, both variances
        T perMode = NumOps.FromDouble((double)n / k);

        // Posterior parameters, initialized by spreading component means across the
        // data range (k-means-style seeding) with broad precision.
        var alpha = Tensor<T>.CreateDefault(new[] { k }, NumOps.Add(alpha0, perMode));
        var betap = Tensor<T>.CreateDefault(new[] { k }, NumOps.Add(beta0, perMode));
        var mp = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0.5, k, 1.0), NumOps.Divide(range, NumOps.FromDouble(k))),
            valMin);
        var ap = Tensor<T>.CreateDefault(new[] { k }, NumOps.Add(a0, NumOps.Multiply(half, perMode)));
        var bp = Tensor<T>.CreateDefault(new[] { k }, NumOps.Add(b0, NumOps.Multiply(NumOps.Multiply(half, dataVar), perMode)));

        var xGrid = Expand(x.Reshape(new[] { n, 1 }), grid);
        var ones = Tensor<T>.CreateDefault(new[] { k }, NumOps.One);
        const int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++)
        {
            // --- Variational E-step: responsibilities from expected log-likelihoods.
            var psiAlphaSum = engine.TensorDigamma(engine.ReduceSum(alpha, new[] { 0 }, true));
            var eLnPi = engine.TensorSubtract(engine.TensorDigamma(alpha), Expand(psiAlphaSum, new[] { k }));   // E[ln π_k]
            var eLnTau = engine.TensorSubtract(engine.TensorDigamma(ap), engine.TensorLog(bp));                // E[ln τ_k]
            var eTau = engine.TensorDivide(ap, bp);                                                              // E[τ_k]
            var diff = engine.TensorSubtract(xGrid, Row(mp, grid));
            // E[τ_k (x-μ_k)^2] = 1/β_k + E[τ_k](x-m_k)^2
            var eTauSq = engine.TensorAdd(
                Row(engine.TensorReciprocal(betap), grid),
                engine.TensorMultiply(Row(eTau, grid), engine.TensorMultiply(diff, diff)));
            var componentTerm = engine.TensorSubtractScalar(
                engine.TensorAdd(eLnPi, engine.TensorMultiplyScalar(eLnTau, half)), halfLogTwoPi);
            var logRho = engine.TensorAdd(Row(componentTerm, grid), engine.TensorMultiplyScalar(eTauSq, negHalf));
            // log-sum-exp normalize
            var r = engine.Softmax(logRho, 1);

            // --- Variational M-step: update posteriors from sufficient statistics.
            var nk = engine.ReduceSum(r, new[] { 0 }, false);
            var hasMass = engine.TensorGreaterThan(nk, massFloor);
            // Massless components divide by 1 instead of their (possibly exactly zero) count and the
            // quotient is then masked out: the division runs before the mask, and a zero divisor
            // throws for decimal.
            var safeNk = engine.TensorWhere(hasMass, nk, ones);
            var weightedSum = engine.ReduceSum(engine.TensorMultiply(r, xGrid), new[] { 0 }, false);
            var xbar = engine.TensorWhere(hasMass, engine.TensorDivide(weightedSum, safeNk), weightedSum);
            var dev = engine.TensorSubtract(xGrid, Row(xbar, grid));
            var weightedSq = engine.ReduceSum(engine.TensorMultiply(r, engine.TensorMultiply(dev, dev)), new[] { 0 }, false);
            var sk = engine.TensorWhere(hasMass, engine.TensorDivide(weightedSq, safeNk), weightedSq);

            alpha = engine.TensorAddScalar(nk, alpha0);
            betap = engine.TensorAddScalar(nk, beta0);
            mp = engine.TensorDivide(engine.TensorAddScalar(engine.TensorMultiply(nk, xbar), beta0M0), betap);
            ap = engine.TensorAddScalar(engine.TensorMultiplyScalar(nk, half), a0);
            var shift = engine.TensorSubtractScalar(xbar, m0);
            var spread = engine.TensorAdd(
                engine.TensorMultiply(nk, sk),
                engine.TensorDivide(
                    engine.TensorMultiplyScalar(engine.TensorMultiply(nk, engine.TensorMultiply(shift, shift)), beta0),
                    betap));
            bp = engine.TensorMax(engine.TensorAddScalar(engine.TensorMultiplyScalar(spread, half), b0), rateFloor);
        }

        // Posterior point estimates. Component weight = E[π_k] = α_k / Σα; mode is
        // kept when its effective count clears a small floor (the Dirichlet prior has
        // already shrunk genuinely-empty components to ≈ alpha0/Σα). A mode is pruned when
        // its effective count (≈ N_k) is below 1 or its weight below 1e-3.
        var weight = engine.TensorDivideScalar(alpha, engine.TensorSum(alpha));
        var effectiveCount = engine.TensorSubtractScalar(alpha, alpha0);
        var pruned = engine.TensorAdd(
            engine.TensorLessThan(effectiveCount, NumOps.One).Cast<int>(),
            engine.TensorLessThan(weight, NumOps.FromDouble(1e-3)).Cast<int>());
        var keptCoordinates = engine.TensorNonzero(engine.TensorEquals(pruned, 0).Cast<int>());
        var kept = keptCoordinates.Reshape(new[] { keptCoordinates.Length });

        Tensor<T> means;
        Tensor<T> stds;
        Tensor<T> weights;
        if (kept.Length == 0)
        {
            // Degenerate column (e.g. constant): one mode at the data mean.
            means = Tensor<T>.CreateDefault(new[] { 1 }, dataMean);
            stds = Tensor<T>.CreateDefault(new[] { 1 }, NumOps.Sqrt(dataVar));
            weights = Tensor<T>.CreateDefault(new[] { 1 }, NumOps.One);
        }
        else
        {
            means = engine.TensorIndexSelect(mp, kept, 0);
            // E[1/τ] ≈ b_k/a_k
            stds = engine.TensorSqrt(engine.TensorMax(
                engine.TensorDivide(engine.TensorIndexSelect(bp, kept, 0), engine.TensorIndexSelect(ap, kept, 0)),
                rateFloor));
            weights = engine.TensorIndexSelect(weight, kept, 0);
        }

        // Renormalize the kept weights so they sum to 1, and return means and stds to data units.
        weights = engine.TensorDivideScalar(weights, engine.TensorSum(weights));
        means = engine.TensorMultiplyScalar(means, scale);
        stds = engine.TensorMultiplyScalar(stds, scale);

        // Engine results may live in the caller's TensorArena and be recycled when its scope ends;
        // the fitted parameters are persistent state, so store detached heap copies.
        return new VGMColumnInfo(Detach(means), Detach(stds), Detach(weights));
    }

    /// <summary>
    /// Copies a 1-D tensor into a freshly allocated heap array, independent of any arena buffer.
    /// </summary>
    private static Tensor<T> Detach(Tensor<T> tensor)
    {
        var copy = new T[tensor.Length];
        tensor.AsSpan().CopyTo(copy);
        return new Tensor<T>(copy, new[] { copy.Length });
    }

    /// <summary>Broadcasts <paramref name="tensor"/> to <paramref name="shape"/> as a dense tensor.</summary>
    private static Tensor<TElement> Expand<TElement>(Tensor<TElement> tensor, int[] shape)
        => AiDotNetEngine.Current.TensorBroadcastTo(tensor, shape).Contiguous();

    /// <summary>Repeats the length-k vector <paramref name="values"/> down every row of an [n, k] grid.</summary>
    private static Tensor<T> Row(Tensor<T> values, int[] grid)
        => Expand(values.Reshape(new[] { 1, grid[1] }), grid);

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

    /// <summary>
    /// Encodes one continuous column as [normalized value, one-hot mode] rows, [rows, 1 + modes].
    /// </summary>
    /// <remarks>
    /// Mode assignment is by SAMPLING from the posterior responsibilities, per
    /// Xu et al. 2019 §4.2 and the official CTGAN DataTransformer
    /// (np.random.choice with p = predict_proba). Argmax collapses every value onto its
    /// single most-likely mode, erasing the multimodal structure the one-hot β is meant to
    /// expose to the generator. The responsibilities ρ_m ∝ w_m · N(value; μ_m, σ_m) are formed
    /// in log space for the whole column and one categorical draw per row is taken by the
    /// engine, seeded from the transformer's RNG (one seed per column, so a seeded transformer
    /// stays reproducible). A single-mode column draws nothing.
    /// </remarks>
    private Tensor<T> TransformContinuousColumn(Matrix<T> data, int col, ColumnTransformInfo transform)
    {
        var engine = AiDotNetEngine.Current;
        var info = _continuousColumnInfos[transform.Index];
        int rows = data.Rows;
        int k = info.NumActiveModes;
        int[] grid = { rows, k };
        var x = Tensor<T>.FromVector(data.GetColumn(col)).Reshape(new[] { rows, 1 });

        Tensor<T> modeOneHot;
        if (k == 1)
        {
            modeOneHot = Tensor<T>.CreateDefault(grid, NumOps.One);
        }
        else
        {
            var s = engine.TensorMax(info.Stds, NumOps.FromDouble(1e-10));
            var z = engine.TensorDivide(engine.TensorSubtract(Expand(x, grid), Row(info.Means, grid)), Row(s, grid));
            var logPrior = engine.TensorSubtract(
                engine.TensorLog(engine.TensorMax(info.Weights, NumOps.FromDouble(1e-12))), engine.TensorLog(s));
            var logp = engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorMultiply(z, z), NumOps.FromDouble(-0.5)), Row(logPrior, grid));
            modeOneHot = engine.TensorCategoricalSample(engine.Softmax(logp, 1), 1, _random.Next());
        }

        // Normalize value relative to selected mode: (value - mean) / (4 * std),
        // clipped to (-1, 1) as in the paper (4σ covers ~99.99% of a Gaussian).
        var selectedMean = engine.ReduceSum(engine.TensorMultiply(modeOneHot, Row(info.Means, grid)), new[] { 1 }, true);
        var selectedStd = engine.ReduceSum(engine.TensorMultiply(modeOneHot, Row(info.Stds, grid)), new[] { 1 }, true);
        var normalized = engine.TensorClamp(
            engine.TensorDivide(engine.TensorSubtract(x, selectedMean), engine.TensorMultiplyScalar(selectedStd, NumOps.FromDouble(4.0))),
            NumOps.FromDouble(-0.99),
            NumOps.FromDouble(0.99));

        // [normalized, one-hot mode indicator (β)]
        return engine.TensorConcatenate(new[] { normalized, modeOneHot }, 1);
    }

    /// <summary>
    /// Encodes one categorical column as one-hot rows [rows, categories], one cell at a time
    /// through <see cref="TransformCategoricalValue"/>.
    /// </summary>
    private Tensor<T> TransformCategoricalColumn(Matrix<T> data, int col, ColumnTransformInfo transform)
    {
        var columnLayout = new ColumnTransformInfo(isContinuous: false, index: transform.Index, startOffset: 0, width: transform.Width);
        var block = new Matrix<T>(data.Rows, transform.Width);
        for (int row = 0; row < data.Rows; row++)
        {
            TransformCategoricalValue(data, row, col, columnLayout, block);
        }

        return Tensor<T>.FromMatrix(block);
    }

    private void TransformCategoricalValue(Matrix<T> data, int row, int col,
        ColumnTransformInfo transform, Matrix<T> result)
    {
        var info = _categoricalColumnInfos[transform.Index];
        double value = NumOps.ToDouble(data[row, col]);
        int offset = transform.StartOffset;

        // A categorical cell holds its category's index, as CTGANDataSampler groups the rows and the inverse transform
        // writes them. Looking the value up by name alone encoded every row of a column declared with label categories
        // ("A", "B", "C") over index data as all zeros, so the critic never saw a real row's category. A value that is
        // not an index of the declared categories is still matched by name.
        double rounded = Math.Round(value);
        int catIdx;
        bool known;
        if (Math.Abs(rounded - value) < 1e-9 && rounded >= 0 && rounded < info.NumCategories)
        {
            catIdx = (int)rounded;
            known = true;
        }
        else
        {
            known = info.CategoryToIndex.TryGetValue(value.ToString(CultureInfo.InvariantCulture), out catIdx);
        }

        // One-hot encode
        if (known)
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

    /// <summary>
    /// Reconstructs one continuous column, [rows, 1]: value = normalized * 4 * std + mean of the
    /// mode whose one-hot indicator is largest (first on ties).
    /// </summary>
    private Tensor<T> InverseTransformContinuousColumn(Tensor<T> transformed, ColumnTransformInfo transform)
    {
        var engine = AiDotNetEngine.Current;
        var info = _continuousColumnInfos[transform.Index];
        int rows = transformed.Shape[0];
        int offset = transform.StartOffset;
        int k = info.NumActiveModes;

        var normalizedValue = engine.TensorSlice(transformed, new[] { 0, offset }, new[] { rows, 1 });
        var selectedMode = FirstArgMaxPerRow(engine.TensorSlice(transformed, new[] { 0, offset + 1 }, new[] { rows, k }));
        var mean = engine.TensorIndexSelect(info.Means, selectedMode, 0).Reshape(new[] { rows, 1 });
        var std = engine.TensorIndexSelect(info.Stds, selectedMode, 0).Reshape(new[] { rows, 1 });

        // Inverse: value = normalized * 4 * std + mean
        return engine.TensorAdd(
            engine.TensorMultiply(engine.TensorMultiplyScalar(normalizedValue, NumOps.FromDouble(4.0)), std), mean);
    }

    /// <summary>
    /// Reconstructs one categorical column, [rows, 1]: the index of the largest one-hot entry
    /// (first on ties), as the category value.
    /// </summary>
    private Tensor<T> InverseTransformCategoricalColumn(Tensor<T> transformed, ColumnTransformInfo transform)
    {
        var engine = AiDotNetEngine.Current;
        var info = _categoricalColumnInfos[transform.Index];
        int rows = transformed.Shape[0];
        if (info.NumCategories == 0)
        {
            return Tensor<T>.CreateDefault(new[] { rows, 1 }, NumOps.Zero);
        }

        var block = engine.TensorSlice(transformed, new[] { 0, transform.StartOffset }, new[] { rows, info.NumCategories });

        // Return the category index as the value
        return FirstArgMaxPerRow(block).Cast<T>().Reshape(new[] { rows, 1 });
    }

    /// <summary>
    /// Per-row index of the largest entry of a [rows, width] block, choosing the first on ties.
    /// </summary>
    /// <remarks>
    /// Matches a left-to-right scan that keeps an entry only when it is strictly greater than the
    /// best so far, starting from the lowest double: NaN never wins (it is ranked as −∞) and a row
    /// with no winner selects index 0.
    /// </remarks>
    private static Tensor<int> FirstArgMaxPerRow(Tensor<T> block)
    {
        var engine = AiDotNetEngine.Current;
        int rows = block.Shape[0];
        int width = block.Shape[1];
        int[] grid = { rows, width };

        var cleaned = engine.TensorNanToNum(block, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        var rowMax = engine.ReduceMax(cleaned, new[] { 1 }, true);
        var isMax = engine.TensorEquals(cleaned, Expand(rowMax, grid)).Cast<int>();

        // candidate = position where the row maximum sits, width elsewhere; the row minimum is the
        // first maximum.
        var position = Expand(engine.TensorArange<int>(0, width, 1).Reshape(new[] { 1, width }), grid);
        var candidate = engine.TensorAddScalar(engine.TensorSubtract(position, engine.TensorMultiplyScalar(isMax, width)), width);
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(candidate), new[] { 1 }, false));
    }

    #endregion

    #region Internal Types

    /// <summary>
    /// Stores VGM parameters for a single continuous column.
    /// </summary>
    internal sealed class VGMColumnInfo
    {
        public Tensor<T> Means { get; }
        public Tensor<T> Stds { get; }
        public Tensor<T> Weights { get; }
        public int NumActiveModes => Means.Length;

        public VGMColumnInfo(Tensor<T> means, Tensor<T> stds, Tensor<T> weights)
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
