using System.Globalization;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Matrix-free orthogonal projection <c>y = x Q</c> (or <c>y = x Qᵀ</c>) parameterized as a product of
/// <c>k</c> Householder reflections <c>Q = H₀ H₁ ⋯ H_{k−1}</c>, where each
/// <c>Hⱼ = I − (2 / (‖uⱼ‖₂² + ε)) uⱼ uⱼᵀ</c> uses only <c>O(k · D)</c> parameters and <c>4 k · M · D</c> FLOPs
/// without ever materializing a <c>D × D</c> matrix.
/// </summary>
/// <typeparam name="T">The numeric type used for inputs, reflector vectors, and outputs.</typeparam>
/// <remarks>
/// <para>
/// Every non-zero reflector vector <c>uⱼ ∈ ℝᴰ</c> defines an exact symmetric orthogonal reflection
/// (<c>Hⱼᵀ = Hⱼ</c>, <c>Hⱼ² = I</c>). Consequently <c>Qᵀ Q = I</c> holds by construction for arbitrary unconstrained
/// gradient updates to <c>uⱼ</c> without SVD, QR retraction, or Cayley matrix inversions, preserving activation
/// and gradient Euclidean norms across arbitrarily deep shared-parameter stacks (e.g. 100+ layers).
/// </para>
/// <para>
/// Setting <see cref="Transpose"/> to <c>true</c> applies the reflections in reverse order
/// (<c>H_{k−1} ⋯ H₁ H₀ = Qᵀ = Q⁻¹</c>), providing the exact inverse/transpose rotation.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = false, TestInputShape = "2, 8", TestConstructorArgs = "8, 2")]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class HouseholderSubspaceLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dimension;
    private readonly int _numReflectors;
    private readonly bool _transpose;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _reflectors;

    /// <summary>Gets the feature dimension <c>D</c> rotated by this layer.</summary>
    public int Dimension => _dimension;

    /// <summary>Gets the number of Householder reflections <c>k</c> composed by this layer.</summary>
    public int NumReflectors => _numReflectors;

    /// <summary>Gets whether the reflection chain is applied in reverse order (<c>Qᵀ = Q⁻¹</c>).</summary>
    public bool Transpose => _transpose;

    /// <summary>Gets the underlying <c>[NumReflectors, Dimension]</c> trainable reflector matrix.</summary>
    public Tensor<T> Reflectors => _reflectors;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a matrix-free orthogonal Householder subspace rotation layer of width <paramref name="dimension"/>
    /// using <paramref name="numReflectors"/> reflections.
    /// </summary>
    /// <param name="dimension">Feature width <c>D</c> (must be positive).</param>
    /// <param name="numReflectors">Number of composed Householder reflections <c>k</c> (must be positive).</param>
    /// <param name="transpose">When <c>true</c>, applies reflections in reverse order (<c>Qᵀ = Q⁻¹</c>).</param>
    public HouseholderSubspaceLayer(int dimension, int numReflectors = 4, bool transpose = false)
        : base([dimension], [dimension])
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        if (numReflectors <= 0) throw new ArgumentOutOfRangeException(nameof(numReflectors));

        _dimension = dimension;
        _numReflectors = numReflectors;
        _transpose = transpose;
        _reflectors = new Tensor<T>([numReflectors, dimension]);
        InitializeLayerWeights(_reflectors, numReflectors, dimension);
        NormalizeRowsInPlace(_reflectors, numReflectors, dimension);
        RegisterTrainableParameter(_reflectors, PersistentTensorRole.Weights);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_dimension <= 0 || inputRank < 1) return null;
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dimension));
        static OutputAxisContract Pass(TensorAxis axis) => new(axis, AxisRelation.Same(axis));
        return inputRank switch
        {
            1 => [features],
            2 => [Pass(TensorAxis.Batch), features],
            3 => [Pass(TensorAxis.Batch), Pass(TensorAxis.Time), features],
            4 => [Pass(TensorAxis.Batch), Pass(TensorAxis.Channels), Pass(TensorAxis.Height), features],
            _ => null
        };
    }

    private sealed class CompactWyMaskEntry
    {
        public CompactWyMaskEntry(
            Tensor<T> strictUpperTwo,
            Tensor<T> identity,
            Tensor<T>[] basisRows,
            Tensor<T>[] basisColumns,
            Tensor<T>? wyOffFwd2,
            Tensor<T>? wyOffTrans2,
            Tensor<T>? twoIdentity2)
        {
            StrictUpperTwo = strictUpperTwo;
            Identity = identity;
            BasisRows = basisRows;
            BasisColumns = basisColumns;
            WyOffFwd2 = wyOffFwd2;
            WyOffTrans2 = wyOffTrans2;
            TwoIdentity2 = twoIdentity2;
        }

        public Tensor<T> StrictUpperTwo { get; }
        public Tensor<T> Identity { get; }
        public Tensor<T>[] BasisRows { get; }
        public Tensor<T>[] BasisColumns { get; }
        public Tensor<T>? WyOffFwd2 { get; }
        public Tensor<T>? WyOffTrans2 { get; }
        public Tensor<T>? TwoIdentity2 { get; }
        public IEngine? RegisteredEngine { get; set; }
    }

    [ThreadStatic]
    private static Dictionary<int, CompactWyMaskEntry>? t_compactWyMasks;

    private static CompactWyMaskEntry GetOrCreateCompactWyMasks(IEngine engine, int k)
    {
        var cache = t_compactWyMasks ??= new Dictionary<int, CompactWyMaskEntry>();
        if (!cache.TryGetValue(k, out var entry))
        {
            using var suspendScope = AiDotNet.Tensors.Engines.TensorScope.Suspend();
            using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

            var ops = MathHelper.GetNumericOperations<T>();
            T zero = ops.Zero;
            T one = ops.One;
            T twoVal = ops.FromDouble(2.0);
            var upperData = new T[k * k];
            var idData = new T[k * k];
            for (int r = 0; r < k; r++)
            {
                for (int c = 0; c < k; c++)
                {
                    int idx = r * k + c;
                    upperData[idx] = r < c ? twoVal : zero;
                    idData[idx] = r == c ? one : zero;
                }
            }

            // BasisRows[r] is e_r^T ([1, k]) and BasisColumns[r] is e_r ([k, 1]): the triangular solve in ApplyChain reads
            // row r of a matrix as e_r^T * A and writes a row back as the outer product e_r * v, so the cache is O(k^2).
            var basisRows = new Tensor<T>[k];
            var basisColumns = new Tensor<T>[k];
            for (int r = 0; r < k; r++)
            {
                var basisData = new T[k];
                for (int c = 0; c < k; c++)
                {
                    basisData[c] = c == r ? one : zero;
                }
                basisRows[r] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([1, k], new Vector<T>(basisData)));
                basisColumns[r] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([k, 1], new Vector<T>((T[])basisData.Clone())));
            }

            Tensor<T>? wyOffFwd2 = null;
            Tensor<T>? wyOffTrans2 = null;
            Tensor<T>? twoIdentity2 = null;
            if (k == 2)
            {
                T negFour = ops.FromDouble(-4.0);
                wyOffFwd2 = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([2, 2], new Vector<T>(new[] { zero, negFour, zero, zero })));
                wyOffTrans2 = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([2, 2], new Vector<T>(new[] { zero, zero, negFour, zero })));
                twoIdentity2 = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([2, 2], new Vector<T>(new[] { twoVal, zero, zero, twoVal })));
            }

            entry = new CompactWyMaskEntry(
                AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([k, k], new Vector<T>(upperData))),
                AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>([k, k], new Vector<T>(idData))),
                basisRows,
                basisColumns,
                wyOffFwd2,
                wyOffTrans2,
                twoIdentity2);
            cache[k] = entry;
        }

        if (!ReferenceEquals(entry.RegisteredEngine, engine))
        {
            engine.RegisterPersistentTensor(entry.StrictUpperTwo, PersistentTensorRole.Constant);
            engine.RegisterPersistentTensor(entry.Identity, PersistentTensorRole.Constant);
            for (int r = 0; r < entry.BasisRows.Length; r++)
            {
                engine.RegisterPersistentTensor(entry.BasisRows[r], PersistentTensorRole.Constant);
                engine.RegisterPersistentTensor(entry.BasisColumns[r], PersistentTensorRole.Constant);
            }
            if (entry.WyOffFwd2 is not null)
                engine.RegisterPersistentTensor(entry.WyOffFwd2, PersistentTensorRole.Constant);
            if (entry.WyOffTrans2 is not null)
                engine.RegisterPersistentTensor(entry.WyOffTrans2, PersistentTensorRole.Constant);
            if (entry.TwoIdentity2 is not null)
                engine.RegisterPersistentTensor(entry.TwoIdentity2, PersistentTensorRole.Constant);
            entry.RegisteredEngine = engine;
        }

        return entry;
    }

    private delegate Tensor<T> HouseholderChainEngineDelegate(
        IEngine engine,
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose,
        double normFloor);

    private static readonly HouseholderChainEngineDelegate? s_householderChainDelegate = ResolveHouseholderChainDelegate();

    private static HouseholderChainEngineDelegate? ResolveHouseholderChainDelegate()
    {
        try
        {
            const System.Reflection.BindingFlags Flags = System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance;
            foreach (var m in typeof(IEngine).GetMethods(Flags))
            {
                if (m.Name != "HouseholderChain" || !m.IsGenericMethodDefinition) continue;
                var ps = m.GetParameters();
                if (ps.Length != 4 || ps[2].ParameterType != typeof(bool) || ps[3].ParameterType != typeof(double))
                    continue;
                var closed = m.MakeGenericMethod(typeof(T));
                var engParam = System.Linq.Expressions.Expression.Parameter(typeof(IEngine), "engine");
                var inParam = System.Linq.Expressions.Expression.Parameter(typeof(Tensor<T>), "input");
                var refParam = System.Linq.Expressions.Expression.Parameter(typeof(Tensor<T>), "reflectors");
                var transParam = System.Linq.Expressions.Expression.Parameter(typeof(bool), "transpose");
                var floorParam = System.Linq.Expressions.Expression.Parameter(typeof(double), "normFloor");
                var body = System.Linq.Expressions.Expression.Call(engParam, closed, inParam, refParam, transParam, floorParam);
                return System.Linq.Expressions.Expression.Lambda<HouseholderChainEngineDelegate>(
                    body, engParam, inParam, refParam, transParam, floorParam).Compile();
            }
        }
        catch
        {
            // Older AiDotNet.Tensors package without IEngine.HouseholderChain falls back to Compact WY below.
        }
        return null;
    }

    /// <summary>
    /// Applies the matrix-free product of Householder reflections stored in <paramref name="reflectors"/>
    /// (<c>[k, D]</c>) to <paramref name="input"/> (<c>[..., D]</c>) using pure tape-tracked <see cref="IEngine"/> ops.
    /// </summary>
    internal static Tensor<T> ApplyChain(IEngine engine, Tensor<T> input, Tensor<T> reflectors, bool transpose = false)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (reflectors is null) throw new ArgumentNullException(nameof(reflectors));
        if (reflectors.Rank != 2)
            throw new ArgumentException($"Reflectors must be rank-2 [k, D], got [{string.Join(",", reflectors._shape)}].", nameof(reflectors));

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        if (k <= 0 || d <= 0)
            throw new ArgumentException("Reflectors shape dimensions must be positive.", nameof(reflectors));
        if (input.Rank < 1 || input.Shape[^1] != d)
            throw new ArgumentException($"Expected last dimension {d}, got [{string.Join(",", input._shape)}].", nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rowCount = input.Length / d;
        if (rowCount == 0)
        {
            return engine.TensorMultiplyScalar(input, numOps.One);
        }

        T two = numOps.FromDouble(2.0);
        // Floor the squared-norm denominator instead of adding epsilon so every reflector above the floor remains an
        // exact orthogonal isometry while zero reflectors yield 0 / sqrt(floor) = 0 (identity). Because both k == 1
        // and k > 1 normalize by sqrt(denom), backward divides by denom (not denom^2), so 1e-30 keeps float reflectors
        // exact down to ||u|| = 1e-15 while keeping 1 / (2 * denom) well below float.MaxValue on sub-floor inputs.
        double floorValue = ReflectorNormFloor();
        if (s_householderChainDelegate is { } fusedChain)
        {
            return fusedChain(engine, input, reflectors, transpose, floorValue);
        }
        Tensor<T> current = input.Rank == 2
            ? input
            : engine.Reshape(input, [rowCount, d]);

        if (k == 1)
        {
            Tensor<T> uHat = NormalizeReflectorRows(engine, reflectors, floorValue);
            Tensor<T> uHatCol = engine.TensorTranspose(uHat);
            Tensor<T> proj = engine.TensorMatMul(current, uHatCol);
            Tensor<T> outer = engine.TensorMatMul(engine.TensorMultiplyScalar(proj, two), uHat);
            current = engine.TensorSubtract(current, outer);
        }
        else
        {
            // Exact Compact WY representation (Schreiber & Van Loan 1989):
            // Normalize rows by sqrt(max(||u_j||_2^2, normFloor)) so H_j = I - 2 * uHat_j * uHat_j^T holds across
            // normal, subnormal, and identically zero reflectors without overflow. Then H_0 ... H_{k-1} = I - UHat^T * T * UHat
            // where N = 2 * strict_triu(UHat * UHat^T) is nilpotent (N^k == 0) and T = 2 * (I + N)^-1.
            var masks = GetOrCreateCompactWyMasks(engine, k);
            Tensor<T> uHat = NormalizeReflectorRows(engine, reflectors, floorValue);
            Tensor<T> uHatCol = engine.TensorTranspose(uHat);
            Tensor<T> gram = engine.TensorMatMul(uHat, uHatCol);
            Tensor<T> tMat;
            if (k == 2 && masks.WyOffFwd2 is not null && masks.WyOffTrans2 is not null && masks.TwoIdentity2 is not null)
            {
                // Exact closed-form 2x2 nilpotent inverse: N^2 == 0 identically for every 2x2 strictly upper-triangular
                // matrix, so 2 * (I + N)^-1 = 2 * (I - N) = 2*I - 4*strict_triu(gram), and its transpose uses the
                // symmetric lower-triangular entry gram[1, 0] == gram[0, 1] directly without extra GEMM/transpose launches.
                Tensor<T> offMask = transpose ? masks.WyOffTrans2 : masks.WyOffFwd2;
                tMat = engine.TensorAdd(engine.TensorMultiply(gram, offMask), masks.TwoIdentity2);
            }
            else
            {
                Tensor<T> nilpotent = engine.TensorMultiply(gram, masks.StrictUpperTwo);

                // M = (I + N)^-1 by back substitution, one row per step from the bottom: (I + N) M = I gives
                // M[i] = e_i - N[i] M, and N[i] is non-zero only right of the diagonal, so it reads only rows below i,
                // which are already final. Row i of M is still e_i before its step, so subtracting e_i * (N[i] M) sets it
                // exactly. Each step forms only that row: N[i] = e_i^T N and N[i] M are [1, k] products, and the write-back
                // is a [k, 1] x [1, k] outer product, so the solve costs O(k^3) rather than k full [k, k] products.
                // Every intermediate is a finished row of M or an identity row, so nothing grows past the final T. The
                // truncated Neumann series this replaces formed the partial sums of (-N)^p, which grow like
                // 2^p * C(k - 1, p - 1) before cancelling and overflow Half at k = 16 for correlated reflectors.
                Tensor<T> series = masks.Identity;
                for (int i = k - 2; i >= 0; i--)
                {
                    Tensor<T> nilpotentRow = engine.TensorMatMul(masks.BasisRows[i], nilpotent);
                    Tensor<T> solvedRow = engine.TensorMatMul(nilpotentRow, series);
                    series = engine.TensorSubtract(series, engine.TensorMatMul(masks.BasisColumns[i], solvedRow));
                }

                tMat = engine.TensorMultiplyScalar(series, two);
                if (transpose)
                {
                    tMat = engine.TensorTranspose(tMat);
                }
            }

            Tensor<T> proj = engine.TensorMatMul(current, uHatCol);
            Tensor<T> mixed = engine.TensorMatMul(proj, tMat);
            Tensor<T> outer = engine.TensorMatMul(mixed, uHat);
            current = engine.TensorSubtract(current, outer);
        }

        return input.Rank == 2
            ? current
            : engine.Reshape(current, input._shape);
    }

    /// <summary>
    /// Returns <c>u_j / sqrt(max(||u_j||^2, floor))</c> for each reflector row without forming <c>||u_j||^2</c> in T.
    /// Squaring in T overflows Half once a row norm passes 256 (256^2 = 65536 &gt; 65504), which made the
    /// denominator infinite and the reflector the identity. Each row is first divided by its largest magnitude
    /// <c>c_j</c>, so every square is at most 1 and the row sum is at most D. Because
    /// <c>(u / c) / sqrt(max(||u / c||^2, floor / c^2)) = u / sqrt(max(||u||^2, floor))</c> for any c &gt; 0, the
    /// values are those of the unscaled formula, and with c detached so are the gradients. c is held at least
    /// <c>sqrt(floor) / 16</c>, which keeps <c>floor / c^2</c> at most 256 and maps a zero row to 0 (identity).
    /// </summary>
    private static double ReflectorNormFloor()
    {
        string typeName = typeof(T).Name;
        return (typeof(T) == typeof(double) || typeof(T) == typeof(decimal))
            ? 1.1754943508222875e-38
            : (typeName == "Half" || typeName.StartsWith("Float8", StringComparison.Ordinal))
                ? 1e-2
                : 1e-30;
    }

    private static Tensor<T> NormalizeReflectorRows(IEngine engine, Tensor<T> reflectors, double floorValue)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double sqrtFloor = Math.Sqrt(floorValue);
        Tensor<T> rowMaxAbs = engine.StopGradient(engine.ReduceMax(engine.TensorAbs(reflectors), [1], keepDims: true));
        Tensor<T> rowScale = engine.TensorMax(rowMaxAbs, numOps.FromDouble(sqrtFloor / 16.0));
        Tensor<T> scaled = engine.TensorDivide(reflectors, rowScale);
        Tensor<T> scaledSq = engine.ReduceSum(engine.TensorMultiply(scaled, scaled), [1], keepDims: true);
        Tensor<T> scaledFloorRoot = engine.TensorMultiplyScalar(
            engine.TensorReciprocal(rowScale), numOps.FromDouble(sqrtFloor));
        Tensor<T> scaledFloor = engine.TensorMultiply(scaledFloorRoot, scaledFloorRoot);
        Tensor<T> denom = engine.TensorMax(scaledSq, scaledFloor);
        return engine.TensorDivide(scaled, engine.TensorSqrt(denom));
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => ApplyChain(Engine, input, _reflectors, _transpose);

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _reflectors.Length) return;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var gradTensor = new Tensor<T>(_reflectors._shape, gradients);
        Engine.TensorSubtractInPlace(_reflectors, Engine.TensorMultiplyScalar(gradTensor, learningRate));
        Engine.InvalidatePersistentTensor(_reflectors);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dimension"] = _dimension.ToString(CultureInfo.InvariantCulture);
        metadata["NumReflectors"] = _numReflectors.ToString(CultureInfo.InvariantCulture);
        metadata["Transpose"] = _transpose.ToString(CultureInfo.InvariantCulture);
        return metadata;
    }

    private void NormalizeRowsInPlace(Tensor<T> reflectors, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // Same overflow-safe normalization as the forward pass: the former sum of squares in T overflowed Half for
        // rows above norm 256, and its 1e-24 guard rounded to 0 in Half, turning a zero row into 0 / 0. The init floor
        // is at most 1e-6 so every row of norm 1e-3 or more becomes exactly unit, as before; for Half its
        // sqrt(1e-6) / 16 = 6.25e-5 scale bound is still a normal Half. Float8 keeps its forward floor, since
        // 6.25e-5 is below its smallest subnormal.
        double initFloor = typeof(T).Name.StartsWith("Float8", StringComparison.Ordinal)
            ? ReflectorNormFloor()
            : Math.Min(ReflectorNormFloor(), 1e-6);
        var normalized = NormalizeReflectorRows(Engine, reflectors, initFloor);
        normalized.AsSpan().Slice(0, reflectors.Length).CopyTo(reflectors.Data.Span);
        Engine.InvalidatePersistentTensor(reflectors);
    }
}
