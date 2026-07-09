using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Public, extensible base class for <b>credit-assignment (learning) rules</b> in the Direct-Feedback-Alignment
/// family. Subclass this to author your own rule with minimal code: the base owns all the shared machinery — the
/// per-layer feedback-matrix cache (lazily built and re-initialized when the network shape changes), the
/// error/target projection and teaching-signal shaping helpers, feedback normalization/sign helpers, seeded random
/// generation, and an overridable per-step feedback-update hook — so a subclass usually only implements
/// <see cref="ComputeTeachingSignals"/> (and, for a learned-feedback rule, <see cref="UpdateFeedback"/>).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g. <see cref="float"/>, <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> when a neural network is wrong, every layer needs a message that says "here's how your
/// output should change." Standard back-propagation computes that message exactly by chaining backwards through
/// all later layers. The rules in this family instead hand each layer a cheaper <i>shortcut</i> message — usually
/// the final output error pushed through a small random (or learned) matrix — and the network still learns. This
/// base class already contains the plumbing those shortcuts need; to invent your own rule you mostly just decide
/// how to turn the output error into each layer's teaching signal.
/// </para>
/// <para>
/// <b>Writing a custom rule.</b> A minimal Direct-Feedback-Alignment-style rule looks like this:
/// <code>
/// public sealed class MyRule&lt;T&gt; : CreditRuleBase&lt;T&gt;
/// {
///     public MyRule(int? seed = null) : base(seed) { }
///     public override string Name =&gt; "MyRule";
///
///     public override void ComputeTeachingSignals(ICreditAssignmentContext&lt;T&gt; context)
///     {
///         // One fixed random [outputs, features_i] matrix per hidden layer, cached and shape-aware:
///         var feedback = EnsureFeedback(context, (layers, i) =&gt;
///             layers[i].IsOutputLayer ? null : (context.OutputError.Shape[1], layers[i].FlatFeatureSize));
///         var error = ErrorMatrix(context); // [batch, outputs]
///         foreach (var layer in context.Layers)
///         {
///             if (layer.IsOutputLayer) continue; // the engine trains the output layer with the exact loss gradient
///             var projected = ProjectThrough(error, feedback[layer.Index]!); // [batch, features_i]
///             layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
///         }
///     }
/// }
/// // Plug it in without touching the library:  builder.ConfigureCreditRule(new MyRule&lt;float&gt;(seed: 42));
/// </code>
/// For a rule whose feedback is <i>learned</i> (e.g. Kolen-Pollack), also override <see cref="UpdateFeedback"/>.
/// </para>
/// </remarks>
public abstract class CreditRuleBase<T> : IFeedbackLearningRule<T>
{
    /// <summary>Optional explicit RNG seed for reproducible feedback matrices (set via the factory or constructor).</summary>
    protected readonly int? Seed;

    // Shared per-layer feedback-matrix cache. Indexed by trainable-layer position; null entries have no matrix
    // (e.g. the output layer, or the first layer for a sequential rule). Rebuilt when the layer shapes change.
    private Matrix<T>?[]? _feedback;
    private int[]? _signature;

    /// <summary>Creates the rule with an optional RNG seed for reproducible feedback matrices.</summary>
    /// <param name="seed">Optional RNG seed; when null the context's shared random source is used.</param>
    protected CreditRuleBase(int? seed = null) => Seed = seed;

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public virtual bool IsExactBackprop => false;

    /// <inheritdoc />
    public virtual void Initialize(ICreditAssignmentContext<T> context) { }

    /// <inheritdoc />
    public abstract void ComputeTeachingSignals(ICreditAssignmentContext<T> context);

    /// <summary>
    /// The cached per-layer feedback matrices (may contain null entries). Populated by <see cref="EnsureFeedback"/>.
    /// </summary>
    protected IReadOnlyList<Matrix<T>?> Feedback => _feedback ?? System.Array.Empty<Matrix<T>?>();

    void IFeedbackLearningRule<T>.OnParametersUpdated(ICreditAssignmentContext<T> context) => UpdateFeedback(context);

    /// <summary>
    /// Overridable per-step hook for rules whose feedback is <b>learned</b>. Called once per training step, after
    /// the step's teaching signals and gradients were produced and while the per-layer activations are still on the
    /// context. The default is a no-op, so fixed-feedback rules need not override it. Learned-feedback rules
    /// (Kolen-Pollack, Direct Kolen-Pollack) override it to nudge their feedback matrices toward the forward weights.
    /// </summary>
    /// <param name="context">The credit-assignment context for the just-computed step.</param>
    protected virtual void UpdateFeedback(ICreditAssignmentContext<T> context) { }

    /// <summary>Returns the RNG this rule should use: its own seeded generator if a seed was supplied, else the context's.</summary>
    protected Random ResolveRandom(ICreditAssignmentContext<T> context)
        => Seed.HasValue ? new Random(Seed.Value) : context.Random;

    // ---- feedback cache -----------------------------------------------------------------------

    /// <summary>
    /// Lazily builds and caches one feedback matrix per trainable layer, sized by <paramref name="size"/>. The
    /// cache is rebuilt automatically whenever the network's per-layer shapes change (so the same rule instance can
    /// be reused across architectures). <paramref name="size"/> receives the ordered trainable layers and an index
    /// and returns the <c>(rows, cols)</c> of that layer's matrix, or <c>null</c> for layers that have no matrix.
    /// </summary>
    /// <param name="context">The credit-assignment context.</param>
    /// <param name="size">Per-layer matrix sizing; return null for layers without a feedback matrix.</param>
    /// <param name="normalizeColumns">When true, each built matrix has its columns L2-normalized (for magnitude-stable variants).</param>
    protected Matrix<T>?[] EnsureFeedback(
        ICreditAssignmentContext<T> context,
        Func<IReadOnlyList<ICreditLayer<T>>, int, (int rows, int cols)?> size,
        bool normalizeColumns = false)
    {
        var layers = context.Layers;
        var sig = BuildSignature(layers);
        if (_feedback is not null && _signature is not null && SignatureEquals(_signature, sig))
            return _feedback;

        var random = ResolveRandom(context);
        var fb = new Matrix<T>?[layers.Count];
        for (int i = 0; i < layers.Count; i++)
        {
            var s = size(layers, i);
            if (s is null) { fb[i] = null; continue; }
            var m = RandomGaussian(s.Value.rows, s.Value.cols, s.Value.rows, random, context.NumOps);
            if (normalizeColumns) NormalizeColumns(m, context.NumOps);
            fb[i] = m;
        }
        _feedback = fb;
        _signature = sig;
        return fb;
    }

    private static int[] BuildSignature(IReadOnlyList<ICreditLayer<T>> layers)
    {
        var sig = new int[layers.Count * 2];
        for (int i = 0; i < layers.Count; i++)
        {
            sig[2 * i] = layers[i].FlatFeatureSize;
            sig[2 * i + 1] = InFeatures(layers[i]);
        }
        return sig;
    }

    private static bool SignatureEquals(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>The flattened feature count of a layer's forward <see cref="ICreditLayer{T}.Input"/> (product of the non-batch axes).</summary>
    protected static int InFeatures(ICreditLayer<T> layer)
    {
        var shape = layer.Input.Shape;
        int flat = 1;
        for (int i = 1; i < shape.Length; i++) flat *= shape[i];
        return flat;
    }

    // ---- projection / shaping ----------------------------------------------------------------

    /// <summary>The output error as a <c>[batch, features]</c> matrix (prediction − target).</summary>
    protected static Matrix<T> ErrorMatrix(ICreditAssignmentContext<T> context) => FlatMatrix(context.OutputError);

    /// <summary>The one-hot / regression target as a <c>[batch, features]</c> matrix.</summary>
    protected static Matrix<T> TargetMatrix(ICreditAssignmentContext<T> context) => FlatMatrix(context.Target);

    /// <summary>Projects a <c>[batch, C]</c> signal (error or target) through a <c>[C, M]</c> feedback matrix, giving <c>[batch, M]</c>.</summary>
    protected static Matrix<T> ProjectThrough(Matrix<T> signal, Matrix<T> feedback) => signal.Multiply(feedback);

    /// <summary>Flattens a <c>[batch, …]</c> tensor to a <c>[batch, flatFeatures]</c> matrix (row-major over the non-batch axes).</summary>
    protected static Matrix<T> FlatMatrix(Tensor<T> t)
    {
        int batch = t.Shape[0];
        int flat = 1;
        for (int i = 1; i < t.Shape.Length; i++) flat *= t.Shape[i];
        var v = t.ToVector();
        var m = new Matrix<T>(batch, flat);
        int idx = 0;
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < flat; j++)
                m[b, j] = v[idx++];
        return m;
    }

    /// <summary>
    /// Builds a constant teaching-signal tensor of shape <paramref name="outputShape"/> from a
    /// <c>[batch, flatFeatures]</c> matrix (row-major over the non-batch axes).
    /// </summary>
    protected static Tensor<T> ToTeachingSignal(Matrix<T> flat, int[] outputShape)
    {
        int batch = flat.Rows;
        int m = flat.Columns;
        var data = new Vector<T>(batch * m);
        int idx = 0;
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < m; j++)
                data[idx++] = flat[b, j];
        return new Tensor<T>(outputShape, data);
    }

    // ---- feedback math helpers ---------------------------------------------------------------

    /// <summary>Element-wise sign matrix (−1 / 0 / +1) of <paramref name="w"/>.</summary>
    protected static Matrix<T> SignMatrix(Matrix<T> w, INumericOperations<T> ops)
    {
        var s = new Matrix<T>(w.Rows, w.Columns);
        for (int i = 0; i < w.Rows; i++)
            for (int j = 0; j < w.Columns; j++)
                s[i, j] = ops.SignOrZero(w[i, j]);
        return s;
    }

    /// <summary>Scales each column of <paramref name="m"/> to unit L2 norm (in place) — used by magnitude-stable variants.</summary>
    protected static void NormalizeColumns(Matrix<T> m, INumericOperations<T> ops)
    {
        for (int j = 0; j < m.Columns; j++)
        {
            double sq = 0;
            for (int i = 0; i < m.Rows; i++) { double v = ops.ToDouble(m[i, j]); sq += v * v; }
            double norm = Math.Sqrt(sq);
            if (norm <= 1e-12) continue;
            T inv = ops.FromDouble(1.0 / norm);
            for (int i = 0; i < m.Rows; i++) m[i, j] = ops.Multiply(m[i, j], inv);
        }
    }

    /// <summary>Rescales each row of <paramref name="signal"/> so its L2 norm matches the same row of <paramref name="reference"/> (in place).</summary>
    protected static void RescaleRowsToMatch(Matrix<T> signal, Matrix<T> reference, INumericOperations<T> ops)
    {
        for (int b = 0; b < signal.Rows; b++)
        {
            double sig = 0, refN = 0;
            for (int j = 0; j < signal.Columns; j++) { double v = ops.ToDouble(signal[b, j]); sig += v * v; }
            for (int c = 0; c < reference.Columns; c++) { double v = ops.ToDouble(reference[b, c]); refN += v * v; }
            double sigNorm = Math.Sqrt(sig);
            if (sigNorm <= 1e-12) continue;
            double scale = Math.Sqrt(refN) / sigNorm;
            T s = ops.FromDouble(scale);
            for (int j = 0; j < signal.Columns; j++) signal[b, j] = ops.Multiply(signal[b, j], s);
        }
    }

    /// <summary>
    /// Mean per-sample outer product <c>result[i,j] = (1/batch) · Σ_b a[b,i]·b[b,j]</c> (i.e. <c>aᵀ·b / batch</c>) —
    /// the increment used to form Kolen-Pollack feedback updates.
    /// </summary>
    protected static Matrix<T> MeanOuter(Matrix<T> a, Matrix<T> b, INumericOperations<T> ops)
    {
        int rows = a.Columns, cols = b.Columns, batch = a.Rows;
        var g = new Matrix<T>(rows, cols);
        T invBatch = ops.FromDouble(1.0 / Math.Max(1, batch));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                T acc = ops.Zero;
                for (int n = 0; n < batch; n++)
                    acc = ops.Add(acc, ops.Multiply(a[n, i], b[n, j]));
                g[i, j] = ops.Multiply(acc, invBatch);
            }
        return g;
    }

    /// <summary>In-place Kolen-Pollack update of a feedback matrix: <c>B ← (1−decay)·B − lr·grad</c>.</summary>
    protected static void KpUpdate(Matrix<T> b, Matrix<T> grad, double lr, double decay, INumericOperations<T> ops)
    {
        T keep = ops.FromDouble(1.0 - decay);
        T rate = ops.FromDouble(lr);
        for (int i = 0; i < b.Rows; i++)
            for (int j = 0; j < b.Columns; j++)
                b[i, j] = ops.Subtract(ops.Multiply(keep, b[i, j]), ops.Multiply(rate, grad[i, j]));
    }

    /// <summary>
    /// Draws an <c>[rows, cols]</c> matrix of i.i.d. Gaussian entries with standard deviation
    /// <c>1/sqrt(fanForScale)</c> (a stable default for feedback matrices), using the supplied RNG.
    /// </summary>
    protected static Matrix<T> RandomGaussian(int rows, int cols, int fanForScale, Random random, INumericOperations<T> ops)
    {
        double std = 1.0 / Math.Sqrt(Math.Max(1, fanForScale));
        var m = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = ops.FromDouble(NextGaussian(random) * std);
        return m;
    }

    /// <summary>Standard-normal sample via Box–Muller.</summary>
    protected static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
