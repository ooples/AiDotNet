using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Shared base + linear-algebra helpers for the built-in credit-assignment rules. Not part of the public API.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal abstract class CreditRuleBase<T> : ICreditRule<T>
{
    /// <summary>Optional explicit RNG seed (set via the <c>CreditRules</c> factory) for reproducible feedback matrices.</summary>
    protected readonly int? Seed;

    protected CreditRuleBase(int? seed = null) => Seed = seed;

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public virtual bool IsExactBackprop => false;

    /// <inheritdoc />
    public virtual void Initialize(ICreditAssignmentContext<T> context) { }

    /// <inheritdoc />
    public abstract void ComputeTeachingSignals(ICreditAssignmentContext<T> context);

    /// <summary>Returns the RNG this rule should use: its own seeded generator if a seed was supplied, else the context's.</summary>
    protected Random ResolveRandom(ICreditAssignmentContext<T> context)
        => Seed.HasValue ? new Random(Seed.Value) : context.Random;

    /// <summary>The output error as a <c>[batch, features]</c> matrix.</summary>
    protected static Matrix<T> ErrorMatrix(ICreditAssignmentContext<T> context)
    {
        var e = context.OutputError; // [B, C]
        var m = new Matrix<T>(e.Shape[0], e.Shape[1]);
        for (int b = 0; b < e.Shape[0]; b++)
            for (int c = 0; c < e.Shape[1]; c++)
                m[b, c] = e[b, c];
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

    /// <summary>The one-hot / regression target as a <c>[batch, features]</c> matrix.</summary>
    protected static Matrix<T> TargetMatrix(ICreditAssignmentContext<T> context) => FlatMatrix(context.Target);

    /// <summary>
    /// Outer-product accumulation <c>result[i,j] = (1/batch) · Σ_b a[b,i] · b[b,j]</c> — i.e. <c>aᵀ·b / batch</c>,
    /// the mean per-sample outer product used to form Kolen-Pollack feedback updates.
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

    /// <summary>Element-wise sign matrix (−1 / 0 / +1) of <paramref name="w"/>.</summary>
    protected static Matrix<T> SignMatrix(Matrix<T> w, INumericOperations<T> ops)
    {
        var s = new Matrix<T>(w.Rows, w.Columns);
        for (int i = 0; i < w.Rows; i++)
            for (int j = 0; j < w.Columns; j++)
                s[i, j] = ops.SignOrZero(w[i, j]);
        return s;
    }

    /// <summary>
    /// Draws an <c>[rows, cols]</c> matrix of i.i.d. Gaussian entries with standard deviation
    /// <c>1/sqrt(fanForScale)</c> (a stable default for fixed feedback matrices), using the supplied RNG.
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
