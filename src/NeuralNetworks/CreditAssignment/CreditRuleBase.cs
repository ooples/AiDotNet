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
