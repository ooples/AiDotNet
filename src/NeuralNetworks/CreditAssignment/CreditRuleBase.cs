using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Shared linear-algebra helpers for the built-in credit-assignment rules. Not part of the public API.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal abstract class CreditRuleBase<T> : ICreditRule<T>
{
    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public virtual void Initialize(ICreditAssignmentContext<T> context) { }

    /// <inheritdoc />
    public abstract void ComputeUpdates(ICreditAssignmentContext<T> context);

    /// <summary>
    /// Element-wise (Hadamard) product of two equally-shaped matrices.
    /// </summary>
    protected static Matrix<T> Hadamard(Matrix<T> a, Matrix<T> b, INumericOperations<T> ops)
    {
        var result = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Columns; j++)
                result[i, j] = ops.Multiply(a[i, j], b[i, j]);
        return result;
    }

    /// <summary>
    /// Fills a layer's <see cref="ICreditLayer{T}.WeightGradient"/> and <see cref="ICreditLayer{T}.BiasGradient"/>
    /// from a pre-activation error signal <paramref name="delta"/> (shape <c>[batch, outputDim]</c>):
    /// <c>dW = (1/B) · deltaᵀ · input</c> and <c>db = (1/B) · Σ_batch delta</c> (mean over the batch, matching
    /// the mean-scaled gradient the optimizer expects).
    /// </summary>
    protected static void SetParameterGradients(ICreditLayer<T> layer, Matrix<T> delta, INumericOperations<T> ops)
    {
        int batch = delta.Rows;
        int outDim = layer.OutputDim;
        int inDim = layer.InputDim;
        var input = layer.Input;
        T invBatch = ops.FromDouble(1.0 / Math.Max(1, batch));

        var weightGrad = new Matrix<T>(outDim, inDim);
        // dW[o, i] = (1/B) * Σ_b delta[b, o] * input[b, i]
        for (int o = 0; o < outDim; o++)
        {
            for (int i = 0; i < inDim; i++)
            {
                T acc = ops.Zero;
                for (int b = 0; b < batch; b++)
                    acc = ops.Add(acc, ops.Multiply(delta[b, o], input[b, i]));
                weightGrad[o, i] = ops.Multiply(acc, invBatch);
            }
        }

        var biasGrad = new Vector<T>(outDim);
        for (int o = 0; o < outDim; o++)
        {
            T acc = ops.Zero;
            for (int b = 0; b < batch; b++)
                acc = ops.Add(acc, delta[b, o]);
            biasGrad[o] = ops.Multiply(acc, invBatch);
        }

        layer.WeightGradient = weightGrad;
        layer.BiasGradient = biasGrad;
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
