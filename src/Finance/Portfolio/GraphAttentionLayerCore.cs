using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// The graph attention computation used by <see cref="GraphAttentionPortfolio{T}"/>
/// (Korangi, Mues and Bravo, arXiv:2407.15532, following Velickovic et al.):
/// <c>a(u,v) = softmax_v( LeakyReLU( a^T [W x_u || W x_v] ) )</c>, aggregated over neighbours and
/// CONCATENATED across heads.
/// </summary>
/// <remarks>
/// <para>
/// What separates this from ordinary dense attention is the MASK. Each node attends only over its
/// graph neighbourhood, and the softmax is normalized within that neighbourhood — so the sparsity that
/// TMFG filtering produced is what the attention actually operates on. Attending over all pairs would
/// discard the filtered graph entirely and reduce the model to a plain transformer over assets.
/// </para>
/// <para>
/// Heads are CONCATENATED, not averaged, following the paper's <c>||</c> operator. Averaging would
/// collapse the heads into a single representation of the same width and lose the distinct relational
/// views the heads learn.
/// </para>
/// <para><b>For Beginners:</b> Each company looks only at the companies it is connected to in the
/// network, decides how much attention to pay to each, and builds a summary from them.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class GraphAttentionLayerCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's LeakyReLU negative slope.</summary>
    /// <remarks>
    /// 0.2 is the value from the original GAT paper, which this work follows. A slope of 0 would make
    /// it a plain ReLU and zero the gradient for every negatively-scored pair, so those edges could
    /// never recover.
    /// </remarks>
    public const double LeakyReLUSlope = 0.2;

    private readonly double _slope;

    /// <summary>Gets the LeakyReLU negative slope.</summary>
    public double Slope => _slope;

    /// <summary>Creates the attention core.</summary>
    /// <param name="slope">LeakyReLU negative slope. Paper/GAT default: 0.2.</param>
    public GraphAttentionLayerCore(double slope = LeakyReLUSlope)
    {
        if (slope < 0.0 || double.IsNaN(slope))
            throw new ArgumentOutOfRangeException(nameof(slope), slope, "slope cannot be negative or NaN.");
        _slope = slope;
    }

    /// <summary>LeakyReLU.</summary>
    public double LeakyReLU(double x) => x >= 0.0 ? x : _slope * x;

    /// <summary>
    /// Attention coefficients over a masked graph, one row per node, normalized within each
    /// neighbourhood.
    /// </summary>
    /// <param name="projected">
    /// Node features after the shared projection <c>W x</c>, shaped <c>[nodes, features]</c>.
    /// </param>
    /// <param name="attentionVector">
    /// The vector <c>a</c>, of length <c>2 * features</c>: the first half pairs with the source node's
    /// features and the second half with the target's.
    /// </param>
    /// <param name="adjacency">
    /// <c>[nodes, nodes]</c> mask; a non-zero entry means the pair is connected. Should include
    /// self-loops.
    /// </param>
    /// <returns><c>[nodes, nodes]</c> coefficients; each row sums to 1 over its neighbours and is 0 elsewhere.</returns>
    /// <remarks>
    /// Non-neighbours receive exactly zero rather than a small value: the softmax runs only over the
    /// neighbourhood, so unconnected assets contribute nothing at all. Implementing the mask as a
    /// post-hoc multiply after a dense softmax would leave the rows no longer summing to 1.
    /// </remarks>
    public Tensor<T> AttentionCoefficients(
        Tensor<T> projected, Vector<T> attentionVector, Tensor<T> adjacency)
    {
        if (projected == null) throw new ArgumentNullException(nameof(projected));
        if (attentionVector == null) throw new ArgumentNullException(nameof(attentionVector));
        if (adjacency == null) throw new ArgumentNullException(nameof(adjacency));
        if (projected.Shape.Length != 2)
            throw new ArgumentException(
                $"projected must be [nodes, features]; got rank {projected.Shape.Length}.", nameof(projected));

        int nodes = projected.Shape[0];
        int features = projected.Shape[1];

        if (attentionVector.Length != 2 * features)
            throw new ArgumentException(
                $"attentionVector must have length {2 * features} (2 * features); got {attentionVector.Length}.",
                nameof(attentionVector));
        if (adjacency.Shape.Length != 2 || adjacency.Shape[0] != nodes || adjacency.Shape[1] != nodes)
            throw new ArgumentException(
                $"adjacency must be [{nodes}, {nodes}].", nameof(adjacency));

        // Split a into its source and target halves once: e_uv = aSrc . Wx_u + aTgt . Wx_v, which is
        // the concatenated inner product written out, and avoids materializing the concatenation for
        // every pair.
        var sourceScore = new double[nodes];
        var targetScore = new double[nodes];
        for (int u = 0; u < nodes; u++)
        {
            double s = 0.0, t = 0.0;
            for (int f = 0; f < features; f++)
            {
                double x = NumOps.ToDouble(projected[(u * features) + f]);
                s += NumOps.ToDouble(attentionVector[f]) * x;
                t += NumOps.ToDouble(attentionVector[features + f]) * x;
            }
            sourceScore[u] = s;
            targetScore[u] = t;
        }

        var result = new Tensor<T>(new[] { nodes, nodes });

        for (int u = 0; u < nodes; u++)
        {
            // Max-subtracted softmax over the neighbourhood only.
            double max = double.NegativeInfinity;
            for (int v = 0; v < nodes; v++)
            {
                if (NumOps.ToDouble(adjacency[(u * nodes) + v]) == 0.0) continue;
                max = Math.Max(max, LeakyReLU(sourceScore[u] + targetScore[v]));
            }

            // An isolated node has no neighbours to normalize over. Leaving its row at zero is the
            // honest result: it aggregates nothing, rather than inventing a uniform distribution over
            // assets it has no established relationship with.
            if (double.IsNegativeInfinity(max)) continue;

            double sum = 0.0;
            var exps = new double[nodes];
            for (int v = 0; v < nodes; v++)
            {
                if (NumOps.ToDouble(adjacency[(u * nodes) + v]) == 0.0) continue;
                exps[v] = Math.Exp(LeakyReLU(sourceScore[u] + targetScore[v]) - max);
                sum += exps[v];
            }

            for (int v = 0; v < nodes; v++)
            {
                if (exps[v] == 0.0) continue;
                result[(u * nodes) + v] = NumOps.FromDouble(exps[v] / sum);
            }
        }

        return result;
    }

    /// <summary>
    /// Aggregates neighbour features under the attention coefficients:
    /// <c>h_u = sum_v alpha_uv * (W x_v)</c>, then applies ReLU.
    /// </summary>
    public Tensor<T> Aggregate(Tensor<T> coefficients, Tensor<T> projected)
    {
        if (coefficients == null) throw new ArgumentNullException(nameof(coefficients));
        if (projected == null) throw new ArgumentNullException(nameof(projected));
        if (projected.Shape.Length != 2)
            throw new ArgumentException("projected must be [nodes, features].", nameof(projected));

        int nodes = projected.Shape[0];
        int features = projected.Shape[1];
        if (coefficients.Shape.Length != 2 || coefficients.Shape[0] != nodes || coefficients.Shape[1] != nodes)
            throw new ArgumentException($"coefficients must be [{nodes}, {nodes}].", nameof(coefficients));

        var result = new Tensor<T>(new[] { nodes, features });
        for (int u = 0; u < nodes; u++)
        {
            for (int f = 0; f < features; f++)
            {
                double sum = 0.0;
                for (int v = 0; v < nodes; v++)
                {
                    double alpha = NumOps.ToDouble(coefficients[(u * nodes) + v]);
                    if (alpha == 0.0) continue;
                    sum += alpha * NumOps.ToDouble(projected[(v * features) + f]);
                }

                // F(.) = ReLU, per the paper.
                result[(u * features) + f] = NumOps.FromDouble(Math.Max(0.0, sum));
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates per-head outputs along the feature axis, as the paper's <c>||</c> operator does.
    /// </summary>
    /// <param name="headOutputs">Per-head <c>[nodes, features]</c> tensors, all the same shape.</param>
    /// <returns><c>[nodes, heads * features]</c>.</returns>
    /// <remarks>
    /// Concatenation, not averaging: averaging would collapse the heads to one representation of the
    /// original width and discard the separate relational views they learn.
    /// </remarks>
    public Tensor<T> ConcatenateHeads(IReadOnlyList<Tensor<T>> headOutputs)
    {
        if (headOutputs == null) throw new ArgumentNullException(nameof(headOutputs));
        if (headOutputs.Count == 0)
            throw new ArgumentException("At least one head is required.", nameof(headOutputs));

        int nodes = headOutputs[0].Shape[0];
        int features = headOutputs[0].Shape[1];
        for (int h = 1; h < headOutputs.Count; h++)
        {
            if (headOutputs[h].Shape.Length != 2 || headOutputs[h].Shape[0] != nodes
                || headOutputs[h].Shape[1] != features)
                throw new ArgumentException(
                    $"Head {h} has shape [{string.Join(",", headOutputs[h].Shape.ToArray())}]; " +
                    $"expected [{nodes}, {features}].", nameof(headOutputs));
        }

        int heads = headOutputs.Count;
        var result = new Tensor<T>(new[] { nodes, heads * features });
        for (int u = 0; u < nodes; u++)
            for (int h = 0; h < heads; h++)
                for (int f = 0; f < features; f++)
                    result[(u * heads * features) + (h * features) + f] = headOutputs[h][(u * features) + f];

        return result;
    }
}
