using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Tasks.Graph;

namespace AiDotNet.Graphs.Extensions;

/// <summary>
/// Model-specific inference extensions on <see cref="AiModelResult{T, TInput, TOutput}"/> for the
/// graph neural network family (GCN, GAT, GraphSAGE, etc. wrapped as
/// <see cref="NodeClassificationModel{T}"/>). Part of #1836.
/// </summary>
/// <remarks>
/// See <c>AiModelResultRadianceFieldExtensions</c> for the full design rationale — extension
/// methods live in the same assembly as <see cref="AiModelResult{T, TInput, TOutput}"/> so they
/// access internal <c>Model</c> directly without exposing it.
/// </remarks>
public static class AiModelResultGraphExtensions
{
    /// <summary>
    /// Runs graph-aware inference: binds the adjacency matrix into every graph-convolutional
    /// layer, then forwards the node features through the network. Returns per-node predictions
    /// (logits for classification, values for regression).
    /// </summary>
    /// <param name="result">The trained model result.</param>
    /// <param name="adjacencyMatrix">Graph adjacency [N, N] — dense or sparse per model configuration.</param>
    /// <param name="nodeFeatures">Node feature tensor [N, F].</param>
    /// <returns>Per-node output tensor [N, OutDim].</returns>
    public static Tensor<T> PredictOnGraph<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> adjacencyMatrix,
        Tensor<T> nodeFeatures)
    {
        var graph = RequireGraphModel(result, nameof(PredictOnGraph));
        if (adjacencyMatrix is null) throw new ArgumentNullException(nameof(adjacencyMatrix));
        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));

        return AiDotNet.Extensions.Telemetry.AiModelResultInferenceTelemetry.TimeAndLog(
            result,
            nameof(PredictOnGraph),
            () =>
            {
                graph.SetAdjacencyMatrix(adjacencyMatrix);
                return graph.ForwardOnGraph(nodeFeatures);
            },
            resultCount: nodeFeatures.Shape.Length > 0 ? nodeFeatures.Shape[0] : (int?)null);
    }

    /// <summary>
    /// Scoring strategy for <see cref="PredictLink"/>. Reference link-prediction impls (GAE,
    /// VGAE, R-GCN) each hard-code one scorer; here callers pick per call.
    /// </summary>
    public enum LinkScorer
    {
        /// <summary>Cosine similarity between the two node embeddings — magnitude-invariant.</summary>
        Cosine,
        /// <summary>Dot product between the two node embeddings — the DistMult / GAE default.</summary>
        DotProduct,
        /// <summary>Sigmoid(dot product) — squashes the score into a probability in (0, 1).</summary>
        SigmoidDot,
    }

    /// <summary>
    /// Predicts a link (edge) score between two nodes by combining their post-embedding
    /// representations through the selected scorer. Reference facades expose only a single
    /// hardcoded scorer per implementation; AiDotNet lets callers pick per call.
    /// </summary>
    public static T PredictLink<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> adjacencyMatrix,
        Tensor<T> nodeFeatures,
        int sourceNode,
        int targetNode,
        LinkScorer scorer = LinkScorer.SigmoidDot)
    {
        var graph = RequireGraphModel(result, nameof(PredictLink));
        if (adjacencyMatrix is null) throw new ArgumentNullException(nameof(adjacencyMatrix));
        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));
        if (sourceNode < 0 || targetNode < 0)
        {
            throw new ArgumentOutOfRangeException(
                sourceNode < 0 ? nameof(sourceNode) : nameof(targetNode),
                "Node indices must be non-negative.");
        }

        graph.SetAdjacencyMatrix(adjacencyMatrix);
        var predictions = graph.ForwardOnGraph(nodeFeatures);
        if (predictions.Shape.Length != 2)
        {
            throw new InvalidOperationException(
                $"PredictLink requires per-node predictions of shape [N, D]; got " +
                $"[{string.Join(",", predictions.Shape)}].");
        }

        int nNodes = predictions.Shape[0];
        int dim    = predictions.Shape[1];
        if (sourceNode >= nNodes || targetNode >= nNodes)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sourceNode),
                $"Node indices ({sourceNode}, {targetNode}) exceed node count ({nNodes}).");
        }

        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        T dot = numOps.Zero;
        T normA = numOps.Zero;
        T normB = numOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            var a = predictions[sourceNode, d];
            var b = predictions[targetNode, d];
            dot = numOps.Add(dot, numOps.Multiply(a, b));
            normA = numOps.Add(normA, numOps.Multiply(a, a));
            normB = numOps.Add(normB, numOps.Multiply(b, b));
        }

        switch (scorer)
        {
            case LinkScorer.DotProduct:
                return dot;
            case LinkScorer.Cosine:
            {
                double na = System.Math.Sqrt(numOps.ToDouble(normA));
                double nb = System.Math.Sqrt(numOps.ToDouble(normB));
                double denom = na * nb;
                return denom > 1e-12
                    ? numOps.FromDouble(numOps.ToDouble(dot) / denom)
                    : numOps.Zero;
            }
            case LinkScorer.SigmoidDot:
            default:
            {
                double x = numOps.ToDouble(dot);
                double sig = 1.0 / (1.0 + System.Math.Exp(-x));
                return numOps.FromDouble(sig);
            }
        }
    }

    /// <summary>
    /// Async graph inference for pipelines that need to score many graphs concurrently
    /// without blocking the caller. Reference impls (PyG, DGL) are synchronous per-graph;
    /// this dispatches to the thread pool with cancellation support.
    /// </summary>
    public static Task<Tensor<T>> PredictOnGraphAsync<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> adjacencyMatrix,
        Tensor<T> nodeFeatures,
        CancellationToken cancellationToken = default)
    {
        var graph = RequireGraphModel(result, nameof(PredictOnGraphAsync));
        if (adjacencyMatrix is null) throw new ArgumentNullException(nameof(adjacencyMatrix));
        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));

        // SetAdjacencyMatrix + ForwardOnGraph mutate model-internal state that isn't
        // guarded against concurrent access. Two PredictOnGraphAsync callers on the same
        // AiModelResult would race: caller A sets its adjacency, caller B overwrites it,
        // caller A forwards against B's adjacency. Serialize via a per-model lock keyed
        // on the graph instance (not `result` — the same underlying model can be shared
        // across multiple result wrappers, and we care about the model's internal state,
        // not the wrapper's identity).
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            lock (GraphInferenceLocks.GetLock(graph))
            {
                cancellationToken.ThrowIfCancellationRequested();
                graph.SetAdjacencyMatrix(adjacencyMatrix);
                return graph.ForwardOnGraph(nodeFeatures);
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Per-model lock table for graph-inference serialization. Uses a
    /// <see cref="System.Runtime.CompilerServices.ConditionalWeakTable{TKey, TValue}"/>
    /// so the lock is GC'd when the model is — no leak.
    /// </summary>
    private static class GraphInferenceLocks
    {
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> Table = new();
        public static object GetLock(object graph) => Table.GetValue(graph, _ => new object());
    }

    /// <summary>
    /// Batched graph inference: score N graphs by iterating the pairs. Reference impls
    /// require the caller to hand-roll this loop.
    /// </summary>
    public static Tensor<T>[] PredictOnGraphBatch<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        IEnumerable<(Tensor<T> Adjacency, Tensor<T> NodeFeatures)> graphs)
    {
        var graph = RequireGraphModel(result, nameof(PredictOnGraphBatch));
        if (graphs is null) throw new ArgumentNullException(nameof(graphs));
        var list = new List<(Tensor<T>, Tensor<T>)>(graphs);
        var outputs = new Tensor<T>[list.Count];
        for (int i = 0; i < list.Count; i++)
        {
            graph.SetAdjacencyMatrix(list[i].Item1);
            outputs[i] = graph.ForwardOnGraph(list[i].Item2);
        }
        return outputs;
    }

    private static AiDotNet.Interfaces.IGraphInferenceModel<T> RequireGraphModel<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
        => AiDotNet.Extensions.Capability.AiModelResultExtensionsCapabilityGate.Require<
            T, TInput, TOutput, AiDotNet.Interfaces.IGraphInferenceModel<T>>(
            result,
            extensionName,
            $"AiDotNet.Interfaces.IGraphInferenceModel<{typeof(T).Name}>",
            hint: "(NodeClassificationModel or any custom GNN subclass implementing it).");
}
