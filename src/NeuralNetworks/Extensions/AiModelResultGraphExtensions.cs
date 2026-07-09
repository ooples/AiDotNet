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

        graph.SetAdjacencyMatrix(adjacencyMatrix);
        return graph.Forward(nodeFeatures);
    }

    /// <summary>
    /// Predicts a link (edge) between two nodes by combining their post-embedding representations
    /// through the model's classification head. Convenience wrapper: computes
    /// <see cref="PredictOnGraph"/> then extracts the two rows and returns their pairwise score.
    /// </summary>
    /// <remarks>
    /// This is a first-pass implementation using per-node logits' dot product as the link score;
    /// a follow-up will add proper decoder-based scoring (bilinear, MLP-over-concat) when the
    /// underlying model exposes its embedding head separately.
    /// </remarks>
    public static T PredictLink<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> adjacencyMatrix,
        Tensor<T> nodeFeatures,
        int sourceNode,
        int targetNode)
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
        var predictions = graph.Forward(nodeFeatures);
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

        // Dot product of the two nodes' representations as the link score.
        var numOps = AiDotNet.Helpers.MathHelper.GetNumericOperations<T>();
        T score = numOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            score = numOps.Add(score, numOps.Multiply(predictions[sourceNode, d], predictions[targetNode, d]));
        }
        return score;
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
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            graph.SetAdjacencyMatrix(adjacencyMatrix);
            return graph.Forward(nodeFeatures);
        }, cancellationToken);
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
            outputs[i] = graph.Forward(list[i].Item2);
        }
        return outputs;
    }

    private static NodeClassificationModel<T> RequireGraphModel<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
    {
        if (result is null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (result.Model is not NodeClassificationModel<T> graph)
        {
            var actualModelType = result.Model?.GetType().FullName ?? "<no model — result not built yet>";
            throw new InvalidOperationException(
                $"AiModelResult.{extensionName} requires the underlying model to be a " +
                $"NodeClassificationModel<{typeof(T).Name}>. The result was built with " +
                $"'{actualModelType}'. Custom graph neural network subclasses need to either " +
                $"inherit from NodeClassificationModel<T> or implement a graph-inference " +
                $"interface (planned #1836 follow-up).");
        }
        return graph;
    }
}
