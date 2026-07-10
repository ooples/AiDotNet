using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for models that support graph-aware inference: bind an adjacency matrix,
/// forward node features, get per-node predictions. Implemented by
/// <see cref="NeuralNetworks.Tasks.Graph.NodeClassificationModel{T}"/> and any custom GNN
/// (GCN, GAT, GraphSAGE, etc.) that follows the same shape.
/// </summary>
/// <remarks>
/// Introduced to replace the concrete-class gate in <c>AiModelResultGraphExtensions</c> —
/// extension methods now gate on this interface so custom GNN subclasses can plug in without
/// deriving from <see cref="NeuralNetworks.Tasks.Graph.NodeClassificationModel{T}"/>.
/// </remarks>
public interface IGraphInferenceModel<T>
{
    /// <summary>
    /// Binds the graph structure into every graph-convolutional layer prior to inference.
    /// </summary>
    void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix);

    /// <summary>
    /// Runs a forward pass through the network. Returns per-node predictions.
    /// </summary>
    Tensor<T> ForwardOnGraph(Tensor<T> nodeFeatures);
}
