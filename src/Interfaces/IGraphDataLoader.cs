using AiDotNet.Data.Abstractions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for graph data loaders that load graph-structured data.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Graph data loaders provide graph-structured data for training graph neural networks.
/// Unlike traditional data loaders that work with vectors or images, graph loaders handle
/// complex graph structures with nodes, edges, and associated features.
/// </para>
/// <para><b>For Beginners:</b> This interface defines what any graph data loader must do.
///
/// Graph neural networks need special data loading because graphs have unique properties:
/// - **Irregular structure**: Graphs don't have fixed sizes like images (28×28)
/// - **Connectivity**: Node relationships are as important as node features
/// - **Variable topology**: Each graph can have different numbers of nodes and edges
///
/// Common graph learning tasks:
/// - **Node Classification**: Predict labels for individual nodes (e.g., categorize users)
/// - **Link Prediction**: Predict missing or future edges (e.g., recommend friends)
/// - **Graph Classification**: Classify entire graphs (e.g., is this molecule toxic?)
/// - **Graph Generation**: Create new valid graphs (e.g., generate new molecules)
///
/// This interface ensures all graph data loaders provide data in a consistent format.
/// </para>
/// </remarks>
public interface IGraphDataLoader<T>
{
    /// <summary>
    /// Loads and returns the next graph or batch of graphs.
    /// </summary>
    /// <returns>A GraphData instance containing the loaded graph(s).</returns>
    /// <remarks>
    /// <para>
    /// Each call returns a graph or batch of graphs with:
    /// - Node features
    /// - Edge connectivity
    /// - Optional edge features
    /// - Optional labels (depending on the task)
    /// - Optional train/val/test masks
    /// </para>
    /// <para><b>For Beginners:</b> This method loads one graph (or batch of graphs) at a time.
    ///
    /// Think of it like loading images in computer vision:
    /// - Image loader returns batches of images
    /// - Graph loader returns batches of graphs
    ///
    /// The key difference is that graphs have variable structure - one graph might have
    /// 10 nodes and 15 edges, while another has 100 nodes and 300 edges.
    /// </para>
    /// </remarks>
    GraphData<T> GetNextBatch();

    /// <summary>
    /// Resets the data loader to the beginning of the dataset.
    /// </summary>
    /// <remarks>
    /// Call this at the start of each epoch to iterate through the dataset from the beginning.
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the total number of graphs in the dataset.
    /// </summary>
    int NumGraphs { get; }

    /// <summary>
    /// Gets the batch size (number of graphs per batch).
    /// </summary>
    int BatchSize { get; }

    /// <summary>
    /// Gets whether the data loader has more batches available.
    /// </summary>
    bool HasNext { get; }
}
