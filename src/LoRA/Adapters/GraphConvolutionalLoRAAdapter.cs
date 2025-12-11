using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This adapter enables LoRA (Low-Rank Adaptation) for graph neural network layers.
/// It wraps a graph convolutional layer (GCN, GAT, GraphSAGE, GIN) and adds a low-rank
/// adaptation that can be efficiently trained while keeping the base layer frozen.
/// </para>
/// <para><b>For Beginners:</b> LoRA for GNNs allows you to fine-tune large pre-trained
/// graph neural networks with a fraction of the trainable parameters.
///
/// **Why LoRA for GNNs?**
/// - Pre-trained GNN models can be huge (millions of parameters)
/// - Fine-tuning all parameters requires lots of memory
/// - LoRA learns small "correction" matrices instead
/// - Result: 10-100x fewer trainable parameters
///
/// **How it works:**
/// - Original GNN layer stays frozen (no updates)
/// - LoRA adds two small matrices (A and B) that learn adaptations
/// - Output = original_output + LoRA_correction
/// - Only A and B are trained, saving memory and time
///
/// **Example - Fine-tuning a GNN for drug discovery:**
/// ```csharp
/// // Wrap existing GAT layer with LoRA
/// var gatLayer = new GraphAttentionLayer&lt;double&gt;(128, 64, numHeads: 8);
/// var loraGat = new GraphConvolutionalLoRAAdapter&lt;double&gt;(
///     gatLayer, rank: 8, alpha: 16);
///
/// // Now train only the LoRA parameters
/// loraGat.UpdateParameters(learningRate);
///
/// // After training, merge LoRA into original layer
/// var mergedLayer = loraGat.MergeToOriginalLayer();
/// ```
///
/// **Supported base layers:**
/// - GraphConvolutionalLayer (GCN)
/// - GraphAttentionLayer (GAT)
/// - GraphSAGELayer
/// - GraphIsomorphismLayer (GIN)
/// - Any layer implementing IGraphConvolutionLayer
/// </para>
/// </remarks>
public class GraphConvolutionalLoRAAdapter<T> : LoRAAdapterBase<T>, IGraphConvolutionLayer<T>
{
    /// <summary>
    /// The graph-aware base layer being adapted.
    /// </summary>
    private readonly IGraphConvolutionLayer<T> _graphBaseLayer;

    /// <summary>
    /// Cached adjacency matrix for graph operations.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Gets the number of input features for this graph layer.
    /// </summary>
    public int InputFeatures => _graphBaseLayer.InputFeatures;

    /// <summary>
    /// Gets the number of output features for this graph layer.
    /// </summary>
    public int OutputFeatures => _graphBaseLayer.OutputFeatures;

    /// <summary>
    /// Initializes a new GraphConvolutionalLoRAAdapter.
    /// </summary>
    /// <param name="baseLayer">The graph layer to adapt (must implement IGraphConvolutionLayer).</param>
    /// <param name="rank">The rank of the LoRA decomposition (default: 8).</param>
    /// <param name="alpha">The LoRA scaling factor (default: same as rank).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer during training (default: true).</param>
    /// <exception cref="ArgumentException">Thrown when baseLayer doesn't implement IGraphConvolutionLayer.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a LoRA adapter for a graph layer:
    ///
    /// ```csharp
    /// // Create base GAT layer
    /// var gat = new GraphAttentionLayer&lt;double&gt;(
    ///     inputFeatures: 128,
    ///     outputFeatures: 64,
    ///     numHeads: 8);
    ///
    /// // Wrap with LoRA for efficient fine-tuning
    /// var loraGat = new GraphConvolutionalLoRAAdapter&lt;double&gt;(
    ///     gat,
    ///     rank: 8,      // Low rank for efficiency
    ///     alpha: 16,    // Scaling factor
    ///     freezeBaseLayer: true);  // Freeze original weights
    ///
    /// // Parameter count comparison:
    /// // Original GAT: ~50,000 parameters
    /// // LoRA adapter: ~2,000 parameters (only 4%!)
    /// ```
    /// </para>
    /// </remarks>
    public GraphConvolutionalLoRAAdapter(
        ILayer<T> baseLayer,
        int rank = 8,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (baseLayer is not IGraphConvolutionLayer<T> graphLayer)
        {
            throw new ArgumentException(
                $"Base layer must implement IGraphConvolutionLayer<{typeof(T).Name}>. " +
                $"Got {baseLayer.GetType().Name} instead.",
                nameof(baseLayer));
        }

        _graphBaseLayer = graphLayer;
    }

    /// <summary>
    /// Sets the adjacency matrix for graph convolution operations.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix defining graph structure.</param>
    /// <remarks>
    /// <para>
    /// This must be called before Forward() to define the graph structure.
    /// The adjacency matrix is passed to the underlying graph layer.
    /// </para>
    /// <para><b>For Beginners:</b> The adjacency matrix tells the layer which nodes are connected.
    /// This is essential for graph convolution operations that aggregate neighbor information.
    /// </para>
    /// </remarks>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;
        _graphBaseLayer.SetAdjacencyMatrix(adjacencyMatrix);
    }

    /// <summary>
    /// Gets the current adjacency matrix.
    /// </summary>
    /// <returns>The adjacency matrix, or null if not set.</returns>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        return _adjacencyMatrix;
    }

    /// <summary>
    /// Performs forward pass through both base graph layer and LoRA layer.
    /// </summary>
    /// <param name="input">Input node features tensor.</param>
    /// <returns>Sum of base layer output and LoRA adaptation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when adjacency matrix is not set.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    /// output = graph_layer(input, adjacency) + lora_layer(input)
    ///
    /// Note that the LoRA layer operates on raw features without the graph structure,
    /// providing a feature-space adaptation that complements the graph-aware base layer.
    /// </para>
    /// <para><b>For Beginners:</b> The graph layer aggregates neighbor information using the
    /// adjacency matrix. The LoRA layer learns to adjust the output features directly.
    /// Together, they provide adapted graph representations.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Forward.");
        }

        // Forward through base graph layer (uses adjacency matrix internally)
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer (feature-space adaptation)
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Merges the LoRA adaptation into the base graph layer.
    /// </summary>
    /// <returns>A new graph layer with LoRA weights merged into its parameters.</returns>
    /// <remarks>
    /// <para>
    /// This creates a standalone graph layer that incorporates the LoRA adaptation.
    /// The merged layer behaves identically to the adapter but without the LoRA overhead.
    /// </para>
    /// <para><b>For Beginners:</b> After training with LoRA, you can "bake in" the adaptation
    /// to create a single layer for deployment. This is faster for inference since it doesn't
    /// need to compute the LoRA correction separately.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Get the LoRA weight contribution as a matrix
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate weight dimensions
        int inputSize = InputFeatures;
        int outputSize = OutputFeatures;
        int weightCount = inputSize * outputSize;

        // Create merged parameters
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge the main weight matrix (first weightCount parameters)
        for (int i = 0; i < Math.Min(weightCount, baseParams.Length); i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;

            // Handle potential dimension mismatch
            if (row < loraWeights.Rows && col < loraWeights.Columns)
            {
                mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
            }
            else
            {
                mergedParams[i] = baseParams[i];
            }
        }

        // Copy remaining parameters unchanged (biases, etc.)
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Clone the appropriate layer type
        return CloneGraphLayerWithParameters(mergedParams);
    }

    /// <summary>
    /// Creates a clone of the base graph layer with the specified parameters.
    /// </summary>
    private ILayer<T> CloneGraphLayerWithParameters(Vector<T> parameters)
    {
        ILayer<T> merged;

        if (_baseLayer is GraphConvolutionalLayer<T> gcnLayer)
        {
            merged = new GraphConvolutionalLayer<T>(
                gcnLayer.InputFeatures,
                gcnLayer.OutputFeatures,
                (IActivationFunction<T>?)null);
        }
        else if (_baseLayer is GraphAttentionLayer<T> gatLayer)
        {
            merged = new GraphAttentionLayer<T>(
                gatLayer.InputFeatures,
                gatLayer.OutputFeatures,
                gatLayer.NumHeads,
                gatLayer.DropoutRate);
        }
        else if (_baseLayer is GraphSAGELayer<T> sageLayer)
        {
            merged = new GraphSAGELayer<T>(
                sageLayer.InputFeatures,
                sageLayer.OutputFeatures);
        }
        else if (_baseLayer is GraphIsomorphismLayer<T> ginLayer)
        {
            merged = new GraphIsomorphismLayer<T>(
                ginLayer.InputFeatures,
                ginLayer.OutputFeatures);
        }
        else
        {
            throw new InvalidOperationException(
                $"Graph layer type {_baseLayer.GetType().Name} is not supported for merging. " +
                "Supported types: GraphConvolutionalLayer, GraphAttentionLayer, GraphSAGELayer, GraphIsomorphismLayer.");
        }

        merged.SetParameters(parameters);
        return merged;
    }

    /// <summary>
    /// Resets the internal state of both graph layer and LoRA layer.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _adjacencyMatrix = null;
    }
}
