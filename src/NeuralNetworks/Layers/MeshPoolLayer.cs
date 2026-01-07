using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements mesh pooling via edge collapse for MeshCNN-style networks.
/// </summary>
/// <remarks>
/// <para>
/// MeshPoolLayer reduces the number of edges in a mesh by collapsing edges based on
/// learned importance scores. This is analogous to pooling in image CNNs but operates
/// on the mesh structure.
/// </para>
/// <para><b>For Beginners:</b> Just like max pooling shrinks an image by combining pixels,
/// mesh pooling shrinks a mesh by combining edges. The layer learns which edges are
/// less important and removes them, simplifying the mesh while preserving important features.
/// 
/// Key concepts:
/// - Edge collapse: Remove an edge by merging its two vertices into one
/// - Importance score: Learned value indicating how important each edge is
/// - Target edges: Number of edges to keep after pooling
/// 
/// The process:
/// 1. Compute importance scores for all edges using current features
/// 2. Sort edges by importance (lowest first)
/// 3. Collapse least important edges until target count is reached
/// 4. Update adjacency information for remaining edges
/// </para>
/// <para>
/// Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MeshPoolLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the target number of edges after pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This determines how much the mesh is simplified. For example, pooling from
    /// 750 edges to 450 edges removes about 40% of the edges.
    /// </para>
    /// </remarks>
    public int TargetEdges { get; private set; }

    /// <summary>
    /// Gets the number of input feature channels per edge.
    /// </summary>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>Always <c>true</c> as importance scores are learned.</value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>false</c> because mesh pooling requires dynamic graph operations.</value>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Gets or sets the edge indices that remain after pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is set during the forward pass and can be used to track which edges
    /// were preserved through pooling.
    /// </para>
    /// </remarks>
    public int[]? RemainingEdgeIndices { get; private set; }

    /// <summary>
    /// Gets or sets the updated edge adjacency after pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After pooling, the edge adjacency must be updated to reflect the new
    /// connectivity of the reduced mesh.
    /// </para>
    /// </remarks>
    public int[,]? UpdatedAdjacency { get; private set; }

    #endregion

    #region Private Fields

    /// <summary>
    /// Learnable weights for computing edge importance scores.
    /// </summary>
    private Tensor<T> _importanceWeights;

    /// <summary>
    /// Cached gradient for importance weights.
    /// </summary>
    private Tensor<T>? _importanceWeightsGradient;

    /// <summary>
    /// Cached input from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached edge adjacency from the last forward pass.
    /// </summary>
    private int[,]? _lastEdgeAdjacency;

    /// <summary>
    /// Cached output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached importance scores from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastImportanceScores;

    /// <summary>
    /// Number of neighboring edges per edge (default 4 for triangular meshes).
    /// </summary>
    private readonly int _numNeighbors;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshPoolLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per edge.</param>
    /// <param name="targetEdges">Target number of edges after pooling.</param>
    /// <param name="numNeighbors">Number of neighboring edges per edge. Default is 4.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a mesh pooling layer that reduces mesh complexity.</para>
    /// <para>
    /// Example: If your mesh has 750 edges and you want to reduce it to 450 edges,
    /// set targetEdges=450. The layer will learn to remove the 300 least important edges.
    /// </para>
    /// </remarks>
    public MeshPoolLayer(
        int inputChannels,
        int targetEdges,
        int numNeighbors = 4)
        : base([inputChannels], [inputChannels], (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (targetEdges <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetEdges), "Target edges must be positive.");
        if (numNeighbors <= 0)
            throw new ArgumentOutOfRangeException(nameof(numNeighbors), "Number of neighbors must be positive.");

        InputChannels = inputChannels;
        TargetEdges = targetEdges;
        _numNeighbors = numNeighbors;

        _importanceWeights = new Tensor<T>([inputChannels]);
        InitializeWeights();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes importance weights with small random values.
    /// </summary>
    private void InitializeWeights()
    {
        // === Vectorized: Weight initialization using TensorRandomUniformRange (Phase C: New IEngine methods) ===
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(InputChannels));

        // Initialize importance weights in [-scale, scale] range
        _importanceWeights = Engine.TensorRandomUniformRange<T>([InputChannels], NumOps.Negate(scale), scale);
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of mesh pooling.
    /// </summary>
    /// <param name="input">Edge features tensor of shape [numEdges, InputChannels].</param>
    /// <returns>Pooled edge features of shape [TargetEdges, InputChannels].</returns>
    /// <exception cref="ArgumentException">Thrown when input has invalid shape.</exception>
    /// <exception cref="InvalidOperationException">Thrown when edge adjacency is not set.</exception>
    /// <remarks>
    /// <para>
    /// This method requires edge adjacency to be set via <see cref="SetEdgeAdjacency"/> before calling.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != InputChannels)
        {
            throw new ArgumentException(
                $"MeshPoolLayer expects input shape [numEdges, {InputChannels}], got [{string.Join(", ", input.Shape)}].",
                nameof(input));
        }

        if (_lastEdgeAdjacency == null)
        {
            throw new InvalidOperationException(
                "Edge adjacency must be set via SetEdgeAdjacency before calling Forward.");
        }

        _lastInput = input;

        int numEdges = input.Shape[0];
        int numToKeep = Math.Min(TargetEdges, numEdges);

        var importanceScores = ComputeImportanceScores(input);
        _lastImportanceScores = importanceScores;

        var sortedIndices = SortEdgesByImportance(importanceScores, numEdges);

        RemainingEdgeIndices = new int[numToKeep];
        for (int i = 0; i < numToKeep; i++)
        {
            RemainingEdgeIndices[i] = sortedIndices[numEdges - 1 - i];
        }
        Array.Sort(RemainingEdgeIndices);

        var outputData = new T[numToKeep * InputChannels];
        for (int i = 0; i < numToKeep; i++)
        {
            int srcEdge = RemainingEdgeIndices[i];
            for (int c = 0; c < InputChannels; c++)
            {
                outputData[i * InputChannels + c] = input[srcEdge, c];
            }
        }

        UpdatedAdjacency = UpdateAdjacency(_lastEdgeAdjacency, RemainingEdgeIndices, numToKeep);

        var output = new Tensor<T>(outputData, [numToKeep, InputChannels]);
        _lastOutput = output;

        return output;
    }

    /// <summary>
    /// Sets the edge adjacency information for the current mesh.
    /// </summary>
    /// <param name="edgeAdjacency">
    /// A 2D array of shape [numEdges, NumNeighbors] containing neighbor edge indices.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when edgeAdjacency is null.</exception>
    public void SetEdgeAdjacency(int[,] edgeAdjacency)
    {
        if (edgeAdjacency == null)
            throw new ArgumentNullException(nameof(edgeAdjacency));

        _lastEdgeAdjacency = edgeAdjacency;
    }

    /// <summary>
    /// Computes importance scores for all edges using vectorized matrix-vector multiplication.
    /// </summary>
    /// <param name="input">Edge features [numEdges, InputChannels].</param>
    /// <returns>Importance scores [numEdges].</returns>
    /// <remarks>
    /// <para>
    /// Uses Engine.TensorMatMul to compute scores = input @ weights^T in a single vectorized operation.
    /// This provides significant speedup over element-wise loops on both CPU (via SIMD) and GPU.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeImportanceScores(Tensor<T> input)
    {
        // Reshape weights to [InputChannels, 1] for matrix-vector multiplication
        var weightsReshaped = _importanceWeights.Reshape(InputChannels, 1);

        // Compute scores = input @ weights (result is [numEdges, 1])
        var scores = Engine.TensorMatMul(input, weightsReshaped);

        // Squeeze to get [numEdges] shape
        return Engine.TensorSqueeze(scores, axis: 1);
    }

    /// <summary>
    /// Sorts edge indices by their importance scores.
    /// </summary>
    /// <param name="importanceScores">Importance scores for each edge.</param>
    /// <param name="numEdges">Number of edges.</param>
    /// <returns>Array of edge indices sorted by ascending importance.</returns>
    /// <remarks>
    /// <para>
    /// Note: Sorting is inherently sequential, but we extract scores in bulk using ToArray().
    /// Future optimization could use GPU-based sorting if available.
    /// </para>
    /// </remarks>
    private int[] SortEdgesByImportance(Tensor<T> importanceScores, int numEdges)
    {
        var indices = Enumerable.Range(0, numEdges).ToArray();

        // Extract scores to array in one operation
        var scoresArray = importanceScores.ToArray();
        var scoreValues = new double[numEdges];

        // Convert to double for sorting (vectorized when possible)
        for (int i = 0; i < numEdges; i++)
        {
            scoreValues[i] = NumOps.ToDouble(scoresArray[i]);
        }

        Array.Sort(scoreValues, indices);
        return indices;
    }

    /// <summary>
    /// Updates edge adjacency after removing edges.
    /// </summary>
    /// <param name="originalAdjacency">Original edge adjacency.</param>
    /// <param name="remainingIndices">Indices of edges that remain.</param>
    /// <param name="numToKeep">Number of edges to keep.</param>
    /// <returns>Updated adjacency for remaining edges.</returns>
    private int[,] UpdateAdjacency(int[,] originalAdjacency, int[] remainingIndices, int numToKeep)
    {
        var indexMap = new Dictionary<int, int>();
        for (int i = 0; i < numToKeep; i++)
        {
            indexMap[remainingIndices[i]] = i;
        }

        var newAdjacency = new int[numToKeep, _numNeighbors];

        for (int i = 0; i < numToKeep; i++)
        {
            int oldIdx = remainingIndices[i];
            for (int n = 0; n < _numNeighbors; n++)
            {
                int oldNeighbor = originalAdjacency[oldIdx, n];
                if (oldNeighbor >= 0 && indexMap.TryGetValue(oldNeighbor, out int newNeighbor))
                {
                    newAdjacency[i, n] = newNeighbor;
                }
                else
                {
                    newAdjacency[i, n] = -1;
                }
            }
        }

        return newAdjacency;
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass for mesh pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses GPU for importance score computation (GEMM) and feature gathering.
    /// Sorting remains on CPU as it requires dynamic branching that is inefficient on GPU.
    /// The operation is:
    /// 1. scores = input @ importanceWeights (GPU GEMM)
    /// 2. sortedIndices = sort(scores) (CPU - inherently sequential)
    /// 3. output = gather(input, topKIndices) (GPU Gather)
    /// </para>
    /// </remarks>
    /// <param name="inputs">Input GPU tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor with pooled features.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_lastEdgeAdjacency == null)
        {
            throw new InvalidOperationException(
                "Edge adjacency must be set via SetEdgeAdjacency before calling ForwardGpu.");
        }

        var input = inputs[0];
        int[] shape = input.Shape;

        if (shape.Length != 2 || shape[1] != InputChannels)
        {
            throw new ArgumentException(
                $"MeshPoolLayer expects input shape [numEdges, {InputChannels}], got [{string.Join(", ", shape)}].");
        }

        int numEdges = shape[0];
        int numToKeep = Math.Min(TargetEdges, numEdges);

        // Upload importance weights to GPU
        using var weightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_importanceWeights.Data));

        // Compute importance scores on GPU: scores = input @ weights
        // [numEdges, InputChannels] @ [InputChannels, 1] -> [numEdges, 1]
        using var scoresBuffer = backend.AllocateBuffer(numEdges);
        backend.Gemm(input.Buffer, weightsBuffer, scoresBuffer, numEdges, 1, InputChannels);

        // Download scores to CPU for sorting (sorting is inherently sequential)
        var scoresData = backend.DownloadBuffer(scoresBuffer);

        // Cache importance scores for backward pass
        _lastImportanceScores = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(scoresData), [numEdges]);

        // Sort edges by importance on CPU
        var sortedIndices = new int[numEdges];
        var scoreValues = new double[numEdges];
        for (int i = 0; i < numEdges; i++)
        {
            sortedIndices[i] = i;
            scoreValues[i] = scoresData[i];
        }
        Array.Sort(scoreValues, sortedIndices);

        // Select top-k edges (highest importance = last after ascending sort)
        RemainingEdgeIndices = new int[numToKeep];
        for (int i = 0; i < numToKeep; i++)
        {
            RemainingEdgeIndices[i] = sortedIndices[numEdges - 1 - i];
        }
        Array.Sort(RemainingEdgeIndices);

        // Download input for caching and CPU operations
        var inputData = backend.DownloadBuffer(input.Buffer);
        _lastInput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(inputData), shape);

        // Gather output features using CPU (since we have the data downloaded)
        var outputData = new float[numToKeep * InputChannels];
        for (int i = 0; i < numToKeep; i++)
        {
            int srcEdge = RemainingEdgeIndices[i];
            int srcOffset = srcEdge * InputChannels;
            int dstOffset = i * InputChannels;
            for (int c = 0; c < InputChannels; c++)
            {
                outputData[dstOffset + c] = inputData[srcOffset + c];
            }
        }

        // Update adjacency for remaining edges (CPU - dynamic topology)
        UpdatedAdjacency = UpdateAdjacency(_lastEdgeAdjacency, RemainingEdgeIndices, numToKeep);

        // Upload output to GPU
        var outputBuffer = backend.AllocateBuffer(outputData);

        // Cache output
        _lastOutput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputData), [numToKeep, InputChannels]);

        return new GpuTensor<T>(backend, outputBuffer, [numToKeep, InputChannels], GpuTensorRole.Activation, ownsBuffer: true);
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass for mesh pooling using vectorized scatter operations.
    /// </summary>
    /// <param name="outputGradient">Gradient with respect to pooled output.</param>
    /// <returns>Gradient with respect to input (sparse, only at kept edges).</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    /// <remarks>
    /// <para>
    /// Uses Engine.TensorScatterAdd to efficiently scatter gradients back to their original positions.
    /// This is much faster than element-wise loops, especially on GPU.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || RemainingEdgeIndices == null || _lastImportanceScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int numEdges = _lastInput.Shape[0];
        int numKept = RemainingEdgeIndices.Length;

        // Create zero-initialized input gradient tensor
        var inputGrad = new Tensor<T>([numEdges, InputChannels]);

        // Create indices tensor for scatter operation
        var indicesTensor = new Tensor<int>(RemainingEdgeIndices, [numKept]);

        // Scatter-add gradients back to original positions
        inputGrad = Engine.TensorScatterAdd(inputGrad, indicesTensor, outputGradient, axis: 0);

        _importanceWeightsGradient = ComputeImportanceWeightsGradient(outputGradient, _lastInput, RemainingEdgeIndices);

        return inputGrad;
    }

    /// <summary>
    /// Computes gradient for importance weights using vectorized operations.
    /// </summary>
    /// <param name="outputGradient">Gradient from output.</param>
    /// <param name="input">Original input features.</param>
    /// <param name="remainingIndices">Indices of kept edges.</param>
    /// <returns>Gradient for importance weights.</returns>
    /// <remarks>
    /// <para>
    /// The gradient is computed as: sum over kept edges of (sum(grad) * input_features).
    /// We use vectorized operations to:
    /// 1. Gather kept edge features using TensorGather
    /// 2. Sum output gradients using ReduceSum
    /// 3. Compute weighted sum using TensorMatMul
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeImportanceWeightsGradient(Tensor<T> outputGradient, Tensor<T> input, int[] remainingIndices)
    {
        int numKept = remainingIndices.Length;

        // Create indices tensor for gathering
        var indicesTensor = new Tensor<int>(remainingIndices, [numKept]);

        // Gather the kept edge features from input
        var keptFeatures = Engine.TensorGather(input, indicesTensor, axis: 0);

        // Sum output gradients along channels to get scale factors [numKept, 1]
        var gradScale = Engine.ReduceSum(outputGradient, [1], keepDims: true);

        // Element-wise multiply scales with features: [numKept, InputChannels]
        var scaledFeatures = Engine.TensorMultiply(keptFeatures, Engine.TensorTile(gradScale, [1, InputChannels]));

        // Sum over kept edges to get final gradient [InputChannels]
        return Engine.ReduceSum(scaledFeatures, [0], keepDims: false);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates the layer parameters using computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for gradient descent.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_importanceWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var scaledGrad = Engine.TensorMultiplyScalar(_importanceWeightsGradient, learningRate);
        _importanceWeights = Engine.TensorSubtract(_importanceWeights, scaledGrad);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>Vector containing importance weights.</returns>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(_importanceWeights.ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing importance weights.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != InputChannels)
            throw new ArgumentException($"Expected {InputChannels} parameters, got {parameters.Length}");

        _importanceWeights = new Tensor<T>(parameters.ToArray(), [InputChannels]);
    }

    /// <summary>
    /// Gets the importance weights tensor.
    /// </summary>
    /// <returns>The importance weights.</returns>
    public override Tensor<T> GetWeights() => _importanceWeights;

    /// <summary>
    /// Gets the bias tensor (null for this layer).
    /// </summary>
    /// <returns>Null as this layer has no biases.</returns>
    public override Tensor<T> GetBiases() => new Tensor<T>([0]);

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => InputChannels;

    /// <summary>
    /// Creates a deep copy of the layer.
    /// </summary>
    /// <returns>A new instance with identical configuration and parameters.</returns>
    public override LayerBase<T> Clone()
    {
        var copy = new MeshPoolLayer<T>(InputChannels, TargetEdges, _numNeighbors);
        copy.SetParameters(GetParameters());
        if (_lastEdgeAdjacency != null)
        {
            copy.SetEdgeAdjacency(_lastEdgeAdjacency);
        }
        return copy;
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets the cached state from forward/backward passes.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastImportanceScores = null;
        _importanceWeightsGradient = null;
        RemainingEdgeIndices = null;
        UpdatedAdjacency = null;
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the layer to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(TargetEdges);
        writer.Write(_numNeighbors);

        var weightArray = _importanceWeights.ToArray();
        for (int i = 0; i < weightArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(weightArray[i]));
        }
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputChannels = reader.ReadInt32();
        TargetEdges = reader.ReadInt32();
        var numNeighbors = reader.ReadInt32();

        _importanceWeights = new Tensor<T>([InputChannels]);
        var weightArray = new T[InputChannels];
        for (int i = 0; i < InputChannels; i++)
        {
            weightArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _importanceWeights = new Tensor<T>(weightArray, [InputChannels]);
    }

    #endregion

    #region JIT Compilation

    /// <summary>
    /// Exports the layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node representing the mesh pooling operation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer is not properly initialized.</exception>
    /// <remarks>
    /// <para>
    /// Mesh pooling is approximated in the computation graph using a learned attention-weighted
    /// aggregation. The edge importance scores are computed and used to weight the features
    /// before reduction, enabling gradient flow through the pooling operation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_importanceWeights == null)
            throw new InvalidOperationException("Layer importance weights not initialized.");

        // Create symbolic input for edge features [numEdges, features]
        var symbolicInput = new Tensor<T>(InputShape);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "mesh_pool_input");
        inputNodes.Add(inputNode);

        // Create importance weights as learnable parameters
        var importanceNode = TensorOperations<T>.Constant(_importanceWeights, "mesh_pool_importance");

        // Compute attention scores via linear transformation
        var scores = TensorOperations<T>.MatrixMultiply(inputNode, importanceNode);

        // Apply softmax to get attention weights
        var attentionWeights = TensorOperations<T>.Softmax(scores);

        // Weighted sum of features (attention-weighted pooling)
        var weightedFeatures = TensorOperations<T>.ElementwiseMultiply(inputNode, attentionWeights);

        // Reduce to get pooled output - use mean across edges
        var pooledOutput = TensorOperations<T>.Mean(weightedFeatures);

        return pooledOutput;
    }

    #endregion
}
