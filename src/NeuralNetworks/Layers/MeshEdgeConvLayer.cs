using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements edge convolution for mesh-based neural networks (MeshCNN style).
/// </summary>
/// <remarks>
/// <para>
/// MeshEdgeConvLayer processes triangle meshes by treating edges as the fundamental unit.
/// Each edge connects two triangular faces, and the layer learns to extract features
/// from the geometric relationship between these faces.
/// </para>
/// <para><b>For Beginners:</b> Think of a mesh as a surface made of connected triangles.
/// Each triangle edge is shared by (at most) two triangles. This layer examines each edge
/// and the two triangles it connects to learn meaningful features about the shape.
/// 
/// Key concepts:
/// - Edge: A line segment connecting two vertices, shared by up to 2 faces
/// - Dihedral angle: The angle between two faces sharing an edge
/// - Edge features: Properties like length, angles, and face normals
/// 
/// The layer learns to recognize patterns in how faces connect, enabling recognition
/// of shapes, surface curvature, and other geometric properties.
/// </para>
/// <para>
/// Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MeshEdgeConvLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the number of input feature channels per edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input channels represent features associated with each edge. Initial edge features
    /// typically include dihedral angle, edge length ratios, and inner angles (5 features).
    /// After processing, this can be any number of learned features.
    /// </para>
    /// </remarks>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets the number of output feature channels per edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each output channel corresponds to a learned filter that detects specific
    /// edge patterns. More channels allow learning more diverse patterns.
    /// </para>
    /// </remarks>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Gets the number of neighboring edges to consider for each edge convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In MeshCNN, each edge has 4 neighboring edges (2 from each adjacent face).
    /// The convolution aggregates features from these neighbors to capture local structure.
    /// </para>
    /// </remarks>
    public int NumNeighbors { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training (backpropagation).
    /// </summary>
    /// <value>Always <c>true</c> for MeshEdgeConvLayer as it has learnable parameters.</value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>true</c> if weights are initialized and activation can be JIT compiled.</value>
    public override bool SupportsJitCompilation => _weights != null && _biases != null && CanActivationBeJitted();

    #endregion

    #region Private Fields

    /// <summary>
    /// Learnable weights for edge convolution [OutputChannels, InputChannels * (1 + NumNeighbors)].
    /// </summary>
    private Tensor<T> _weights;

    /// <summary>
    /// Learnable bias values [OutputChannels].
    /// </summary>
    private Tensor<T> _biases;

    /// <summary>
    /// Cached weight gradients from backward pass.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Cached bias gradients from backward pass.
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Cached input from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached edge adjacency from the last forward pass.
    /// </summary>
    private int[,]? _lastEdgeAdjacency;

    /// <summary>
    /// Cached output before activation from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Cached output after activation from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshEdgeConvLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per edge.</param>
    /// <param name="outputChannels">Number of output feature channels per edge.</param>
    /// <param name="numNeighbors">Number of neighboring edges per edge. Default is 4 (standard MeshCNN).</param>
    /// <param name="activation">The activation function to apply. Defaults to ReLU.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates an edge convolution layer for processing mesh data.</para>
    /// <para>
    /// Typical usage:
    /// - inputChannels: 5 for initial edge features (dihedral angle, ratios, inner angles)
    /// - outputChannels: 32, 64, 128, etc. depending on network depth
    /// - numNeighbors: 4 (each edge has 4 neighboring edges in a triangular mesh)
    /// </para>
    /// </remarks>
    public MeshEdgeConvLayer(
        int inputChannels,
        int outputChannels,
        int numNeighbors = 4,
        IActivationFunction<T>? activation = null)
        : base(
            [inputChannels],
            [outputChannels],
            activation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, numNeighbors);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        NumNeighbors = numNeighbors;

        int weightInputSize = inputChannels * (1 + numNeighbors);
        _weights = new Tensor<T>([outputChannels, weightInputSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshEdgeConvLayer{T}"/> class with vector activation.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per edge.</param>
    /// <param name="outputChannels">Number of output feature channels per edge.</param>
    /// <param name="numNeighbors">Number of neighboring edges per edge. Default is 4.</param>
    /// <param name="vectorActivation">The vector activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Vector activation functions operate on entire vectors at once, which can be more efficient
    /// for operations like Softmax that need to consider all elements together.
    /// </para>
    /// </remarks>
    public MeshEdgeConvLayer(
        int inputChannels,
        int outputChannels,
        int numNeighbors = 4,
        IVectorActivationFunction<T>? vectorActivation = null)
        : base(
            [inputChannels],
            [outputChannels],
            vectorActivation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, numNeighbors);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        NumNeighbors = numNeighbors;

        int weightInputSize = inputChannels * (1 + numNeighbors);
        _weights = new Tensor<T>([outputChannels, weightInputSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    #endregion

    #region Static Helper Methods

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels.</param>
    /// <param name="numNeighbors">Number of neighbors.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is invalid.</exception>
    private static void ValidateParameters(int inputChannels, int outputChannels, int numNeighbors)
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "Output channels must be positive.");
        if (numNeighbors <= 0)
            throw new ArgumentOutOfRangeException(nameof(numNeighbors), "Number of neighbors must be positive.");
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes weights using He (Kaiming) initialization and biases to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// He initialization scales weights based on fan-in to prevent vanishing/exploding gradients.
    /// Formula: weight ~ N(0, sqrt(2 / fan_in))
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        int fanIn = InputChannels * (1 + NumNeighbors);
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));
        double scaleDouble = NumOps.ToDouble(scale);

        var random = RandomHelper.CreateSecureRandom();
        var weightData = _weights.ToArray();

        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] = NumOps.FromDouble((random.NextDouble() * 2.0 - 1.0) * scaleDouble);
        }
        _weights = new Tensor<T>(weightData, _weights.Shape);

        var biasData = new T[OutputChannels];
        for (int i = 0; i < biasData.Length; i++)
        {
            biasData[i] = NumOps.Zero;
        }
        _biases = new Tensor<T>(biasData, _biases.Shape);
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of edge convolution.
    /// </summary>
    /// <param name="input">
    /// Edge features tensor of shape [numEdges, InputChannels].
    /// </param>
    /// <returns>
    /// Output tensor of shape [numEdges, OutputChannels] after convolution and activation.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when input has invalid shape.</exception>
    /// <exception cref="InvalidOperationException">Thrown when edge adjacency is not set.</exception>
    /// <remarks>
    /// <para>
    /// This method requires edge adjacency to be set via <see cref="SetEdgeAdjacency"/> before calling.
    /// The convolution aggregates features from each edge and its neighbors.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != InputChannels)
        {
            throw new ArgumentException(
                $"MeshEdgeConvLayer expects input shape [numEdges, {InputChannels}], got [{string.Join(", ", input.Shape)}].",
                nameof(input));
        }

        if (_lastEdgeAdjacency == null)
        {
            throw new InvalidOperationException(
                "Edge adjacency must be set via SetEdgeAdjacency before calling Forward.");
        }

        _lastInput = input;

        int numEdges = input.Shape[0];
        int aggregatedFeatures = InputChannels * (1 + NumNeighbors);

        var aggregatedInput = AggregateEdgeFeatures(input, _lastEdgeAdjacency, numEdges, aggregatedFeatures);

        // Forward: output = aggregatedInput @ weights.T + biases
        var weightsTransposed = Engine.TensorTranspose(_weights);
        var preActivation = Engine.TensorMatMul(aggregatedInput, weightsTransposed);
        preActivation = AddBiases(preActivation);
        _lastPreActivation = preActivation;

        var output = ApplyActivation(preActivation);
        _lastOutput = output;

        return output;
    }

    /// <summary>
    /// Sets the edge adjacency information for the current mesh.
    /// </summary>
    /// <param name="edgeAdjacency">
    /// A 2D array of shape [numEdges, NumNeighbors] containing neighbor edge indices.
    /// Use -1 for boundary edges with fewer neighbors.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when edgeAdjacency is null.</exception>
    /// <exception cref="ArgumentException">Thrown when second dimension doesn't match NumNeighbors.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells the layer how edges are connected in your mesh.</para>
    /// <para>
    /// For each edge, provide the indices of its neighboring edges. In a standard triangular mesh,
    /// each edge has 4 neighbors (2 per adjacent face). For boundary edges with fewer neighbors,
    /// use -1 as a placeholder.
    /// </para>
    /// </remarks>
    public void SetEdgeAdjacency(int[,] edgeAdjacency)
    {
        if (edgeAdjacency == null)
            throw new ArgumentNullException(nameof(edgeAdjacency));

        if (edgeAdjacency.GetLength(1) != NumNeighbors)
        {
            throw new ArgumentException(
                $"Edge adjacency must have {NumNeighbors} neighbors per edge, got {edgeAdjacency.GetLength(1)}.",
                nameof(edgeAdjacency));
        }

        _lastEdgeAdjacency = edgeAdjacency;
    }

    /// <summary>
    /// Aggregates features from each edge and its neighbors using vectorized operations.
    /// </summary>
    /// <param name="input">Input edge features [numEdges, InputChannels].</param>
    /// <param name="adjacency">Edge adjacency [numEdges, NumNeighbors].</param>
    /// <param name="numEdges">Number of edges.</param>
    /// <param name="aggregatedSize">Size of aggregated feature vector.</param>
    /// <returns>Aggregated features [numEdges, aggregatedSize].</returns>
    /// <remarks>
    /// <para>
    /// This method uses vectorized gather operations to efficiently collect neighbor features.
    /// The aggregated output concatenates self-features with neighbor features:
    /// [self_features | neighbor_1_features | neighbor_2_features | ... | neighbor_N_features]
    /// </para>
    /// </remarks>
    private Tensor<T> AggregateEdgeFeatures(Tensor<T> input, int[,] adjacency, int numEdges, int aggregatedSize)
    {
        // Create result tensor [numEdges, aggregatedSize]
        var result = new Tensor<T>([numEdges, aggregatedSize]);

        // Step 1: Copy self-features (first InputChannels columns)
        // Use TensorSetSlice to copy input to the first InputChannels columns of result
        var selfSlice = Engine.TensorSlice(result, [0, 0], [numEdges, InputChannels]);
        Engine.TensorCopy(input, selfSlice);

        // Step 2: Gather neighbor features for each neighbor position
        for (int n = 0; n < NumNeighbors; n++)
        {
            int featureOffset = InputChannels * (1 + n);

            // Create indices tensor for this neighbor position
            var neighborIndices = new int[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                int idx = adjacency[e, n];
                // Clamp invalid indices to 0 and we'll zero them out after
                neighborIndices[e] = (idx >= 0 && idx < numEdges) ? idx : 0;
            }

            var indicesTensor = new Tensor<int>(neighborIndices, [numEdges]);

            // Gather neighbor features using vectorized operation
            var gathered = Engine.TensorGather(input, indicesTensor, axis: 0);

            // Create mask for invalid neighbors and zero them out
            var maskData = new T[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                int idx = adjacency[e, n];
                maskData[e] = (idx >= 0 && idx < numEdges) ? NumOps.One : NumOps.Zero;
            }
            var mask = new Tensor<T>(maskData, [numEdges, 1]);

            // Apply mask (multiply gathered features by mask to zero out invalid neighbors)
            gathered = Engine.TensorMultiply(gathered, Engine.TensorTile(mask, [1, InputChannels]));

            // Set the gathered features into the result at the appropriate offset
            Engine.TensorSetSlice(result, gathered, [0, featureOffset]);
        }

        return result;
    }

    /// <summary>
    /// Adds bias values to each output channel using vectorized operations.
    /// </summary>
    /// <param name="convOutput">The convolution output tensor [numEdges, OutputChannels].</param>
    /// <returns>Tensor with biases added to each channel.</returns>
    private Tensor<T> AddBiases(Tensor<T> convOutput)
    {
        int numEdges = convOutput.Shape[0];
        var biasExpanded = _biases.Reshape(1, OutputChannels);
        return Engine.TensorBroadcastAdd(convOutput, biasExpanded);
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass to compute gradients for training.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    /// <remarks>
    /// <para>
    /// Routes to manual or autodiff implementation based on UseAutodiff property.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to output.</param>
    /// <returns>The gradient of the loss with respect to input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastPreActivation == null || _lastOutput == null || _lastEdgeAdjacency == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        int numEdges = _lastInput.Shape[0];
        int aggregatedFeatures = InputChannels * (1 + NumNeighbors);

        var aggregatedInput = AggregateEdgeFeatures(_lastInput, _lastEdgeAdjacency, numEdges, aggregatedFeatures);
        
        // Compute weight gradient: delta.T @ aggregatedInput
        var deltaTransposed = Engine.TensorTranspose(delta);
        _weightsGradient = Engine.TensorMatMul(deltaTransposed, aggregatedInput);
        _weightsGradient = _weightsGradient.Reshape(_weights.Shape);

        _biasesGradient = ComputeBiasGradient(delta);

        // Compute input gradient: delta @ weights
        var aggregatedGrad = Engine.TensorMatMul(delta, _weights);
        var inputGrad = ScatterGradients(aggregatedGrad, _lastEdgeAdjacency, numEdges);

        return inputGrad;
    }

    /// <summary>
    /// Backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to output.</param>
    /// <returns>The gradient of the loss with respect to input.</returns>
    /// <remarks>
    /// <para>
    /// Currently routes to manual implementation. Full autodiff integration pending
    /// the addition of mesh-specific operations to the computation graph.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // TODO: Implement proper autodiff when mesh graph operations are available
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Computes the bias gradient by summing gradients over edges using vectorized reduction.
    /// </summary>
    /// <param name="delta">The gradient tensor [numEdges, OutputChannels].</param>
    /// <returns>Bias gradient tensor [OutputChannels].</returns>
    /// <remarks>
    /// <para>
    /// Uses Engine.ReduceSum for vectorized summation along the edge dimension (axis 0).
    /// This provides significant speedup on both CPU (via SIMD) and GPU.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeBiasGradient(Tensor<T> delta)
    {
        // Use vectorized reduce sum along axis 0 (sum over all edges)
        return Engine.ReduceSum(delta, [0], keepDims: false);
    }

    /// <summary>
    /// Scatters gradients from aggregated features back to original edges using vectorized operations.
    /// </summary>
    /// <param name="aggregatedGrad">Gradient of aggregated features [numEdges, aggregatedFeatures].</param>
    /// <param name="adjacency">Edge adjacency [numEdges, NumNeighbors].</param>
    /// <param name="numEdges">Number of edges.</param>
    /// <returns>Input gradient tensor [numEdges, InputChannels].</returns>
    /// <remarks>
    /// <para>
    /// This method scatters gradients back using vectorized operations:
    /// 1. Self-gradients: Direct slice copy from first InputChannels columns
    /// 2. Neighbor gradients: Use TensorScatterAdd for each neighbor position
    /// </para>
    /// </remarks>
    private Tensor<T> ScatterGradients(Tensor<T> aggregatedGrad, int[,] adjacency, int numEdges)
    {
        // Initialize input gradient tensor
        var inputGrad = new Tensor<T>([numEdges, InputChannels]);

        // Step 1: Add self-gradients (first InputChannels columns)
        var selfGrad = Engine.TensorSlice(aggregatedGrad, [0, 0], [numEdges, InputChannels]);
        inputGrad = Engine.TensorAdd(inputGrad, selfGrad);

        // Step 2: Scatter neighbor gradients for each neighbor position
        for (int n = 0; n < NumNeighbors; n++)
        {
            int featureOffset = InputChannels * (1 + n);

            // Extract gradient slice for this neighbor position
            var neighborGrad = Engine.TensorSlice(aggregatedGrad, [0, featureOffset], [numEdges, InputChannels]);

            // Create indices tensor for scatter
            var neighborIndices = new int[numEdges];
            var validMask = new T[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                int idx = adjacency[e, n];
                if (idx >= 0 && idx < numEdges)
                {
                    neighborIndices[e] = idx;
                    validMask[e] = NumOps.One;
                }
                else
                {
                    neighborIndices[e] = 0; // Won't matter since mask is zero
                    validMask[e] = NumOps.Zero;
                }
            }

            var indicesTensor = new Tensor<int>(neighborIndices, [numEdges]);

            // Mask the gradients to zero out invalid neighbors
            var mask = new Tensor<T>(validMask, [numEdges, 1]);
            var maskedGrad = Engine.TensorMultiply(neighborGrad, Engine.TensorTile(mask, [1, InputChannels]));

            // Scatter-add the gradients back to their original positions
            inputGrad = Engine.TensorScatterAdd(inputGrad, indicesTensor, maskedGrad, axis: 0);
        }

        return inputGrad;
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates the layer parameters using the computed gradients and learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for gradient descent.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called.</exception>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var scaledWeightGrad = Engine.TensorMultiplyScalar(_weightsGradient, learningRate);
        _weights = Engine.TensorSubtract(_weights, scaledWeightGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
        _biases = Engine.TensorSubtract(_biases, scaledBiasGrad);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all weight and bias parameters.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters (weights followed by biases).</param>
    /// <exception cref="ArgumentException">Thrown when parameter count does not match expected.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _weights.Length + _biases.Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, but got {parameters.Length}");

        int index = 0;
        _weights = new Tensor<T>(_weights.Shape, parameters.Slice(index, _weights.Length));
        index += _weights.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(index, _biases.Length));
    }

    /// <summary>
    /// Gets the weight tensor.
    /// </summary>
    /// <returns>The weights tensor.</returns>
    public override Tensor<T> GetWeights() => _weights;

    /// <summary>
    /// Gets the bias tensor.
    /// </summary>
    /// <returns>The bias tensor.</returns>
    public override Tensor<T> GetBiases() => _biases;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => _weights.Length + _biases.Length;

    /// <summary>
    /// Creates a deep copy of the layer.
    /// </summary>
    /// <returns>A new instance with identical configuration and parameters.</returns>
    public override LayerBase<T> Clone()
    {
        MeshEdgeConvLayer<T> copy;

        if (UsingVectorActivation)
        {
            copy = new MeshEdgeConvLayer<T>(InputChannels, OutputChannels, NumNeighbors, VectorActivation);
        }
        else
        {
            copy = new MeshEdgeConvLayer<T>(InputChannels, OutputChannels, NumNeighbors, ScalarActivation);
        }

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
        _lastPreActivation = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
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
        writer.Write(OutputChannels);
        writer.Write(NumNeighbors);

        var weightArray = _weights.ToArray();
        for (int i = 0; i < weightArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(weightArray[i]));
        }

        var biasArray = _biases.ToArray();
        for (int i = 0; i < biasArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(biasArray[i]));
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
        OutputChannels = reader.ReadInt32();
        NumNeighbors = reader.ReadInt32();

        int weightSize = OutputChannels * InputChannels * (1 + NumNeighbors);
        _weights = new Tensor<T>([OutputChannels, InputChannels * (1 + NumNeighbors)]);
        var weightArray = new T[weightSize];
        for (int i = 0; i < weightSize; i++)
        {
            weightArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _weights = new Tensor<T>(weightArray, _weights.Shape);

        _biases = new Tensor<T>([OutputChannels]);
        var biasArray = new T[OutputChannels];
        for (int i = 0; i < OutputChannels; i++)
        {
            biasArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _biases = new Tensor<T>(biasArray, _biases.Shape);
    }

    #endregion

    #region JIT Compilation

    /// <summary>
    /// Exports the layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <exception cref="NotSupportedException">Thrown because MeshEdgeConv JIT is not yet implemented.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "MeshEdgeConvLayer.ExportComputationGraph requires graph-specific operations " +
            "for edge aggregation which are not yet available in TensorOperations.");
    }

    #endregion
}
