using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements spiral convolution for mesh vertex processing.
/// </summary>
/// <remarks>
/// <para>
/// SpiralConvLayer operates on mesh vertices by aggregating features from neighbors
/// in a consistent spiral ordering. This enables translation-equivariant convolutions
/// on irregular mesh structures by defining a canonical ordering of vertex neighbors.
/// </para>
/// <para><b>For Beginners:</b> Unlike image convolutions where neighbors are in a grid,
/// mesh vertices have irregular connectivity. Spiral convolution solves this by:
/// 
/// 1. Starting at each vertex
/// 2. Visiting neighbors in a consistent spiral pattern (like a clock hand)
/// 3. Gathering features from each neighbor in order
/// 4. Applying learned weights to the ordered features
/// 
/// This creates a consistent "template" for convolution regardless of mesh topology.
/// 
/// Applications:
/// - 3D shape analysis and classification
/// - Facial expression recognition
/// - Body pose estimation
/// - Medical surface analysis
/// </para>
/// <para>
/// Reference: "Neural 3D Morphable Models: Spiral Convolutional Networks" by Bouritsas et al.
/// Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpiralConvLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the number of input feature channels per vertex.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input channels represent features at each vertex. For 3D coordinates, this is 3.
    /// After processing, this can include normals, colors, or learned features.
    /// </para>
    /// </remarks>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets the number of output feature channels per vertex.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each output channel represents a learned feature detector. More channels
    /// enable learning more diverse vertex patterns.
    /// </para>
    /// </remarks>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Gets the spiral sequence length (number of neighbors in the spiral).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The spiral length determines how many neighbors are considered for each vertex.
    /// Longer spirals capture more context but increase computation. Typical values
    /// are 9-16 for detailed meshes.
    /// </para>
    /// </remarks>
    public int SpiralLength { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training (backpropagation).
    /// </summary>
    /// <value>Always <c>true</c> for SpiralConvLayer as it has learnable parameters.</value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>true</c> if weights are initialized and activation can be JIT compiled.</value>
    public override bool SupportsJitCompilation => _weights != null && _biases != null && CanActivationBeJitted();

    #endregion

    #region Private Fields

    /// <summary>
    /// Learnable weights [OutputChannels, InputChannels * SpiralLength].
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
    /// Cached pre-activation output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Cached output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Spiral indices for each vertex [numVertices, SpiralLength].
    /// </summary>
    private int[,]? _spiralIndices;

    /// <summary>
    /// Cached gathered neighbor features for backward pass.
    /// </summary>
    private Tensor<T>? _gatheredFeatures;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="SpiralConvLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per vertex.</param>
    /// <param name="outputChannels">Number of output feature channels per vertex.</param>
    /// <param name="spiralLength">Length of the spiral sequence (number of neighbors).</param>
    /// <param name="numVertices">Expected number of vertices (for shape calculation).</param>
    /// <param name="activation">Activation function to apply. Defaults to ReLU.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a spiral convolution layer for mesh processing.</para>
    /// <para>
    /// The spiral indices must be set via <see cref="SetSpiralIndices"/> before forward pass.
    /// </para>
    /// </remarks>
    public SpiralConvLayer(
        int inputChannels,
        int outputChannels,
        int spiralLength,
        int numVertices = 1,
        IActivationFunction<T>? activation = null)
        : base(
            [numVertices, inputChannels],
            [numVertices, outputChannels],
            activation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, spiralLength);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        SpiralLength = spiralLength;

        int weightSize = inputChannels * spiralLength;
        _weights = new Tensor<T>([outputChannels, weightSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpiralConvLayer{T}"/> class with vector activation.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per vertex.</param>
    /// <param name="outputChannels">Number of output feature channels per vertex.</param>
    /// <param name="spiralLength">Length of the spiral sequence (number of neighbors).</param>
    /// <param name="numVertices">Expected number of vertices (for shape calculation).</param>
    /// <param name="vectorActivation">Vector activation function to apply.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    public SpiralConvLayer(
        int inputChannels,
        int outputChannels,
        int spiralLength,
        int numVertices = 1,
        IVectorActivationFunction<T>? vectorActivation = null)
        : base(
            [numVertices, inputChannels],
            [numVertices, outputChannels],
            vectorActivation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, spiralLength);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        SpiralLength = spiralLength;

        int weightSize = inputChannels * spiralLength;
        _weights = new Tensor<T>([outputChannels, weightSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels.</param>
    /// <param name="spiralLength">Spiral sequence length.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is non-positive.</exception>
    private static void ValidateParameters(int inputChannels, int outputChannels, int spiralLength)
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "Output channels must be positive.");
        if (spiralLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(spiralLength), "Spiral length must be positive.");
    }

    /// <summary>
    /// Initializes weights using He (Kaiming) initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// He initialization scales weights based on fan-in to prevent vanishing/exploding gradients.
    /// Formula: weight ~ N(0, sqrt(2 / fan_in))
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        int fanIn = InputChannels * SpiralLength;
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));
        double scaleDouble = NumOps.ToDouble(scale);

        var random = RandomHelper.CreateSecureRandom();
        var weightData = new T[_weights.Length];

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

    #region Spiral Configuration

    /// <summary>
    /// Sets the spiral indices for the mesh being processed.
    /// </summary>
    /// <param name="spiralIndices">
    /// A 2D array of shape [numVertices, SpiralLength] containing neighbor vertex indices
    /// in spiral order for each vertex.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when spiralIndices is null.</exception>
    /// <exception cref="ArgumentException">Thrown when spiral length doesn't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before processing a mesh, you must define the spiral
    /// ordering of neighbors for each vertex. This method sets that ordering.</para>
    /// <para>
    /// The spiral indices should be computed from the mesh topology using a consistent
    /// algorithm that starts at a fixed reference direction and visits neighbors in order.
    /// </para>
    /// </remarks>
    public void SetSpiralIndices(int[,] spiralIndices)
    {
        if (spiralIndices == null)
            throw new ArgumentNullException(nameof(spiralIndices));

        if (spiralIndices.GetLength(1) != SpiralLength)
        {
            throw new ArgumentException(
                $"Spiral indices second dimension ({spiralIndices.GetLength(1)}) " +
                $"must match SpiralLength ({SpiralLength}).",
                nameof(spiralIndices));
        }

        _spiralIndices = spiralIndices;
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of spiral convolution.
    /// </summary>
    /// <param name="input">
    /// Vertex features tensor with shape [numVertices, InputChannels] or
    /// [batch, numVertices, InputChannels].
    /// </param>
    /// <returns>
    /// Output tensor with shape [numVertices, OutputChannels] or
    /// [batch, numVertices, OutputChannels].
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when spiral indices are not set.</exception>
    /// <exception cref="ArgumentException">Thrown when input has invalid shape.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Gathers neighbor features according to spiral indices
    /// 2. Concatenates gathered features into a flat vector per vertex
    /// 3. Applies linear transformation (weights + bias)
    /// 4. Applies activation function
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_spiralIndices == null)
        {
            throw new InvalidOperationException(
                "Spiral indices must be set via SetSpiralIndices before calling Forward.");
        }

        _lastInput = input;

        bool hasBatch = input.Rank == 3;
        int numVertices = hasBatch ? input.Shape[1] : input.Shape[0];
        int inputChannels = hasBatch ? input.Shape[2] : input.Shape[1];

        if (inputChannels != InputChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inputChannels}) must match layer InputChannels ({InputChannels}).",
                nameof(input));
        }

        Tensor<T> output;

        if (hasBatch)
        {
            int batchSize = input.Shape[0];
            output = ProcessBatched(input, batchSize, numVertices);
        }
        else
        {
            output = ProcessSingle(input, numVertices);
        }

        _lastPreActivation = output;
        var activated = ApplyActivation(output);
        _lastOutput = activated;

        return activated;
    }

    /// <summary>
    /// Processes a single (non-batched) input tensor.
    /// </summary>
    /// <param name="input">Input tensor [numVertices, InputChannels].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Output tensor [numVertices, OutputChannels].</returns>
    private Tensor<T> ProcessSingle(Tensor<T> input, int numVertices)
    {
        var gathered = GatherSpiralFeatures(input, numVertices);
        _gatheredFeatures = gathered;

        var transposedWeights = Engine.TensorTranspose(_weights);
        var output = Engine.TensorMatMul(gathered, transposedWeights);
        output = AddBiases(output, numVertices);

        return output;
    }

    /// <summary>
    /// Processes a batched input tensor.
    /// </summary>
    /// <param name="input">Input tensor [batch, numVertices, InputChannels].</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="numVertices">Number of vertices per sample.</param>
    /// <returns>Output tensor [batch, numVertices, OutputChannels].</returns>
    private Tensor<T> ProcessBatched(Tensor<T> input, int batchSize, int numVertices)
    {
        var outputData = new T[batchSize * numVertices * OutputChannels];
        
        // Thread-local storage for gathered features per batch sample
        var localGatheredFeatures = new Tensor<T>[batchSize];
        var transposedWeights = Engine.TensorTranspose(_weights);

        Parallel.For(0, batchSize, b =>
        {
            var singleInput = ExtractBatchSlice(input, b, numVertices);
            
            // Gather spiral features (thread-safe, result stored per-batch)
            var gathered = GatherSpiralFeatures(singleInput, numVertices);
            localGatheredFeatures[b] = gathered;
            
            // Compute output using pre-transposed weights
            var singleOutput = Engine.TensorMatMul(gathered, transposedWeights);
            singleOutput = AddBiases(singleOutput, numVertices);
            
            // Apply activation
            singleOutput = ApplyActivation(singleOutput);
            
            var singleData = singleOutput.ToArray();
            int offset = b * numVertices * OutputChannels;
            Array.Copy(singleData, 0, outputData, offset, singleData.Length);
        });

        // Combine gathered features for backward pass (sum across batch)
        // For backward pass, we need the gathered features. Store the first batch's for gradient computation.
        // A more complete solution would store all or use a different gradient strategy.
        if (batchSize > 0 && localGatheredFeatures[0] != null)
        {
            _gatheredFeatures = CombineGatheredFeatures(localGatheredFeatures, batchSize, numVertices);
        }

        return new Tensor<T>(outputData, [batchSize, numVertices, OutputChannels]);
    }

    /// <summary>
    /// Combines gathered features from all batch samples for backward pass.
    /// </summary>
    /// <param name="localGatheredFeatures">Array of gathered features per batch sample.</param>
    /// <param name="batchSize">Number of batch samples.</param>
    /// <param name="numVertices">Number of vertices per sample.</param>
    /// <returns>Combined gathered features tensor.</returns>
    private Tensor<T> CombineGatheredFeatures(Tensor<T>[] localGatheredFeatures, int batchSize, int numVertices)
    {
        int featureDim = SpiralLength * InputChannels;
        var combinedData = new T[batchSize * numVertices * featureDim];
        
        for (int b = 0; b < batchSize; b++)
        {
            var batchData = localGatheredFeatures[b].ToArray();
            int offset = b * numVertices * featureDim;
            Array.Copy(batchData, 0, combinedData, offset, batchData.Length);
        }
        
        return new Tensor<T>(combinedData, [batchSize, numVertices, featureDim]);
    }

    /// <summary>
    /// Extracts a single sample from a batched tensor.
    /// </summary>
    /// <param name="batched">Batched tensor [batch, vertices, channels].</param>
    /// <param name="batchIndex">Index of the batch to extract.</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Single sample tensor [vertices, channels].</returns>
    private Tensor<T> ExtractBatchSlice(Tensor<T> batched, int batchIndex, int numVertices)
    {
        int channels = batched.Shape[2];
        var data = new T[numVertices * channels];

        for (int v = 0; v < numVertices; v++)
        {
            for (int c = 0; c < channels; c++)
            {
                data[v * channels + c] = batched[batchIndex, v, c];
            }
        }

        return new Tensor<T>(data, [numVertices, channels]);
    }

    /// <summary>
    /// Gathers features from neighbors according to spiral indices.
    /// </summary>
    /// <param name="input">Input vertex features [numVertices, InputChannels].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Gathered features [numVertices, InputChannels * SpiralLength].</returns>
    /// <remarks>
    /// <para>
    /// Uses vectorized TensorGather operations to efficiently collect neighbor features.
    /// Each spiral position is gathered independently and concatenated.
    /// </para>
    /// </remarks>
    private Tensor<T> GatherSpiralFeatures(Tensor<T> input, int numVertices)
    {
        if (_spiralIndices == null)
            throw new InvalidOperationException("Spiral indices not set.");

        int gatheredSize = InputChannels * SpiralLength;

        // Create result tensor
        var gathered = new Tensor<T>([numVertices, gatheredSize]);

        // Gather features for each spiral position using vectorized operations
        for (int s = 0; s < SpiralLength; s++)
        {
            int featureOffset = s * InputChannels;

            // Create indices for this spiral position
            var spiralPositionIndices = new int[numVertices];
            var validMask = new T[numVertices];

            for (int v = 0; v < numVertices; v++)
            {
                int idx = _spiralIndices[v, s];
                if (idx >= 0 && idx < numVertices)
                {
                    spiralPositionIndices[v] = idx;
                    validMask[v] = NumOps.One;
                }
                else
                {
                    spiralPositionIndices[v] = 0; // Placeholder, will be masked to zero
                    validMask[v] = NumOps.Zero;
                }
            }

            var indicesTensor = new Tensor<int>(spiralPositionIndices, [numVertices]);

            // Gather neighbor features
            var neighborFeatures = Engine.TensorGather(input, indicesTensor, axis: 0);

            // Apply mask to zero out invalid neighbors
            var mask = new Tensor<T>(validMask, [numVertices, 1]);
            neighborFeatures = Engine.TensorMultiply(neighborFeatures, Engine.TensorTile(mask, [1, InputChannels]));

            // Set the gathered features into the result at the appropriate offset
            Engine.TensorSetSlice(gathered, neighborFeatures, [0, featureOffset]);
        }

        return gathered;
    }

    /// <summary>
    /// Adds biases to each vertex output using vectorized broadcast.
    /// </summary>
    /// <param name="output">Output tensor [numVertices, OutputChannels].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Output with biases added.</returns>
    private Tensor<T> AddBiases(Tensor<T> output, int numVertices)
    {
        var biasExpanded = _biases.Reshape(1, OutputChannels);
        return Engine.TensorBroadcastAdd(output, biasExpanded);
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass to compute gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastPreActivation == null || _lastOutput == null || _gatheredFeatures == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_spiralIndices == null)
            throw new InvalidOperationException("Spiral indices not set.");

        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        bool hasBatch = _lastInput.Rank == 3;
        int numVertices = hasBatch ? _lastInput.Shape[1] : _lastInput.Shape[0];

        var transposedDelta = Engine.TensorTranspose(delta);
        _weightsGradient = Engine.TensorMatMul(transposedDelta, _gatheredFeatures);
        _biasesGradient = Engine.ReduceSum(delta, [0], keepDims: false);

        var gatheredGrad = Engine.TensorMatMul(delta, _weights);

        var inputGrad = ScatterSpiralGradients(gatheredGrad, numVertices);

        if (hasBatch)
        {
            int batchSize = _lastInput.Shape[0];
            inputGrad = inputGrad.Reshape(batchSize, numVertices, InputChannels);
        }

        return inputGrad;
    }

    /// <summary>
    /// Backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    /// <remarks>
    /// <para>
    /// Currently routes to manual implementation. Full autodiff integration pending
    /// the addition of spiral-specific operations to the computation graph.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // TODO: Implement proper autodiff when spiral graph operations are available
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Scatters gradients back to input vertices according to spiral indices using vectorized operations.
    /// </summary>
    /// <param name="gatheredGrad">Gradients for gathered features [numVertices, InputChannels * SpiralLength].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Input gradients [numVertices, InputChannels].</returns>
    /// <remarks>
    /// <para>
    /// Uses vectorized TensorScatterAdd operations to efficiently scatter gradients back.
    /// Each spiral position is processed independently for better parallelism.
    /// </para>
    /// </remarks>
    private Tensor<T> ScatterSpiralGradients(Tensor<T> gatheredGrad, int numVertices)
    {
        if (_spiralIndices == null)
            throw new InvalidOperationException("Spiral indices not set.");

        // Initialize input gradient tensor
        var inputGrad = new Tensor<T>([numVertices, InputChannels]);

        // Scatter gradients for each spiral position
        for (int s = 0; s < SpiralLength; s++)
        {
            int featureOffset = s * InputChannels;

            // Extract gradient slice for this spiral position
            var spiralGrad = Engine.TensorSlice(gatheredGrad, [0, featureOffset], [numVertices, InputChannels]);

            // Create indices and mask for scatter
            var neighborIndices = new int[numVertices];
            var validMask = new T[numVertices];

            for (int v = 0; v < numVertices; v++)
            {
                int idx = _spiralIndices[v, s];
                if (idx >= 0 && idx < numVertices)
                {
                    neighborIndices[v] = idx;
                    validMask[v] = NumOps.One;
                }
                else
                {
                    neighborIndices[v] = 0; // Placeholder, masked out
                    validMask[v] = NumOps.Zero;
                }
            }

            var indicesTensor = new Tensor<int>(neighborIndices, [numVertices]);

            // Apply mask to zero out invalid gradients
            var mask = new Tensor<T>(validMask, [numVertices, 1]);
            var maskedGrad = Engine.TensorMultiply(spiralGrad, Engine.TensorTile(mask, [1, InputChannels]));

            // Scatter-add gradients back to original positions
            inputGrad = Engine.TensorScatterAdd(inputGrad, indicesTensor, maskedGrad, axis: 0);
        }

        return inputGrad;
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates layer parameters using computed gradients.
    /// </summary>
    /// <param name="learningRate">Learning rate for gradient descent.</param>
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
    /// <returns>Vector containing all weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Sets all trainable parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    /// <exception cref="ArgumentException">Thrown when parameter count is incorrect.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _weights.Length + _biases.Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int idx = 0;
        _weights = new Tensor<T>(_weights.Shape, parameters.Slice(idx, _weights.Length));
        idx += _weights.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(idx, _biases.Length));
    }

    /// <summary>
    /// Gets the weight tensor.
    /// </summary>
    /// <returns>Weights [OutputChannels, InputChannels * SpiralLength].</returns>
    public override Tensor<T> GetWeights() => _weights;

    /// <summary>
    /// Gets the bias tensor.
    /// </summary>
    /// <returns>Biases [OutputChannels].</returns>
    public override Tensor<T> GetBiases() => _biases;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => _weights.Length + _biases.Length;

    /// <summary>
    /// Creates a deep copy of this layer.
    /// </summary>
    /// <returns>A new SpiralConvLayer with identical configuration and parameters.</returns>
    public override LayerBase<T> Clone()
    {
        SpiralConvLayer<T> copy;

        if (UsingVectorActivation)
        {
            copy = new SpiralConvLayer<T>(
                InputChannels, OutputChannels, SpiralLength, InputShape[0], VectorActivation);
        }
        else
        {
            copy = new SpiralConvLayer<T>(
                InputChannels, OutputChannels, SpiralLength, InputShape[0], ScalarActivation);
        }

        copy.SetParameters(GetParameters());

        if (_spiralIndices != null)
        {
            copy.SetSpiralIndices(_spiralIndices);
        }

        return copy;
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets cached state from forward/backward passes.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _lastOutput = null;
        _gatheredFeatures = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the layer to a binary stream.
    /// </summary>
    /// <param name="writer">Binary writer for serialization.</param>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(SpiralLength);

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
    /// <param name="reader">Binary reader for deserialization.</param>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputChannels = reader.ReadInt32();
        OutputChannels = reader.ReadInt32();
        SpiralLength = reader.ReadInt32();

        int weightSize = InputChannels * SpiralLength;
        _weights = new Tensor<T>([OutputChannels, weightSize]);
        var weightArray = new T[_weights.Length];
        for (int i = 0; i < weightArray.Length; i++)
        {
            weightArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _weights = new Tensor<T>(weightArray, _weights.Shape);

        _biases = new Tensor<T>([OutputChannels]);
        var biasArray = new T[_biases.Length];
        for (int i = 0; i < biasArray.Length; i++)
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
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer is not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_weights == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        var symbolicInput = new Tensor<T>(InputShape);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "spiral_conv_input");
        inputNodes.Add(inputNode);

        var weightNode = TensorOperations<T>.Constant(_weights, "spiral_conv_weights");
        var biasNode = TensorOperations<T>.Constant(_biases, "spiral_conv_bias");

        var transposedWeights = TensorOperations<T>.Transpose(weightNode);
        var matmulNode = TensorOperations<T>.MatrixMultiply(inputNode, transposedWeights);
        var biasedNode = TensorOperations<T>.Add(matmulNode, biasNode);

        var activatedOutput = ApplyActivationToGraph(biasedNode);
        return activatedOutput;
    }

    #endregion
}
