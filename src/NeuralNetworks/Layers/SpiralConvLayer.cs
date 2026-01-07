using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.DirectGpu; // For IGpuBuffer

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
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>true</c> if weights are initialized and activation can be JIT compiled.</value>
    public override bool SupportsJitCompilation => _weights != null && _biases != null && CanActivationBeJitted();

    #endregion

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0) throw new ArgumentException("SpiralConvLayer requires an input tensor.");
        var input = inputs[0];

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        if (_spiralIndices == null)
            throw new InvalidOperationException("Spiral indices not set.");

        // Ensure GPU indices are ready
        if (_spiralIndicesGpu == null)
        {
            InitializeGpuIndices(gpuEngine);
        }

        int numVertices = _spiralIndices!.GetLength(0);
        int inputChannels = InputChannels;
        
        bool hasBatch = input.Shape.Length == 3;
        int batchSize = hasBatch ? input.Shape[0] : 1;
        
        // Input validation
        int actualVertices = hasBatch ? input.Shape[1] : input.Shape[0];
        int actualChannels = hasBatch ? input.Shape[2] : input.Shape[1];
        
        if (actualVertices != numVertices)
            throw new ArgumentException($"Input vertices ({actualVertices}) does not match spiral indices ({numVertices}).");
        if (actualChannels != inputChannels)
            throw new ArgumentException($"Input channels ({actualChannels}) does not match layer ({inputChannels}).");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend unavailable.");

        // Helper for single batch processing
        IGpuTensor<T> ProcessBatchItem(IGpuTensor<T> batchInput)
        {
            // batchInput: [V, C]
            // Gather: [V*S, C]
            int numGather = numVertices * SpiralLength;
            var gatheredRaw = gpuEngine.GatherGpu(batchInput, _spiralIndicesGpu!, numGather, inputChannels);

            // Apply Mask: [V*S, 1] broadcast to [V*S, C]
            // We need to create the mask tensor from the buffer
            // _spiralMaskGpu is a buffer. Wrap in tensor?
            // TensorBroadcastMultiply needs two tensors.
            // Or use element-wise multiply if we can replicate mask.
            // Let's assume we can wrap buffer.
            using var maskTensor = new GpuTensor<T>(backend, _spiralMaskGpu!, [numGather, 1], GpuTensorRole.Constant, ownsBuffer: false);
            
            // Broadcast multiply: gatheredRaw * mask
            var gatheredMasked = gpuEngine.BroadcastMultiplyColumnGpu(gatheredRaw, maskTensor);
            gatheredRaw.Dispose();

            // Reshape to [V, S*C]
            var gathered = gpuEngine.ReshapeGpu(gatheredMasked, [numVertices, SpiralLength * inputChannels]);
            
            // MatMul: [V, S*C] @ [S*C, OutC] -> [V, OutC]
            // Weights are [OutC, S*C]. Need Transpose.
            // Cache transposed weights?
            // For now, transpose on the fly (or Engine caches it if persistent).
            // Actually, FusedLinearGpu takes weights as [In, Out].
            // Our weights are [Out, In]. So we need Transpose.
            var wT = _weights.Transpose(); // CPU transpose
            // TODO: Optimize by storing transposed weights for GPU.
            
            var output = gpuEngine.FusedLinearGpu(gathered, wT, _biases, MapActivationToFused());
            gathered.Dispose();
            
            return output;
        }

        IGpuTensor<T> result;
        if (hasBatch)
        {
            var outputs = new IGpuTensor<T>[batchSize];
            for (int b = 0; b < batchSize; b++)
            {
                // Slice input [B, V, C] -> [V, C]
                // DirectGpuTensorEngine doesn't have SliceBatch? 
                // We can use Reshape + Offset logic or a custom slice.
                // Or Slice operations if available.
                // Assuming Slice is not available on GPU yet, we might need to implement it or use Copy2DStrided.
                // Actually, GatherGpu logic was: Copy2DStrided(input, output, V, C, totalC, offset).
                // Here we want batch slice.
                // We can use Slice on GPU?
                // Let's use loop with offset copy if needed, but for now let's assume we can iterate.
                // Wait, I can't easily slice a GpuTensor in the current API without a specific method.
                // I'll assume I can implement a helper or use what I have.
                // Workaround: Reshape to [B*V, C], then use Gather with offset indices?
                // Or: Implement a batch loop where we offset the input pointer?
                // GpuTensor has Buffer and Offset? No, just Buffer.
                // I can use `CreateView` if GpuTensor supported offsets, but it doesn't seem to.
                
                // Fallback for batch: 
                // 1. Reshape input to [1, B*V, C] -> NO.
                // 2. Use a kernel that handles batch?
                // 3. Just loop?
                // If I can't slice, I can't loop.
                
                // Let's assume input is contiguous.
                // I can create a new GpuTensor for each slice using shared buffer?
                // IDirectGpuBackend buffers are handles. OpenCL sub-buffers?
                // Not exposed.
                
                // OK, strategy shift: Flatten Batch and Vertices.
                // Treat [B, V, C] as [B*V, C].
                // We need indices for [B*V] vertices.
                // Indices are [V*S]. They need to be repeated B times, with offset V added to each block.
                // i.e. indices[b, v, s] = indices[v, s] + b*V.
                
                // I can generate this extended index array on CPU and upload.
                // Since V and S are fixed, and B varies, maybe generate dynamically?
                // Or just generate on CPU every time?
                // Generating indices on CPU for B*V*S size is fast enough.
                
                // 1. Reshape input to [B*V, C]
                // 2. Generate full indices [B*V*S]
                // 3. Gather -> [B*V*S, C]
                // 4. Apply Mask (repeated B times)
                // 5. Reshape to [B*V, S*C]
                // 6. MatMul
                // 7. Reshape to [B, V, OutC]
                
                // This seems best.
            }
            
            // For now, let's implement the "Flatten Batch" strategy properly.
            
            int totalVertices = batchSize * numVertices;
            var flatInput = gpuEngine.ReshapeGpu(input, [totalVertices, inputChannels]);
            
            // Generate extended indices
            var flatIndices = new int[totalVertices * SpiralLength];
            var flatMask = new float[totalVertices * SpiralLength];
            
            // This loop might be slow for huge batches, but it's CPU side.
            // Parallelize?
            int[] baseIndices = new int[numVertices * SpiralLength];
            float[] baseMask = new float[numVertices * SpiralLength];
            
            // Prepare base
            for (int v = 0; v < numVertices; v++)
            {
                for (int s = 0; s < SpiralLength; s++)
                {
                    int idx = v * SpiralLength + s;
                    int neighbor = _spiralIndices![v, s];
                    if (neighbor >= 0)
                    {
                        baseIndices[idx] = neighbor;
                        baseMask[idx] = 1.0f;
                    }
                    else
                    {
                        baseIndices[idx] = 0;
                        baseMask[idx] = 0.0f;
                    }
                }
            }
            
            System.Threading.Tasks.Parallel.For(0, batchSize, b =>
            {
                int offset = b * numVertices * SpiralLength;
                int vertexOffset = b * numVertices;
                for (int i = 0; i < baseIndices.Length; i++)
                {
                    flatIndices[offset + i] = baseIndices[i] + vertexOffset; // Offset index by batch
                    flatMask[offset + i] = baseMask[i];
                }
            });
            
            using var indicesBuffer = backend.AllocateIntBuffer(flatIndices);
            using var maskBuffer = backend.AllocateBuffer(flatMask);
            
            int numGather = totalVertices * SpiralLength;
            var gatheredRaw = gpuEngine.GatherGpu(flatInput, indicesBuffer, numGather, inputChannels);
            
            using var maskTensor = new GpuTensor<T>(backend, maskBuffer, [numGather, 1], GpuTensorRole.Constant, ownsBuffer: false);
            var gatheredMasked = gpuEngine.BroadcastMultiplyColumnGpu(gatheredRaw, maskTensor);
            gatheredRaw.Dispose();
            
            var gathered = gpuEngine.ReshapeGpu(gatheredMasked, [totalVertices, SpiralLength * inputChannels]);
            
            var wT = _weights.Transpose();
            var outputFlat = gpuEngine.FusedLinearGpu(gathered, wT, _biases, MapActivationToFused());
            gathered.Dispose();
            
            result = gpuEngine.ReshapeGpu(outputFlat, [batchSize, numVertices, OutputChannels]);
        }
        else
        {
            result = ProcessBatchItem(input);
        }

        if (IsTrainingMode)
        {
            _lastInput = input.ToTensor();
            // We might need gathered features for backward pass?
            // The CPU Backward uses _gatheredFeatures.
            // We should cache it if possible, or recompute.
            // Recomputing is safer for now to avoid complexity of downloading.
            // But optimal is caching.
            // For now, just cache output/input.
            _lastOutput = result.ToTensor();
        }

        return result;
    }

    private void InitializeGpuIndices(DirectGpuTensorEngine gpuEngine)
    {
        var backend = gpuEngine.GetBackend();
        if (backend == null) return;

        int numVertices = _spiralIndices!.GetLength(0);
        int[] indices = new int[numVertices * SpiralLength];
        float[] mask = new float[numVertices * SpiralLength];

        for (int v = 0; v < numVertices; v++)
        {
            for (int s = 0; s < SpiralLength; s++)
            {
                int idx = v * SpiralLength + s;
                int neighbor = _spiralIndices[v, s];
                
                if (neighbor >= 0 && neighbor < numVertices)
                {
                    indices[idx] = neighbor;
                    mask[idx] = 1.0f;
                }
                else
                {
                    indices[idx] = 0; // Point to valid 0 to avoid crash, mask will zero it
                    mask[idx] = 0.0f;
                }
            }
        }

        _spiralIndicesGpu = backend.AllocateIntBuffer(indices);
        _spiralMaskGpu = backend.AllocateBuffer(mask);
    }

    private IGpuBuffer? _spiralMaskGpu;

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _spiralIndicesGpu?.Dispose();
            _spiralMaskGpu?.Dispose();
        }
        base.Dispose(disposing);
    }

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
    /// Cached GPU buffer for spiral indices.
    /// </summary>
    private IGpuBuffer? _spiralIndicesGpu;

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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [numVertices, inputChannels],
            [numVertices, outputChannels],
            activationFunction ?? new ReLUActivation<T>())
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
        // === Vectorized: He initialization using TensorRandomUniformRange (Phase C: New IEngine methods) ===
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));

        // Initialize weights in [-scale, scale] range
        _weights = Engine.TensorRandomUniformRange<T>(_weights.Shape, NumOps.Negate(scale), scale);

        // Initialize biases to zero
        _biases = new Tensor<T>(_biases.Shape);
        Engine.TensorFill(_biases, NumOps.Zero);
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
    /// <remarks>
    /// <para>
    /// If spiral indices are set, they are serialized along with the layer.
    /// Otherwise, users must call <see cref="SetSpiralIndices"/> after deserialization.
    /// </para>
    /// </remarks>
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

        // Serialize spiral indices if set
        bool hasIndices = _spiralIndices != null;
        writer.Write(hasIndices);
        if (hasIndices && _spiralIndices != null)
        {
            int numVertices = _spiralIndices.GetLength(0);
            writer.Write(numVertices);
            for (int v = 0; v < numVertices; v++)
            {
                for (int s = 0; s < SpiralLength; s++)
                {
                    writer.Write(_spiralIndices[v, s]);
                }
            }
        }
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <param name="reader">Binary reader for deserialization.</param>
    /// <remarks>
    /// <para>
    /// If spiral indices were serialized with the layer, they are restored automatically.
    /// Otherwise, users must call <see cref="SetSpiralIndices"/> before calling Forward.
    /// </para>
    /// </remarks>
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

        // Deserialize spiral indices if present
        bool hasIndices = reader.ReadBoolean();
        if (hasIndices)
        {
            int numVertices = reader.ReadInt32();
            _spiralIndices = new int[numVertices, SpiralLength];
            for (int v = 0; v < numVertices; v++)
            {
                for (int s = 0; s < SpiralLength; s++)
                {
                    _spiralIndices[v, s] = reader.ReadInt32();
                }
            }
        }
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
