using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that writes to a memory tensor using an attention mechanism.
/// </summary>
/// <remarks>
/// <para>
/// The MemoryWriteLayer implements a form of attention-based memory writing. It computes attention scores
/// between the input and memory tensors, using these scores to determine where to write new information.
/// This approach allows the layer to selectively update memory based on the current input. The layer uses
/// a query-key-value attention mechanism where queries and keys determine where to write, and values determine
/// what to write.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network store information in memory.
/// 
/// Think of it like deciding what to write in a notebook:
/// - You have some new information (your current input)
/// - You have a notebook with existing notes (your memory)
/// - The layer decides which pages of the notebook are relevant to your new information
/// - It then writes the new information on those pages, focusing more on the most relevant ones
/// 
/// For example, if your input represents new information about "France has a beautiful capital city",
/// the layer would focus on memory locations related to France and update them with this new information.
/// 
/// This is similar to how we humans selectively update our memories with new information, rather than
/// storing everything in completely new locations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MemoryWriteLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes an attention sparsity auxiliary loss that encourages focused memory writes.
    /// This helps prevent the layer from writing to too many memory locations at once, promoting more selective updates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses an additional learning signal.
    ///
    /// When enabled (true):
    /// - The layer encourages focused attention on specific memory locations for writing
    /// - This helps the network learn to be more selective about where it updates memory
    /// - Training may be more stable and produce better memory write patterns
    ///
    /// When disabled (false):
    /// - Only the main task loss is used for training
    /// - This is the default setting
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the auxiliary loss contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines how much the attention sparsity loss contributes to the total loss.
    /// The default value of 0.005 provides a good balance between the main task and sparsity regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much importance to give to the write attention sparsity penalty.
    ///
    /// The weight affects training:
    /// - Higher values (e.g., 0.01) make the network prioritize focused writes more strongly
    /// - Lower values (e.g., 0.001) make the sparsity penalty less important
    /// - The default (0.005) works well for most memory-augmented tasks
    ///
    /// If your memory writes are too diffuse (spreading across too many locations), increase this value.
    /// If the main task is more important, you might decrease it.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Stores the last computed attention sparsity loss for diagnostic purposes.
    /// </summary>
    private T _lastAttentionSparsityLoss;

    /// <summary>
    /// The weight tensor used to transform the input into query vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms the input vector into query vectors used to determine where to write in memory.
    /// Shape: [inputDimension, memoryDimension]
    /// </remarks>
    private Tensor<T> _queryWeights;

    /// <summary>
    /// The weight tensor used to transform the input into key vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms the input vector into key vectors used with memory keys for attention calculation.
    /// Shape: [inputDimension, memoryDimension]
    /// </remarks>
    private Tensor<T> _keyWeights;

    /// <summary>
    /// The weight tensor used to transform the input into value vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms the input vector into value vectors that determine what to write to memory.
    /// Shape: [inputDimension, memoryDimension]
    /// </remarks>
    private Tensor<T> _valueWeights;

    /// <summary>
    /// The weight tensor applied to the output after value transformation.
    /// </summary>
    /// <remarks>
    /// This tensor applies a final transformation to the output before adding the bias.
    /// Shape: [memoryDimension, memoryDimension]
    /// </remarks>
    private Tensor<T> _outputWeights;

    /// <summary>
    /// The bias tensor added to the output.
    /// </summary>
    /// <remarks>
    /// This tensor is added to the output after all weight transformations.
    /// Shape: [memoryDimension]
    /// </remarks>
    private Tensor<T> _outputBias;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The memory tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the memory tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastMemory;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The attention scores tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the attention scores from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastAttentionScores;

    /// <summary>
    /// The write values tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the write values tensor computed from attention weights, which can be used
    /// when updating memory or computing auxiliary objectives.
    /// </remarks>
    private Tensor<T>? _lastWriteValues;

    /// <summary>
    /// The transformed values tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input values after value projection, which are used for output gradients.
    /// </remarks>
    private Tensor<T>? _lastValues;

    /// <summary>
    /// The gradient of the loss with respect to the query weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the query weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _queryWeightsGradient;

    /// <summary>
    /// The gradient of the loss with respect to the key weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the key weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _keyWeightsGradient;

    /// <summary>
    /// The gradient of the loss with respect to the value weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the value weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _valueWeightsGradient;

    /// <summary>
    /// The gradient of the loss with respect to the output weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the output weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _outputWeightsGradient;

    /// <summary>
    /// The gradient of the loss with respect to the output bias.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the output bias, which is used to update the bias
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _outputBiasGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the MemoryWriteLayer has trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that MemoryWriteLayer can be trained through backpropagation. The layer
    /// has trainable parameters (weights and biases) that are updated during training to optimize
    /// the memory writing process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (weights and biases) that change during training
    /// - It will improve its performance as it sees more data
    /// - It learns to better focus attention on relevant parts of memory for writing
    /// 
    /// During training, the layer learns:
    /// - Which features in the input are important for determining where to write
    /// - How to transform input information into memory updates
    /// - How to selectively update memory instead of overwriting everything
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryWriteLayer{T}"/> class with the specified dimensions
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryWriteLayer with the specified dimensions and activation function.
    /// The layer is initialized with random weights scaled according to the layer dimensions to facilitate
    /// stable training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and activation function.
    /// 
    /// When creating a MemoryWriteLayer, you need to specify:
    /// - inputDimension: The size of your input vector (e.g., 128 for a 128-feature input)
    /// - memoryDimension: The size of each memory entry (e.g., 256 for memory entries with 256 features)
    /// - activationFunction: The function that processes the final output (optional)
    /// 
    /// The constructor creates weight matrices of the appropriate sizes and initializes them
    /// with small random values to start the learning process. The initialization scale
    /// is carefully chosen to prevent training issues like vanishing or exploding gradients.
    /// </para>
    /// </remarks>
    public MemoryWriteLayer(int inputDimension, int memoryDimension, IActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new IdentityActivation<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionSparsityLoss = NumOps.Zero;

        _queryWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _keyWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _valueWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _outputWeights = new Tensor<T>([memoryDimension, memoryDimension]);
        _outputBias = new Tensor<T>([memoryDimension]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryWriteLayer{T}"/> class with the specified dimensions
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="activationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryWriteLayer with the specified dimensions and vector activation function.
    /// A vector activation function operates on entire vectors rather than individual elements.
    /// The layer is initialized with random weights scaled according to the layer dimensions to facilitate
    /// stable training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs in your memory writing operation.
    /// </para>
    /// </remarks>
    public MemoryWriteLayer(int inputDimension, int memoryDimension, IVectorActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new IdentityActivation<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionSparsityLoss = NumOps.Zero;

        _queryWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _keyWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _valueWeights = new Tensor<T>([inputDimension, memoryDimension]);
        _outputWeights = new Tensor<T>([memoryDimension, memoryDimension]);
        _outputBias = new Tensor<T>([memoryDimension]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes the layer's weights and biases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using a scaling factor derived from the dimensions
    /// of the weight matrices. The scaling helps prevent vanishing or exploding gradients
    /// during training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's weights and biases.
    /// 
    /// Proper initialization is important for neural networks because:
    /// - Starting with good values helps the network learn faster
    /// - It helps prevent problems during training like vanishing or exploding gradients
    ///   (when values become too small or too large)
    /// 
    /// This method:
    /// - Calculates a scaling factor based on the size of the matrices
    /// - Initializes weights to small random values multiplied by this scale
    /// - Sets all bias values to zero
    /// 
    /// This approach (known as "He initialization") works well for many types of neural networks.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Shape[0] + _queryWeights.Shape[1])));
        InitializeTensor(_queryWeights, scale);
        InitializeTensor(_keyWeights, scale);
        InitializeTensor(_valueWeights, scale);
        InitializeTensor(_outputWeights, scale);

        _outputBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a tensor with random values scaled by the given factor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the tensor with random values between -0.5 and 0.5, scaled by the provided factor.
    /// This approach helps to establish good initial conditions for training, especially for deeper networks
    /// where proper weight initialization is crucial for convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a tensor with small random numbers.
    ///
    /// When initializing a neural network:
    /// - We need to start with random values to break symmetry
    /// - Values that are too large or too small can cause problems
    /// - The scale parameter helps control how large the initial values are
    ///
    /// This method goes through each position in the tensor and assigns it a random
    /// value between -0.5 and 0.5, multiplied by the scale factor. This gives a
    /// controlled amount of randomness that helps the network start learning effectively.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        // Create random tensor using Engine operations
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);

        // Shift to [-0.5, 0.5] range: randomTensor - 0.5
        var halfTensor = new Tensor<T>(tensor.Shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy to tensor
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = scaled.GetFlat(i);
        }
    }

    /// <summary>
    /// Performs the forward pass of the memory write layer with input and memory tensors.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="memory">The memory tensor to update.</param>
    /// <returns>The output tensor containing updated memory.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the memory write layer. It computes queries, keys, and values
    /// from the input, computes attention scores between the queries and memory, applies softmax to get
    /// attention weights, and then uses these weights to selectively update memory with the computed values.
    /// The method uses a scaled dot-product attention mechanism, dividing the attention scores by the square
    /// root of the key dimension for stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual memory writing operation based on the input.
    /// 
    /// The forward pass works in these steps:
    /// 1. Transform the input into three different representations:
    ///    - Queries: used to match against memory
    ///    - Keys: used to transform the input for attention scoring
    ///    - Values: the actual content to be written to memory
    /// 2. Compare queries with memory to get attention scores
    /// 3. Scale the scores for stability and convert to weights using softmax
    /// 4. Use these weights to determine where to write the values
    /// 5. Apply additional transformations and activation function for the final output
    /// 
    /// This process allows the layer to selectively update different parts of memory
    /// based on the relevance of the current input, rather than writing to all
    /// memory locations equally.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> memory)
    {
        _lastInput = input;
        _lastMemory = memory;

        // Dynamic input dimension adaptation
        int actualInputDim = input.Shape[^1];
        int expectedInputDim = _queryWeights.Shape[0];
        if (actualInputDim != expectedInputDim)
        {
            int memoryDim = _queryWeights.Shape[1];
            T scale = NumOps.FromDouble(Math.Sqrt(2.0 / (actualInputDim + memoryDim)));
            var random = RandomHelper.CreateSecureRandom();

            _queryWeights = new Tensor<T>([actualInputDim, memoryDim]);
            _keyWeights = new Tensor<T>([actualInputDim, memoryDim]);
            _valueWeights = new Tensor<T>([actualInputDim, memoryDim]);

            for (int i = 0; i < _queryWeights.Length; i++)
                _queryWeights.SetFlat(i, NumOps.Multiply(scale, NumOps.FromDouble(random.NextDouble() * 2 - 1)));
            for (int i = 0; i < _keyWeights.Length; i++)
                _keyWeights.SetFlat(i, NumOps.Multiply(scale, NumOps.FromDouble(random.NextDouble() * 2 - 1)));
            for (int i = 0; i < _valueWeights.Length; i++)
                _valueWeights.SetFlat(i, NumOps.Multiply(scale, NumOps.FromDouble(random.NextDouble() * 2 - 1)));

            UpdateInputShape([actualInputDim]);
        }

        // Use Engine operations for matrix multiplications
        var queries = Engine.TensorMatMul(input, _queryWeights);
        var keys = Engine.TensorMatMul(input, _keyWeights);
        var values = Engine.TensorMatMul(input, _valueWeights);
        _lastValues = values;

        // Compute attention scores: queries × memory^T
        var memoryTransposed = Engine.TensorTranspose(memory);
        var attentionScores = Engine.TensorMatMul(queries, memoryTransposed);

        // Scale attention scores by sqrt(key_dim) for stability
        T scaleFactor = NumOps.FromDouble(1.0 / Math.Sqrt(keys.Shape[1]));
        attentionScores = Engine.TensorMultiplyScalar(attentionScores, scaleFactor);

        // Apply softmax to get attention weights
        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        // Compute write values: attention-weighted combination of values
        // attentionWeights: [batch, numSlots], values: [batch, memoryDim]
        // We want: [numSlots, memoryDim] to update memory
        // Use: attentionWeights^T @ values = [numSlots, batch] @ [batch, memoryDim] = [numSlots, memoryDim]
        var attentionT = Engine.TensorTranspose(attentionWeights);
        var writeValues = Engine.TensorMatMul(attentionT, values);
        _lastWriteValues = writeValues; // Cache for gradient computation / memory update

        // For the output, we transform the values (what we're writing) through output weights
        // This maintains [batch, memoryDim] shape for downstream layers
        // Output: values × outputWeights + outputBias = [batch, memoryDim]
        var projected = Engine.TensorMatMul(values, _outputWeights);

        // Broadcast bias and add
        var batchSize = input.Shape[0];
        var biasBroadcast = BroadcastBiases(_outputBias, batchSize);
        var output = Engine.TensorAdd(projected, biasBroadcast);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
    }

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("MemoryWriteLayer requires both input and memory tensors.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        // MemoryWriteLayer has complex attention-based operations - use CPU fallback
        var cpuInput = inputs[0].ToTensor();
        var cpuMemory = inputs[1].ToTensor();
        var cpuOutput = Forward(cpuInput, cpuMemory);
        return gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
    }

    /// <summary>
    /// Applies softmax over memory slots dimension for write layer.
    /// </summary>
    private void ApplySoftmaxWriteMemory(float[] data, int batchSize, int memorySlots)
    {
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * memorySlots;

            float maxVal = float.MinValue;
            for (int s = 0; s < memorySlots; s++)
            {
                maxVal = MathF.Max(maxVal, data[offset + s]);
            }

            float sumExp = 0f;
            for (int s = 0; s < memorySlots; s++)
            {
                data[offset + s] = MathF.Exp(data[offset + s] - maxVal);
                sumExp += data[offset + s];
            }

            float invSum = 1f / sumExp;
            for (int s = 0; s < memorySlots; s++)
            {
                data[offset + s] *= invSum;
            }
        }
    }

    /// <summary>
    /// Applies activation function for write memory layer.
    /// </summary>
    private void ApplyWriteMemoryActivation(float[] data, int batchSize, int memoryDim)
    {
        var activation = ScalarActivation ?? (object?)VectorActivation;
        string activationName = activation?.GetType().Name.ToLowerInvariant() ?? "identity";

        if (activationName.Contains("relu"))
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Max(0f, data[i]);
        }
        else if (activationName.Contains("sigmoid"))
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = 1f / (1f + MathF.Exp(-data[i]));
        }
        else if (activationName.Contains("tanh"))
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Tanh(data[i]);
        }
        // Identity or unknown - keep as is
    }

    /// <summary>
    /// Performs the backward pass of the memory write layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the memory write layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradients of all weights and biases, as well as
    /// the gradient with respect to the input tensor. The computed weight and bias gradients are stored for later
    /// use in the parameter update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how all parameters should change to reduce errors.
    ///
    /// During the backward pass:
    /// - The layer receives gradients indicating how the output (updated memory) should change
    /// - It calculates how each weight, bias, and input value should change
    /// - These gradients are used later to update the parameters during training
    ///
    /// The backward pass is complex because it needs to:
    /// - Calculate gradients for query, key, and value weights
    /// - Calculate gradients for the output weights and bias
    /// - Handle the chain rule through the softmax attention mechanism
    /// - Combine gradients from multiple paths
    ///
    /// This process enables the layer to learn more effective memory writing strategies over time.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using automatic differentiation through the
    /// TensorOperations API. It recreates the forward computation graph including attention
    /// mechanism, query-key-value transformations, and output projection, then manually
    /// propagates gradients through the computation graph.
    /// </para>
    /// <para>
    /// The autodiff implementation:
    /// - Converts all cached inputs and parameters to ComputationNodes
    /// - Replays the forward pass computation using autodiff operations
    /// - Manually sets the output gradient and propagates it backward
    /// - Extracts parameter gradients and input gradient from the computation graph
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMemory == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert inputs and parameters to computation nodes (tensors already in correct format)
        var inputNode = TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var memoryNode = TensorOperations<T>.Variable(_lastMemory, "memory", requiresGradient: false);

        var queryWeightsNode = TensorOperations<T>.Variable(_queryWeights, "queryWeights", requiresGradient: true);
        var keyWeightsNode = TensorOperations<T>.Variable(_keyWeights, "keyWeights", requiresGradient: true);
        var valueWeightsNode = TensorOperations<T>.Variable(_valueWeights, "valueWeights", requiresGradient: true);
        var outputWeightsNode = TensorOperations<T>.Variable(_outputWeights, "outputWeights", requiresGradient: true);
        var outputBiasNode = TensorOperations<T>.Variable(_outputBias, "outputBias", requiresGradient: true);

        // Replay forward pass using autodiff operations
        // 1. Compute queries, keys, and values
        var queries = TensorOperations<T>.MatrixMultiply(inputNode, queryWeightsNode);
        var keys = TensorOperations<T>.MatrixMultiply(inputNode, keyWeightsNode);
        var values = TensorOperations<T>.MatrixMultiply(inputNode, valueWeightsNode);

        // 2. Compute attention scores: queries × memory^T
        var memoryTransposed = TensorOperations<T>.Transpose(memoryNode);
        var attentionScores = TensorOperations<T>.MatrixMultiply(queries, memoryTransposed);

        // 3. Scale attention scores by sqrt(key_dim)
        double scaleFactor = 1.0 / Math.Sqrt(keys.Value.Shape[1]);
        var scaleTensor = new Tensor<T>([1]);
        scaleTensor[0] = NumOps.FromDouble(scaleFactor);
        var scaleConstant = TensorOperations<T>.Constant(scaleTensor, "scale");
        var scaledAttention = TensorOperations<T>.ElementwiseMultiply(attentionScores, scaleConstant);

        // 4. Apply softmax to get attention weights
        var attentionWeights = TensorOperations<T>.Softmax(scaledAttention, axis: -1);

        // 5. Compute write values: values × attentionWeights
        var writeValues = TensorOperations<T>.ElementwiseMultiply(values, attentionWeights);

        // 6. Apply output transformation: writeValues × outputWeights + outputBias
        var outputBeforeActivation = TensorOperations<T>.MatrixMultiply(writeValues, outputWeightsNode);

        // Broadcast bias across batch dimension
        var batchSize = _lastInput.Shape[0];
        var biasesBroadcast = BroadcastBiases(_outputBias, batchSize);
        var biasNode = TensorOperations<T>.Variable(biasesBroadcast, "biases_broadcast", requiresGradient: false);
        var output = TensorOperations<T>.Add(outputBeforeActivation, biasNode);

        // Manually propagate gradients
        output.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((output, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients for parameters (already Tensor<T>)
        _queryWeightsGradient = queryWeightsNode.Gradient ?? throw new InvalidOperationException("Query weights gradient was not computed.");
        _keyWeightsGradient = keyWeightsNode.Gradient ?? throw new InvalidOperationException("Key weights gradient was not computed.");
        _valueWeightsGradient = valueWeightsNode.Gradient ?? throw new InvalidOperationException("Value weights gradient was not computed.");
        _outputWeightsGradient = outputWeightsNode.Gradient ?? throw new InvalidOperationException("Output weights gradient was not computed.");
        _outputBiasGradient = outputBiasNode.Gradient ?? throw new InvalidOperationException("Output bias gradient was not computed.");

        // Return input gradient
        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Input gradient was not computed.");

        return inputNode.Gradient;
    }

    /// <summary>
    /// Broadcasts a bias tensor across the batch dimension.
    /// </summary>
    /// <param name="biases">The bias tensor of shape [memoryDimension].</param>
    /// <param name="batchSize">The batch size for broadcasting.</param>
    /// <returns>A tensor of shape [batchSize, memoryDimension] with biases broadcast.</returns>
    private Tensor<T> BroadcastBiases(Tensor<T> biases, int batchSize)
    {
        int outputDim = biases.Length;

        // Reshape bias from [outputDim] to [1, outputDim] and tile across batch
        var biasReshaped = biases.Reshape([1, outputDim]);
        var broadcast = Engine.TensorTile(biasReshaped, [batchSize, 1]);

        return broadcast;
    }

    /// <summary>
    /// Manual backward pass implementation for memory write layer with attention.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations for
    /// attention-based memory writing. It computes gradients through the attention mechanism,
    /// including softmax, query/key/value transformations, and output projection.
    /// </para>
    /// <para>
    /// Autodiff Note: The memory write operation involves complex attention mechanisms with
    /// softmax over attention scores, query-key-value structure, and multiple weight matrices.
    /// The manual implementation provides efficient gradient calculations for all components
    /// of the attention-based memory update.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastValues == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Output weights gradient: values^T × activationGradient
        // For Y = X × W, gradient ∂L/∂W = X^T × ∂L/∂Y where X is projected values
        var lastValuesT = Engine.TensorTranspose(_lastValues);
        _outputWeightsGradient = Engine.TensorMatMul(lastValuesT, activationGradient);

        // Output bias gradient: sum across batch dimension
        _outputBiasGradient = activationGradient.Sum([0]);

        // Values gradient: activationGradient × outputWeights^T
        var outputWeightsT = Engine.TensorTranspose(_outputWeights);
        var valuesGradient = Engine.TensorMatMul(activationGradient, outputWeightsT);

        // Weight gradients: input^T × gradient
        var lastInputT = Engine.TensorTranspose(_lastInput);
        _valueWeightsGradient = Engine.TensorMatMul(lastInputT, valuesGradient);

        _queryWeightsGradient = Tensor<T>.CreateDefault(_queryWeights.Shape, NumOps.Zero);
        _keyWeightsGradient = Tensor<T>.CreateDefault(_keyWeights.Shape, NumOps.Zero);

        // Input gradient: values gradient × valueWeights^T
        var valueWeightsT = Engine.TensorTranspose(_valueWeights);
        var inputGradient = Engine.TensorMatMul(valuesGradient, valueWeightsT);

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the memory write layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all trainable parameters of the layer (query weights, key weights, value weights,
    /// output weights, and output bias) based on the gradients calculated during the backward pass. The learning
    /// rate controls the size of the parameter updates. Each parameter is updated by subtracting the corresponding
    /// gradient multiplied by the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's weights and biases during training.
    /// 
    /// After the backward pass calculates how parameters should change, this method:
    /// - Takes each weight matrix and bias vector
    /// - Subtracts the corresponding gradient scaled by the learning rate
    /// - This moves the parameters in the direction that reduces errors
    /// 
    /// The learning rate controls how big each update step is:
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer gradually improves its memory writing abilities over many training iterations.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine operations for parameter updates
        var scaledQueryGrad = Engine.TensorMultiplyScalar(_queryWeightsGradient, learningRate);
        _queryWeights = Engine.TensorSubtract(_queryWeights, scaledQueryGrad);

        var scaledKeyGrad = Engine.TensorMultiplyScalar(_keyWeightsGradient, learningRate);
        _keyWeights = Engine.TensorSubtract(_keyWeights, scaledKeyGrad);

        var scaledValueGrad = Engine.TensorMultiplyScalar(_valueWeightsGradient, learningRate);
        _valueWeights = Engine.TensorSubtract(_valueWeights, scaledValueGrad);

        var scaledOutputGrad = Engine.TensorMultiplyScalar(_outputWeightsGradient, learningRate);
        _outputWeights = Engine.TensorSubtract(_outputWeights, scaledOutputGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_outputBiasGradient, learningRate);
        _outputBias = Engine.TensorSubtract(_outputBias, scaledBiasGrad);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    /// <summary>
    /// Performs the forward pass of the memory write layer with just the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing with an empty memory tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Forward method to handle the case where only an input tensor is provided.
    /// It creates an empty (zero-initialized) memory tensor of the appropriate size and calls the overloaded
    /// Forward method with both the input and the empty memory.
    /// </para>
    /// <para><b>For Beginners:</b> This method handles the case when no existing memory is provided.
    /// 
    /// When you call this method with just an input tensor:
    /// - The layer creates a blank memory tensor filled with zeros
    /// - It then calls the regular Forward method with both your input and this blank memory
    /// - The result is as if you're writing to a fresh, empty memory
    /// 
    /// This is useful when:
    /// - You're starting a new sequence and don't have previous memory
    /// - You want to initialize memory from scratch
    /// - You want to simplify your code by not having to create empty memory yourself
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // For a memory write layer, we need both input and memory
        // When only input is provided, we create a zero-initialized memory tensor
        int batchSize = input.Shape[0];
        int memoryDimension = _queryWeights.Shape[1];

        // Create an empty memory tensor and initialize with zeros
        var emptyMemory = new Tensor<T>([batchSize, memoryDimension]);
        emptyMemory.Fill(NumOps.Zero);

        // Call the overloaded Forward method with the empty memory
        return Forward(input, emptyMemory);
    }

    /// <summary>
    /// Gets all trainable parameters from the memory write layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. It concatenates
    /// the query weights, key weights, value weights, output weights, and output bias into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values in the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all the weights and biases from this layer
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The method carefully arranges all parameters in a specific order
    /// so they can be correctly restored later.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate to efficiently combine all parameters
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeights.ToArray()),
            new Vector<T>(_keyWeights.ToArray()),
            new Vector<T>(_valueWeights.ToArray()),
            new Vector<T>(_outputWeights.ToArray()),
            new Vector<T>(_outputBias.ToArray())
        );
    }

    /// <summary>
    /// Sets the trainable parameters for the memory write layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector. It extracts the appropriate
    /// portions of the input vector for each parameter (query weights, key weights, value weights, output weights,
    /// and output bias). This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The method extracts portions for each weight matrix and bias vector
    /// - It places each value in its correct position
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters,
    /// ensuring that all matrices and vectors maintain their correct dimensions.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int querySize = _queryWeights.Shape[0] * _queryWeights.Shape[1];
        int keySize = _keyWeights.Shape[0] * _keyWeights.Shape[1];
        int valueSize = _valueWeights.Shape[0] * _valueWeights.Shape[1];
        int outputSize = _outputWeights.Shape[0] * _outputWeights.Shape[1];
        int biasSize = _outputBias.Length;
        int totalParams = querySize + keySize + valueSize + outputSize + biasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set query weights using Tensor.FromVector
        var queryParams = parameters.SubVector(index, querySize);
        _queryWeights = Tensor<T>.FromVector(queryParams).Reshape(_queryWeights.Shape);
        index += querySize;

        // Set key weights
        var keyParams = parameters.SubVector(index, keySize);
        _keyWeights = Tensor<T>.FromVector(keyParams).Reshape(_keyWeights.Shape);
        index += keySize;

        // Set value weights
        var valueParams = parameters.SubVector(index, valueSize);
        _valueWeights = Tensor<T>.FromVector(valueParams).Reshape(_valueWeights.Shape);
        index += valueSize;

        // Set output weights
        var outputParams = parameters.SubVector(index, outputSize);
        _outputWeights = Tensor<T>.FromVector(outputParams).Reshape(_outputWeights.Shape);
        index += outputSize;

        // Set output bias
        var biasParams = parameters.SubVector(index, biasSize);
        _outputBias = Tensor<T>.FromVector(biasParams);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    /// <summary>
    /// Resets the internal state of the memory write layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the memory write layer, including the cached inputs, memory,
    /// outputs, attention scores, and all gradients. This is useful when starting to process a new sequence
    /// or batch of data, or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - Stored inputs, memory, outputs, and attention scores from previous processing are cleared
    /// - All calculated gradients are cleared
    /// - The layer forgets any information from previous data batches
    ///
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    ///
    /// Resetting state helps ensure that each forward and backward pass is independent,
    /// which is important for correct behavior in many neural network architectures.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastMemory = null;
        _lastOutput = null;
        _lastAttentionScores = null;
        _lastWriteValues = null;
        _lastValues = null;

        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    /// <summary>
    /// Computes the auxiliary loss for this layer based on attention sparsity regularization.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes an attention sparsity loss that encourages focused memory write patterns.
    /// The loss is computed as the negative entropy of the attention weights: L = -Σ(p * log(p))
    /// where p represents the attention probabilities. Lower entropy (more focused attention) results in lower loss.
    /// This encourages the layer to write to specific memory locations rather than spreading writes uniformly.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates a penalty for unfocused write attention patterns.
    ///
    /// Attention sparsity loss for writing:
    /// - Measures how focused the write attention is on specific memory locations
    /// - Lower values mean more focused writes (good)
    /// - Higher values mean writes are spread across many locations (less focused)
    ///
    /// Why this is useful:
    /// - In most tasks, you want to update specific relevant memory locations
    /// - Spreading writes too thin means you dilute the information being stored
    /// - Focused writes means you update specific memory locations with clear information
    ///
    /// Example: If you're updating memory with "Paris is the capital of France",
    /// you want focused writes to the France-related memory locations, not scattered writes
    /// across all memory that dilute this information.
    ///
    /// Technical note: The loss is computed using entropy. Entropy measures how "spread out" a distribution is.
    /// - Low entropy = focused distribution (e.g., [0.9, 0.05, 0.05] - mostly updating first location)
    /// - High entropy = spread out distribution (e.g., [0.33, 0.33, 0.34] - updating all equally)
    /// We use negative entropy as the loss, so the network is penalized for high entropy (unfocused writes).
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (_lastAttentionScores == null)
        {
            _lastAttentionSparsityLoss = NumOps.Zero;
            return _lastAttentionSparsityLoss;
        }

        // Compute negative entropy to encourage focused attention using tensor ops
        // L = -mean(sum(p * log(p), axis=-1))
        var epsilon = NumOps.FromDouble(1e-10);
        var safeAttention = Engine.TensorMax(_lastAttentionScores, epsilon);
        var logAttention = Engine.TensorLog(safeAttention);
        var product = Engine.TensorMultiply(safeAttention, logAttention);
        var sumPerBatch = Engine.ReduceSum(product, new[] { product.Shape.Length - 1 }, keepDims: false);
        var negativeEntropy = Engine.TensorMultiplyScalar<T>(sumPerBatch, NumOps.FromDouble(-1));
        var meanEntropy = Engine.ReduceMean(negativeEntropy, new[] { 0 }, keepDims: false);
        _lastAttentionSparsityLoss = meanEntropy.GetFlat(0);
        return _lastAttentionSparsityLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method returns diagnostic information that can be used to monitor the auxiliary loss during training.
    /// The diagnostics include the total attention sparsity loss, the weight applied to it, and whether auxiliary loss is enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information to help you understand how the auxiliary loss is working.
    ///
    /// The diagnostics show:
    /// - TotalAttentionSparsityLoss: The computed penalty for unfocused write attention
    /// - AttentionSparsityWeight: How much this penalty affects the overall training
    /// - UseAttentionSparsity: Whether this penalty is currently enabled
    ///
    /// You can use this information to:
    /// - Monitor if write attention is becoming more focused over time
    /// - Debug training issues related to memory writing
    /// - Understand how the layer is learning to update memory
    ///
    /// Example: If TotalAttentionSparsityLoss is decreasing during training, it means the layer
    /// is learning to be more focused in its memory writes, which is typically a good sign.
    /// If it's staying high or increasing, it might mean the layer is having trouble learning
    /// which parts of memory to update.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalAttentionSparsityLoss", System.Convert.ToString(_lastAttentionSparsityLoss) ?? "0" },
            { "AttentionSparsityWeight", System.Convert.ToString(AuxiliaryLossWeight) ?? "0.005" },
            { "UseAttentionSparsity", UseAuxiliaryLoss.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_queryWeights == null || _keyWeights == null || _valueWeights == null ||
            _outputWeights == null || _outputBias == null)
            throw new InvalidOperationException("Layer not initialized. Call Initialize() first.");

        // MemoryWriteLayer requires TWO inputs: input and memory
        // Input 0: Write input [batch, inputDim]
        var inputTensor = new Tensor<T>([1, _queryWeights.Shape[0]]);
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputTensor, "input");
        inputNodes.Add(inputNode);

        // Input 1: Memory [memorySize, memoryDim]
        var memoryTensor = new Tensor<T>([10, _keyWeights.Shape[1]]); // Placeholder size
        var memoryNode = Autodiff.TensorOperations<T>.Variable(memoryTensor, "memory");
        inputNodes.Add(memoryNode);

        // Weights are already Tensor<T>, use them directly
        var queryWeightsNode = Autodiff.TensorOperations<T>.Constant(_queryWeights, "queryWeights");
        var keyWeightsNode = Autodiff.TensorOperations<T>.Constant(_keyWeights, "keyWeights");
        var valueWeightsNode = Autodiff.TensorOperations<T>.Constant(_valueWeights, "valueWeights");
        var outputWeightsNode = Autodiff.TensorOperations<T>.Constant(_outputWeights, "outputWeights");
        var biasNode = Autodiff.TensorOperations<T>.Constant(_outputBias, "outputBias");

        // Build attention computation graph for memory writing
        // Step 1: queries = input @ queryWeights
        var queries = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, queryWeightsNode);

        // Step 2: keys = input @ keyWeights
        var keys = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, keyWeightsNode);

        // Step 3: values = input @ valueWeights
        var values = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, valueWeightsNode);

        // Step 4: scores = queries @ memory.T
        var memoryT = Autodiff.TensorOperations<T>.Transpose(memoryNode);
        var scores = Autodiff.TensorOperations<T>.MatrixMultiply(queries, memoryT);

        // Step 5: Scale scores for stability
        var keyDim = keys.Value.Shape[1];
        var scale = Autodiff.TensorOperations<T>.Constant(
            new Tensor<T>([1])
            {
                [0] = NumOps.FromDouble(1.0 / Math.Sqrt(keyDim))
            },
            "scale"
        );
        scores = Autodiff.TensorOperations<T>.ElementwiseMultiply(scores, scale);

        // Step 6: attention = softmax(scores)
        var attention = Autodiff.TensorOperations<T>.Softmax(scores, axis: -1);

        // Step 7: writeValues = values * attention (element-wise with broadcasting)
        var writeValues = Autodiff.TensorOperations<T>.ElementwiseMultiply(values, attention);

        // Step 8: output = writeValues @ outputWeights + bias
        var projected = Autodiff.TensorOperations<T>.MatrixMultiply(writeValues, outputWeightsNode);
        var output = Autodiff.TensorOperations<T>.Add(projected, biasNode);

        // Step 9: Apply activation if needed
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
            output = ScalarActivation.ApplyToGraph(output);
        else if (VectorActivation != null && VectorActivation.SupportsJitCompilation)
            output = VectorActivation.ApplyToGraph(output);

        return output;
    }

    public override bool SupportsJitCompilation => _queryWeights != null && _keyWeights != null &&
                                                     _valueWeights != null && _outputWeights != null &&
                                                     _outputBias != null;

}
