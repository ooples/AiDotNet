using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that reads from a memory tensor using an attention mechanism.
/// </summary>
/// <remarks>
/// <para>
/// The MemoryReadLayer implements a form of attention-based memory access. It computes attention scores
/// between the input and memory tensors, using these scores to create a weighted sum of memory values.
/// This approach allows the layer to selectively retrieve information from memory based on the current input.
/// The layer consists of key weights (for attention computation), value weights (for transforming memory values),
/// and output weights (for final processing).
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network retrieve information from memory.
/// 
/// Think of it like searching for relevant information in a book:
/// - You have a query (your current input)
/// - You have a memory (like pages of a book)
/// - The layer finds which parts of the memory are most relevant to your query
/// - It then combines those relevant parts to produce an output
/// 
/// For example, if your input represents a question like "What's the capital of France?",
/// the layer would look through memory to find information about France, give more attention
/// to content about its capital, and then combine this information to produce the answer "Paris".
/// 
/// This is similar to how modern language models can retrieve and use stored information
/// when answering questions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Memory)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, NormalizesInput = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "1, 4", TestConstructorArgs = "4, 4, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class MemoryReadLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes an attention sparsity auxiliary loss that encourages focused memory access.
    /// This helps prevent the layer from attending to too many memory locations at once, promoting more selective retrieval.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses an additional learning signal.
    ///
    /// When enabled (true):
    /// - The layer encourages focused attention on specific memory locations
    /// - This helps the network learn to be more selective about what information it retrieves
    /// - Training may be more stable and produce better memory access patterns
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
    /// <para><b>For Beginners:</b> This controls how much importance to give to the attention sparsity penalty.
    ///
    /// The weight affects training:
    /// - Higher values (e.g., 0.01) make the network prioritize focused attention more strongly
    /// - Lower values (e.g., 0.001) make the sparsity penalty less important
    /// - The default (0.005) works well for most memory-augmented tasks
    ///
    /// If your memory attention is too diffuse (spreading across too many locations), increase this value.
    /// If the main task is more important, you might decrease it.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Stores the last computed attention sparsity loss for diagnostic purposes.
    /// </summary>
    private T _lastAttentionSparsityLoss;

    /// <summary>
    /// The weight tensor used to transform the input into query keys.
    /// </summary>
    /// <remarks>
    /// This tensor transforms the input vector into a key vector that is used to query the memory.
    /// Shape: [inputDimension, memoryDimension]
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;

    /// <summary>
    /// The weight tensor used to transform the memory values after attention.
    /// </summary>
    /// <remarks>
    /// This tensor transforms the retrieved memory values into the output space.
    /// Shape: [memoryDimension, outputDimension]
    /// </remarks>
    private Tensor<T> _valueWeights;

    /// <summary>
    /// The weight tensor applied to the output after value transformation.
    /// </summary>
    /// <remarks>
    /// This tensor applies a final transformation to the output before adding the bias.
    /// Shape: [outputDimension, outputDimension]
    /// </remarks>
    private Tensor<T> _outputWeights;

    /// <summary>
    /// The bias tensor added to the output.
    /// </summary>
    /// <remarks>
    /// This tensor is added to the output after all weight transformations.
    /// Shape: [outputDimension]
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    /// The transformed tensor from the most recent forward pass (input to output weights).
    /// </summary>
    /// <remarks>
    /// This field stores the transformed tensor (result of readValues × valueWeights) from the most
    /// recent forward pass, which is needed during the backward pass for output weights gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastTransformed;

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
    /// Always <c>true</c> because the MemoryReadLayer has trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that MemoryReadLayer can be trained through backpropagation. The layer
    /// has trainable parameters (weights and biases) that are updated during training to optimize
    /// the memory reading process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (weights and biases) that change during training
    /// - It will improve its performance as it sees more data
    /// - It learns to better focus attention on relevant parts of memory
    /// 
    /// During training, the layer learns:
    /// - Which features in the input are important for querying memory
    /// - How to transform retrieved memory information
    /// - How to combine everything into a useful output
    /// </para>
    /// </remarks>
    public override int ParameterCount => _keyWeights.Length + _valueWeights.Length + _outputWeights.Length + _outputBias.Length;
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryReadLayer{T}"/> class with the specified dimensions
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="outputDimension">The size of the output vector.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryReadLayer with the specified dimensions and activation function.
    /// The layer is initialized with random weights scaled according to the layer dimensions to facilitate
    /// stable training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and activation function.
    /// 
    /// When creating a MemoryReadLayer, you need to specify:
    /// - inputDimension: The size of your query vector (e.g., 128 for a 128-feature query)
    /// - memoryDimension: The size of each memory entry (e.g., 256 for memory entries with 256 features)
    /// - outputDimension: The size of the output you want (e.g., 64 for a 64-feature result)
    /// - activationFunction: The function that processes the final output (optional)
    /// 
    /// The constructor creates weight matrices of the appropriate sizes and initializes them
    /// with small random values to start the learning process. The initialization scale
    /// is carefully chosen to prevent vanishing or exploding gradients during training.
    /// </para>
    /// </remarks>
    public MemoryReadLayer(int memoryDimension, int outputDimension, IActivationFunction<T>? activationFunction = null)
        : base(new[] { -1 }, new[] { outputDimension }, activationFunction ?? new IdentityActivation<T>())
    {
        if (memoryDimension <= 0) throw new ArgumentOutOfRangeException(nameof(memoryDimension));
        if (outputDimension <= 0) throw new ArgumentOutOfRangeException(nameof(outputDimension));

        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionSparsityLoss = NumOps.Zero;

        _memoryDimension = memoryDimension;
        _keyWeights = new Tensor<T>([0, 0]);
        _valueWeights = new Tensor<T>([memoryDimension, outputDimension]);
        _outputWeights = new Tensor<T>([outputDimension, outputDimension]);
        _outputBias = new Tensor<T>([outputDimension]);
    }

    private int _memoryDimension;

    /// <summary>
    /// Resolves input feature size from input.Shape on first forward and allocates key weights.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException(
                $"MemoryReadLayer requires rank>=1 input; got rank {rank}.", nameof(input));

        int inputDimension = input.Shape[rank - 1];
        int outputDimension = OutputShape[0];

        _keyWeights = new Tensor<T>([inputDimension, _memoryDimension]);
        InitializeParameters();
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);

        ResolveShapes(new[] { inputDimension }, new[] { outputDimension });
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryReadLayer{T}"/> class with the specified dimensions
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="outputDimension">The size of the output vector.</param>
    /// <param name="activationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryReadLayer with the specified dimensions and vector activation function.
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
    /// that consider the relationships between different outputs in your memory reading operation.
    /// </para>
    /// </remarks>
    public MemoryReadLayer(int memoryDimension, int outputDimension, IVectorActivationFunction<T> activationFunction)
        : base(new[] { -1 }, new[] { outputDimension }, activationFunction ?? new IdentityActivation<T>())
    {
        if (memoryDimension <= 0) throw new ArgumentOutOfRangeException(nameof(memoryDimension));
        if (outputDimension <= 0) throw new ArgumentOutOfRangeException(nameof(outputDimension));

        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionSparsityLoss = NumOps.Zero;

        _memoryDimension = memoryDimension;
        _keyWeights = new Tensor<T>([0, 0]);
        _valueWeights = new Tensor<T>([memoryDimension, outputDimension]);
        _outputWeights = new Tensor<T>([outputDimension, outputDimension]);
        _outputBias = new Tensor<T>([outputDimension]);
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
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_keyWeights.Shape[0] + _keyWeights.Shape[1])));
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
        var randomTensor = Tensor<T>.CreateRandom(tensor._shape);

        // Shift to [-0.5, 0.5] range: randomTensor - 0.5
        var halfTensor = new Tensor<T>(tensor._shape);
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
    /// Performs the forward pass of the memory read layer with input and memory tensors.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="memory">The memory tensor to read from.</param>
    /// <returns>The output tensor after memory reading and processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the memory read layer. It computes attention scores
    /// between the input and memory, applies softmax to get attention weights, retrieves a weighted sum
    /// of memory values, applies transformations through the value and output weights, and finally adds
    /// the bias and applies the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual memory reading operation based on the input.
    /// 
    /// The forward pass works in these steps:
    /// 1. Use the input to create query keys by applying the key weights
    /// 2. Compare these keys with each memory entry to get attention scores
    /// 3. Convert the scores to weights using softmax (making them sum to 1.0)
    /// 4. Use these weights to create a weighted sum of memory values
    /// 5. Transform this retrieved information through value and output weights
    /// 6. Add bias and apply activation function for the final output
    /// 
    /// This is similar to how attention works in many modern AI systems:
    /// the input "attends" to relevant parts of memory, focusing more on what's important
    /// for the current task and less on irrelevant information.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> memory)
    {
        _lastInput = input;
        _lastMemory = memory;

        // Use Engine operations for matrix multiplications
        var keys = Engine.TensorMatMul(input, _keyWeights);

        // Compute attention scores: keys × memory^T
        var memoryTransposed = Engine.TensorTranspose(memory);
        var attentionScores = Engine.TensorMatMul(keys, memoryTransposed);

        // Apply softmax to get attention weights
        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        // Read from memory: attentionWeights × memory
        var readValues = Engine.TensorMatMul(attentionWeights, memory);

        // Apply value and output transformations
        var transformed = Engine.TensorMatMul(readValues, _valueWeights);
        _lastTransformed = transformed; // Cache for output weights gradient computation
        var projected = Engine.TensorMatMul(transformed, _outputWeights);

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

    /// <summary>
    /// Broadcasts a bias tensor across the batch dimension.
    /// </summary>
    /// <param name="biases">The bias tensor of shape [outputDimension].</param>
    /// <param name="batchSize">The batch size for broadcasting.</param>
    /// <returns>A tensor of shape [batchSize, outputDimension] with biases broadcast.</returns>
    private Tensor<T> BroadcastBiases(Tensor<T> biases, int batchSize)
    {
        int outputDim = biases.Length;

        // Reshape bias from [outputDim] to [1, outputDim] and tile across batch
        var biasReshaped = biases.Reshape([1, outputDim]);
        var broadcast = Engine.TensorTile(biasReshaped, [batchSize, 1]);

        return broadcast;
    }

    /// <summary>
    /// Combines gradients for input and memory into a single tensor.
    /// </summary>
    /// <param name="inputGradient">The gradient with respect to the input tensor.</param>
    /// <param name="memoryGradient">The gradient with respect to the memory tensor.</param>
    /// <returns>A combined tensor containing both gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method combines the gradients for the input and memory tensors into a single tensor
    /// by concatenating them along the first dimension. This allows the backward pass to return
    /// a single gradient tensor that contains information about how both inputs should change.
    /// </para>
    /// <para><b>For Beginners:</b> This method packages two sets of gradients into one tensor.
    ///
    /// Since the MemoryReadLayer has two inputs (the input tensor and the memory tensor),
    /// the backward pass needs to calculate gradients for both. This method:
    ///
    /// - Takes the separate gradients for input and memory
    /// - Combines them into a single tensor by stacking them together
    /// - Returns this combined tensor to the previous layer
    ///
    /// Later, these combined gradients can be split apart again if needed to
    /// update both the input and memory pathways in the neural network.
    /// </para>
    /// </remarks>
    private static Tensor<T> CombineGradients(Tensor<T> inputGradient, Tensor<T> memoryGradient)
    {
        // Only concatenate when shapes match on all non-concatenated axes.
        if (inputGradient.Shape.Length != memoryGradient.Shape.Length)
        {
            return inputGradient;
        }

        for (int i = 1; i < inputGradient.Shape.Length; i++)
        {
            if (inputGradient.Shape[i] != memoryGradient.Shape[i])
            {
                return inputGradient;
            }
        }

        return Tensor<T>.Concatenate([inputGradient, memoryGradient], 0);
    }

    /// <summary>
    /// Updates the parameters of the memory read layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all trainable parameters of the layer (key weights, value weights, output weights,
    /// and output bias) based on the gradients calculated during the backward pass. The learning rate controls
    /// the size of the parameter updates. Each parameter is updated by subtracting the corresponding gradient
    /// multiplied by the learning rate.
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
    /// This is how the layer gradually improves its performance over many training iterations.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine operations for parameter updates
        var scaledKeyGrad = Engine.TensorMultiplyScalar(_keyWeightsGradient, learningRate);
        _keyWeights = Engine.TensorSubtract(_keyWeights, scaledKeyGrad);

        var scaledValueGrad = Engine.TensorMultiplyScalar(_valueWeightsGradient, learningRate);
        _valueWeights = Engine.TensorSubtract(_valueWeights, scaledValueGrad);

        var scaledOutputGrad = Engine.TensorMultiplyScalar(_outputWeightsGradient, learningRate);
        _outputWeights = Engine.TensorSubtract(_outputWeights, scaledOutputGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_outputBiasGradient, learningRate);
        _outputBias = Engine.TensorSubtract(_outputBias, scaledBiasGrad);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    /// <summary>
    /// Declares named input ports for this multi-input layer.
    /// </summary>
    public override IReadOnlyList<LayerPort> InputPorts =>
    [
        new LayerPort("query", GetInputShape()),
        new LayerPort("memory", GetOutputShape(), Required: false)
    ];

    /// <summary>
    /// Named multi-input forward pass.
    /// </summary>
    public override Tensor<T> Forward(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (!inputs.TryGetValue("query", out var query) || query == null)
            throw new ArgumentException("MemoryReadLayer requires 'query'.", nameof(inputs));
        // Memory is optional — falls back to default memory (parity with Forward(Tensor<T>))
        if (!inputs.TryGetValue("memory", out var memory) || memory == null)
            return Forward(query);
        return Forward(query, memory);
    }

    /// <summary>
    /// Performs a forward pass using a default identity-like memory tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor produced using the default memory tensor.</returns>
    /// <remarks>
    /// <para>
    /// This overload provides a default identity-like memory tensor so the layer can be used in
    /// generic pipelines that only pass a single input tensor. For custom memory contents, use
    /// Forward(input, memory) instead.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you use the layer without manually supplying a memory tensor.
    ///
    /// The layer creates a simple "identity" memory that passes values through, which is useful for
    /// quick tests or when a pipeline only supports a single input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        // Support single-argument Forward by using an identity-like memory matrix
        // This allows MemoryReadLayer to work in generic layer pipelines
        int memoryDimension = _keyWeights.Shape[1]; // Get memory dimension from key weights

        // Create a default identity memory that passes through values
        // Shape: [memoryDimension, memoryDimension] - acts as identity for attention
        var defaultMemory = new Tensor<T>([memoryDimension, memoryDimension]);
        for (int i = 0; i < memoryDimension; i++)
        {
            for (int j = 0; j < memoryDimension; j++)
            {
                defaultMemory[i, j] = (i == j) ? NumOps.One : NumOps.Zero;
            }
        }

        return Forward(input, defaultMemory);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        int[] inputShape = input._shape;
        int batchSize = inputShape.Length >= 2 ? inputShape[0] : 1;
        int inputDim = inputShape.Length >= 2 ? inputShape[1] : inputShape[0];

        // Derive dimensions from weight tensor shapes
        // _keyWeights: [inputDim, keyDim]
        // _valueWeights: [memDim, valueDim]
        // _outputWeights: [valueDim, outputDim]
        // _outputBias: [outputDim]
        int keyDim = _keyWeights.Shape[1];
        int memDim = _valueWeights.Shape[0];
        int valueDim = _valueWeights.Shape[1];
        int outputDim = _outputBias.Shape[0];

        // Get memory from inputs or use default memory if available
        Tensor<T>? memoryTensor = inputs.Length >= 2 ? inputs[1] : null;
        int memorySlots;
        IGpuBuffer memoryBuffer;
        bool memoryBufferOwned = false;

        if (memoryTensor is not null)
        {
            memorySlots = memoryTensor.Shape[0];
            memoryBuffer = memoryTensor.Buffer;
        }
        else if (_lastMemory is not null)
        {
            memorySlots = _lastMemory.Shape[0];
            float[] memoryData = DirectGpuEngine.ToFloatArray<T>(_lastMemory.Data.ToArray());
            memoryBuffer = backend.AllocateBuffer(memoryData);
            memoryBufferOwned = true;
        }
        else
        {
            throw new InvalidOperationException("Memory not provided and no cached memory available.");
        }

        // Upload weights to GPU
        float[] keyWeightData = DirectGpuEngine.ToFloatArray<T>(_keyWeights.Data.ToArray());
        float[] valueWeightData = DirectGpuEngine.ToFloatArray<T>(_valueWeights.Data.ToArray());
        float[] outputWeightData = DirectGpuEngine.ToFloatArray<T>(_outputWeights.Data.ToArray());
        float[] outputBiasData = DirectGpuEngine.ToFloatArray<T>(_outputBias.Data.ToArray());

        using var keyWeightsBuffer = backend.AllocateBuffer(keyWeightData);
        using var valueWeightsBuffer = backend.AllocateBuffer(valueWeightData);
        using var outputWeightsBuffer = backend.AllocateBuffer(outputWeightData);
        using var outputBiasBuffer = backend.AllocateBuffer(outputBiasData);

        // Step 1: Compute keys = input @ keyWeights -> [batch, keyDim]
        int keysSize = batchSize * keyDim;
        using var keysBuffer = backend.AllocateBuffer(keysSize);
        backend.Gemm(input.Buffer, keyWeightsBuffer, keysBuffer, batchSize, keyDim, inputDim);

        // Step 2: Compute attention scores = keys @ memory^T -> [batch, memorySlots]
        // Memory is [memSlots, memDim], need [memDim, memSlots] for matmul
        int memorySize = memorySlots * memDim;
        using var memoryTransposedBuffer = backend.AllocateBuffer(memorySize);
        backend.BatchedTranspose(memoryBuffer, memoryTransposedBuffer, 1, memorySlots, memDim);

        int scoresSize = batchSize * memorySlots;
        using var scoresBuffer = backend.AllocateBuffer(scoresSize);
        backend.Gemm(keysBuffer, memoryTransposedBuffer, scoresBuffer, batchSize, memorySlots, keyDim);

        // Step 3: Apply softmax to get attention weights
        using var attentionBuffer = backend.AllocateBuffer(scoresSize);
        backend.Softmax(scoresBuffer, attentionBuffer, batchSize, memorySlots);

        // Step 4: Read from memory: attention @ memory -> [batch, memDim]
        int readSize = batchSize * memDim;
        using var readBuffer = backend.AllocateBuffer(readSize);
        backend.Gemm(attentionBuffer, memoryBuffer, readBuffer, batchSize, memDim, memorySlots);

        // Step 5: Apply value transformation: read @ valueWeights -> [batch, valueDim]
        int transformedSize = batchSize * valueDim;
        using var transformedBuffer = backend.AllocateBuffer(transformedSize);
        backend.Gemm(readBuffer, valueWeightsBuffer, transformedBuffer, batchSize, valueDim, memDim);

        // Step 6: Apply output projection: transformed @ outputWeights -> [batch, outputDim]
        int outputSize = batchSize * outputDim;
        using var projectedBuffer = backend.AllocateBuffer(outputSize);
        backend.Gemm(transformedBuffer, outputWeightsBuffer, projectedBuffer, batchSize, outputDim, valueDim);

        // Step 7: Add bias - copy projected data then apply BiasAdd in-place
        var withBiasBuffer = backend.AllocateBuffer(outputSize);
        backend.Copy(projectedBuffer, withBiasBuffer, outputSize);
        backend.BiasAdd(withBiasBuffer, outputBiasBuffer, withBiasBuffer, batchSize, outputDim);

        // Step 8: Apply activation using helper method
        var activationType = GetFusedActivationType();
        IGpuBuffer outputBuffer;
        if (activationType != FusedActivationType.None)
        {
            var activatedBuffer = backend.AllocateBuffer(outputSize);
            ApplyGpuActivation(backend, withBiasBuffer, activatedBuffer, outputSize, activationType);
            withBiasBuffer.Dispose();
            outputBuffer = activatedBuffer;
        }
        else
        {
            outputBuffer = withBiasBuffer;
        }

        if (memoryBufferOwned)
        {
            memoryBuffer.Dispose();
        }

        return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, [batchSize, outputDim], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Gets all trainable parameters from the memory read layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. It concatenates
    /// the key weights, value weights, output weights, and output bias into a single vector. This is useful
    /// for optimization algorithms that operate on all parameters at once, or for saving and loading model weights.
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
            new Vector<T>(_keyWeights.ToArray()),
            new Vector<T>(_valueWeights.ToArray()),
            new Vector<T>(_outputWeights.ToArray()),
            new Vector<T>(_outputBias.ToArray())
        );
    }

    /// <summary>
    /// Sets the trainable parameters for the memory read layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector. It extracts the appropriate
    /// portions of the input vector for each parameter (key weights, value weights, output weights, and output bias).
    /// This is useful for loading saved model weights or for implementing optimization algorithms that operate
    /// on all parameters at once.
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
        int keySize = _keyWeights.Shape[0] * _keyWeights.Shape[1];
        int valueSize = _valueWeights.Shape[0] * _valueWeights.Shape[1];
        int outputSize = _outputWeights.Shape[0] * _outputWeights.Shape[1];
        int biasSize = _outputBias.Length;
        int totalParams = keySize + valueSize + outputSize + biasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set key weights using Tensor.FromVector
        var keyParams = parameters.SubVector(index, keySize);
        _keyWeights = Tensor<T>.FromVector(keyParams).Reshape(_keyWeights._shape);
        index += keySize;

        // Set value weights
        var valueParams = parameters.SubVector(index, valueSize);
        _valueWeights = Tensor<T>.FromVector(valueParams).Reshape(_valueWeights._shape);
        index += valueSize;

        // Set output weights
        var outputParams = parameters.SubVector(index, outputSize);
        _outputWeights = Tensor<T>.FromVector(outputParams).Reshape(_outputWeights._shape);
        index += outputSize;

        // Set output bias
        var biasParams = parameters.SubVector(index, biasSize);
        _outputBias = Tensor<T>.FromVector(biasParams);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    /// <summary>
    /// Resets the internal state of the memory read layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the memory read layer, including the cached inputs, memory,
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
    public override Vector<T> GetParameterGradients()
    {
        if (_keyWeightsGradient == null || _valueWeightsGradient == null ||
            _outputWeightsGradient == null || _outputBiasGradient == null)
            return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_keyWeightsGradient.ToArray()),
            new Vector<T>(_valueWeightsGradient.ToArray()),
            new Vector<T>(_outputWeightsGradient.ToArray()),
            new Vector<T>(_outputBiasGradient.ToArray()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _keyWeightsGradient = null; _valueWeightsGradient = null;
        _outputWeightsGradient = null; _outputBiasGradient = null;
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastMemory = null;
        _lastOutput = null;
        _lastAttentionScores = null;
        _lastTransformed = null;

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
    /// This method computes an attention sparsity loss that encourages focused memory access patterns.
    /// The loss is computed as the negative entropy of the attention weights: L = -Σ(p * log(p))
    /// where p represents the attention probabilities. Lower entropy (more focused attention) results in lower loss.
    /// This encourages the layer to attend to specific memory locations rather than spreading attention uniformly.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates a penalty for unfocused attention patterns.
    ///
    /// Attention sparsity loss:
    /// - Measures how focused the attention is on specific memory locations
    /// - Lower values mean more focused attention (good)
    /// - Higher values mean attention is spread across many locations (less focused)
    ///
    /// Why this is useful:
    /// - In most tasks, you want to retrieve specific relevant information from memory
    /// - Spreading attention too thin means you get a "blurry" mix of information
    /// - Focused attention means you get clear, specific information
    ///
    /// Example: If you're answering "What is the capital of France?" from memory,
    /// you want focused attention on the entry about Paris, not a mix of all French cities.
    ///
    /// Technical note: The loss is computed using entropy. Entropy measures how "spread out" a distribution is.
    /// - Low entropy = focused distribution (e.g., [0.9, 0.05, 0.05] - mostly on first item)
    /// - High entropy = spread out distribution (e.g., [0.33, 0.33, 0.34] - spread evenly)
    /// We use negative entropy as the loss, so the network is penalized for high entropy (unfocused attention).
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastAttentionScores == null)
        {
            _lastAttentionSparsityLoss = NumOps.Zero;
            return _lastAttentionSparsityLoss;
        }

        // Compute negative entropy to encourage low entropy (focused attention) using tensor ops
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
    /// - TotalAttentionSparsityLoss: The computed penalty for unfocused attention
    /// - AttentionSparsityWeight: How much this penalty affects the overall training
    /// - UseAttentionSparsity: Whether this penalty is currently enabled
    ///
    /// You can use this information to:
    /// - Monitor if attention is becoming more focused over time
    /// - Debug training issues related to memory access
    /// - Understand how the layer is learning to retrieve information
    ///
    /// Example: If TotalAttentionSparsityLoss is decreasing during training, it means the layer
    /// is learning to be more focused in its memory access, which is typically a good sign.
    /// If it's staying high or increasing, it might mean the layer is having trouble learning
    /// which parts of memory are relevant.
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

}
