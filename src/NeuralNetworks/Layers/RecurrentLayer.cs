#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Helpers;


namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a recurrent neural network layer that processes sequential data by maintaining a hidden state.
/// </summary>
/// <remarks>
/// <para>
/// The RecurrentLayer implements a basic recurrent neural network (RNN) that processes sequence data by 
/// maintaining and updating a hidden state over time steps. For each element in the sequence, the layer 
/// computes a new hidden state based on the current input and the previous hidden state. This allows the 
/// network to capture temporal dependencies and patterns in sequential data.
/// </para>
/// <para><b>For Beginners:</b> This layer is designed to work with data that comes in sequences.
/// 
/// Think of the RecurrentLayer as having a memory that helps it understand sequences:
/// - When reading a sentence word by word, it remembers previous words to understand context
/// - When analyzing time series data, it remembers past values to predict future trends
/// - When processing video frames, it remembers earlier frames to track movement
/// 
/// Unlike regular layers that process each input independently, this layer:
/// - Takes both the current input and its own memory (hidden state) to make decisions
/// - Updates its memory after seeing each item in the sequence
/// - Passes this updated memory forward to the next time step
/// 
/// For example, when processing the sentence "The cat sat on the mat":
/// - At the word "cat", it remembers "The" came before
/// - At the word "sat", it remembers both "The" and "cat" came before
/// - This context helps it understand the full meaning of the sentence
/// 
/// This ability to maintain information across a sequence makes recurrent layers 
/// powerful for tasks involving text, time series, audio, and other sequential data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, HasTrainingMode = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class RecurrentLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Tensor storing the weight parameters for connections between inputs and hidden neurons.
    /// </summary>
    /// <remarks>
    /// This tensor has dimensions [hiddenSize, inputSize], where each row represents the weights
    /// for one hidden neuron. These weights determine how each input feature influences each
    /// hidden neuron and are trainable parameters of the layer.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputWeights;

    /// <summary>
    /// Tensor storing the weight parameters for connections between previous hidden state and current hidden state.
    /// </summary>
    /// <remarks>
    /// This tensor has dimensions [hiddenSize, hiddenSize], where each row represents the weights
    /// for one hidden neuron. These weights determine how the previous hidden state influences the
    /// current hidden state and are what gives the recurrent layer its "memory" capability.
    /// </remarks>
    private Tensor<T> _hiddenWeights;

    /// <summary>
    /// Tensor storing the bias parameters for each hidden neuron.
    /// </summary>
    /// <remarks>
    /// This tensor has length hiddenSize, where each element is a constant value added to the
    /// weighted sum for the corresponding hidden neuron. Biases allow the network to shift the
    /// activation function, giving it more flexibility to fit the data.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _biases;

    /// <summary>
    /// Gets the total number of trainable parameters in this recurrent layer.
    /// </summary>
    /// <remarks>
    /// The parameter count includes all weights and biases:
    /// - Input weights: inputSize × hiddenSize
    /// - Hidden weights: hiddenSize × hiddenSize
    /// - Biases: hiddenSize
    /// </remarks>
    public override int ParameterCount =>
        _inputWeights.Length + _hiddenWeights.Length + _biases.Length;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// sequence of input vectors that were processed in the most recent forward pass. The tensor
    /// is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores the hidden state tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached hidden state is needed during the backward pass to compute gradients. It holds the
    /// sequence of hidden state vectors that were computed in the most recent forward pass. The tensor
    /// is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastHiddenState;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached output is needed during the backward pass to compute certain derivatives.
    /// It holds the sequence of output vectors that were produced in the most recent forward pass.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the gradients of the loss with respect to the input weight parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for all input weight parameters during the backward pass.
    /// It has the same dimensions as the _inputWeights tensor and is used to update the input weights during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _inputWeightsGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the hidden weight parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for all hidden weight parameters during the backward pass.
    /// It has the same dimensions as the _hiddenWeights tensor and is used to update the hidden weights during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _hiddenWeightsGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the bias parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for all bias parameters during the backward pass.
    /// It has the same length as the _biases tensor and is used to update the biases during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for RecurrentLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the RecurrentLayer has trainable parameters (input weights, hidden weights, and biases)
    /// that can be optimized during the training process using backpropagation through time (BPTT). The gradients of
    /// these parameters are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (weights and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the weights and biases will
    /// automatically adjust to better recognize patterns in your sequence data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="RecurrentLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state and output at each time step.</param>
    /// <param name="activationFunction">The activation function to apply to the hidden state. Defaults to Tanh if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RecurrentLayer with the specified dimensions and a scalar activation function.
    /// The weights are initialized using Xavier/Glorot initialization to improve training dynamics, and the biases
    /// are initialized to zero. A scalar activation function is applied element-wise to each hidden neuron independently.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new recurrent layer for your neural network using a simple activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer at each time step
    /// - hiddenSize: How many memory units (neurons) the layer has
    /// - activationFunction: How to transform the hidden state (defaults to tanh)
    /// 
    /// The hiddenSize determines the "memory capacity" of the layer:
    /// - Larger values can remember more information about the sequence
    /// - But also require more computation and might be harder to train
    /// 
    /// Tanh is commonly used as the activation function because:
    /// - It outputs values between -1 and 1
    /// - It has a nice gradient for training
    /// - It works well for capturing both positive and negative patterns
    /// 
    /// The layer starts with carefully initialized weights to help training proceed smoothly.
    /// </para>
    /// </remarks>
    public RecurrentLayer(int inputSize, int hiddenSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [hiddenSize], activationFunction ?? new TanhActivation<T>())
    {
        _inputWeights = new Tensor<T>([hiddenSize, inputSize]);
        _hiddenWeights = new Tensor<T>([hiddenSize, hiddenSize]);
        _biases = new Tensor<T>([hiddenSize]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_inputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_hiddenWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RecurrentLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state and output at each time step.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply to the hidden state. Defaults to Tanh if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RecurrentLayer with the specified dimensions and a vector activation function.
    /// The weights are initialized using Xavier/Glorot initialization to improve training dynamics, and the biases
    /// are initialized to zero. A vector activation function is applied to the entire hidden state vector at once,
    /// which allows for interactions between different hidden neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new recurrent layer for your neural network using an advanced activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer at each time step
    /// - hiddenSize: How many memory units (neurons) the layer has
    /// - vectorActivationFunction: How to transform the entire hidden state as a group
    /// 
    /// A vector activation means all hidden neurons are calculated together, which can capture relationships between them.
    /// This is an advanced option that might be useful for specific types of sequence problems.
    /// 
    /// This constructor works the same as the scalar version, but allows for more sophisticated activation patterns
    /// across the hidden state. Most RNN implementations use the scalar version with tanh activation.
    /// </para>
    /// </remarks>
    public RecurrentLayer(int inputSize, int hiddenSize, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputSize], [hiddenSize], vectorActivationFunction ?? new TanhActivation<T>())
    {
        _inputWeights = new Tensor<T>([hiddenSize, inputSize]);
        _hiddenWeights = new Tensor<T>([hiddenSize, hiddenSize]);
        _biases = new Tensor<T>([hiddenSize]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_inputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_hiddenWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Performs the forward pass of the recurrent layer.
    /// </summary>
    /// <param name="input">The input tensor to process, with shape [sequenceLength, batchSize, inputSize].</param>
    /// <returns>The output tensor after recurrent processing, with shape [sequenceLength, batchSize, hiddenSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the recurrent layer. It processes each element in the input sequence
    /// in order, updating the hidden state at each time step based on the current input and the previous hidden state.
    /// The initial hidden state is set to zero. The method caches the input, hidden states, and outputs for use during
    /// the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your sequence data through the recurrent layer.
    /// 
    /// During the forward pass:
    /// 1. The layer starts with an empty memory (hidden state of zeros)
    /// 2. For each item in the sequence (like each word in a sentence):
    ///    - It takes both the current input and its current memory
    ///    - It calculates a new memory state based on these values
    ///    - It saves this memory for the next item in the sequence
    /// 3. The outputs at each time step become the overall output of the layer
    /// 
    /// The formula at each step is approximately:
    /// new_memory = activation(input_weights × current_input + hidden_weights × previous_memory + bias)
    /// 
    /// This step-by-step processing allows the layer to build up an understanding of the entire sequence.
    /// The layer saves all inputs, hidden states, and outputs for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims for rank > 3
        Tensor<T> input3D;
        int sequenceLength;
        int batchSize;

        // Shape ops via Engine so the gradient tape records them.
        if (rank == 1)
        {
            // 1D: [inputSize] -> treat as single timestep, single batch
            sequenceLength = 1;
            batchSize = 1;
            int inputSize = input.Shape[0];
            input3D = Engine.Reshape(input, new[] { 1, 1, inputSize });
        }
        else if (rank == 2)
        {
            // 2D: [sequenceLength, inputSize] -> add batch dim
            sequenceLength = input.Shape[0];
            batchSize = 1;
            int inputSize = input.Shape[1];
            input3D = Engine.Reshape(input, new[] { sequenceLength, 1, inputSize });
        }
        else if (rank == 3)
        {
            // Standard 3D: [sequenceLength, batchSize, inputSize]
            sequenceLength = input.Shape[0];
            batchSize = input.Shape[1];
            input3D = input;
        }
        else
        {
            // Higher-rank: collapse middle dims into batch
            sequenceLength = input.Shape[0];
            int flatBatch = 1;
            for (int d = 1; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            int inputSize = input.Shape[rank - 1];
            input3D = Engine.Reshape(input, new[] { sequenceLength, flatBatch, inputSize });
        }

        // Cache input only if training
        if (IsTrainingMode)
        {
            _lastInput = input3D;
        }

        int hiddenSize = _inputWeights.Shape[0];
        int actualInputSize = input3D.Shape[2]; // [seqLen, batch, inputSize]
        int expectedInputSize = _inputWeights.Shape[1];

        if (actualInputSize != expectedInputSize)
        {
            throw new ArgumentException(
                $"Input size mismatch: expected {expectedInputSize} but got {actualInputSize}.",
                nameof(input));
        }

        // Allocate output tensor (cannot rent — reshape at end creates new tensor,
        // leaving rented buffer unreturned and potentially reused with stale data)
        var output = new Tensor<T>([sequenceLength, batchSize, hiddenSize]);
        var hiddenState = new Tensor<T>([sequenceLength + 1, batchSize, hiddenSize]);
        hiddenState.Fill(NumOps.Zero);

        // Process sequence using tensor operations. All projections via
        // Engine.* so the gradient tape records the forward chain.
        // _inputWeights shape: [hiddenSize, inputSize]
        // _hiddenWeights shape: [hiddenSize, hiddenSize]
        // _biases shape: [hiddenSize]
        var inputWeightsT = Engine.TensorPermute(_inputWeights, new[] { 1, 0 }); // [inputSize, hiddenSize]
        var hiddenWeightsT = Engine.TensorPermute(_hiddenWeights, new[] { 1, 0 }); // [hiddenSize, hiddenSize]

        for (int t = 0; t < sequenceLength; t++)
        {
            // VECTORIZED: Extract input slice for time step t: [batchSize, inputSize]
            var inputAtT = Engine.TensorSliceAxis(input3D, 0, t); // [batchSize, inputSize]

            // VECTORIZED: Extract previous hidden state: [batchSize, hiddenSize]
            var prevHidden = Engine.TensorSliceAxis(hiddenState, 0, t); // [batchSize, hiddenSize]

            // Compute: h_t = activation(input @ W_input^T + h_{t-1} @ W_hidden^T + biases)
            // Tape-tracked matmul (was .MatrixMultiply which bypasses the tape).
            var inputContribution = Engine.TensorMatMul(inputAtT, inputWeightsT); // [batchSize, hiddenSize]
            var hiddenContribution = Engine.TensorMatMul(prevHidden, hiddenWeightsT); // [batchSize, hiddenSize]

            // Sum contributions and add biases using Engine operations
            var preActivation = Engine.TensorAdd(inputContribution, hiddenContribution);
            preActivation = Engine.TensorBroadcastAdd(preActivation, _biases); // Broadcasting biases across batch

            // Apply activation
            var newHidden = ApplyActivation(preActivation);

            // VECTORIZED: Store results using tensor slice operations
            Engine.TensorSetSliceAxis(output, newHidden, 0, t);
            Engine.TensorSetSliceAxis(hiddenState, newHidden, 0, t + 1);
        }

        _lastHiddenState = hiddenState;
        _lastOutput = output;

        // Restore original batch dimensions for any-rank support — via Engine
        // for tape recording.
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Output shape: [sequenceLength, ...middleDims, hiddenSize]
            int[] newShape = new int[_originalInputShape.Length];
            newShape[0] = sequenceLength;
            for (int d = 1; d < _originalInputShape.Length - 1; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 1] = hiddenSize;
            output = Engine.Reshape(output, newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // 2D input -> 2D output (remove batch dim)
            output = Engine.Reshape(output, new[] { sequenceLength, hiddenSize });
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 1)
        {
            // 1D input -> 1D output (just the hidden size)
            output = Engine.Reshape(output, new[] { hiddenSize });
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after RNN processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input._shape;
        int rank = shape.Length;
        int hiddenSize = _inputWeights.Shape[0];
        int inputSize = _inputWeights.Shape[1];

        // Determine sequence length, batch size from shape
        int sequenceLength;
        int batchSize;
        if (rank == 1)
        {
            sequenceLength = 1;
            batchSize = 1;
        }
        else if (rank == 2)
        {
            sequenceLength = shape[0];
            batchSize = 1;
        }
        else if (rank == 3)
        {
            sequenceLength = shape[0];
            batchSize = shape[1];
        }
        else
        {
            // Higher rank: collapse middle dims into batch
            sequenceLength = shape[0];
            int flatBatch = 1;
            for (int d = 1; d < rank - 1; d++)
                flatBatch *= shape[d];
            batchSize = flatBatch;
        }

        int hiddenBufferSize = batchSize * hiddenSize;
        int inputSliceSize = batchSize * inputSize;
        int outputSize = sequenceLength * batchSize * hiddenSize;
        int[] outputShape = [sequenceLength, batchSize, hiddenSize];

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Upload transposed weights to GPU
        using var inputWeightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_inputWeights).ToArray()));
        using var hiddenWeightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_hiddenWeights).ToArray()));
        using var biasesBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biases.ToArray()));

        // Allocate hidden state buffers
        var currentHBuffer = backend.AllocateBuffer(hiddenBufferSize);
        backend.Fill(currentHBuffer, 0.0f, hiddenBufferSize); // Initialize to zeros

        // Allocate temporary buffers
        using var tempBuffer1 = backend.AllocateBuffer(hiddenBufferSize);
        using var tempBuffer2 = backend.AllocateBuffer(hiddenBufferSize);
        using var preActivationBuffer = backend.AllocateBuffer(hiddenBufferSize);
        var newHBuffer = backend.AllocateBuffer(hiddenBufferSize);

        try
        {
            // Process each time step
            for (int t = 0; t < sequenceLength; t++)
            {
                // Get input slice for this timestep
                int inputSliceOffset = t * inputSliceSize;
                var inputSlice = input.CreateView(inputSliceOffset, [batchSize, inputSize]);

                // h_t = activation(input @ W_input^T + h_{t-1} @ W_hidden^T + biases)
                // Compute input contribution: input @ W_input^T
                backend.Gemm(inputSlice.Buffer, inputWeightsBuffer, tempBuffer1, batchSize, hiddenSize, inputSize);

                // Compute hidden contribution: h_{t-1} @ W_hidden^T
                backend.Gemm(currentHBuffer, hiddenWeightsBuffer, tempBuffer2, batchSize, hiddenSize, hiddenSize);

                // Sum contributions: preAct = inputContrib + hiddenContrib
                backend.Add(tempBuffer1, tempBuffer2, preActivationBuffer, hiddenBufferSize);

                // Add biases
                backend.BiasAdd(preActivationBuffer, biasesBuffer, preActivationBuffer, batchSize, hiddenSize);

                // Apply activation (Tanh is the default for RNN)
                backend.Tanh(preActivationBuffer, newHBuffer, hiddenBufferSize);

                // Store hidden state in output
                int outputOffset = t * hiddenBufferSize;
                backend.Copy2DStrided(newHBuffer, outputBuffer, 1, hiddenBufferSize, outputSize, outputOffset);

                // Swap hidden state buffers
                var tempH = currentHBuffer;
                currentHBuffer = newHBuffer;
                newHBuffer = tempH;
            }

            // Dispose the buffer we're not returning
            newHBuffer.Dispose();
            currentHBuffer.Dispose();

            var outputTensor = GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);

            // Cache for GPU-resident training
            if (IsTrainingMode)
            {
                _gpuLastInput = input;
                _gpuLastOutput = outputTensor;
            }

            return outputTensor;
        }
        catch
        {
            // Clean up on error
            outputBuffer.Dispose();
            currentHBuffer.Dispose();
            newHBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Broadcasts biases across the batch dimension.
    /// </summary>
    private Tensor<T> BroadcastBiases(Tensor<T> biases, int batchSize)
    {
        // VECTORIZED: Use TensorExpandDims + TensorTile for broadcasting
        var biasLength = biases.Length;
        var biases2D = biases.Reshape([1, biasLength]); // [1, biasLength]
        var broadcasted = Engine.TensorTile(biases2D, [batchSize, 1]); // [batchSize, biasLength]
        return broadcasted;
    }


    private Tensor<T>? _inputWeightsVelocity;
    private Tensor<T>? _hiddenWeightsVelocity;
    private Tensor<T>? _biasesVelocity;

    #region GPU Training Fields
    private Tensor<T>? _gpuLastInput;
    private Tensor<T>? _gpuLastOutput;

    // GPU weight buffers
    private Tensor<T>? _gpuInputWeights;
    private Tensor<T>? _gpuHiddenWeights;
    private Tensor<T>? _gpuBiases;

    // GPU gradient buffers
    private Tensor<T>? _gpuInputWeightsGradient;
    private Tensor<T>? _gpuHiddenWeightsGradient;
    private Tensor<T>? _gpuBiasesGradient;

    // GPU velocity buffers (SGD momentum)
    private Tensor<T>? _gpuInputWeightsVelocity;
    private Tensor<T>? _gpuHiddenWeightsVelocity;
    private Tensor<T>? _gpuBiasesVelocity;

    // GPU Adam first moment buffers
    private Tensor<T>? _gpuInputWeightsM;
    private Tensor<T>? _gpuHiddenWeightsM;
    private Tensor<T>? _gpuBiasesM;

    // GPU Adam second moment buffers
    private Tensor<T>? _gpuInputWeightsV;
    private Tensor<T>? _gpuHiddenWeightsV;
    private Tensor<T>? _gpuBiasesV;
    #endregion

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Updates the parameters of the recurrent layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the input weights, hidden weights, and biases of the recurrent layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The input weight values are adjusted based on their gradients
    /// 2. The hidden weight values are adjusted based on their gradients
    /// 3. The bias values are adjusted based on their gradients
    /// 4. The learning rate controls how big each update step is
    /// 
    /// These updates help the layer:
    /// - Pay more attention to important input features
    /// - Better remember relevant information from previous time steps
    /// - Adjust its baseline activation levels
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_inputWeightsGradient == null || _hiddenWeightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_inputWeightsVelocity == null)
            {
                _inputWeightsVelocity = new Tensor<T>(_inputWeights._shape);
                _inputWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_inputWeightsVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_hiddenWeightsVelocity == null)
            {
                _hiddenWeightsVelocity = new Tensor<T>(_hiddenWeights._shape);
                _hiddenWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_hiddenWeightsVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_biasesVelocity == null)
            {
                _biasesVelocity = new Tensor<T>(_biases._shape);
                _biasesVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_biasesVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_inputWeights, _inputWeightsGradient, _inputWeightsVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_hiddenWeights, _hiddenWeightsGradient, _hiddenWeightsVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_biases, _biasesGradient, _biasesVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            var scaledInputGrad = Engine.TensorMultiplyScalar(_inputWeightsGradient, learningRate);
            var scaledHiddenGrad = Engine.TensorMultiplyScalar(_hiddenWeightsGradient, learningRate);
            var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);

            _inputWeights = Engine.TensorSubtract(_inputWeights, scaledInputGrad);
            _hiddenWeights = Engine.TensorSubtract(_hiddenWeights, scaledHiddenGrad);
            _biases = Engine.TensorSubtract(_biases, scaledBiasGrad);

            Engine.InvalidatePersistentTensor(_inputWeights);
            Engine.InvalidatePersistentTensor(_hiddenWeights);
            Engine.InvalidatePersistentTensor(_biases);
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the recurrent layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (input weights, hidden weights, and biases).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the recurrent layer as a single vector.
    /// The input weights are stored first, followed by the hidden weights, and finally the biases.
    /// This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the recurrent layer.
    /// 
    /// The parameters:
    /// - Are the weights and biases that the recurrent layer learns during training
    /// - Control how the layer processes sequence information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The input weights are stored first in the vector, followed by the hidden weights, and finally the biases.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // VECTORIZED: Concatenate tensor data using Vector.Concatenate
        return Vector<T>.Concatenate(
            _inputWeights.ToVector(),
            _hiddenWeights.ToVector(),
            _biases.ToVector()
        );
    }

    /// <summary>
    /// Gets all parameter gradients of the recurrent layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameter gradients (input weight gradients, hidden weight gradients, and bias gradients).</returns>
    public override Vector<T> GetParameterGradients()
    {
        // If gradients haven't been computed yet, return zero gradients
        if (_inputWeightsGradient == null || _hiddenWeightsGradient == null || _biasesGradient == null)
        {
            return new Vector<T>(ParameterCount);
        }

        // VECTORIZED: Concatenate gradient data using Vector.Concatenate
        return Vector<T>.Concatenate(
            _inputWeightsGradient.ToVector(),
            _hiddenWeightsGradient.ToVector(),
            _biasesGradient.ToVector()
        );
    }

    /// <summary>
    /// Sets the trainable parameters of the recurrent layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (input weights, hidden weights, and biases) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the recurrent layer from a single vector.
    /// The vector should contain the input weight values first, followed by the hidden weight values,
    /// and finally the bias values. This is useful for loading saved model weights or for
    /// implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights and biases in the recurrent layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the input weights
    /// - The middle part of the vector is used for the hidden weights
    /// - The last part of the vector is used for the biases
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int inputWeightsSize = _inputWeights.Length;
        int hiddenWeightsSize = _hiddenWeights.Length;
        int totalParams = inputWeightsSize + hiddenWeightsSize + _biases.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Create new tensors to ensure independence from cloned layers
        int idx = 0;
        _inputWeights = new Tensor<T>(_inputWeights._shape);
        for (int i = 0; i < inputWeightsSize; i++)
            _inputWeights[i] = parameters[idx++];

        _hiddenWeights = new Tensor<T>(_hiddenWeights._shape);
        for (int i = 0; i < hiddenWeightsSize; i++)
            _hiddenWeights[i] = parameters[idx++];

        _biases = new Tensor<T>(_biases._shape);
        for (int i = 0; i < _biases.Length; i++)
            _biases[i] = parameters[idx++];

        RegisterTrainableParameter(_inputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_hiddenWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputSize"] = _inputWeights.Shape[1].ToString();
        metadata["HiddenSize"] = _inputWeights.Shape[0].ToString();
        return metadata;
    }

    /// <summary>
    /// Resets the internal state of the recurrent layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the recurrent layer, including the cached inputs, hidden states,
    /// and outputs from the forward pass, and the gradients from the backward pass. This is useful when starting to
    /// process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs, hidden states, and outputs from previous calculations are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous sequences
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence of data
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// 
    /// The weights and biases (the learned parameters) are not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputWeightsGradient = null;
        _hiddenWeightsGradient = null;
        _biasesGradient = null;
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastHiddenState = null;
        _lastOutput = null;
        _inputWeightsGradient = null;
        _hiddenWeightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Initializes the weights and biases of the recurrent layer with proper scaling.
    /// </summary>
    /// <remarks>
    /// This private method initializes the weights using Xavier/Glorot initialization, which scales
    /// the random values based on the dimensions of the matrices to help with training dynamics.
    /// The input weights and hidden weights are initialized separately with their own scaling factors,
    /// and the biases are initialized to zero. This initialization strategy helps prevent vanishing
    /// or exploding gradients during training.
    /// </remarks>
    private void InitializeParameters()
    {
        // VECTORIZED: Initialize weights and biases (Xavier/Glorot initialization)
        int hiddenSize = _inputWeights.Shape[0];
        int inputSize = _inputWeights.Shape[1];

        T inputScale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(2.0, (hiddenSize + inputSize))));
        T hiddenScale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(2.0, (hiddenSize + hiddenSize))));
        T half = NumOps.FromDouble(0.5);

        // Generate random input weights: (random - 0.5) * scale
        var inputRandom = Tensor<T>.CreateRandom(_inputWeights.Length, 1).Reshape(_inputWeights._shape);
        var inputHalf = new Tensor<T>(_inputWeights._shape);
        inputHalf.Fill(half);
        var inputCentered = Engine.TensorSubtract(inputRandom, inputHalf);
        _inputWeights = Engine.TensorMultiplyScalar(inputCentered, inputScale);

        // Generate random hidden weights: (random - 0.5) * scale
        var hiddenRandom = Tensor<T>.CreateRandom(_hiddenWeights.Length, 1).Reshape(_hiddenWeights._shape);
        var hiddenHalf = new Tensor<T>(_hiddenWeights._shape);
        hiddenHalf.Fill(half);
        var hiddenCentered = Engine.TensorSubtract(hiddenRandom, hiddenHalf);
        _hiddenWeights = Engine.TensorMultiplyScalar(hiddenCentered, hiddenScale);

        // Initialize biases to zero (standard practice per Elman 1990)
        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Updates parameters on GPU using the configured optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Ensure GPU weight buffers exist
        _gpuInputWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _inputWeights, GpuTensorRole.Weight);
        _gpuHiddenWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _hiddenWeights, GpuTensorRole.Weight);
        _gpuBiases ??= GpuTensorHelper.UploadToGpu<T>(backend, _biases, GpuTensorRole.Weight);

        // Ensure optimizer state exists
        EnsureRecurrentOptimizerState(config, backend);

        // Apply updates for input weights
        if (_gpuInputWeightsGradient is not null)
        {
            var inputWeightsState = BuildRecurrentOptimizerState("inputWeights");
            config.ApplyUpdate(backend, _gpuInputWeights.Buffer, _gpuInputWeightsGradient.Buffer, inputWeightsState, _inputWeights.Length);
        }

        // Apply updates for hidden weights
        if (_gpuHiddenWeightsGradient is not null)
        {
            var hiddenWeightsState = BuildRecurrentOptimizerState("hiddenWeights");
            config.ApplyUpdate(backend, _gpuHiddenWeights.Buffer, _gpuHiddenWeightsGradient.Buffer, hiddenWeightsState, _hiddenWeights.Length);
        }

        // Apply updates for biases
        if (_gpuBiasesGradient is not null)
        {
            var biasesState = BuildRecurrentOptimizerState("biases");
            config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasesGradient.Buffer, biasesState, _biases.Length);
        }

        // Download updated weights back to CPU tensors
        _inputWeights = _gpuInputWeights;
        _hiddenWeights = _gpuHiddenWeights;
        _biases = _gpuBiases;

        // Notify engine that tensor data has changed
        Engine.InvalidatePersistentTensor(_inputWeights);
        Engine.InvalidatePersistentTensor(_hiddenWeights);
        Engine.InvalidatePersistentTensor(_biases);
    }

    private void EnsureRecurrentOptimizerState(IGpuOptimizerConfig config, IDirectGpuBackend backend)
    {
        var optimizerType = config.OptimizerType;

        // Ensure velocity buffers for SGD momentum, NAG, LARS
        if (optimizerType == GpuOptimizerType.Sgd || optimizerType == GpuOptimizerType.Nag || optimizerType == GpuOptimizerType.Lars)
        {
            _gpuInputWeightsVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_inputWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuHiddenWeightsVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_hiddenWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }

        // Ensure Adam moment buffers
        if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            _gpuInputWeightsM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_inputWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuInputWeightsV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_inputWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuHiddenWeightsM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_hiddenWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuHiddenWeightsV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_hiddenWeights.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    private GpuOptimizerState BuildRecurrentOptimizerState(string paramName)
    {
        return paramName switch
        {
            "inputWeights" => new GpuOptimizerState
            {
                Velocity = _gpuInputWeightsVelocity?.Buffer,
                M = _gpuInputWeightsM?.Buffer,
                V = _gpuInputWeightsV?.Buffer
            },
            "hiddenWeights" => new GpuOptimizerState
            {
                Velocity = _gpuHiddenWeightsVelocity?.Buffer,
                M = _gpuHiddenWeightsM?.Buffer,
                V = _gpuHiddenWeightsV?.Buffer
            },
            "biases" => new GpuOptimizerState
            {
                Velocity = _gpuBiasesVelocity?.Buffer,
                M = _gpuBiasesM?.Buffer,
                V = _gpuBiasesV?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }
}
