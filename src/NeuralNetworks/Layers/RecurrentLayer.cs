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
public class RecurrentLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Matrix storing the weight parameters for connections between inputs and hidden neurons.
    /// </summary>
    /// <remarks>
    /// This matrix has dimensions [hiddenSize, inputSize], where each row represents the weights
    /// for one hidden neuron. These weights determine how each input feature influences each
    /// hidden neuron and are trainable parameters of the layer.
    /// </remarks>
    private Matrix<T> _inputWeights;
    
    /// <summary>
    /// Matrix storing the weight parameters for connections between previous hidden state and current hidden state.
    /// </summary>
    /// <remarks>
    /// This matrix has dimensions [hiddenSize, hiddenSize], where each row represents the weights
    /// for one hidden neuron. These weights determine how the previous hidden state influences the
    /// current hidden state and are what gives the recurrent layer its "memory" capability.
    /// </remarks>
    private Matrix<T> _hiddenWeights;
    
    /// <summary>
    /// Vector storing the bias parameters for each hidden neuron.
    /// </summary>
    /// <remarks>
    /// This vector has length hiddenSize, where each element is a constant value added to the
    /// weighted sum for the corresponding hidden neuron. Biases allow the network to shift the
    /// activation function, giving it more flexibility to fit the data.
    /// </remarks>
    private Vector<T> _biases;
    
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
    /// This matrix holds the accumulated gradients for all input weight parameters during the backward pass.
    /// It has the same dimensions as the _inputWeights matrix and is used to update the input weights during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _inputWeightsGradient;
    
    /// <summary>
    /// Stores the gradients of the loss with respect to the hidden weight parameters.
    /// </summary>
    /// <remarks>
    /// This matrix holds the accumulated gradients for all hidden weight parameters during the backward pass.
    /// It has the same dimensions as the _hiddenWeights matrix and is used to update the hidden weights during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _hiddenWeightsGradient;
    
    /// <summary>
    /// Stores the gradients of the loss with respect to the bias parameters.
    /// </summary>
    /// <remarks>
    /// This vector holds the accumulated gradients for all bias parameters during the backward pass.
    /// It has the same length as the _biases vector and is used to update the biases during
    /// the parameter update step. The vector is null before the first backward pass or after a reset.
    /// </remarks>
    private Vector<T>? _biasesGradient;

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
        _inputWeights = new Matrix<T>(hiddenSize, inputSize);
        _hiddenWeights = new Matrix<T>(hiddenSize, hiddenSize);
        _biases = new Vector<T>(hiddenSize);

        InitializeParameters();
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
        _inputWeights = new Matrix<T>(hiddenSize, inputSize);
        _hiddenWeights = new Matrix<T>(hiddenSize, hiddenSize);
        _biases = new Vector<T>(hiddenSize);

        InitializeParameters();
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
    /// new_memory = activation(input_weights � current_input + hidden_weights � previous_memory + bias)
    /// 
    /// This step-by-step processing allows the layer to build up an understanding of the entire sequence.
    /// The layer saves all inputs, hidden states, and outputs for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int sequenceLength = input.Shape[0];
        int batchSize = input.Shape[1];
        int inputSize = input.Shape[2];
        int hiddenSize = _inputWeights.Rows;

        var output = new Tensor<T>([sequenceLength, batchSize, hiddenSize]);
        var hiddenState = new Tensor<T>([sequenceLength + 1, batchSize, hiddenSize]);

        // Initialize the first hidden state with zeros
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                hiddenState[0, b, h] = NumOps.Zero;
            }
        }

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                var inputVector = new Vector<T>(inputSize);
                var prevHiddenVector = new Vector<T>(hiddenSize);

                for (int i = 0; i < inputSize; i++)
                {
                    inputVector[i] = input[t, b, i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    prevHiddenVector[h] = hiddenState[t, b, h];
                }

                var newHiddenVector = _inputWeights.Multiply(inputVector)
                    .Add(_hiddenWeights.Multiply(prevHiddenVector))
                    .Add(_biases);

                newHiddenVector = ApplyActivation(newHiddenVector);

                for (int h = 0; h < hiddenSize; h++)
                {
                    output[t, b, h] = newHiddenVector[h];
                    hiddenState[t + 1, b, h] = newHiddenVector[h];
                }
            }
        }

        _lastHiddenState = hiddenState;
        _lastOutput = output;

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the recurrent layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the recurrent layer, which is used during training
    /// to propagate error gradients back through the network. It implements backpropagation through time (BPTT)
    /// by starting at the end of the sequence and working backward, accumulating gradients for the weights and biases.
    /// For each time step, it calculates gradients with respect to the input, the hidden state, and the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer should change to reduce errors.
    ///
    /// During the backward pass:
    /// 1. The layer starts from the end of the sequence and works backward
    /// 2. At each time step:
    ///    - It receives error gradients from two sources: the layer above and the future time step
    ///    - It calculates how each of its weights and biases should change
    ///    - It calculates how the error should flow back to the previous layer and to the previous time step
    ///
    /// This is like figuring out how a mistake at the end of a sentence affects your understanding
    /// of each word that came before it. The further back in time, the more complex these relationships become.
    ///
    /// This process, called "backpropagation through time," is what allows recurrent networks to learn from sequences.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using Backpropagation Through Time (BPTT).
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations optimized for
    /// recurrent neural networks. It performs backpropagation through time (BPTT), processing the
    /// sequence in reverse order and accumulating gradients across time steps.
    /// </para>
    /// <para>
    /// Autodiff Note: Implementing BPTT with automatic differentiation is complex due to temporal
    /// dependencies and the need to accumulate gradients across time steps. The manual implementation
    /// provides efficient and correct gradient calculations for recurrent layers.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int sequenceLength = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];
        int inputSize = _lastInput.Shape[2];
        int hiddenSize = _inputWeights.Rows;

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        var inputWeightsGradient = new Matrix<T>(_inputWeights.Rows, _inputWeights.Columns);
        var hiddenWeightsGradient = new Matrix<T>(_hiddenWeights.Rows, _hiddenWeights.Columns);
        var biasesGradient = new Vector<T>(_biases.Length);

        var nextHiddenGradient = new Tensor<T>([batchSize, hiddenSize]);

        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            var currentGradient = new Tensor<T>([batchSize, hiddenSize]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    currentGradient[b, h] = NumOps.Add(outputGradient[t, b, h], nextHiddenGradient[b, h]);
                }

                var inputVector = new Vector<T>(inputSize);
                var prevHiddenVector = new Vector<T>(hiddenSize);
                var gradientVector = new Vector<T>(hiddenSize);

                for (int i = 0; i < inputSize; i++)
                {
                    inputVector[i] = _lastInput[t, b, i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    prevHiddenVector[h] = _lastHiddenState[t, b, h];
                    gradientVector[h] = currentGradient[b, h];
                }

                var activationDerivative = ApplyActivationDerivative(prevHiddenVector, gradientVector);

                inputWeightsGradient = inputWeightsGradient.Add(Matrix<T>.OuterProduct(activationDerivative, inputVector));
                hiddenWeightsGradient = hiddenWeightsGradient.Add(Matrix<T>.OuterProduct(activationDerivative, prevHiddenVector));
                biasesGradient = biasesGradient.Add(activationDerivative);

                var inputGradientVector = _inputWeights.Transpose().Multiply(activationDerivative);
                var hiddenGradientVector = _hiddenWeights.Transpose().Multiply(activationDerivative);

                for (int i = 0; i < inputSize; i++)
                {
                    inputGradient[t, b, i] = inputGradientVector[i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    nextHiddenGradient[b, h] = hiddenGradientVector[h];
                }
            }
        }

        _inputWeightsGradient = inputWeightsGradient;
        _hiddenWeightsGradient = hiddenWeightsGradient;
        _biasesGradient = biasesGradient;

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with BPTT.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients through time.
    /// It processes the sequence in reverse order, accumulating gradients for each time step.
    /// This implementation can be useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// <para>
    /// Note: This autodiff implementation processes each time step individually and accumulates
    /// gradients, similar to manual BPTT but using autodiff operations for each step.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int sequenceLength = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];
        int inputSize = _lastInput.Shape[2];
        int hiddenSize = _inputWeights.Rows;

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        var inputWeightsGradient = new Matrix<T>(_inputWeights.Rows, _inputWeights.Columns);
        var hiddenWeightsGradient = new Matrix<T>(_hiddenWeights.Rows, _hiddenWeights.Columns);
        var biasesGradient = new Vector<T>(_biases.Length);

        var nextHiddenGradient = new Tensor<T>([batchSize, hiddenSize]);

        // Convert weights and biases to tensors for autodiff
        var inputWeightsTensor = MatrixToTensor(_inputWeights);
        var hiddenWeightsTensor = MatrixToTensor(_hiddenWeights);
        var biasesTensor = VectorToTensor(_biases);

        // Process sequence in reverse (BPTT)
        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            // Extract input and hidden state for this time step
            var inputAtT = new Tensor<T>([batchSize, inputSize]);
            var hiddenAtT = new Tensor<T>([batchSize, hiddenSize]);
            var gradAtT = new Tensor<T>([batchSize, hiddenSize]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < inputSize; i++)
                {
                    inputAtT[b, i] = _lastInput[t, b, i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    hiddenAtT[b, h] = _lastHiddenState[t, b, h];
                    gradAtT[b, h] = NumOps.Add(outputGradient[t, b, h], nextHiddenGradient[b, h]);
                }
            }

            // Create computation nodes for this time step
            var inputNode = Autodiff.TensorOperations<T>.Variable(inputAtT, "input", requiresGradient: true);
            var hiddenNode = Autodiff.TensorOperations<T>.Variable(hiddenAtT, "hidden", requiresGradient: true);
            var inputWeightsNode = Autodiff.TensorOperations<T>.Variable(inputWeightsTensor, "input_weights", requiresGradient: true);
            var hiddenWeightsNode = Autodiff.TensorOperations<T>.Variable(hiddenWeightsTensor, "hidden_weights", requiresGradient: true);
            var biasesNode = Autodiff.TensorOperations<T>.Variable(biasesTensor, "biases", requiresGradient: true);

            // Forward pass for this time step using autodiff: h_new = activation(W_input @ input + W_hidden @ hidden + bias)
            var inputWeightsTransposed = Autodiff.TensorOperations<T>.Transpose(inputWeightsNode);
            var hiddenWeightsTransposed = Autodiff.TensorOperations<T>.Transpose(hiddenWeightsNode);

            var inputContribution = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, inputWeightsTransposed);
            var hiddenContribution = Autodiff.TensorOperations<T>.MatrixMultiply(hiddenNode, hiddenWeightsTransposed);

            // Broadcast biases across batch dimension
            var biasesBroadcast = BroadcastBiases(biasesTensor, batchSize);
            var biasNode = Autodiff.TensorOperations<T>.Variable(biasesBroadcast, "biases_broadcast", requiresGradient: false);

            var preActivation = Autodiff.TensorOperations<T>.Add(inputContribution, hiddenContribution);
            preActivation = Autodiff.TensorOperations<T>.Add(preActivation, biasNode);

            // Apply activation using autodiff
            var activated = ApplyActivationAutodiff(preActivation);

            // Set gradient and perform backward pass
            activated.Gradient = gradAtT;

            var topoOrder = GetTopologicalOrder(activated);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            // Accumulate gradients
            if (inputWeightsNode.Gradient != null)
            {
                var stepInputWeightsGrad = TensorToMatrix(inputWeightsNode.Gradient);
                inputWeightsGradient = inputWeightsGradient.Add(stepInputWeightsGrad);
            }

            if (hiddenWeightsNode.Gradient != null)
            {
                var stepHiddenWeightsGrad = TensorToMatrix(hiddenWeightsNode.Gradient);
                hiddenWeightsGradient = hiddenWeightsGradient.Add(stepHiddenWeightsGrad);
            }

            if (biasesNode.Gradient != null)
            {
                var stepBiasesGrad = TensorToVector(biasesNode.Gradient);
                biasesGradient = biasesGradient.Add(stepBiasesGrad);
            }

            // Store input gradient and next hidden gradient for next iteration
            if (inputNode.Gradient != null)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < inputSize; i++)
                    {
                        inputGradient[t, b, i] = inputNode.Gradient[b, i];
                    }
                }
            }

            if (hiddenNode.Gradient != null)
            {
                nextHiddenGradient = hiddenNode.Gradient;
            }
        }

        _inputWeightsGradient = inputWeightsGradient;
        _hiddenWeightsGradient = hiddenWeightsGradient;
        _biasesGradient = biasesGradient;

        return inputGradient;
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));

                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies activation function using autodiff operations.
    /// </summary>
    private Autodiff.ComputationNode<T> ApplyActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        if (ScalarActivation is ReLUActivation<T>)
        {
            return Autodiff.TensorOperations<T>.ReLU(input);
        }
        else if (ScalarActivation is SigmoidActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Sigmoid(input);
        }
        else if (ScalarActivation is TanhActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Tanh(input);
        }
        else
        {
            // For unsupported activations, return input unchanged
            return input;
        }
    }

    /// <summary>
    /// Broadcasts biases across the batch dimension.
    /// </summary>
    private Tensor<T> BroadcastBiases(Tensor<T> biases, int batchSize)
    {
        var biasLength = biases.Length;
        var broadcasted = new Tensor<T>(new int[] { batchSize, biasLength });

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < biasLength; j++)
            {
                broadcasted[i, j] = biases[j];
            }
        }

        return broadcasted;
    }

    /// <summary>
    /// Converts a Matrix to a 2D Tensor.
    /// </summary>
    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(new int[] { matrix.Rows, matrix.Columns });
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                tensor[i, j] = matrix[i, j];
            }
        }
        return tensor;
    }

    /// <summary>
    /// Converts a Vector to a 1D Tensor.
    /// </summary>
    private Tensor<T> VectorToTensor(Vector<T> vector)
    {
        // Use Tensor.FromVector for efficient conversion
        return Tensor<T>.FromVector(vector);
    }

    /// <summary>
    /// Converts a 2D Tensor to a Matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Rank != 2)
            throw new ArgumentException("Tensor must be 2D to convert to Matrix");

        var matrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                matrix[i, j] = tensor[i, j];
            }
        }
        return matrix;
    }

    /// <summary>
    /// Converts a Tensor to a Vector.
    /// </summary>
    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        // Handle both 1D and 2D tensors (for column vectors)
        int length = tensor.Length;
        var vector = new Vector<T>(length);

        for (int i = 0; i < length; i++)
        {
            vector[i] = tensor[i];
        }

        return vector;
    }

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

        _inputWeights = _inputWeights.Subtract(_inputWeightsGradient.Multiply(learningRate));
        _hiddenWeights = _hiddenWeights.Subtract(_hiddenWeightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
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
        // Calculate total number of parameters
        int totalParams = _inputWeights.Rows * _inputWeights.Columns + 
                          _hiddenWeights.Rows * _hiddenWeights.Columns + 
                          _biases.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy input weights
        for (int i = 0; i < _inputWeights.Rows; i++)
        {
            for (int j = 0; j < _inputWeights.Columns; j++)
            {
                parameters[index++] = _inputWeights[i, j];
            }
        }
    
        // Copy hidden weights
        for (int i = 0; i < _hiddenWeights.Rows; i++)
        {
            for (int j = 0; j < _hiddenWeights.Columns; j++)
            {
                parameters[index++] = _hiddenWeights[i, j];
            }
        }
    
        // Copy biases
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
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
        int totalParams = _inputWeights.Rows * _inputWeights.Columns + 
                          _hiddenWeights.Rows * _hiddenWeights.Columns + 
                          _biases.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set input weights
        for (int i = 0; i < _inputWeights.Rows; i++)
        {
            for (int j = 0; j < _inputWeights.Columns; j++)
            {
                _inputWeights[i, j] = parameters[index++];
            }
        }
    
        // Set hidden weights
        for (int i = 0; i < _hiddenWeights.Rows; i++)
        {
            for (int j = 0; j < _hiddenWeights.Columns; j++)
            {
                _hiddenWeights[i, j] = parameters[index++];
            }
        }
    
        // Set biases
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
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
        // Initialize weights and biases (e.g., Xavier/Glorot initialization)
        T inputScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputWeights.Rows + _inputWeights.Columns)));
        T hiddenScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_hiddenWeights.Rows + _hiddenWeights.Columns)));

        for (int i = 0; i < _inputWeights.Rows; i++)
        {
            for (int j = 0; j < _inputWeights.Columns; j++)
            {
                _inputWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), inputScale);
            }

            for (int j = 0; j < _hiddenWeights.Columns; j++)
            {
                _hiddenWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), hiddenScale);
            }

            _biases[i] = NumOps.Zero;
        }
    }
}