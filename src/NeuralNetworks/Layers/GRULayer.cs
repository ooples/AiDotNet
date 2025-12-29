using AiDotNet.Autodiff;


namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Gated Recurrent Unit (GRU) layer for processing sequential data.
/// </summary>
/// <remarks>
/// <para>
/// The GRU (Gated Recurrent Unit) layer is a type of recurrent neural network layer that is designed to
/// capture dependencies over time in sequential data. It addresses the vanishing gradient problem that
/// standard recurrent neural networks face when dealing with long sequences. The GRU uses update and reset
/// gates to control the flow of information, allowing the network to retain relevant information over
/// many time steps while forgetting irrelevant details.
/// </para>
/// <para><b>For Beginners:</b> This layer helps neural networks understand sequences of data, like sentences or time series.
/// 
/// Think of the GRU as having a "memory" that helps it understand context:
/// - When reading a sentence, it remembers important words from earlier
/// - When analyzing stock prices, it remembers relevant trends from previous days
/// - It uses special "gates" to decide what information to keep or forget
/// 
/// For example, in the sentence "The clouds were dark and it started to ___", 
/// the GRU would recognize the context and predict "rain" because it remembers
/// the earlier words about dark clouds.
/// 
/// GRUs are simpler versions of LSTMs (Long Short-Term Memory) but often perform similarly well
/// while being more efficient to train.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GRULayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensors for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight tensors transform the input at each time step:
    /// - _Wz: Weights for the update gate that determines how much of the previous hidden state to keep
    /// - _Wr: Weights for the reset gate that determines how much of the previous hidden state to reset
    /// - _Wh: Weights for the candidate hidden state that contains new information
    /// </para>
    /// <para><b>For Beginners:</b> These weights transform your input data into useful information.
    ///
    /// Think of these weight tensors as "filters" that extract different types of information:
    /// - _Wz helps decide how much old information to keep
    /// - _Wr helps decide how much old information to reset or forget
    /// - _Wh helps create new information based on the current input
    ///
    /// During training, these weights are adjusted to better recognize important patterns in your data.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wz, _Wr, _Wh;

    /// <summary>
    /// The weight tensors that transform the previous hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight tensors process the previous hidden state:
    /// - _Uz: Weights that transform the previous hidden state for the update gate
    /// - _Ur: Weights that transform the previous hidden state for the reset gate
    /// - _Uh: Weights that transform the previous hidden state for the candidate hidden state
    /// </para>
    /// <para><b>For Beginners:</b> These weights process the "memory" from previous time steps.
    ///
    /// These tensors help the GRU work with information from earlier in the sequence:
    /// - _Uz helps decide which parts of the memory to keep
    /// - _Ur helps decide which parts of the memory to reset
    /// - _Uh helps combine memory with new information
    ///
    /// For example, when reading text, these weights help the network remember important context
    /// from words that appeared earlier in the sentence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Uz, _Ur, _Uh;

    /// <summary>
    /// The bias tensors for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These bias tensors provide an offset to the transformations:
    /// - _bz: Bias for the update gate
    /// - _br: Bias for the reset gate
    /// - _bh: Bias for the candidate hidden state
    /// </para>
    /// <para><b>For Beginners:</b> These biases are like "default settings" for each gate.
    ///
    /// Biases help the network by:
    /// - Providing a starting point for each gate's operation
    /// - Allowing outputs to be non-zero even when inputs are zero
    /// - Giving the model flexibility to fit data better
    ///
    /// They're like the "baseline" settings that get adjusted during training.
    /// </para>
    /// </remarks>
    private Tensor<T> _bz, _br, _bh;

    /// <summary>
    /// Gradients for the weight matrices during backpropagation.
    /// </summary>
    private Tensor<T>? _dWz, _dWr, _dWh, _dUz, _dUr, _dUh;

    /// <summary>
    /// Gradients for the bias vectors during backpropagation.
    /// </summary>
    private Tensor<T>? _dbz, _dbr, _dbh;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The final hidden state from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastHiddenState;

    /// <summary>
    /// The activation values for the update gate (z), reset gate (r), and candidate hidden state (h)
    /// from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastZ, _lastR, _lastH;

    /// <summary>
    /// All hidden states from the last forward pass, used when returning sequences.
    /// </summary>
    private List<Tensor<T>>? _allHiddenStates;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The size of the input feature vector at each time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the dimensionality of the input feature vector at each time step in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of features in each element of your sequence.
    /// 
    /// For example:
    /// - In text processing, this might be the size of word embeddings (e.g., 300 dimensions)
    /// - In time series, this would be the number of measurements at each time point
    /// - In audio processing, this could be the number of frequency bands
    /// </para>
    /// </remarks>
    private readonly int _inputSize;

    /// <summary>
    /// The size of the hidden state vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines the dimensionality of the GRU's hidden state and, consequently, the capacity
    /// of the layer to capture and represent patterns in the sequence data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the GRU's "memory".
    /// 
    /// The hidden size determines:
    /// - How much information the GRU can remember about the sequence
    /// - The complexity of patterns it can recognize
    /// - The capacity of the model (larger = more capacity but more parameters to train)
    /// 
    /// A larger hidden size allows the layer to capture more complex patterns but requires
    /// more data and time to train effectively.
    /// </para>
    /// </remarks>
    private readonly int _hiddenSize;

    /// <summary>
    /// Determines whether the layer returns the full sequence of hidden states or just the final state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If true, the forward pass returns all hidden states for all time steps. If false, only the final
    /// hidden state is returned.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls what information the layer provides as output.
    /// 
    /// Two options:
    /// - true: Returns information about every element in the sequence (useful for sequence-to-sequence tasks)
    /// - false: Returns only the final "summary" after processing the entire sequence
    /// 
    /// For example, in sentiment analysis, you might only need the final state (false)
    /// to classify the sentiment of a whole text, but for translation, you'd want
    /// information about each position (true).
    /// </para>
    /// </remarks>
    private readonly bool _returnSequences;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// The activation function applied to the candidate hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the candidate hidden state. Typically, a tanh activation function
    /// is used to maintain values between -1 and 1, which helps with the stability of the recurrent network.
    /// </para>
    /// <para><b>For Beginners:</b> This function shapes the new information the GRU creates.
    /// 
    /// The activation function:
    /// - Adds non-linearity, helping the network learn complex patterns
    /// - Usually tanh (hyperbolic tangent), which outputs values between -1 and 1
    /// - Helps keep the values in a manageable range to prevent explosion
    /// 
    /// Think of it as a way to normalize and shape the information before it's combined with memory.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _activation;

    /// <summary>
    /// The activation function applied to the update and reset gates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the update and reset gates. Typically, a sigmoid activation function
    /// is used, which outputs values between 0 and 1, allowing the gates to control the flow of information.
    /// </para>
    /// <para><b>For Beginners:</b> This function controls how the gates operate.
    /// 
    /// The recurrent activation:
    /// - Usually sigmoid, which outputs values between 0 and 1
    /// - Creates "gate" values that act like percentages or dials
    /// - Value of 0 means "block completely" while 1 means "let everything through"
    /// 
    /// This helps the GRU decide how much information to remember versus forget at each step.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _recurrentActivation;

    /// <summary>
    /// The vector activation function applied to the candidate hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire candidate hidden state vector at once rather
    /// than element-wise. This can capture dependencies between different elements of the hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to shape new information.
    /// 
    /// A vector activation:
    /// - Works on the entire vector at once, not just individual values
    /// - Can capture relationships between different features
    /// - Potentially more powerful than element-wise activations
    /// 
    /// This is useful for capturing complex patterns that depend on multiple features interacting together.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorActivation;

    /// <summary>
    /// The vector activation function applied to the update and reset gates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire update and reset gate vectors at once rather
    /// than element-wise. This can capture dependencies between different elements of the gates.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to control the gates.
    /// 
    /// A vector recurrent activation:
    /// - Works on entire gate vectors at once
    /// - Can make decisions based on all features together
    /// - Potentially more powerful than individual feature activations
    /// 
    /// This allows for more sophisticated memory management, where decisions about
    /// what to remember depend on the relationships between multiple features.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorRecurrentActivation;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The total number of weight and bias parameters in the GRU layer.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property calculates the total number of trainable parameters in the GRU layer, which includes
    /// all the weights and biases for the gates and candidate hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer needs to learn.
    /// 
    /// The formula counts:
    /// - Weights connecting inputs to the GRU (Wz, Wr, Wh)
    /// - Weights connecting the previous hidden state (Uz, Ur, Uh)
    /// - Bias values for each gate and candidate state (bz, br, bh)
    /// 
    /// A higher parameter count means the model can capture more complex patterns
    /// but requires more data and time to train effectively.
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _hiddenSize * _inputSize * 3 +  // Wz, Wr, Wh
        _hiddenSize * _hiddenSize * 3 + // Uz, Ur, Uh
        _hiddenSize * 3;                // bz, br, bh

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The GRULayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The GRU layer always supports training because it has weights and biases that can be updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="GRULayer{T}"/> class with the specified dimensions, return behavior, and element-wise activation functions.
    /// </summary>
    /// <param name="inputSize">The size of the input feature vector at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state vector.</param>
    /// <param name="returnSequences">If <c>true</c>, returns all hidden states; if <c>false</c>, returns only the final hidden state.</param>
    /// <param name="activation">The activation function for the candidate hidden state. Defaults to tanh if not specified.</param>
    /// <param name="recurrentActivation">The activation function for the gates. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GRU layer with the specified dimensions and element-wise activation functions.
    /// The weights are initialized randomly with a scale factor based on the hidden size, and the biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU layer with standard activation functions.
    /// 
    /// When creating a GRU layer, you specify:
    /// - inputSize: How many features each element in your sequence has
    /// - hiddenSize: How large the GRU's "memory" should be
    /// - returnSequences: Whether you want information about every element or just a final summary
    /// - activation: How to shape new information (default is tanh, outputting values between -1 and 1)
    /// - recurrentActivation: How the gates should work (default is sigmoid, outputting values between 0 and 1)
    /// 
    /// For example, if processing sentences where each word is represented by a 100-dimensional vector,
    /// and you want a 200-dimensional memory, you would use inputSize=100 and hiddenSize=200.
    /// </para>
    /// </remarks>
    public GRULayer(int inputSize, int hiddenSize,
                    bool returnSequences = false,
                    IActivationFunction<T>? activation = null,
                    IActivationFunction<T>? recurrentActivation = null)
        : base([inputSize], [hiddenSize], activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _returnSequences = returnSequences;
        _activation = activation ?? new TanhActivation<T>();
        _recurrentActivation = recurrentActivation ?? new SigmoidActivation<T>();

        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _hiddenSize)));

        _Wz = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wr = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wh = InitializeTensor(_hiddenSize, _inputSize, scale);

        _Uz = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeTensor(_hiddenSize, _hiddenSize, scale);

        _bz = new Tensor<T>([_hiddenSize]);
        _br = new Tensor<T>([_hiddenSize]);
        _bh = new Tensor<T>([_hiddenSize]);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GRULayer{T}"/> class with the specified dimensions, return behavior, and vector activation functions.
    /// </summary>
    /// <param name="inputSize">The size of the input feature vector at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state vector.</param>
    /// <param name="returnSequences">If <c>true</c>, returns all hidden states; if <c>false</c>, returns only the final hidden state.</param>
    /// <param name="vectorActivation">The vector activation function for the candidate hidden state. Defaults to tanh if not specified.</param>
    /// <param name="vectorRecurrentActivation">The vector activation function for the gates. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GRU layer with the specified dimensions and vector activation functions.
    /// Vector activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU layer with more advanced vector-based activation functions.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// This constructor is useful when you need the layer to understand how different
    /// features interact with each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public GRULayer(int inputSize, int hiddenSize,
                    bool returnSequences = false,
                    IVectorActivationFunction<T>? vectorActivation = null,
                    IVectorActivationFunction<T>? vectorRecurrentActivation = null)
        : base([inputSize], [hiddenSize], vectorActivation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _returnSequences = returnSequences;
        _vectorActivation = vectorActivation ?? new TanhActivation<T>();
        _vectorRecurrentActivation = vectorRecurrentActivation ?? new SigmoidActivation<T>();

        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _hiddenSize)));

        _Wz = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wr = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wh = InitializeTensor(_hiddenSize, _inputSize, scale);

        _Uz = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeTensor(_hiddenSize, _hiddenSize, scale);

        _bz = new Tensor<T>([_hiddenSize]);
        _br = new Tensor<T>([_hiddenSize]);
        _bh = new Tensor<T>([_hiddenSize]);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="rows">The number of rows in the tensor.</param>
    /// <param name="cols">The number of columns in the tensor.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <returns>A new tensor with scaled random values.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with the specified dimensions and fills it with random values
    /// between -0.5 and 0.5, scaled by the provided scale factor. This type of initialization helps
    /// with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a tensor of random starting values for weights.
    ///
    /// The method:
    /// - Creates a new tensor with the specified size
    /// - Fills it with random numbers between -0.5 and 0.5
    /// - Multiplies these numbers by a scale factor to control their size
    ///
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// The scale factor helps prevent values from being too large or too small at the start of training.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeTensor(int rows, int cols, T scale)
    {
        // Create random tensor using Tensor<T>.CreateRandom [0, 1]
        var randomTensor = Tensor<T>.CreateRandom(rows, cols);

        // Shift to [-0.5, 0.5] range: random - 0.5
        var halfTensor = new Tensor<T>([rows, cols]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        return Engine.TensorMultiplyScalar(shifted, scale);
    }

    /// <summary>
    /// Performs the forward pass of the GRU layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape should be [batchSize, sequenceLength, inputSize].</param>
    /// <returns>The output tensor. If returnSequences is true, shape will be [batchSize, sequenceLength, hiddenSize]; otherwise, [batchSize, hiddenSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the GRU layer. It processes the input sequence step by step,
    /// updating the hidden state at each time step according to the GRU equations. The update gate (z) controls
    /// how much of the previous hidden state to keep, the reset gate (r) controls how much of the previous hidden
    /// state to reset, and the candidate hidden state (h_candidate) contains new information from the current input.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your sequence data through the GRU.
    /// 
    /// For each element in your sequence (like each word in a sentence):
    /// 1. The update gate (z) decides how much of the old memory to keep
    /// 2. The reset gate (r) decides how much of the old memory to forget
    /// 3. The layer creates new information based on the current input and relevant memory
    /// 4. It combines the kept memory and new information to update its understanding
    /// 
    /// This process repeats for each element in the sequence, with the memory
    /// evolving to capture the relevant context from the entire sequence.
    /// 
    /// The final output depends on the returnSequences setting:
    /// - If true: Returns information about every element in the sequence
    /// - If false: Returns only the final memory state
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims into batch for rank > 3
        Tensor<T> input3D;
        int batchSize;
        int sequenceLength;

        if (rank == 2)
        {
            // 2D input [sequenceLength, inputSize] -> add batch dim
            batchSize = 1;
            sequenceLength = input.Shape[0];
            input3D = input.Reshape([1, sequenceLength, _inputSize]);
        }
        else if (rank == 3)
        {
            // Standard 3D input [batchSize, sequenceLength, inputSize]
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            input3D = input;
        }
        else
        {
            // Higher-rank tensor: collapse leading dims into batch
            sequenceLength = input.Shape[rank - 2];
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            input3D = input.Reshape([flatBatch, sequenceLength, _inputSize]);
        }

        _lastInput = input3D;

        // Reset hidden state if needed
        if (_lastHiddenState == null)
        {
            _lastHiddenState = new Tensor<T>([batchSize, _hiddenSize]);
        }

        // Initialize list to store all hidden states if returning sequences
        if (_returnSequences)
        {
            _allHiddenStates = new List<Tensor<T>>(sequenceLength);
        }

        // Process each time step
        Tensor<T> currentHiddenState = _lastHiddenState;

        // Store the last activation values for backpropagation
        Tensor<T>? lastZ = null;
        Tensor<T>? lastR = null;
        Tensor<T>? lastH_candidate = null;

        // Pre-transpose weights for efficiency (weights are [hiddenSize, inputSize], need [inputSize, hiddenSize] for matmul)
        var WzT = Engine.TensorTranspose(_Wz);
        var WrT = Engine.TensorTranspose(_Wr);
        var WhT = Engine.TensorTranspose(_Wh);
        var UzT = Engine.TensorTranspose(_Uz);
        var UrT = Engine.TensorTranspose(_Ur);
        var UhT = Engine.TensorTranspose(_Uh);

        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract current time step input: slice axis 1 (sequence) at timestep t
            var xt = input3D.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);

            var z = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WzT), Engine.TensorMatMul(currentHiddenState, UzT)), _bz), true);
            var r = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WrT), Engine.TensorMatMul(currentHiddenState, UrT)), _br), true);
            var h_candidate = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WhT), Engine.TensorMatMul(r.ElementwiseMultiply(currentHiddenState), UhT)), _bh), false);
            // Vectorized: compute (1 - z) using Tensor operations

            var ones = new Tensor<T>(z.Shape);

            ones.Fill(NumOps.One);

            var oneMinusZ = ones.Subtract(z);


            var h = z.ElementwiseMultiply(currentHiddenState).Add(

                oneMinusZ.ElementwiseMultiply(h_candidate)

            );

            currentHiddenState = h;

            // Save the last timestep's activations
            if (t == sequenceLength - 1)
            {
                lastZ = z;
                lastR = r;
                lastH_candidate = h_candidate;
            }

            if (_returnSequences && _allHiddenStates != null)
            {
                _allHiddenStates.Add(h.Clone());
            }
        }

        _lastZ = lastZ;
        _lastR = lastR;
        _lastH = lastH_candidate;
        _lastHiddenState = currentHiddenState;

        // Return either the sequence of hidden states or just the final state
        Tensor<T> output;
        if (_returnSequences && _allHiddenStates != null)
        {
            // Each hidden state is [batchSize, hiddenSize]
            // Concatenate along axis 1, then reshape to [batchSize, sequenceLength, hiddenSize]
            output = Tensor<T>.Concatenate([.. _allHiddenStates], 1).Reshape([batchSize, sequenceLength, _hiddenSize]);
        }
        else
        {
            output = currentHiddenState;
        }

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            int[] newShape;
            if (_returnSequences)
            {
                // Output shape: [...leadingDims, sequenceLength, hiddenSize]
                newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 2] = sequenceLength;
                newShape[_originalInputShape.Length - 1] = _hiddenSize;
            }
            else
            {
                // Output shape: [...leadingDims, hiddenSize]
                newShape = new int[_originalInputShape.Length - 1];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 2] = _hiddenSize;
            }
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2 && !_returnSequences)
        {
            // 2D input -> 1D output (remove batch dim)
            output = output.Reshape([_hiddenSize]);
        }

        return output;
    }

    /// <summary>
    /// Applies the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <param name="isRecurrent">If <c>true</c>, applies the recurrent activation function; otherwise, applies the regular activation function.</param>
    /// <returns>The activated tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies either the recurrent activation function (for gates) or the regular activation function
    /// (for the candidate hidden state) to the input tensor. It supports both element-wise and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the appropriate "shaping function" to the values.
    /// 
    /// Depending on what's being processed:
    /// - For gates (update and reset), it uses the recurrent activation (usually sigmoid)
    /// - For new information (candidate hidden state), it uses the regular activation (usually tanh)
    /// 
    /// This helps shape the values into the right ranges:
    /// - Gates need to be between 0 and 1 (like percentages)
    /// - New information is often kept between -1 and 1 for stability
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivation(Tensor<T> input, bool isRecurrent)
    {
        if (isRecurrent)
        {
            if (_vectorRecurrentActivation != null)
            {
                // Use centralized ActivationHelper for optimized activation dispatch
                return ActivationHelper.ApplyActivation(_vectorRecurrentActivation, input, Engine);
            }
            else if (_recurrentActivation != null)
            {
                return input.Transform((x, _) => _recurrentActivation.Activate(x));
            }
        }
        else
        {
            if (_vectorActivation != null)
            {
                // Use centralized ActivationHelper for optimized activation dispatch
                return ActivationHelper.ApplyActivation(_vectorActivation, input, Engine);
            }
            else if (_activation != null)
            {
                return input.Transform((x, _) => _activation.Activate(x));
            }
        }

        throw new InvalidOperationException("No activation function specified");
    }

    /// <summary>
    /// Performs the backward pass of the GRU layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the GRU layer, which is used during training to propagate
    /// error gradients back through the network. It calculates the gradients for all the weights and biases,
    /// and returns the gradient with respect to the layer's input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    ///
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change to reduce the overall error
    /// 2. It calculates how each of its weights and biases should change to produce better output
    /// 3. It calculates how its input should change, which will be used by earlier layers
    ///
    /// This complex calculation essentially runs the GRU's logic in reverse, tracking how
    /// changes to the output would affect each internal part of the layer.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using Backpropagation Through Time (BPTT) for GRU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations optimized for
    /// GRU networks. It performs backpropagation through time (BPTT), processing the
    /// sequence in reverse order and computing gradients for all gate parameters (reset, update)
    /// and hidden states.
    /// </para>
    /// <para>
    /// Autodiff Note: GRU backward pass involves complex interactions between reset and update gates.
    /// Implementing this with automatic differentiation would require handling temporal dependencies
    /// and gate-specific gradient flows. The manual implementation provides efficient and correct
    /// gradient calculations for all GRU components.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastZ == null || _lastR == null || _lastH == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];

        // Initialize gradients
        var dWz = new Tensor<T>([_hiddenSize, _inputSize]);
        var dWr = new Tensor<T>([_hiddenSize, _inputSize]);
        var dWh = new Tensor<T>([_hiddenSize, _inputSize]);
        var dUz = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dUr = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dUh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dbz = new Tensor<T>([_hiddenSize]);
        var dbr = new Tensor<T>([_hiddenSize]);
        var dbh = new Tensor<T>([_hiddenSize]);

        // Initialize input gradients
        var dInputs = new Tensor<T>([batchSize, sequenceLength, _inputSize]);

        // If we don't have sequence data, just process the final time step
        if (!_returnSequences || _allHiddenStates == null || _allHiddenStates.Count == 0)
        {
            // Handle single timestep case (simple backward pass)
            var dh = outputGradient;
            // Vectorized: compute (1 - _lastZ) using Tensor operations
            var ones1 = new Tensor<T>(_lastZ.Shape);
            ones1.Fill(NumOps.One);
            var oneMinusLastZ = ones1.Subtract(_lastZ);

            var dh_candidate = dh.ElementwiseMultiply(oneMinusLastZ);
            var dz = dh.ElementwiseMultiply(_lastHiddenState.Subtract(_lastH));

            var dr = ApplyActivationDerivative(_lastH, isRecurrent: false)
                .ElementwiseMultiply(dh_candidate)
                .ElementwiseMultiply(_lastHiddenState.Multiply(_Uh));

            var dx = dz.Multiply(_Wz.Transpose())
                    .Add(dr.Multiply(_Wr.Transpose()))
                    .Add(dh_candidate.Multiply(_Wh.Transpose()));

            // Reshape dx to match input format for the last timestep using Engine.TensorSetSlice
            var dxReshaped = dx.Reshape([batchSize, 1, _inputSize]);
            dInputs = Engine.TensorSetSlice(dInputs, dxReshaped, [0, sequenceLength - 1, 0]);

            // Calculate gradients for weights and biases
            dWz = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dz);
            dWr = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dr);
            dWh = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dh_candidate);

            // For U gradients, we need the previous hidden state
            // If not available, use zeros
            Tensor<T> prevHidden;
            if (sequenceLength > 1 && _allHiddenStates != null && _allHiddenStates.Count >= sequenceLength - 1)
            {
                prevHidden = _allHiddenStates[sequenceLength - 2];
            }
            else
            {
                prevHidden = new Tensor<T>([batchSize, _hiddenSize]);
            }

            dUz = prevHidden.Transpose([1, 0]).Multiply(dz);
            dUr = prevHidden.Transpose([1, 0]).Multiply(dr);
            dUh = prevHidden.Transpose([1, 0]).Multiply(dh_candidate.ElementwiseMultiply(_lastR));

            dbz = dz.Sum([0]);
            dbr = dr.Sum([0]);
            dbh = dh_candidate.Sum([0]);
        }
        else
        {
            // For returning sequences, we need to backpropagate through all time steps

            // Split the output gradient into time steps if returningSequences is true
            Tensor<T>[] dhOutSequence = new Tensor<T>[sequenceLength];
            if (_returnSequences)
            {
                // Split the gradient for each time step
                for (int t = 0; t < sequenceLength; t++)
                {
                    dhOutSequence[t] = outputGradient.Slice(1, t, t + 1).Reshape([batchSize, _hiddenSize]);
                }
            }
            else
            {
                // If not returning sequences, only the last time step has gradient from output
                for (int t = 0; t < sequenceLength - 1; t++)
                {
                    dhOutSequence[t] = new Tensor<T>([batchSize, _hiddenSize]);
                }
                dhOutSequence[sequenceLength - 1] = outputGradient;
            }

            // Initialize hidden state gradient for the next timestep
            var dhNext = new Tensor<T>([batchSize, _hiddenSize]);

            // We need to store activations for each time step during forward pass
            // If we haven't stored them, we'll recompute them now
            List<Tensor<T>> timeStepHidden = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepZ = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepR = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepHCandidate = new List<Tensor<T>>(sequenceLength);

            // We know the last state values
            timeStepHidden.Add(_lastHiddenState);
            timeStepZ.Add(_lastZ);
            timeStepR.Add(_lastR);
            timeStepHCandidate.Add(_lastH);

            // If we have _allHiddenStates, use those states
            if (_allHiddenStates != null && _allHiddenStates.Count == sequenceLength)
            {
                timeStepHidden = _allHiddenStates;
            }
            else
            {
                // Recompute hidden states for each time step
                // This is computationally expensive but ensures correctness
                var currentH = new Tensor<T>([batchSize, _hiddenSize]);

                for (int t = 0; t < sequenceLength; t++)
                {
                    var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);

                    var z = ComputeGate(xt, currentH, _Wz, _Uz, _bz, true);
                    var r = ComputeGate(xt, currentH, _Wr, _Ur, _br, true);
                    var h_candidate = ComputeGate(xt, r.ElementwiseMultiply(currentH), _Wh, _Uh, _bh, false);

                    // Vectorized: compute (1 - z) using Tensor operations
                    var ones2 = new Tensor<T>(z.Shape);
                    ones2.Fill(NumOps.One);
                    var oneMinusZ2 = ones2.Subtract(z);

                    var newH = z.ElementwiseMultiply(currentH).Add(
                        oneMinusZ2.ElementwiseMultiply(h_candidate)
                    );

                    currentH = newH;

                    if (t < sequenceLength - 1)
                    {
                        timeStepHidden[t] = currentH.Clone();
                        timeStepZ[t] = z.Clone();
                        timeStepR[t] = r.Clone();
                        timeStepHCandidate[t] = h_candidate.Clone();
                    }
                }
            }

            // Backpropagate through time
            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                // Combine gradient from output and from next timestep
                var dh = dhNext.Add(dhOutSequence[t]);

                // Get activations for this timestep
                var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);
                var h = timeStepHidden[t];
                var z = timeStepZ[t];
                var r = timeStepR[t];
                var h_candidate = timeStepHCandidate[t];

                // Get previous hidden state (or zeros for t=0)
                var h_prev = t > 0 ? timeStepHidden[t - 1] : new Tensor<T>([batchSize, _hiddenSize]);

                // Calculate gradients for this timestep
                // Vectorized: compute (1 - z) using Tensor operations
                var ones3 = new Tensor<T>(z.Shape);
                ones3.Fill(NumOps.One);
                var oneMinusZ3 = ones3.Subtract(z);

                var dh_candidate = dh.ElementwiseMultiply(oneMinusZ3);
                var dz = dh.ElementwiseMultiply(h_prev.Subtract(h_candidate));

                var dr = ApplyActivationDerivative(h_candidate, isRecurrent: false)
                    .ElementwiseMultiply(dh_candidate)
                    .ElementwiseMultiply(h_prev.Multiply(_Uh));

                // Input gradient for this timestep
                var dxt = dz.Multiply(_Wz.Transpose())
                        .Add(dr.Multiply(_Wr.Transpose()))
                        .Add(dh_candidate.Multiply(_Wh.Transpose()));

                // Store input gradients using Engine.TensorSetSlice
                var dxtReshaped = dxt.Reshape([batchSize, 1, _inputSize]);
                dInputs = Engine.TensorSetSlice(dInputs, dxtReshaped, [0, t, 0]);

                // Gradient for next timestep's hidden state
                dhNext = dz.Multiply(_Uz.Transpose())
                        .Add(dr.Multiply(_Ur.Transpose()))
                        .Add(dh_candidate.ElementwiseMultiply(r).Multiply(_Uh.Transpose()));

                // Accumulate weight gradients
                dWz = dWz.Add(xt.Transpose([1, 0]).Multiply(dz));
                dWr = dWr.Add(xt.Transpose([1, 0]).Multiply(dr));
                dWh = dWh.Add(xt.Transpose([1, 0]).Multiply(dh_candidate));

                dUz = dUz.Add(h_prev.Transpose([1, 0]).Multiply(dz));
                dUr = dUr.Add(h_prev.Transpose([1, 0]).Multiply(dr));
                dUh = dUh.Add(h_prev.Transpose([1, 0]).Multiply(dh_candidate.ElementwiseMultiply(r)));

                dbz = dbz.Add(dz.Sum([0]));
                dbr = dbr.Add(dr.Sum([0]));
                dbh = dbh.Add(dh_candidate.Sum([0]));
            }
        }

        // Store gradients for use in UpdateParameters
        _dWz = dWz; _dWr = dWr; _dWh = dWh;
        _dUz = dUz; _dUr = dUr; _dUh = dUh;
        _dbz = dbz; _dbr = dbr; _dbh = dbh;

        return dInputs;
    }

    /// <summary>
    /// Computes a gate activation for the GRU layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="hidden">The hidden state tensor.</param>
    /// <param name="W">The input weight matrix.</param>
    /// <param name="U">The hidden state weight matrix.</param>
    /// <param name="b">The bias vector.</param>
    /// <param name="isRecurrent">If true, applies recurrent activation; otherwise, applies regular activation.</param>
    /// <returns>The computed gate activation.</returns>
    private Tensor<T> ComputeGate(Tensor<T> input, Tensor<T> hidden, Tensor<T> W, Tensor<T> U, Tensor<T> b, bool isRecurrent)
    {
        var gate = input.Multiply(W).Add(hidden.Multiply(U)).Add(b);
        return ApplyActivation(gate, isRecurrent);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all the weight matrices and bias vectors of the GRU layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// This is typically called after the backward pass during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - All weights and biases are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to understand and predict sequences.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_dWz == null || _dWr == null || _dWh == null ||
            _dUz == null || _dUr == null || _dUh == null ||
            _dbz == null || _dbr == null || _dbh == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine operations for GPU/CPU acceleration
        _Wz = Engine.TensorSubtract(_Wz, Engine.TensorMultiplyScalar(_dWz, learningRate));
        _Wr = Engine.TensorSubtract(_Wr, Engine.TensorMultiplyScalar(_dWr, learningRate));
        _Wh = Engine.TensorSubtract(_Wh, Engine.TensorMultiplyScalar(_dWh, learningRate));

        _Uz = Engine.TensorSubtract(_Uz, Engine.TensorMultiplyScalar(_dUz, learningRate));
        _Ur = Engine.TensorSubtract(_Ur, Engine.TensorMultiplyScalar(_dUr, learningRate));
        _Uh = Engine.TensorSubtract(_Uh, Engine.TensorMultiplyScalar(_dUh, learningRate));

        _bz = Engine.TensorSubtract(_bz, Engine.TensorMultiplyScalar(_dbz, learningRate));
        _br = Engine.TensorSubtract(_br, Engine.TensorMultiplyScalar(_dbr, learningRate));
        _bh = Engine.TensorSubtract(_bh, Engine.TensorMultiplyScalar(_dbh, learningRate));
    }

    /// <summary>
    /// Updates the parameters of the layer with the given vector of parameter values.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets all the weight matrices and bias vectors of the GRU layer from a single vector of parameters.
    /// The parameters are arranged in the following order: Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you directly set all the learnable values in the layer.
    /// 
    /// The parameters vector contains all weights and biases in a specific order:
    /// 1. Weights for input to update gate (Wz)
    /// 2. Weights for input to reset gate (Wr)
    /// 3. Weights for input to candidate hidden state (Wh)
    /// 4. Weights for hidden state to update gate (Uz)
    /// 5. Weights for hidden state to reset gate (Ur)
    /// 6. Weights for hidden state to candidate hidden state (Uh)
    /// 7. Biases for update gate (bz)
    /// 8. Biases for reset gate (br)
    /// 9. Biases for candidate hidden state (bh)
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int inputWeightSize = _hiddenSize * _inputSize;
        int hiddenWeightSize = _hiddenSize * _hiddenSize;
        int biasSize = _hiddenSize;
        int idx = 0;

        // Extract and reshape weight tensors using Tensor.FromVector
        _Wz = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Wr = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Wh = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Uz = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _Ur = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _Uh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _bz = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);
        idx += biasSize;

        _br = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);
        idx += biasSize;

        _bh = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// The parameters are arranged in the following order: Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for
    /// saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// It gathers all parameters in this specific order:
    /// 1. Weights for input to update gate (Wz)
    /// 2. Weights for input to reset gate (Wr)
    /// 3. Weights for input to candidate hidden state (Wh)
    /// 4. Weights for hidden state to update gate (Uz)
    /// 5. Weights for hidden state to reset gate (Ur)
    /// 6. Weights for hidden state to candidate hidden state (Uh)
    /// 7. Biases for update gate (bz)
    /// 8. Biases for reset gate (br)
    /// 9. Biases for candidate hidden state (bh)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate for production-grade parameter extraction
        return Vector<T>.Concatenate(
            new Vector<T>(_Wz.ToArray()),
            new Vector<T>(_Wr.ToArray()),
            new Vector<T>(_Wh.ToArray()),
            new Vector<T>(_Uz.ToArray()),
            new Vector<T>(_Ur.ToArray()),
            new Vector<T>(_Uh.ToArray()),
            new Vector<T>(_bz.ToArray()),
            new Vector<T>(_br.ToArray()),
            new Vector<T>(_bh.ToArray())
        );
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input, hidden state, activation values, and all hidden states if returning sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The hidden state (memory) is cleared
    /// - All stored information about previous inputs is removed
    /// - All gate activations are reset
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// 
    /// For example, if you've processed one sentence and want to start with a new sentence,
    /// you should reset the state to prevent the new sentence from being influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastHiddenState = null;
        _lastZ = null;
        _lastR = null;
        _lastH = null;
        _allHiddenStates = null;
    }

    /// <summary>
    /// Exports the GRU layer's single time-step computation as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the hidden state at one time step.</returns>
    /// <remarks>
    /// <para>
    /// This method exports a single GRU cell computation for JIT compilation.
    /// The graph computes: h_t = GRUCell(x_t, h_{t-1})
    /// using the standard GRU equations with update gate, reset gate, and candidate hidden state.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create placeholders for single time-step inputs
        // x_t shape: [batchSize, inputSize]
        var inputPlaceholder = new Tensor<T>(new int[] { 1, _inputSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "x_t");

        // h_{t-1} shape: [batchSize, hiddenSize]
        var prevHiddenPlaceholder = new Tensor<T>(new int[] { 1, _hiddenSize });
        var prevHiddenNode = TensorOperations<T>.Variable(prevHiddenPlaceholder, "h_prev");

        // Create weight and bias nodes (storage is already Tensor<T>)
        var WzNode = TensorOperations<T>.Variable(_Wz, "W_z");
        var WrNode = TensorOperations<T>.Variable(_Wr, "W_r");
        var WhNode = TensorOperations<T>.Variable(_Wh, "W_h");
        var UzNode = TensorOperations<T>.Variable(_Uz, "U_z");
        var UrNode = TensorOperations<T>.Variable(_Ur, "U_r");
        var UhNode = TensorOperations<T>.Variable(_Uh, "U_h");
        var bzNode = TensorOperations<T>.Variable(_bz, "b_z");
        var brNode = TensorOperations<T>.Variable(_br, "b_r");
        var bhNode = TensorOperations<T>.Variable(_bh, "b_h");

        // Add inputs to the list
        inputNodes.Add(inputNode);
        inputNodes.Add(prevHiddenNode);
        inputNodes.Add(WzNode);
        inputNodes.Add(WrNode);
        inputNodes.Add(WhNode);
        inputNodes.Add(UzNode);
        inputNodes.Add(UrNode);
        inputNodes.Add(UhNode);
        inputNodes.Add(bzNode);
        inputNodes.Add(brNode);
        inputNodes.Add(bhNode);

        // Build GRU computation graph (single time step)
        // Update gate: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
        var WzT = TensorOperations<T>.Transpose(WzNode);
        var UzT = TensorOperations<T>.Transpose(UzNode);
        var z_input = TensorOperations<T>.MatrixMultiply(inputNode, WzT);
        var z_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, UzT);
        var z_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(z_input, z_hidden), bzNode);
        var z_t = TensorOperations<T>.Sigmoid(z_preact);

        // Reset gate: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)
        var WrT = TensorOperations<T>.Transpose(WrNode);
        var UrT = TensorOperations<T>.Transpose(UrNode);
        var r_input = TensorOperations<T>.MatrixMultiply(inputNode, WrT);
        var r_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, UrT);
        var r_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(r_input, r_hidden), brNode);
        var r_t = TensorOperations<T>.Sigmoid(r_preact);

        // Candidate hidden state: h_candidate = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)
        var WhT = TensorOperations<T>.Transpose(WhNode);
        var UhT = TensorOperations<T>.Transpose(UhNode);
        var h_input = TensorOperations<T>.MatrixMultiply(inputNode, WhT);
        var r_gated = TensorOperations<T>.ElementwiseMultiply(r_t, prevHiddenNode);
        var h_hidden = TensorOperations<T>.MatrixMultiply(r_gated, UhT);
        var h_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(h_input, h_hidden), bhNode);
        var h_candidate = TensorOperations<T>.Tanh(h_preact);

        // Final hidden state: h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h_candidate
        var z_gated = TensorOperations<T>.ElementwiseMultiply(z_t, prevHiddenNode);

        // Compute (1 - z_t)
        var onesTensor = new Tensor<T>(new int[] { 1, _hiddenSize });
        onesTensor.Fill(NumOps.One);
        var onesNode = TensorOperations<T>.Constant(onesTensor);
        var one_minus_z = TensorOperations<T>.Subtract(onesNode, z_t);

        var candidate_gated = TensorOperations<T>.ElementwiseMultiply(one_minus_z, h_candidate);
        var h_t = TensorOperations<T>.Add(z_gated, candidate_gated);

        return h_t;
    }

    /// <summary>
    /// Gets whether this layer currently supports JIT compilation.
    /// </summary>
    /// <value>
    /// True for GRU layers, as single time-step JIT compilation is supported.
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies the derivative of the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply the derivative to.</param>
    /// <param name="isRecurrent">If <c>true</c>, applies the derivative of the recurrent activation function; otherwise, applies the derivative of the regular activation function.</param>
    /// <returns>The tensor with the activation derivative applied.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of either the recurrent activation function (for gates) or the regular
    /// activation function (for the candidate hidden state) to the input tensor. This is used during the backward
    /// pass to calculate gradients. It supports both element-wise and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the activation function is to changes.
    ///
    /// During backpropagation (learning):
    /// - We need to know how much a small change in input affects the output
    /// - This method calculates that sensitivity for either the gate functions or the main activation
    /// - It's a crucial part of the math that allows neural networks to learn
    ///
    /// Think of it as measuring the "slope" of the activation function at each point,
    /// which tells us how to adjust the parameters to improve the network's performance.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivationDerivative(Tensor<T> input, bool isRecurrent)
    {
        if (isRecurrent)
        {
            if (_vectorRecurrentActivation != null)
                return _vectorRecurrentActivation.Derivative(input);
            else if (_recurrentActivation != null)
                return input.Transform((x, _) => _recurrentActivation.Derivative(x));
        }
        else
        {
            if (_vectorActivation != null)
                return _vectorActivation.Derivative(input);
            else if (_activation != null)
                return input.Transform((x, _) => _activation.Derivative(x));
        }

        throw new InvalidOperationException("No activation function specified.");
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with full BPTT.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements full Backpropagation Through Time (BPTT) using automatic differentiation.
    /// It builds a complete computation graph across all timesteps and backpropagates through the
    /// entire sequence, correctly computing gradients for all parameters and inputs.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];

        // Initialize gradient accumulators
        _dWz = new Tensor<T>([_hiddenSize, _inputSize]);
        _dWr = new Tensor<T>([_hiddenSize, _inputSize]);
        _dWh = new Tensor<T>([_hiddenSize, _inputSize]);
        _dUz = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dUr = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dUh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dbz = new Tensor<T>([_hiddenSize]);
        _dbr = new Tensor<T>([_hiddenSize]);
        _dbh = new Tensor<T>([_hiddenSize]);

        // Initialize input gradients tensor
        var dInputs = new Tensor<T>([batchSize, sequenceLength, _inputSize]);

        // Create input nodes for each timestep
        var inputNodes = new List<ComputationNode<T>>(sequenceLength);
        for (int t = 0; t < sequenceLength; t++)
        {
            var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);
            inputNodes.Add(Autodiff.TensorOperations<T>.Variable(xt, $"input_t{t}", requiresGradient: true));
        }

        // Create parameter nodes (shared across all timesteps)
        var WzNode = Autodiff.TensorOperations<T>.Variable(_Wz, "Wz", requiresGradient: true);
        var WrNode = Autodiff.TensorOperations<T>.Variable(_Wr, "Wr", requiresGradient: true);
        var WhNode = Autodiff.TensorOperations<T>.Variable(_Wh, "Wh", requiresGradient: true);
        var UzNode = Autodiff.TensorOperations<T>.Variable(_Uz, "Uz", requiresGradient: true);
        var UrNode = Autodiff.TensorOperations<T>.Variable(_Ur, "Ur", requiresGradient: true);
        var UhNode = Autodiff.TensorOperations<T>.Variable(_Uh, "Uh", requiresGradient: true);
        var bzNode = Autodiff.TensorOperations<T>.Variable(_bz, "bz", requiresGradient: true);
        var brNode = Autodiff.TensorOperations<T>.Variable(_br, "br", requiresGradient: true);
        var bhNode = Autodiff.TensorOperations<T>.Variable(_bh, "bh", requiresGradient: true);

        // Transpose weight matrices once (shared across timesteps)
        var WzT = Autodiff.TensorOperations<T>.Transpose(WzNode);
        var WrT = Autodiff.TensorOperations<T>.Transpose(WrNode);
        var WhT = Autodiff.TensorOperations<T>.Transpose(WhNode);
        var UzT = Autodiff.TensorOperations<T>.Transpose(UzNode);
        var UrT = Autodiff.TensorOperations<T>.Transpose(UrNode);
        var UhT = Autodiff.TensorOperations<T>.Transpose(UhNode);

        // Broadcast biases for batch operations
        var bzBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_bz, batchSize), "bz_broadcast", requiresGradient: false);
        var brBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_br, batchSize), "br_broadcast", requiresGradient: false);
        var bhBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_bh, batchSize), "bh_broadcast", requiresGradient: false);

        // Build computation graph through all timesteps (forward unrolling)
        var hiddenStates = new List<ComputationNode<T>>(sequenceLength + 1);

        // Initial hidden state (zeros)
        var h0 = new Tensor<T>([batchSize, _hiddenSize]);
        hiddenStates.Add(Autodiff.TensorOperations<T>.Variable(h0, "h0", requiresGradient: false));

        // Process each timestep, building the full computation graph
        for (int t = 0; t < sequenceLength; t++)
        {
            var xt = inputNodes[t];
            var hPrev = hiddenStates[t];

            // Compute update gate: z = sigmoid(xt @ Wz.T + h_prev @ Uz.T + bz)
            var z_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WzT);
            var z_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(hPrev, UzT);
            var z_sum = Autodiff.TensorOperations<T>.Add(z_input, z_hidden);
            var z_preact = Autodiff.TensorOperations<T>.Add(z_sum, bzBroadcast);
            var z = Autodiff.TensorOperations<T>.Sigmoid(z_preact);

            // Compute reset gate: r = sigmoid(xt @ Wr.T + h_prev @ Ur.T + br)
            var r_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WrT);
            var r_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(hPrev, UrT);
            var r_sum = Autodiff.TensorOperations<T>.Add(r_input, r_hidden);
            var r_preact = Autodiff.TensorOperations<T>.Add(r_sum, brBroadcast);
            var r = Autodiff.TensorOperations<T>.Sigmoid(r_preact);

            // Compute candidate hidden state: h_candidate = tanh(xt @ Wh.T + (r * h_prev) @ Uh.T + bh)
            var h_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WhT);
            var r_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(r, hPrev);
            var h_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(r_gated, UhT);
            var h_sum = Autodiff.TensorOperations<T>.Add(h_input, h_hidden);
            var h_preact = Autodiff.TensorOperations<T>.Add(h_sum, bhBroadcast);
            var h_candidate = Autodiff.TensorOperations<T>.Tanh(h_preact);

            // Compute final hidden state: h = z * h_prev + (1 - z) * h_candidate
            var z_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(z, hPrev);
            var onesT = CreateOnesLike(z.Value);
            var onesNode = Autodiff.TensorOperations<T>.Constant(onesT);
            var one_minus_z = Autodiff.TensorOperations<T>.Subtract(onesNode, z);
            var candidate_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(one_minus_z, h_candidate);
            var h_new = Autodiff.TensorOperations<T>.Add(z_gated, candidate_gated);

            hiddenStates.Add(h_new);
        }

        // Determine output node(s) and set gradients
        if (_returnSequences)
        {
            // For return sequences, we need to backprop from all hidden states
            // Split the output gradient for each timestep
            for (int t = 0; t < sequenceLength; t++)
            {
                var gradT = outputGradient.Slice(1, t, t + 1).Reshape([batchSize, _hiddenSize]);
                var hNode = hiddenStates[t + 1]; // +1 because hiddenStates[0] is h0

                // Accumulate gradients if already set
                if (hNode.Gradient == null)
                {
                    hNode.Gradient = gradT;
                }
                else
                {
                    hNode.Gradient = Engine.TensorAdd(hNode.Gradient, gradT);
                }
            }
        }
        else
        {
            // For non-sequence output, only the final hidden state receives gradient
            var finalHidden = hiddenStates[sequenceLength];
            var outputGradFlat = outputGradient.Shape.Length == 2
                ? outputGradient
                : outputGradient.Reshape([batchSize, _hiddenSize]);
            finalHidden.Gradient = outputGradFlat;
        }

        // Perform topological sort and backward pass (inlined)
        // Start from all nodes that have gradients set
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        var topoStack = new Stack<(ComputationNode<T> node, bool processed)>();

        // Add all output nodes to the stack
        for (int t = sequenceLength; t >= 1; t--)
        {
            if (hiddenStates[t].Gradient != null && !visited.Contains(hiddenStates[t]))
            {
                topoStack.Push((hiddenStates[t], false));
            }
        }

        while (topoStack.Count > 0)
        {
            var (currentNode, processed) = topoStack.Pop();
            if (visited.Contains(currentNode)) continue;

            if (processed)
            {
                visited.Add(currentNode);
                topoOrder.Add(currentNode);
            }
            else
            {
                topoStack.Push((currentNode, true));
                foreach (var parent in currentNode.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        topoStack.Push((parent, false));
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

        // Extract parameter gradients (accumulated across all timesteps via autodiff)
        if (WzNode.Gradient != null) _dWz = WzNode.Gradient;
        if (WrNode.Gradient != null) _dWr = WrNode.Gradient;
        if (WhNode.Gradient != null) _dWh = WhNode.Gradient;
        if (UzNode.Gradient != null) _dUz = UzNode.Gradient;
        if (UrNode.Gradient != null) _dUr = UrNode.Gradient;
        if (UhNode.Gradient != null) _dUh = UhNode.Gradient;
        if (bzNode.Gradient != null) _dbz = bzNode.Gradient.Sum([0]);
        if (brNode.Gradient != null) _dbr = brNode.Gradient.Sum([0]);
        if (bhNode.Gradient != null) _dbh = bhNode.Gradient.Sum([0]);

        // Extract input gradients for each timestep using Engine.TensorSetSlice
        for (int t = 0; t < sequenceLength; t++)
        {
            var inputGrad = inputNodes[t].Gradient;
            if (inputGrad != null)
            {
                var inputGradReshaped = inputGrad.Reshape([batchSize, 1, _inputSize]);
                dInputs = Engine.TensorSetSlice(dInputs, inputGradReshaped, [0, t, 0]);
            }
        }

        return dInputs;
    }

    /// <summary>
    /// Broadcasts a 1D tensor across the batch dimension.
    /// </summary>
    /// <param name="vector">The 1D tensor to broadcast.</param>
    /// <param name="batchSize">The batch size to broadcast to.</param>
    /// <returns>A 2D tensor with the vector broadcast across rows.</returns>
    private Tensor<T> BroadcastVector(Tensor<T> vector, int batchSize)
    {
        // Use Engine.TensorTile to broadcast the vector across the batch dimension
        // First reshape vector from [length] to [1, length], then tile along first axis
        var reshapedVector = vector.Reshape([1, vector.Length]);
        return Engine.TensorTile(reshapedVector, [batchSize, 1]);
    }

    /// <summary>
    /// Creates a tensor of ones with the same shape as the input tensor.
    /// </summary>
    private Tensor<T> CreateOnesLike(Tensor<T> tensor)
    {
        var ones = new Tensor<T>(tensor.Shape);
        ones.Fill(NumOps.One);
        return ones;
    }
}
