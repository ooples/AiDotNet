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
    /// The weight matrices for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight matrices transform the input at each time step:
    /// - _Wz: Weights for the update gate that determines how much of the previous hidden state to keep
    /// - _Wr: Weights for the reset gate that determines how much of the previous hidden state to reset
    /// - _Wh: Weights for the candidate hidden state that contains new information
    /// </para>
    /// <para><b>For Beginners:</b> These weights transform your input data into useful information.
    /// 
    /// Think of these weight matrices as "filters" that extract different types of information:
    /// - _Wz helps decide how much old information to keep
    /// - _Wr helps decide how much old information to reset or forget
    /// - _Wh helps create new information based on the current input
    /// 
    /// During training, these weights are adjusted to better recognize important patterns in your data.
    /// </para>
    /// </remarks>
    private Matrix<T> _Wz, _Wr, _Wh = default!;

    /// <summary>
    /// The weight matrices that transform the previous hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight matrices process the previous hidden state:
    /// - _Uz: Weights that transform the previous hidden state for the update gate
    /// - _Ur: Weights that transform the previous hidden state for the reset gate
    /// - _Uh: Weights that transform the previous hidden state for the candidate hidden state
    /// </para>
    /// <para><b>For Beginners:</b> These weights process the "memory" from previous time steps.
    /// 
    /// These matrices help the GRU work with information from earlier in the sequence:
    /// - _Uz helps decide which parts of the memory to keep
    /// - _Ur helps decide which parts of the memory to reset
    /// - _Uh helps combine memory with new information
    /// 
    /// For example, when reading text, these weights help the network remember important context
    /// from words that appeared earlier in the sentence.
    /// </para>
    /// </remarks>
    private Matrix<T> _Uz, _Ur, _Uh = default!;

    /// <summary>
    /// The bias vectors for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These bias vectors provide an offset to the transformations:
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
    private Vector<T> _bz, _br, _bh;

    /// <summary>
    /// Gradients for the weight matrices during backpropagation.
    /// </summary>
    private Tensor<T>? _dWz, _dWr, _dWh, _dUz, _dUr, _dUh = default!;

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
    private Tensor<T>? _lastZ, _lastR, _lastH = default!;

    /// <summary>
    /// All hidden states from the last forward pass, used when returning sequences.
    /// </summary>
    private List<Tensor<T>>? _allHiddenStates;

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

        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenSize));

        _Wz = InitializeMatrix(_hiddenSize, _inputSize, scale);
        _Wr = InitializeMatrix(_hiddenSize, _inputSize, scale);
        _Wh = InitializeMatrix(_hiddenSize, _inputSize, scale);

        _Uz = InitializeMatrix(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeMatrix(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeMatrix(_hiddenSize, _hiddenSize, scale);

        _bz = new Vector<T>(_hiddenSize);
        _br = new Vector<T>(_hiddenSize);
        _bh = new Vector<T>(_hiddenSize);
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
    /// Vector<double> activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU layer with more advanced vector-based activation functions.
    /// 
    /// Vector<double> activation functions:
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

        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenSize));

        _Wz = InitializeMatrix(_hiddenSize, _inputSize, scale);
        _Wr = InitializeMatrix(_hiddenSize, _inputSize, scale);
        _Wh = InitializeMatrix(_hiddenSize, _inputSize, scale);

        _Uz = InitializeMatrix(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeMatrix(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeMatrix(_hiddenSize, _hiddenSize, scale);

        _bz = new Vector<T>(_hiddenSize);
        _br = new Vector<T>(_hiddenSize);
        _bh = new Vector<T>(_hiddenSize);
    }

    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="cols">The number of columns in the matrix.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <returns>A new matrix with scaled random values.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new matrix with the specified dimensions and fills it with random values
    /// between -0.5 and 0.5, scaled by the provided scale factor. This type of initialization helps
    /// with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a matrix of random starting values for weights.
    /// 
    /// The method:
    /// - Creates a new matrix with the specified size
    /// - Fills it with random numbers between -0.5 and 0.5
    /// - Multiplies these numbers by a scale factor to control their size
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// The scale factor helps prevent values from being too large or too small at the start of training.
    /// </para>
    /// </remarks>
    private Matrix<T> InitializeMatrix(int rows, int cols, T scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }

        return matrix;
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
        _lastInput = input;
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];

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

        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract current time step input
            var xt = input.Slice(0, t).Reshape([batchSize, _inputSize]);
    
            var z = ApplyActivation(xt.Multiply(_Wz).Add(currentHiddenState.Multiply(_Uz)).Add(_bz), true);
            var r = ApplyActivation(xt.Multiply(_Wr).Add(currentHiddenState.Multiply(_Ur)).Add(_br), true);
            var h_candidate = ApplyActivation(xt.Multiply(_Wh).Add(r.ElementwiseMultiply(currentHiddenState.Multiply(_Uh))).Add(_bh), false);
            var h = z.ElementwiseMultiply(currentHiddenState).Add(
                z.Transform((x, _) => NumOps.Subtract(NumOps.One, x)).ElementwiseMultiply(h_candidate)
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
        if (_returnSequences && _allHiddenStates != null)
        {
            // Concatenate all hidden states along time dimension
            return Tensor<T>.Concatenate([.. _allHiddenStates], 1);
        }
        else
        {
            return currentHiddenState;
        }
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
                return _vectorRecurrentActivation.Activate(input);
            else if (_recurrentActivation != null)
                return input.Transform((x, _) => _recurrentActivation.Activate(x));
        }
        else
        {
            if (_vectorActivation != null)
                return _vectorActivation.Activate(input);
            else if (_activation != null)
                return input.Transform((x, _) => _activation.Activate(x));
        }

        throw new InvalidOperationException("No activation function specified.");
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
            var dh_candidate = dh.ElementwiseMultiply(
                _lastZ.Transform((x, _) => NumOps.Subtract(NumOps.One, x))
            );
            var dz = dh.ElementwiseMultiply(_lastHiddenState.Subtract(_lastH));

            var dr = ApplyActivationDerivative(_lastH, isRecurrent: false)
                .ElementwiseMultiply(dh_candidate)
                .ElementwiseMultiply(_lastHiddenState.Multiply(_Uh));

            var dx = dz.Multiply(_Wz.Transpose())
                    .Add(dr.Multiply(_Wr.Transpose()))
                    .Add(dh_candidate.Multiply(_Wh.Transpose()));

            // Reshape dx to match input format for the last timestep
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    dInputs[i, sequenceLength - 1, j] = dx[i, j];
                }
            }

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

                    var newH = z.ElementwiseMultiply(currentH).Add(
                        z.Transform((x, _) => NumOps.Subtract(NumOps.One, x)).ElementwiseMultiply(h_candidate)
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
                var dh_candidate = dh.ElementwiseMultiply(
                    z.Transform((x, _) => NumOps.Subtract(NumOps.One, x))
                );
                var dz = dh.ElementwiseMultiply(h_prev.Subtract(h_candidate));

                var dr = ApplyActivationDerivative(h_candidate, isRecurrent: false)
                    .ElementwiseMultiply(dh_candidate)
                    .ElementwiseMultiply(h_prev.Multiply(_Uh));

                // Input gradient for this timestep
                var dxt = dz.Multiply(_Wz.Transpose())
                        .Add(dr.Multiply(_Wr.Transpose()))
                        .Add(dh_candidate.Multiply(_Wh.Transpose()));

                // Store input gradients
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < _inputSize; j++)
                    {
                        dInputs[i, t, j] = dxt[i, j];
                    }
                }

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
    private Tensor<T> ComputeGate(Tensor<T> input, Tensor<T> hidden, Matrix<T> W, Matrix<T> U, Vector<T> b, bool isRecurrent)
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

        _Wz = _Wz.Subtract(_dWz.ToMatrix().Multiply(learningRate));
        _Wr = _Wr.Subtract(_dWr.ToMatrix().Multiply(learningRate));
        _Wh = _Wh.Subtract(_dWh.ToMatrix().Multiply(learningRate));

        _Uz = _Uz.Subtract(_dUz.ToMatrix().Multiply(learningRate));
        _Ur = _Ur.Subtract(_dUr.ToMatrix().Multiply(learningRate));
        _Uh = _Uh.Subtract(_dUh.ToMatrix().Multiply(learningRate));

        _bz = _bz.Subtract(_dbz.ToVector().Multiply(learningRate));
        _br = _br.Subtract(_dbr.ToVector().Multiply(learningRate));
        _bh = _bh.Subtract(_dbh.ToVector().Multiply(learningRate));
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
        int startIndex = 0;
    
        // Update Wz
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _Wz[i, j] = parameters[startIndex++];
            }
        }
    
        // Update Wr
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _Wr[i, j] = parameters[startIndex++];
            }
        }
    
        // Update Wh
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _Wh[i, j] = parameters[startIndex++];
            }
        }
    
        // Update Uz
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                _Uz[i, j] = parameters[startIndex++];
            }
        }
    
        // Update Ur
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                _Ur[i, j] = parameters[startIndex++];
            }
        }
    
        // Update Uh
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                _Uh[i, j] = parameters[startIndex++];
            }
        }
    
        // Update bz
        for (int i = 0; i < _hiddenSize; i++)
        {
            _bz[i] = parameters[startIndex++];
        }
    
        // Update br
        for (int i = 0; i < _hiddenSize; i++)
        {
            _br[i] = parameters[startIndex++];
        }
    
        // Update bh
        for (int i = 0; i < _hiddenSize; i++)
        {
            _bh[i] = parameters[startIndex++];
        }
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
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Get Wz parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                parameters[index++] = _Wz[i, j];
            }
        }
    
        // Get Wr parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                parameters[index++] = _Wr[i, j];
            }
        }
    
        // Get Wh parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                parameters[index++] = _Wh[i, j];
            }
        }
    
        // Get Uz parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                parameters[index++] = _Uz[i, j];
            }
        }
    
        // Get Ur parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                parameters[index++] = _Ur[i, j];
            }
        }
    
        // Get Uh parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                parameters[index++] = _Uh[i, j];
            }
        }
    
        // Get bz parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            parameters[index++] = _bz[i];
        }
    
        // Get br parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            parameters[index++] = _br[i];
        }
    
        // Get bh parameters
        for (int i = 0; i < _hiddenSize; i++)
        {
            parameters[index++] = _bh[i];
        }
    
        return parameters;
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
}