using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Convolutional Long Short-Term Memory (ConvLSTM) layer for processing sequential spatial data.
/// </summary>
/// <remarks>
/// <para>
/// ConvLSTM combines convolutional operations with LSTM (Long Short-Term Memory) to handle
/// spatial-temporal data. It's particularly useful for tasks involving sequences of images or
/// spatial data, such as video prediction, weather forecasting, and spatiotemporal sequence prediction.
/// </para>
/// <para>
/// Key features of ConvLSTM:
/// - Maintains spatial information throughout the processing
/// - Captures both spatial and temporal dependencies
/// - Uses convolutional operations instead of matrix multiplications in the LSTM cell
/// - Suitable for data with both spatial and temporal structure
/// </para>
/// <para><b>For Beginners:</b> ConvLSTM is like a smart video analyzer that remembers spatial patterns over time.
/// 
/// Imagine you're watching a video of clouds moving across the sky:
/// 1. ConvLSTM looks at each frame (like a photo) in the video sequence
/// 2. It remembers important spatial features (like cloud shapes) from previous frames
/// 3. It uses this memory to predict how these features might change in future frames
/// 
/// This layer is particularly good at:
/// - Predicting what might happen next in a video
/// - Analyzing patterns in weather maps over time
/// - Understanding how spatial arrangements change in a sequence
/// 
/// Unlike simpler layers that treat each frame independently, ConvLSTM connects the dots
/// between frames, making it powerful for tasks involving moving images or changing spatial data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class ConvLSTMLayer<T> : LayerBase<T>
{
    private readonly int _kernelSize;
    private readonly int _filters;
    private readonly int _padding;
    private readonly int _strides;

    private Tensor<T> _weightsFi; // Forget gate input weights
    private Tensor<T> _weightsIi; // Input gate input weights
    private Tensor<T> _weightsCi; // Cell state input weights
    private Tensor<T> _weightsOi; // Output gate input weights

    private Tensor<T> _weightsFh; // Forget gate hidden weights
    private Tensor<T> _weightsIh; // Input gate hidden weights
    private Tensor<T> _weightsCh; // Cell state hidden weights
    private Tensor<T> _weightsOh; // Output gate hidden weights

    private Tensor<T> _biasF; // Forget gate bias
    private Tensor<T> _biasI; // Input gate bias
    private Tensor<T> _biasC; // Cell state bias
    private Tensor<T> _biasO; // Output gate bias

    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastHiddenState;
    private Tensor<T>? _lastCellState;
    private Dictionary<string, object> _gradients = new Dictionary<string, object>();
    private readonly Dictionary<string, Tensor<T>> _momentums = new Dictionary<string, Tensor<T>>();
    private const double MomentumFactor = 0.9;

    private readonly SigmoidActivation<T> _sigmoidActivation = new();

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> indicating that the layer supports training; this value is always true for ConvLSTM layers.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the ConvLSTM layer can be trained through backpropagation.
    /// ConvLSTM layers always return true as they contain trainable parameters (weights and biases).
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// ConvLSTM layers always return true because they have parameters (like weights and biases) 
    /// that can be updated during training to learn patterns in spatio-temporal data (like videos or weather data).
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the ConvLSTMLayer class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [batchSize, timeSteps, height, width, channels].</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="filters">The number of output filters (channels) for the layer.</param>
    /// <param name="padding">The padding added to the input to maintain spatial dimensions.</param>
    /// <param name="strides">The stride of the convolution, controlling how the filter moves across the input.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the ConvLSTM layer with the specified parameters, initializing
    /// weights and biases for the forget, input, cell, and output gates.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you set up your ConvLSTM layer, like choosing the tools for your video analyzer.
    /// 
    /// - inputShape: Describes the size and structure of your input data (e.g., video dimensions)
    /// - kernelSize: How big of an area the layer looks at in each step (like the size of its "eye")
    /// - filters: How many different patterns or features the layer will try to detect
    /// - padding: Extra space added around the edges to maintain the spatial size
    /// - strides: How far the layer's "eye" moves in each step
    /// 
    /// For example, to analyze weather satellite images:
    /// ```csharp
    /// var convLSTM = new ConvLSTMLayer<float>(
    ///     inputShape: [batchSize: 32, timeSteps: 24, height: 64, width: 64, channels: 1],
    ///     kernelSize: 3,
    ///     filters: 32,
    ///     padding: 1,
    ///     strides: 1
    /// );
    /// ```
    /// This sets up a layer that can process 24 hours of 64x64 pixel weather images,
    /// looking for 32 different types of weather patterns.
    /// </para>
    /// </remarks>
    public ConvLSTMLayer(int[] inputShape, int kernelSize, int filters, int padding = 1, int strides = 1, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, kernelSize, filters, padding, strides), activationFunction ?? new TanhActivation<T>())
    {
        _kernelSize = kernelSize;
        _filters = filters;
        _padding = padding;
        _strides = strides;

        int inputChannels = InputShape[3];

        // Initialize weights
        _weightsFi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsIi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsCi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsOi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);

        _weightsFh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsIh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsCh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsOh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);

        // Initialize biases
        _biasF = new Tensor<T>([1, 1, 1, _filters]);
        _biasI = new Tensor<T>([1, 1, 1, _filters]);
        _biasC = new Tensor<T>([1, 1, 1, _filters]);
        _biasO = new Tensor<T>([1, 1, 1, _filters]);

        // Initialize weights with small random values
        InitializeWeights(_weightsFi);
        InitializeWeights(_weightsIi);
        InitializeWeights(_weightsCi);
        InitializeWeights(_weightsOi);
        InitializeWeights(_weightsFh);
        InitializeWeights(_weightsIh);
        InitializeWeights(_weightsCh);
        InitializeWeights(_weightsOh);

        // Initialize biases to zero
        InitializeBiases(_biasF);
        InitializeBiases(_biasI);
        InitializeBiases(_biasC);
        InitializeBiases(_biasO);
    }

    /// <summary>
    /// Initializes a new instance of the ConvLSTMLayer class with a vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [batch, time, height, width, channels].</param>
    /// <param name="kernelSize">The size of the convolutional kernel (filter).</param>
    /// <param name="filters">The number of output filters (channels) for the layer.</param>
    /// <param name="padding">The padding added to the input.</param>
    /// <param name="strides">The stride of the convolution.</param>
    /// <param name="vectorActivationFunction">The vector activation function to use. Defaults to Tanh if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor allows using a vector activation function that can process entire tensors at once,
    /// which may be more efficient for certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the first one, but uses a special type of activation function.
    /// 
    /// A vector activation function:
    /// - Processes entire groups of numbers at once, rather than one at a time
    /// - Can be faster for large datasets
    /// - Works the same way as the regular activation function, just with different internal machinery
    /// 
    /// You would use this version if you're working with very large datasets where processing
    /// speed is important, or if you have a specific vector activation function you want to use.
    /// </para>
    /// </remarks>
    public ConvLSTMLayer(int[] inputShape, int kernelSize, int filters, int padding = 1, int strides = 1, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, kernelSize, filters, padding, strides), vectorActivationFunction ?? new TanhActivation<T>())
    {
        _kernelSize = kernelSize;
        _filters = filters;
        _padding = padding;
        _strides = strides;

        int inputChannels = InputShape[3];

        // Initialize weights
        _weightsFi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsIi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsCi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);
        _weightsOi = new Tensor<T>([_kernelSize, _kernelSize, inputChannels, _filters]);

        _weightsFh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsIh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsCh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);
        _weightsOh = new Tensor<T>([_kernelSize, _kernelSize, _filters, _filters]);

        // Initialize biases
        _biasF = new Tensor<T>([1, 1, 1, _filters]);
        _biasI = new Tensor<T>([1, 1, 1, _filters]);
        _biasC = new Tensor<T>([1, 1, 1, _filters]);
        _biasO = new Tensor<T>([1, 1, 1, _filters]);

        // Initialize weights with small random values
        InitializeWeights(_weightsFi);
        InitializeWeights(_weightsIi);
        InitializeWeights(_weightsCi);
        InitializeWeights(_weightsOi);
        InitializeWeights(_weightsFh);
        InitializeWeights(_weightsIh);
        InitializeWeights(_weightsCh);
        InitializeWeights(_weightsOh);

        // Initialize biases to zero
        InitializeBiases(_biasF);
        InitializeBiases(_biasI);
        InitializeBiases(_biasC);
        InitializeBiases(_biasO);
    }

    /// <summary>
    /// Calculates the output shape of the layer based on input dimensions and layer parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="filters">The number of output filters.</param>
    /// <param name="padding">The padding added to the input.</param>
    /// <param name="strides">The stride of the convolution.</param>
    /// <returns>The calculated output shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the dimensions of the output tensor based on the input dimensions
    /// and the convolution parameters (kernel size, padding, strides).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how big the output data will be.
    /// 
    /// It's like figuring out the dimensions of a photo after cropping and resizing:
    /// - The input shape tells us the original dimensions
    /// - The kernel size is like the size of the cropping tool
    /// - Padding adds extra space around the edges
    /// - Strides determine how far to move the cropping tool each time
    /// - Filters determine how many different "versions" of the output we'll have
    /// 
    /// For example, if you have a 64×64 image and use a kernel size of 3, padding of 1,
    /// and strides of 1, the output height and width will still be 64×64, preserving
    /// the spatial dimensions.
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int kernelSize, int filters, int padding, int strides)
    {
        int outputHeight = (inputShape[1] - kernelSize + 2 * padding) / strides + 1;
        int outputWidth = (inputShape[2] - kernelSize + 2 * padding) / strides + 1;
        return new int[] { inputShape[0], outputHeight, outputWidth, filters };
    }

    /// <summary>
    /// Initializes the weights of the layer with small random values.
    /// </summary>
    /// <param name="weights">The weights tensor to initialize.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using a scaled random distribution,
    /// which helps with training stability and convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives the layer's weights their starting values.
    /// 
    /// Think of this like setting the initial position of knobs on a control panel:
    /// - The weights need to start somewhere before training
    /// - We use small random values (both positive and negative)
    /// - The values are scaled based on the size of the weight tensor
    /// 
    /// Random initialization is important because:
    /// - It breaks symmetry (if all weights started at the same value, they would all learn the same thing)
    /// - The small scale helps prevent numerical issues during early training
    /// </para>
    /// </remarks>
    private void InitializeWeights(Tensor<T> weights)
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (weights.Shape[0] * weights.Shape[1] * weights.Shape[2])));
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Initializes all bias values to zero.
    /// </summary>
    /// <param name="biases">The bias tensor to initialize.</param>
    /// <remarks>
    /// <para>
    /// This method sets all values in the bias tensor to zero, which is a common
    /// initialization strategy for biases in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets all the bias values to zero at the start.
    /// 
    /// Biases are like offset values that the layer adds after multiplication:
    /// - They help the neural network fit the data better
    /// - Unlike weights, biases are typically initialized to zero
    /// - During training, they'll be adjusted away from zero as needed
    /// 
    /// Setting biases to zero initially is a standard practice that works well for
    /// most neural network architectures.
    /// </para>
    /// </remarks>
    private void InitializeBiases(Tensor<T> biases)
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Performs the forward pass of the ConvLSTM layer.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, timeSteps, height, width, channels].</param>
    /// <returns>The output tensor after processing through the ConvLSTM layer.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass processes the input sequence through the ConvLSTM cells, updating
    /// the hidden state and cell state at each time step. It applies the convolutional
    /// operations within the LSTM structure to maintain spatial information.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like running your video through the analyzer.
    /// 
    /// During the forward pass, for each frame in the sequence:
    /// 1. The layer looks at the current frame and its memory of previous frames
    /// 2. It updates its memory based on what it sees in the current frame
    /// 3. It produces an output that combines information from the current frame and its memory
    /// 
    /// This process allows the layer to:
    /// - Remember important features from earlier in the sequence
    /// - Understand how spatial patterns are changing over time
    /// - Produce outputs that consider both the current input and the history
    /// 
    /// The result is a new sequence that captures the layer's understanding of the
    /// spatial-temporal patterns in your input data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims for rank > 5
        Tensor<T> input5D;
        int batchSize;
        int timeSteps;
        int height;
        int width;
        int channels;

        if (rank == 4)
        {
            // 4D: [timeSteps, height, width, channels] -> add batch dim
            batchSize = 1;
            timeSteps = input.Shape[0];
            height = input.Shape[1];
            width = input.Shape[2];
            channels = input.Shape[3];
            input5D = input.Reshape([1, timeSteps, height, width, channels]);
        }
        else if (rank == 5)
        {
            // Standard 5D: [batchSize, timeSteps, height, width, channels]
            batchSize = input.Shape[0];
            timeSteps = input.Shape[1];
            height = input.Shape[2];
            width = input.Shape[3];
            // channels not needed - input5D is already properly shaped
            input5D = input;
        }
        else if (rank > 5)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            timeSteps = input.Shape[rank - 4];
            height = input.Shape[rank - 3];
            width = input.Shape[rank - 2];
            channels = input.Shape[rank - 1];
            input5D = input.Reshape([flatBatch, timeSteps, height, width, channels]);
        }
        else
        {
            throw new ArgumentException($"ConvLSTMLayer requires at least 4D input, got {rank}D");
        }

        _lastInput = input5D;

        var output = new Tensor<T>([batchSize, timeSteps, height, width, _filters]);
        _lastHiddenState = new Tensor<T>([batchSize, height, width, _filters]);
        _lastCellState = new Tensor<T>([batchSize, height, width, _filters]);

        for (int t = 0; t < timeSteps; t++)
        {
            // Slice along dimension 1 (time) to get [batchSize, height, width, channels]
            var xt = input5D.GetSliceAlongDimension(t, 1);
            (_lastHiddenState, _lastCellState) = ConvLSTMCell(xt, _lastHiddenState, _lastCellState);
            // Set slice along dimension 1 (time) in output
            output.SetSlice(1, t, _lastHiddenState);
        }

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 5)
        {
            // Output shape: [...leadingDims, timeSteps, height, width, filters]
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 4] = timeSteps;
            newShape[_originalInputShape.Length - 3] = height;
            newShape[_originalInputShape.Length - 2] = width;
            newShape[_originalInputShape.Length - 1] = _filters;
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 4)
        {
            // 4D input -> 4D output (remove batch dim)
            output = output.Reshape([timeSteps, height, width, _filters]);
        }

        return output;
    }

    /// <summary>
    /// Processes a single time step through the ConvLSTM cell.
    /// </summary>
    /// <param name="input">Input tensor for the current time step with shape [batchSize, height, width, channels].</param>
    /// <param name="prevHiddenState">Previous hidden state tensor with shape [batchSize, height, width, filters].</param>
    /// <param name="prevCellState">Previous cell state tensor with shape [batchSize, height, width, filters].</param>
    /// <returns>A tuple containing the new hidden state and cell state.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core ConvLSTM cell operations:
    /// 1. Forget gate: Controls what information to discard from the cell state
    /// 2. Input gate: Controls what new information to store in the cell state
    /// 3. Candidate cell: Creates new candidate values to add to the cell state
    /// 4. Output gate: Controls what parts of the cell state to output
    /// 5. Updates cell state and hidden state based on these gates
    /// </para>
    /// <para><b>For Beginners:</b> This method processes a single frame in your sequence.
    /// 
    /// The ConvLSTM cell has four main components (called "gates"):
    /// 
    /// 1. Forget gate: Decides what to throw away from the previous memory
    ///    - Like deciding which parts of yesterday's weather aren't relevant today
    /// 
    /// 2. Input gate: Decides what new information to add to memory
    ///    - Like determining which parts of today's weather data are important to remember
    /// 
    /// 3. Cell input: Creates new candidate information to potentially add to memory
    ///    - Like identifying new weather patterns forming in today's data
    /// 
    /// 4. Output gate: Decides what information to use for the current output
    ///    - Like deciding which parts of the memory are relevant for today's forecast
    /// 
    /// These gates work together to maintain a "memory" that updates intelligently as
    /// the layer processes each frame in sequence.
    /// </para>
    /// </remarks>
    private (Tensor<T> hiddenState, Tensor<T> cellState) ConvLSTMCell(Tensor<T> input, Tensor<T> prevHiddenState, Tensor<T> prevCellState)
    {
        // Compute gate pre-activations by convolving input and hidden state, then adding bias
        // Use Engine.TensorBroadcastAdd for biases since they have shape [1,1,1,filters] and need to broadcast
        // to the convolution output shape [batchSize, height, width, filters]
        var forgetPreact = Engine.TensorAdd(Convolve(input, _weightsFi), Convolve(prevHiddenState, _weightsFh));
        forgetPreact = Engine.TensorBroadcastAdd(forgetPreact, _biasF);
        var forgetGate = Engine.Sigmoid(forgetPreact);

        var inputPreact = Engine.TensorAdd(Convolve(input, _weightsIi), Convolve(prevHiddenState, _weightsIh));
        inputPreact = Engine.TensorBroadcastAdd(inputPreact, _biasI);
        var inputGate = Engine.Sigmoid(inputPreact);

        var candidatePreact = Engine.TensorAdd(Convolve(input, _weightsCi), Convolve(prevHiddenState, _weightsCh));
        candidatePreact = Engine.TensorBroadcastAdd(candidatePreact, _biasC);
        var candidateCell = ApplyActivation(candidatePreact);

        var outputPreact = Engine.TensorAdd(Convolve(input, _weightsOi), Convolve(prevHiddenState, _weightsOh));
        outputPreact = Engine.TensorBroadcastAdd(outputPreact, _biasO);
        var outputGate = Engine.Sigmoid(outputPreact);

        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * candidate
        var newCellState = Engine.TensorAdd(
            Engine.TensorMultiply(forgetGate, prevCellState),
            Engine.TensorMultiply(inputGate, candidateCell));

        // Compute new hidden state: h_t = o_t * activation(c_t)
        var newHiddenState = Engine.TensorMultiply(outputGate, ApplyActivation(newCellState));

        return (newHiddenState, newCellState);
    }

    /// <summary>
    /// Performs a 2D convolution operation between an input tensor and a kernel.
    /// </summary>
    /// <param name="input">Input tensor with shape [batchSize, height, width, channels].</param>
    /// <param name="kernel">Kernel tensor with shape [kernelHeight, kernelWidth, inputChannels, outputChannels].</param>
    /// <returns>Output tensor with shape [batchSize, outputHeight, outputWidth, outputChannels].</returns>
    private Tensor<T> Convolve(Tensor<T> input, Tensor<T> kernel)
    {
        // Transpose input from [B, H, W, C] to [B, C, H, W] for Engine
        var inputNCHW = input.Transpose(new[] { 0, 3, 1, 2 });

        // Transpose kernel from [kH, kW, inC, outC] to [outC, inC, kH, kW] for Engine
        var kernelNCHW = kernel.Transpose(new[] { 3, 2, 0, 1 });

        var stride = new int[] { _strides, _strides };
        var padding = new int[] { _padding, _padding };
        var dilation = new int[] { 1, 1 };

        // Use GPU-accelerated Conv2D
        var outputNCHW = Engine.Conv2D(inputNCHW, kernelNCHW, stride, padding, dilation);

        // Transpose output back to [B, H, W, outC]
        return outputNCHW.Transpose(new[] { 0, 2, 3, 1 });
    }

    /// <summary>
    /// Performs the backward pass of the ConvLSTM layer, computing gradients for all parameters.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer with shape [batchSize, timeSteps, height, width, filters]</param>
    /// <returns>Gradient with respect to the input with shape [batchSize, timeSteps, height, width, channels]</returns>
    /// <remarks>
    /// <para>
    /// This method implements backpropagation through time (BPTT) for the ConvLSTM layer:
    /// 1. Initializes gradient tensors for all parameters
    /// 2. Iterates backward through time steps
    /// 3. Computes gradients for each time step using BackwardStep
    /// 4. Accumulates gradients across all time steps
    /// 5. Stores gradients for later use in parameter updates
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out how to improve the layer during training.
    ///
    /// During the backward pass:
    /// - The layer receives information about how to adjust its output to reduce errors
    /// - It works backwards through the sequence (from the most recent frame to the earliest)
    /// - It calculates how each of its internal values (weights and biases) should change
    /// - It also calculates how the input should have been different to reduce errors
    ///
    /// Think of it like a coach reviewing a game film backwards, noting what each player
    /// should have done differently at each moment to get a better outcome.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with proper BPTT graph construction.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements Backpropagation Through Time (BPTT) by unrolling the computation graph
    /// across all time steps. It correctly handles:
    /// - Temporal dependencies (hidden and cell states)
    /// - Spatial convolutions (using correct NHWC/NCHW conversions)
    /// - Complex gating logic
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Normalize outputGradient to 5D to match canonical _lastInput shape
        var outGrad5D = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length == 4)
        {
            // 4D output gradient -> 5D (add batch dim)
            outGrad5D = outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2], outputGradient.Shape[3]]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 5)
        {
            // Higher-rank output gradient -> 5D (flatten leading dims)
            int flatBatch = 1;
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                flatBatch *= _originalInputShape[d];
            outGrad5D = outputGradient.Reshape([flatBatch, outputGradient.Shape[_originalInputShape.Length - 4], outputGradient.Shape[_originalInputShape.Length - 3], outputGradient.Shape[_originalInputShape.Length - 2], outputGradient.Shape[_originalInputShape.Length - 1]]);
        }

        // 1. Create Variables for all parameters
        var wFi = Autodiff.TensorOperations<T>.Variable(_weightsFi, "weightsFi", true);
        var wIi = Autodiff.TensorOperations<T>.Variable(_weightsIi, "weightsIi", true);
        var wCi = Autodiff.TensorOperations<T>.Variable(_weightsCi, "weightsCi", true);
        var wOi = Autodiff.TensorOperations<T>.Variable(_weightsOi, "weightsOi", true);

        var wFh = Autodiff.TensorOperations<T>.Variable(_weightsFh, "weightsFh", true);
        var wIh = Autodiff.TensorOperations<T>.Variable(_weightsIh, "weightsIh", true);
        var wCh = Autodiff.TensorOperations<T>.Variable(_weightsCh, "weightsCh", true);
        var wOh = Autodiff.TensorOperations<T>.Variable(_weightsOh, "weightsOh", true);

        var bF = Autodiff.TensorOperations<T>.Variable(_biasF, "biasF", true);
        var bI = Autodiff.TensorOperations<T>.Variable(_biasI, "biasI", true);
        var bC = Autodiff.TensorOperations<T>.Variable(_biasC, "biasC", true);
        var bO = Autodiff.TensorOperations<T>.Variable(_biasO, "biasO", true);

        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", true);

        // 2. Pre-process weights for Conv2D (NHWC -> NCHW equivalent for kernels: [kH, kW, In, Out] -> [Out, In, kH, kW])
        // _weightsFi shape: [kH, kW, In, Out]
        // Target: [Out, In, kH, kW] => Permute(3, 2, 0, 1)
        var wFi_T = Autodiff.TensorOperations<T>.Permute(wFi, 3, 2, 0, 1);
        var wIi_T = Autodiff.TensorOperations<T>.Permute(wIi, 3, 2, 0, 1);
        var wCi_T = Autodiff.TensorOperations<T>.Permute(wCi, 3, 2, 0, 1);
        var wOi_T = Autodiff.TensorOperations<T>.Permute(wOi, 3, 2, 0, 1);

        var wFh_T = Autodiff.TensorOperations<T>.Permute(wFh, 3, 2, 0, 1);
        var wIh_T = Autodiff.TensorOperations<T>.Permute(wIh, 3, 2, 0, 1);
        var wCh_T = Autodiff.TensorOperations<T>.Permute(wCh, 3, 2, 0, 1);
        var wOh_T = Autodiff.TensorOperations<T>.Permute(wOh, 3, 2, 0, 1);

        // Reshape biases for broadcasting: [1, 1, 1, Filters] -> [1, Filters, 1, 1] (NCHW)
        // Input bias is [1, 1, 1, F]
        var bF_T = Autodiff.TensorOperations<T>.Reshape(bF, 1, _filters, 1, 1);
        var bI_T = Autodiff.TensorOperations<T>.Reshape(bI, 1, _filters, 1, 1);
        var bC_T = Autodiff.TensorOperations<T>.Reshape(bC, 1, _filters, 1, 1);
        var bO_T = Autodiff.TensorOperations<T>.Reshape(bO, 1, _filters, 1, 1);

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];
        int height = _lastInput.Shape[2];
        int width = _lastInput.Shape[3];
        int channels = _lastInput.Shape[4];

        // Initialize states (Zero tensors)
        var hiddenState = Autodiff.TensorOperations<T>.Constant(new Tensor<T>([batchSize, _filters, height, width]), "h0"); // NCHW for internal
        var cellState = Autodiff.TensorOperations<T>.Constant(new Tensor<T>([batchSize, _filters, height, width]), "c0");   // NCHW for internal

        var outputNodes = new List<Autodiff.ComputationNode<T>>();
        var stride = new int[] { _strides, _strides };
        var padding = new int[] { _padding, _padding };

        // 3. Unroll BPTT Loop
        for (int t = 0; t < timeSteps; t++)
        {
            // Slice time step t: [Batch, 1, H, W, C] -> [Batch, H, W, C]
            // Note: Using axis 1 for time dimension slice
            var xt_raw = Autodiff.TensorOperations<T>.Slice(inputNode, t, 1, 1, axis: 1);
            var xt_NHWC = Autodiff.TensorOperations<T>.Reshape(xt_raw, batchSize, height, width, channels);

            // Permute input to NCHW: [Batch, C, H, W]
            var xt = Autodiff.TensorOperations<T>.Permute(xt_NHWC, 0, 3, 1, 2);

            // Gates Calculation (All in NCHW format)
            // Forget Gate
            var f_x = Autodiff.TensorOperations<T>.Conv2D(xt, wFi_T, bF_T, stride, padding);
            var f_h = Autodiff.TensorOperations<T>.Conv2D(hiddenState, wFh_T, stride: stride, padding: padding);
            var f_sum = Autodiff.TensorOperations<T>.Add(f_x, f_h);
            var f = Autodiff.TensorOperations<T>.Sigmoid(f_sum);

            // Input Gate
            var i_x = Autodiff.TensorOperations<T>.Conv2D(xt, wIi_T, bI_T, stride, padding);
            var i_h = Autodiff.TensorOperations<T>.Conv2D(hiddenState, wIh_T, stride: stride, padding: padding);
            var i_sum = Autodiff.TensorOperations<T>.Add(i_x, i_h);
            var i_gate = Autodiff.TensorOperations<T>.Sigmoid(i_sum);

            // Cell Candidate
            var c_x = Autodiff.TensorOperations<T>.Conv2D(xt, wCi_T, bC_T, stride, padding);
            var c_h = Autodiff.TensorOperations<T>.Conv2D(hiddenState, wCh_T, stride: stride, padding: padding);
            var c_sum = Autodiff.TensorOperations<T>.Add(c_x, c_h);
            // Apply layer activation (defaults to Tanh for cell candidate in standard LSTM)
            // But constructor allows custom activation. LayerBase ApplyActivation uses it.
            // Standard ConvLSTM uses Tanh/Sigmoid/Tanh structure.
            // We'll use Tanh for standard compliance or ApplyActivationToGraph if it matches?
            // The constructor sets base activation to TanhActivation.
            // Let's use Tanh directly as per standard LSTM logic, or delegate?
            // Forward uses ApplyActivation for candidate cell.
            var c_cand = ApplyActivationToGraph(c_sum);

            // Output Gate
            var o_x = Autodiff.TensorOperations<T>.Conv2D(xt, wOi_T, bO_T, stride, padding);
            var o_h = Autodiff.TensorOperations<T>.Conv2D(hiddenState, wOh_T, stride: stride, padding: padding);
            var o_sum = Autodiff.TensorOperations<T>.Add(o_x, o_h);
            var o = Autodiff.TensorOperations<T>.Sigmoid(o_sum);

            // Update Cell State
            var c_forget = Autodiff.TensorOperations<T>.ElementwiseMultiply(f, cellState);
            var c_input = Autodiff.TensorOperations<T>.ElementwiseMultiply(i_gate, c_cand);
            var newC = Autodiff.TensorOperations<T>.Add(c_forget, c_input);

            // Update Hidden State
            // Forward uses ApplyActivation(newCellState) ... usually Tanh
            var c_activated = ApplyActivationToGraph(newC);
            var newH = Autodiff.TensorOperations<T>.ElementwiseMultiply(o, c_activated);

            // Store for next step
            cellState = newC;
            hiddenState = newH;

            // Convert Output back to NHWC for consistency with layer output
            var output_NHWC = Autodiff.TensorOperations<T>.Permute(newH, 0, 2, 3, 1);

            // Reshape to [Batch, 1, H, W, C] for concatenation
            var output_step = Autodiff.TensorOperations<T>.Reshape(output_NHWC, batchSize, 1, height, width, _filters);
            outputNodes.Add(output_step);
        }

        // 4. Concatenate outputs along time axis (axis 1)
        var finalOutput = Autodiff.TensorOperations<T>.Concat(outputNodes, axis: 1);

        // 5. Backward Pass
        finalOutput.Gradient = outGrad5D;

        // Topo sort and execute backward
        finalOutput.Backward(); // Uses built-in topo sort and execution

        // 6. Extract and Store Gradients in Dictionary
        _gradients = new Dictionary<string, object>
        {
            ["weightsFi"] = wFi.Gradient ?? Tensor<T>.CreateDefault(_weightsFi.Shape, NumOps.Zero),
            ["weightsIi"] = wIi.Gradient ?? Tensor<T>.CreateDefault(_weightsIi.Shape, NumOps.Zero),
            ["weightsCi"] = wCi.Gradient ?? Tensor<T>.CreateDefault(_weightsCi.Shape, NumOps.Zero),
            ["weightsOi"] = wOi.Gradient ?? Tensor<T>.CreateDefault(_weightsOi.Shape, NumOps.Zero),
            ["weightsFh"] = wFh.Gradient ?? Tensor<T>.CreateDefault(_weightsFh.Shape, NumOps.Zero),
            ["weightsIh"] = wIh.Gradient ?? Tensor<T>.CreateDefault(_weightsIh.Shape, NumOps.Zero),
            ["weightsCh"] = wCh.Gradient ?? Tensor<T>.CreateDefault(_weightsCh.Shape, NumOps.Zero),
            ["weightsOh"] = wOh.Gradient ?? Tensor<T>.CreateDefault(_weightsOh.Shape, NumOps.Zero),
            ["biasF"] = bF.Gradient ?? Tensor<T>.CreateDefault(_biasF.Shape, NumOps.Zero),
            ["biasI"] = bI.Gradient ?? Tensor<T>.CreateDefault(_biasI.Shape, NumOps.Zero),
            ["biasC"] = bC.Gradient ?? Tensor<T>.CreateDefault(_biasC.Shape, NumOps.Zero),
            ["biasO"] = bO.Gradient ?? Tensor<T>.CreateDefault(_biasO.Shape, NumOps.Zero)
        };

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore higher-rank gradients to their original shape
        if (_originalInputShape != null && _originalInputShape.Length != 5)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Manual backward pass implementation using Backpropagation Through Time (BPTT) for ConvLSTM.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations optimized for
    /// ConvLSTM networks. It performs backpropagation through time (BPTT), processing the
    /// sequence in reverse order and computing gradients for all convolutional gate parameters,
    /// hidden states, and cell states.
    /// </para>
    /// <para>
    /// Autodiff Note: ConvLSTM backward pass combines the complexity of LSTM gates with
    /// convolutional operations across spatial dimensions. Implementing this with automatic
    /// differentiation would require handling temporal dependencies, spatial convolutions,
    /// and gate-specific gradient flows. The manual implementation provides efficient and
    /// correct gradient calculations for all ConvLSTM components.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // Normalize outputGradient to 5D to match canonical _lastInput shape
        var outGrad5D = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length == 4)
        {
            // 4D output gradient -> 5D (add batch dim)
            outGrad5D = outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2], outputGradient.Shape[3]]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 5)
        {
            // Higher-rank output gradient -> 5D (flatten leading dims)
            int flatBatch = 1;
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                flatBatch *= _originalInputShape[d];
            outGrad5D = outputGradient.Reshape([flatBatch, outputGradient.Shape[_originalInputShape.Length - 4], outputGradient.Shape[_originalInputShape.Length - 3], outputGradient.Shape[_originalInputShape.Length - 2], outputGradient.Shape[_originalInputShape.Length - 1]]);
        }

        int batchSize = _lastInput!.Shape[0];
        int timeSteps = _lastInput.Shape[1];

        var dInput = new Tensor<T>(_lastInput.Shape);
        var dWeightsFi = new Tensor<T>(_weightsFi.Shape);
        var dWeightsIi = new Tensor<T>(_weightsIi.Shape);
        var dWeightsCi = new Tensor<T>(_weightsCi.Shape);
        var dWeightsOi = new Tensor<T>(_weightsOi.Shape);
        var dWeightsFh = new Tensor<T>(_weightsFh.Shape);
        var dWeightsIh = new Tensor<T>(_weightsIh.Shape);
        var dWeightsCh = new Tensor<T>(_weightsCh.Shape);
        var dWeightsOh = new Tensor<T>(_weightsOh.Shape);
        var dBiasF = new Tensor<T>(_biasF.Shape);
        var dBiasI = new Tensor<T>(_biasI.Shape);
        var dBiasC = new Tensor<T>(_biasC.Shape);
        var dBiasO = new Tensor<T>(_biasO.Shape);

        var dNextH = new Tensor<T>(_lastHiddenState!.Shape);
        var dNextC = new Tensor<T>(_lastCellState!.Shape);

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            var currentDh = outGrad5D.GetSlice(t).Add(dNextH);
            var xt = _lastInput.GetSlice(t);
            var prevH = t > 0 ? _lastHiddenState.GetSlice(t - 1) : new Tensor<T>(_lastHiddenState.Shape);
            var prevC = t > 0 ? _lastCellState.GetSlice(t - 1) : new Tensor<T>(_lastCellState.Shape);

            var (dxt, dprevH, dprevC, cellGrads) = BackwardStep(xt, prevH, prevC, currentDh, dNextC);

            dInput.SetSlice(t, dxt);
            if (t > 0)
            {
                dNextH = dprevH;
                dNextC = dprevC;
            }

            // Accumulate gradients
            dWeightsFi = dWeightsFi.Add(cellGrads.dWfi);
            dWeightsIi = dWeightsIi.Add(cellGrads.dWii);
            dWeightsCi = dWeightsCi.Add(cellGrads.dWci);
            dWeightsOi = dWeightsOi.Add(cellGrads.dWoi);
            dWeightsFh = dWeightsFh.Add(cellGrads.dWfh);
            dWeightsIh = dWeightsIh.Add(cellGrads.dWih);
            dWeightsCh = dWeightsCh.Add(cellGrads.dWch);
            dWeightsOh = dWeightsOh.Add(cellGrads.dWoh);
            dBiasF = dBiasF.Add(cellGrads.dbf);
            dBiasI = dBiasI.Add(cellGrads.dbi);
            dBiasC = dBiasC.Add(cellGrads.dbc);
            dBiasO = dBiasO.Add(cellGrads.dbo);
        }

        // Store gradients for use in UpdateParameters
        _gradients = new Dictionary<string, object>
        {
            ["weightsFi"] = dWeightsFi,
            ["weightsIi"] = dWeightsIi,
            ["weightsCi"] = dWeightsCi,
            ["weightsOi"] = dWeightsOi,
            ["weightsFh"] = dWeightsFh,
            ["weightsIh"] = dWeightsIh,
            ["weightsCh"] = dWeightsCh,
            ["weightsOh"] = dWeightsOh,
            ["biasF"] = dBiasF,
            ["biasI"] = dBiasI,
            ["biasC"] = dBiasC,
            ["biasO"] = dBiasO
        };

        // Restore higher-rank gradients to their original shape
        if (_originalInputShape != null && _originalInputShape.Length != 5)
        {
            return dInput.Reshape(_originalInputShape);
        }

        return dInput;
    }

    /// <summary>
    /// Applies the derivative of the activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply the activation derivative to</param>
    /// <returns>A tensor with the activation derivative applied element-wise</returns>
    /// <remarks>
    /// <para>
    /// This method handles both vector and scalar activation functions based on the layer configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the activation function is to changes.
    /// 
    /// During training, we need to know:
    /// - How much the activation function's output changes when its input changes
    /// - This is called the "derivative" and helps determine how to adjust the weights
    /// - It's a key part of the learning process
    /// 
    /// For example, if a small change in input causes a large change in output, the
    /// derivative will be large, and the weights connected to that activation will
    /// receive larger updates during training.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivationDerivative(Tensor<T> input)
    {
        if (UsingVectorActivation)
        {
            return VectorActivation!.Derivative(input);
        }
        else
        {
            return input.Transform((x, _) => ScalarActivation!.Derivative(x));
        }
    }

    /// <summary>
    /// Performs the backward step for a single time step in the ConvLSTM layer.
    /// </summary>
    /// <param name="xt">Input at the current time step</param>
    /// <param name="prevH">Hidden state from the previous time step</param>
    /// <param name="prevC">Cell state from the previous time step</param>
    /// <param name="dh">Gradient of the loss with respect to the hidden state</param>
    /// <param name="dc">Gradient of the loss with respect to the cell state</param>
    /// <returns>
    /// A tuple containing:
    /// - dxt: Gradient with respect to the input
    /// - dprevH: Gradient with respect to the previous hidden state
    /// - dprevC: Gradient with respect to the previous cell state
    /// - cellGrads: Gradients for all weights and biases
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method implements the backpropagation through a single ConvLSTM cell:
    /// 1. First computes the forward pass to get intermediate values
    /// 2. Computes gradients for each gate (forget, input, cell, output)
    /// 3. Computes gradients for all weights and biases
    /// 4. Returns gradients needed for continued backpropagation
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates updates for a single frame in the sequence.
    /// 
    /// For each frame:
    /// - It first calculates what the layer actually did at this frame (forward step)
    /// - Then it computes how each gate should have behaved differently
    /// - It determines how the weights and biases should change to improve performance
    /// - It also calculates how changes to this frame affect previous frames
    /// 
    /// This is like analyzing a single play in a game to see:
    /// - What each player did
    /// - What they should have done instead
    /// - How their coaches should adjust their training
    /// - How this play was affected by previous plays
    /// </para>
    /// </remarks>
    private (Tensor<T> dxt, Tensor<T> dprevH, Tensor<T> dprevC, CellGradients cellGrads) BackwardStep(
        Tensor<T> xt, Tensor<T> prevH, Tensor<T> prevC, Tensor<T> dh, Tensor<T> dc)
    {
        var (f, i, c, o, newC, newH) = ForwardStep(xt, prevH, prevC);

        var do_ = dh.Multiply(ApplyActivation(newC));
        var dNewC = dh.Multiply(o).Multiply(ApplyActivationDerivative(newC)).Add(dc);
        var df = dNewC.Multiply(prevC);
        var di = dNewC.Multiply(c);
        var dc_ = dNewC.Multiply(i);
        var dprevC = dNewC.Multiply(f);

        var dWfi = Convolve(xt.Transpose([1, 2, 3, 0]), df);
        var dWii = Convolve(xt.Transpose([1, 2, 3, 0]), di);
        var dWci = Convolve(xt.Transpose([1, 2, 3, 0]), dc_);
        var dWoi = Convolve(xt.Transpose([1, 2, 3, 0]), do_);

        var dWfh = Convolve(prevH.Transpose([1, 2, 3, 0]), df);
        var dWih = Convolve(prevH.Transpose([1, 2, 3, 0]), di);
        var dWch = Convolve(prevH.Transpose([1, 2, 3, 0]), dc_);
        var dWoh = Convolve(prevH.Transpose([1, 2, 3, 0]), do_);

        var dbf = df.Sum([0, 1, 2]).Reshape(_biasF.Shape);
        var dbi = di.Sum([0, 1, 2]).Reshape(_biasI.Shape);
        var dbc = dc_.Sum([0, 1, 2]).Reshape(_biasC.Shape);
        var dbo = do_.Sum([0, 1, 2]).Reshape(_biasO.Shape);

        var dxt = Convolve(df, _weightsFi.Transpose([1, 0, 2, 3]))
            .Add(Convolve(di, _weightsIi.Transpose([1, 0, 2, 3])))
            .Add(Convolve(dc_, _weightsCi.Transpose([1, 0, 2, 3])))
            .Add(Convolve(do_, _weightsOi.Transpose([1, 0, 2, 3])));

        var dprevH = Convolve(df, _weightsFh.Transpose([1, 0, 2, 3]))
            .Add(Convolve(di, _weightsIh.Transpose([1, 0, 2, 3])))
            .Add(Convolve(dc_, _weightsCh.Transpose([1, 0, 2, 3])))
            .Add(Convolve(do_, _weightsOh.Transpose([1, 0, 2, 3])));

        var cellGrads = new CellGradients(dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo);

        return (dxt, dprevH, dprevC, cellGrads);
    }

    /// <summary>
    /// Performs a single forward step of the ConvLSTM cell for one time step.
    /// </summary>
    /// <param name="xt">The input tensor at the current time step</param>
    /// <param name="prevH">The hidden state from the previous time step</param>
    /// <param name="prevC">The cell state from the previous time step</param>
    /// <returns>
    /// A tuple containing:
    /// - f: Forget gate activations
    /// - i: Input gate activations
    /// - c: Cell input activations
    /// - o: Output gate activations
    /// - newC: New cell state
    /// - newH: New hidden state
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method implements the core ConvLSTM cell operations:
    /// </para>
    /// <para>
    /// 1. Forget gate (f): Determines what information to discard from the cell state
    /// 2. Input gate (i): Determines what new information to store in the cell state
    /// 3. Cell input (c): Creates candidate values that could be added to the cell state
    /// 4. Output gate (o): Determines what parts of the cell state to output
    /// 5. New cell state (newC): Updates the cell state using the forget and input gates
    /// 6. New hidden state (newH): Creates the output based on the cell state and output gate
    /// </para>
    /// <para><b>For Beginners:</b> This method processes one frame through all four gates of the ConvLSTM.
    /// 
    /// For each frame, the cell:
    /// 1. Forget gate (f): Decides what to forget from previous memory
    ///    - Like clearing outdated information from a whiteboard
    /// 
    /// 2. Input gate (i): Decides what new information to accept
    ///    - Like deciding which new notes to add to the whiteboard
    /// 
    /// 3. Cell input (c): Creates potential new information
    ///    - Like drafting new notes that might be added to the whiteboard
    /// 
    /// 4. Output gate (o): Filters what to output from memory
    ///    - Like deciding which parts of the whiteboard to share with others
    /// 
    /// 5. Updates memory (newC): Combines old memory with new information
    ///    - Like erasing some old notes and adding new ones to the whiteboard
    /// 
    /// 6. Creates output (newH): Produces the final output for this frame
    ///    - Like taking a photo of the relevant parts of the whiteboard
    /// </para>
    /// </remarks>
    private (Tensor<T> f, Tensor<T> i, Tensor<T> c, Tensor<T> o, Tensor<T> newC, Tensor<T> newH) ForwardStep(
            Tensor<T> xt, Tensor<T> prevH, Tensor<T> prevC)
    {
        // Use Engine.Sigmoid for vectorized/GPU-accelerated sigmoid activations
        var f = Engine.Sigmoid(Convolve(xt, _weightsFi).Add(Convolve(prevH, _weightsFh)).Add(_biasF));
        var i = Engine.Sigmoid(Convolve(xt, _weightsIi).Add(Convolve(prevH, _weightsIh)).Add(_biasI));
        var c = ApplyActivation(Convolve(xt, _weightsCi).Add(Convolve(prevH, _weightsCh)).Add(_biasC));
        var o = Engine.Sigmoid(Convolve(xt, _weightsOi).Add(Convolve(prevH, _weightsOh)).Add(_biasO));

        var newC = f.Multiply(prevC).Add(i.Multiply(c));
        var newH = o.Multiply(ApplyActivation(newC));

        return (f, i, c, o, newC, newH);
    }

    /// <summary>
    /// Structure to hold gradients for all parameters of the ConvLSTM cell.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This structure organizes gradients for all weights and biases in the ConvLSTM cell:
    /// </para>
    /// <para>
    /// - dWfi, dWii, dWci, dWoi: Gradients for input weights
    /// - dWfh, dWih, dWch, dWoh: Gradients for hidden state weights
    /// - dbf, dbi, dbc, dbo: Gradients for biases
    /// </para>
    /// <para><b>For Beginners:</b> This structure keeps track of all the updates for weights and biases.
    /// 
    /// Think of this as a organized container with slots for each update:
    /// - Updates for weights that process the input (dWfi, dWii, dWci, dWoi)
    /// - Updates for weights that process the previous output (dWfh, dWih, dWch, dWoh)
    /// - Updates for bias values (dbf, dbi, dbc, dbo)
    /// 
    /// Having a structure like this makes it easier to gather all updates in one place
    /// before applying them to the actual parameters.
    /// </para>
    /// </remarks>
    private struct CellGradients
    {
        public Tensor<T> dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo;

        /// <summary>
        /// Initializes a new instance of the CellGradients structure with the specified gradient tensors.
        /// </summary>
        /// <param name="dWfi">Gradient for forget gate input weights</param>
        /// <param name="dWii">Gradient for input gate input weights</param>
        /// <param name="dWci">Gradient for cell input weights</param>
        /// <param name="dWoi">Gradient for output gate input weights</param>
        /// <param name="dWfh">Gradient for forget gate hidden weights</param>
        /// <param name="dWih">Gradient for input gate hidden weights</param>
        /// <param name="dWch">Gradient for cell hidden weights</param>
        /// <param name="dWoh">Gradient for output gate hidden weights</param>
        /// <param name="dbf">Gradient for forget gate bias</param>
        /// <param name="dbi">Gradient for input gate bias</param>
        /// <param name="dbc">Gradient for cell bias</param>
        /// <param name="dbo">Gradient for output gate bias</param>
        /// <remarks>
        /// This constructor initializes all the gradient fields in the structure with the provided tensors.
        /// </remarks>
        public CellGradients(Tensor<T> dWfi, Tensor<T> dWii, Tensor<T> dWci, Tensor<T> dWoi,
            Tensor<T> dWfh, Tensor<T> dWih, Tensor<T> dWch, Tensor<T> dWoh,
            Tensor<T> dbf, Tensor<T> dbi, Tensor<T> dbc, Tensor<T> dbo)
        {
            this.dWfi = dWfi; this.dWii = dWii; this.dWci = dWci; this.dWoi = dWoi;
            this.dWfh = dWfh; this.dWih = dWih; this.dWch = dWch; this.dWoh = dWoh;
            this.dbf = dbf; this.dbi = dbi; this.dbc = dbc; this.dbo = dbo;
        }
    }

    /// <summary>
    /// Updates all trainable parameters of the layer using the computed gradients and specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate controlling how much to adjust parameters</param>
    /// <remarks>
    /// <para>
    /// This method applies gradient descent with momentum to update all weights and biases:
    /// </para>
    /// <para>
    /// 1. First checks if gradients are available from a previous backward pass
    /// 2. Updates all input weights (weightsFi, weightsIi, weightsCi, weightsOi)
    /// 3. Updates all hidden weights (weightsFh, weightsIh, weightsCh, weightsOh)
    /// 4. Updates all biases (biasF, biasI, biasC, biasO)
    /// 5. Clears gradients after all updates are complete
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the calculated updates to all weights and biases.
    /// 
    /// After figuring out how parameters should change:
    /// - The learningRate controls how big each adjustment is
    /// - Smaller values make small, cautious changes
    /// - Larger values make bigger, more aggressive changes
    /// 
    /// The method also uses "momentum," which is like inertia:
    /// - If parameters have been moving in a certain direction, they tend to keep going
    /// - This helps navigate flat regions and avoid getting stuck in local minima
    /// - Think of it like rolling a ball downhill - it builds up speed in the right direction
    /// 
    /// After updating all parameters, the gradients are cleared to prepare for the next training batch.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_gradients.Count == 0)
        {
            throw new InvalidOperationException("No gradients available. Ensure backward pass is called before updating parameters.");
        }

        _weightsFi = UpdateParameterWithMomentum(_weightsFi, "weightsFi", learningRate);
        _weightsIi = UpdateParameterWithMomentum(_weightsIi, "weightsIi", learningRate);
        _weightsCi = UpdateParameterWithMomentum(_weightsCi, "weightsCi", learningRate);
        _weightsOi = UpdateParameterWithMomentum(_weightsOi, "weightsOi", learningRate);
        _weightsFh = UpdateParameterWithMomentum(_weightsFh, "weightsFh", learningRate);
        _weightsIh = UpdateParameterWithMomentum(_weightsIh, "weightsIh", learningRate);
        _weightsCh = UpdateParameterWithMomentum(_weightsCh, "weightsCh", learningRate);
        _weightsOh = UpdateParameterWithMomentum(_weightsOh, "weightsOh", learningRate);

        _biasF = UpdateParameterWithMomentum(_biasF, "biasF", learningRate);
        _biasI = UpdateParameterWithMomentum(_biasI, "biasI", learningRate);
        _biasC = UpdateParameterWithMomentum(_biasC, "biasC", learningRate);
        _biasO = UpdateParameterWithMomentum(_biasO, "biasO", learningRate);

        // Clear gradients after update
        _gradients.Clear();
    }

    /// <summary>
    /// Updates a single parameter tensor using gradient descent with momentum.
    /// </summary>
    /// <param name="parameter">The parameter tensor to update</param>
    /// <param name="paramName">The name of the parameter (used to look up its gradient)</param>
    /// <param name="learningRate">The learning rate for the update</param>
    /// <remarks>
    /// <para>
    /// This method implements gradient descent with momentum for a single parameter:
    /// </para>
    /// <para>
    /// 1. Retrieves the gradient for the parameter from the _gradients dictionary
    /// 2. Retrieves or initializes the momentum for the parameter
    /// 3. For each element in the parameter:
    ///    a. Updates the momentum using the formula: momentum = momentumFactor * momentum + learningRate * gradient
    ///    b. Updates the parameter using the formula: parameter = parameter - momentum
    /// </para>
    /// <para><b>For Beginners:</b> This method updates one set of weights or biases using momentum.
    /// 
    /// The update process works like this:
    /// - First, it finds the stored gradient (direction of improvement) for this parameter
    /// - Then, it either retrieves or creates a "momentum" value for this parameter
    /// - For each value in the parameter:
    ///   * It updates the momentum by combining the previous momentum with the new gradient
    ///   * It adjusts the parameter value by subtracting the momentum
    /// 
    /// Think of momentum like pushing a ball down a hill:
    /// - The gradient shows which way is downhill
    /// - The momentum keeps the ball rolling in a consistent direction
    /// - Small bumps in the terrain (noisy gradients) don't easily knock the ball off course
    /// </para>
    /// </remarks>
    private Tensor<T> UpdateParameterWithMomentum(Tensor<T> parameter, string paramName, T learningRate)
    {
        if (!_gradients.TryGetValue(paramName, out var gradientObj) || gradientObj is not Tensor<T> gradient)
        {
            throw new InvalidOperationException($"Gradient for {paramName} not found or invalid.");
        }

        if (!_momentums.TryGetValue(paramName, out var momentum))
        {
            momentum = new Tensor<T>(parameter.Shape);
            // Initialize to zero (new Tensor does this)
        }

        // momentum = momentumFactor * momentum + learningRate * gradient
        var momFactor = NumOps.FromDouble(MomentumFactor);
        var term1 = Engine.TensorMultiplyScalar(momentum, momFactor);
        var term2 = Engine.TensorMultiplyScalar(gradient, learningRate);
        var newMomentum = Engine.TensorAdd(term1, term2);

        _momentums[paramName] = newMomentum;

        // parameter = parameter - momentum
        return Engine.TensorSubtract(parameter, newMomentum);
    }

    /// <summary>
    /// Retrieves all trainable parameters of the ConvLSTM layer as a flattened vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases of the layer</returns>
    /// <remarks>
    /// <para>
    /// This method flattens all trainable parameters into a single vector in the following order:
    /// </para>
    /// <para>
    /// 1. Input weights: _weightsFi, _weightsIi, _weightsCi, _weightsOi
    /// 2. Hidden weights: _weightsFh, _weightsIh, _weightsCh, _weightsOh
    /// 3. Biases: _biasF, _biasI, _biasC, _biasO
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values into one long list.
    /// 
    /// It's like taking all the knobs and dials from the control panel and listing them in a single row:
    /// - First, it counts how many total numbers need to be stored
    /// - Then it creates a vector (a one-dimensional array) of that size
    /// - Finally, it copies all the weights and biases into this vector in a specific order
    /// 
    /// This is useful for:
    /// - Saving all parameters to a file
    /// - Loading parameters from a file
    /// - Certain optimization techniques that work with all parameters at once
    /// - Tracking how many learnable parameters the layer has in total
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = 0;

        // Input weights
        totalParams += _weightsFi.Length;
        totalParams += _weightsIi.Length;
        totalParams += _weightsCi.Length;
        totalParams += _weightsOi.Length;

        // Hidden weights
        totalParams += _weightsFh.Length;
        totalParams += _weightsIh.Length;
        totalParams += _weightsCh.Length;
        totalParams += _weightsOh.Length;

        // Biases
        totalParams += _biasF.Length;
        totalParams += _biasI.Length;
        totalParams += _biasC.Length;
        totalParams += _biasO.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy input weights
        CopyTensorToVector(_weightsFi, parameters, ref index);
        CopyTensorToVector(_weightsIi, parameters, ref index);
        CopyTensorToVector(_weightsCi, parameters, ref index);
        CopyTensorToVector(_weightsOi, parameters, ref index);

        // Copy hidden weights
        CopyTensorToVector(_weightsFh, parameters, ref index);
        CopyTensorToVector(_weightsIh, parameters, ref index);
        CopyTensorToVector(_weightsCh, parameters, ref index);
        CopyTensorToVector(_weightsOh, parameters, ref index);

        // Copy biases
        CopyTensorToVector(_biasF, parameters, ref index);
        CopyTensorToVector(_biasI, parameters, ref index);
        CopyTensorToVector(_biasC, parameters, ref index);
        CopyTensorToVector(_biasO, parameters, ref index);

        return parameters;
    }

    /// <summary>
    /// Helper method to copy values from a tensor to a vector.
    /// </summary>
    /// <param name="tensor">Source tensor containing values to copy</param>
    /// <param name="vector">Destination vector where values will be copied</param>
    /// <param name="startIndex">Starting index in the destination vector, updated after copying</param>
    /// <remarks>
    /// <para>
    /// This method iterates through all elements in the tensor and copies them sequentially
    /// to the vector starting at the specified index. The startIndex parameter is updated
    /// to point to the next available position in the vector after copying.
    /// </para>
    /// <para><b>For Beginners:</b> This method copies values from a multi-dimensional array to a simple list.
    /// 
    /// Think of it like taking items off shelves in a store (the tensor) and placing them
    /// in a single line (the vector):
    /// - We start at a specific position in the line (startIndex)
    /// - We go through each item on the shelves one by one
    /// - We place each item in the line, moving forward one position each time
    /// - When we're done, we update our position marker to show where we stopped
    /// 
    /// This is a utility method that helps when converting between different data formats.
    /// </para>
    /// </remarks>
    private static void CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[startIndex++] = tensor[i];
        }
    }

    /// <summary>
    /// Sets all trainable parameters of the ConvLSTM layer from a flattened vector.
    /// </summary>
    /// <param name="parameters">Vector containing all weights and biases to set</param>
    /// <remarks>
    /// <para>
    /// This method updates all trainable parameters from a single vector in the following order:
    /// </para>
    /// <para>
    /// 1. Input weights: _weightsFi, _weightsIi, _weightsCi, _weightsOi
    /// 2. Hidden weights: _weightsFh, _weightsIh, _weightsCh, _weightsOh
    /// 3. Biases: _biasF, _biasI, _biasC, _biasO
    /// </para>
    /// <para><b>For Beginners:</b> This method loads all learnable values from a single list.
    /// 
    /// It's the opposite of GetParameters():
    /// - It takes a long list of numbers (the parameters vector)
    /// - It distributes these numbers back into the appropriate weight and bias tensors
    /// - It follows the same order that was used when creating the vector
    /// 
    /// This is useful when:
    /// - Loading a previously saved model
    /// - Initializing with pre-trained weights
    /// - Testing with specific parameter values
    /// - Implementing advanced optimization techniques
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        // Set input weights
        CopyVectorToTensor(parameters, _weightsFi, ref index);
        CopyVectorToTensor(parameters, _weightsIi, ref index);
        CopyVectorToTensor(parameters, _weightsCi, ref index);
        CopyVectorToTensor(parameters, _weightsOi, ref index);

        // Set hidden weights
        CopyVectorToTensor(parameters, _weightsFh, ref index);
        CopyVectorToTensor(parameters, _weightsIh, ref index);
        CopyVectorToTensor(parameters, _weightsCh, ref index);
        CopyVectorToTensor(parameters, _weightsOh, ref index);

        // Set biases
        CopyVectorToTensor(parameters, _biasF, ref index);
        CopyVectorToTensor(parameters, _biasI, ref index);
        CopyVectorToTensor(parameters, _biasC, ref index);
        CopyVectorToTensor(parameters, _biasO, ref index);
    }

    /// <summary>
    /// Helper method to copy values from a vector to a tensor.
    /// </summary>
    /// <param name="vector">Source vector containing values to copy</param>
    /// <param name="tensor">Destination tensor where values will be copied</param>
    /// <param name="startIndex">Starting index in the source vector, updated after copying</param>
    /// <remarks>
    /// <para>
    /// This method iterates through all elements in the tensor and sets them sequentially
    /// from the vector starting at the specified index. The startIndex parameter is updated
    /// to point to the next position in the vector after copying.
    /// </para>
    /// <para><b>For Beginners:</b> This method copies values from a simple list to a multi-dimensional array.
    /// 
    /// Think of it like taking items from a single line (the vector) and placing them
    /// back onto shelves in a store (the tensor):
    /// - We start at a specific position in the line (startIndex)
    /// - We fill each position in the tensor one by one
    /// - We take items from the line in order, moving forward one position each time
    /// - When we're done, we update our position marker to show where we stopped
    /// 
    /// This is a utility method that helps when converting between different data formats.
    /// </para>
    /// </remarks>
    private static void CopyVectorToTensor(Vector<T> vector, Tensor<T> tensor, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = vector[startIndex++];
        }
    }

    /// <summary>
    /// Resets the internal state of the ConvLSTM layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all cached values and gradients from previous forward and backward passes:
    /// </para>
    /// <para>
    /// 1. Clears the cached input tensor (_lastInput)
    /// 2. Clears the cached hidden state (_lastHiddenState)
    /// 3. Clears the cached cell state (_lastCellState)
    /// 4. Clears all accumulated gradients
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// It's like erasing a whiteboard to start a new lesson:
    /// - The layer forgets the last input it processed
    /// - It clears its internal memory states (hidden and cell states)
    /// - It discards any stored gradients from previous training
    /// 
    /// This is important when:
    /// - Starting to process a new, unrelated sequence
    /// - Beginning a new training epoch
    /// - Testing the model on different data
    /// - You want to ensure that information from previous sequences
    ///   doesn't influence the processing of new sequences
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastHiddenState = null;
        _lastCellState = null;

        // Clear gradients
        _gradients.Clear();
    }

    /// <summary>
    /// Exports the ConvLSTM computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which input nodes will be added. The method adds:
    /// <list type="bullet">
    /// <item><description>x_t: Current input tensor [batch, height, width, channels]</description></item>
    /// <item><description>h_prev: Previous hidden state [batch, height, width, filters]</description></item>
    /// <item><description>c_prev: Previous cell state [batch, height, width, filters]</description></item>
    /// </list>
    /// </param>
    /// <returns>A computation node representing the new hidden state h_t.</returns>
    /// <remarks>
    /// <para>
    /// This method exports a single timestep of the ConvLSTM cell for JIT compilation.
    /// The computation graph implements the full ConvLSTM equations using Conv2D operations:
    /// </para>
    /// <para>
    /// <b>Gates (all use Conv2D operations):</b>
    /// <list type="bullet">
    /// <item><description>Forget gate: f_t = σ(Conv2D(x_t, W_fi) + Conv2D(h_{t-1}, W_fh) + b_f)</description></item>
    /// <item><description>Input gate: i_t = σ(Conv2D(x_t, W_ii) + Conv2D(h_{t-1}, W_ih) + b_i)</description></item>
    /// <item><description>Cell candidate: c̃_t = tanh(Conv2D(x_t, W_ci) + Conv2D(h_{t-1}, W_ch) + b_c)</description></item>
    /// <item><description>Output gate: o_t = σ(Conv2D(x_t, W_oi) + Conv2D(h_{t-1}, W_oh) + b_o)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>State updates:</b>
    /// <list type="bullet">
    /// <item><description>Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t</description></item>
    /// <item><description>Hidden state: h_t = o_t ⊙ tanh(c_t)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a blueprint for running ConvLSTM faster.
    ///
    /// For processing sequences:
    /// 1. Initialize h_prev and c_prev to zeros for the first timestep
    /// 2. Call the JIT-compiled graph for each timestep in your sequence
    /// 3. Pass the output hidden state as h_prev for the next timestep
    /// 4. Track cell state separately if needed for stateful operation
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // ConvLSTM expects input shape: [batch, height, width, channels]
        // For JIT, we work with single-timestep input (no time dimension)
        int height = InputShape[1];
        int width = InputShape[2];
        int inputChannels = InputShape[3];

        // Create input placeholder: x_t with shape [batch, height, width, channels]
        var inputPlaceholder = new Tensor<T>([1, height, width, inputChannels]);
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "x_t");
        inputNodes.Add(inputNode);

        // Create previous hidden state placeholder: h_{t-1} with shape [batch, height, width, filters]
        int outHeight = OutputShape[1];
        int outWidth = OutputShape[2];
        var prevHiddenPlaceholder = new Tensor<T>([1, outHeight, outWidth, _filters]);
        var prevHiddenNode = TensorOperations<T>.Variable(prevHiddenPlaceholder, "h_prev");
        inputNodes.Add(prevHiddenNode);

        // Create previous cell state placeholder: c_{t-1} with shape [batch, height, width, filters]
        var prevCellPlaceholder = new Tensor<T>([1, outHeight, outWidth, _filters]);
        var prevCellNode = TensorOperations<T>.Variable(prevCellPlaceholder, "c_prev");
        inputNodes.Add(prevCellNode);

        // Create variable nodes for all weights (input weights)
        var wFi = TensorOperations<T>.Variable(_weightsFi, "W_fi");
        var wIi = TensorOperations<T>.Variable(_weightsIi, "W_ii");
        var wCi = TensorOperations<T>.Variable(_weightsCi, "W_ci");
        var wOi = TensorOperations<T>.Variable(_weightsOi, "W_oi");

        // Create variable nodes for all weights (hidden/recurrent weights)
        var wFh = TensorOperations<T>.Variable(_weightsFh, "W_fh");
        var wIh = TensorOperations<T>.Variable(_weightsIh, "W_ih");
        var wCh = TensorOperations<T>.Variable(_weightsCh, "W_ch");
        var wOh = TensorOperations<T>.Variable(_weightsOh, "W_oh");

        // Create variable nodes for biases
        var bF = TensorOperations<T>.Variable(_biasF, "b_f");
        var bI = TensorOperations<T>.Variable(_biasI, "b_i");
        var bC = TensorOperations<T>.Variable(_biasC, "b_c");
        var bO = TensorOperations<T>.Variable(_biasO, "b_o");

        // Pre-process weights for Conv2D: [kH, kW, In, Out] -> [Out, In, kH, kW]
        // This matches the permutation used in BackwardViaAutodiff for consistency
        var wFi_T = TensorOperations<T>.Permute(wFi, 3, 2, 0, 1);
        var wIi_T = TensorOperations<T>.Permute(wIi, 3, 2, 0, 1);
        var wCi_T = TensorOperations<T>.Permute(wCi, 3, 2, 0, 1);
        var wOi_T = TensorOperations<T>.Permute(wOi, 3, 2, 0, 1);

        var wFh_T = TensorOperations<T>.Permute(wFh, 3, 2, 0, 1);
        var wIh_T = TensorOperations<T>.Permute(wIh, 3, 2, 0, 1);
        var wCh_T = TensorOperations<T>.Permute(wCh, 3, 2, 0, 1);
        var wOh_T = TensorOperations<T>.Permute(wOh, 3, 2, 0, 1);

        // Reshape biases for NCHW broadcasting: [1, 1, 1, Filters] -> [1, Filters, 1, 1]
        var bF_T = TensorOperations<T>.Reshape(bF, 1, _filters, 1, 1);
        var bI_T = TensorOperations<T>.Reshape(bI, 1, _filters, 1, 1);
        var bC_T = TensorOperations<T>.Reshape(bC, 1, _filters, 1, 1);
        var bO_T = TensorOperations<T>.Reshape(bO, 1, _filters, 1, 1);

        // Permute inputs from NHWC to NCHW for Conv2D operations
        var inputNCHW = TensorOperations<T>.Permute(inputNode, 0, 3, 1, 2);
        var prevHiddenNCHW = TensorOperations<T>.Permute(prevHiddenNode, 0, 3, 1, 2);
        var prevCellNCHW = TensorOperations<T>.Permute(prevCellNode, 0, 3, 1, 2);

        // Stride and padding arrays for Conv2D
        var stride = new int[] { _strides, _strides };
        var padding = new int[] { _padding, _padding };

        // ========== Forget Gate: f_t = sigmoid(Conv2D(x_t, W_fi) + Conv2D(h_{t-1}, W_fh) + b_f) ==========
        var f_input = TensorOperations<T>.Conv2D(inputNCHW, wFi_T, bF_T, stride, padding);
        var f_hidden = TensorOperations<T>.Conv2D(prevHiddenNCHW, wFh_T, stride: stride, padding: padding);
        var f_preact = TensorOperations<T>.Add(f_input, f_hidden);
        var f_t = TensorOperations<T>.Sigmoid(f_preact);

        // ========== Input Gate: i_t = sigmoid(Conv2D(x_t, W_ii) + Conv2D(h_{t-1}, W_ih) + b_i) ==========
        var i_input = TensorOperations<T>.Conv2D(inputNCHW, wIi_T, bI_T, stride, padding);
        var i_hidden = TensorOperations<T>.Conv2D(prevHiddenNCHW, wIh_T, stride: stride, padding: padding);
        var i_preact = TensorOperations<T>.Add(i_input, i_hidden);
        var i_t = TensorOperations<T>.Sigmoid(i_preact);

        // ========== Cell Candidate: c̃_t = tanh(Conv2D(x_t, W_ci) + Conv2D(h_{t-1}, W_ch) + b_c) ==========
        var c_input = TensorOperations<T>.Conv2D(inputNCHW, wCi_T, bC_T, stride, padding);
        var c_hidden = TensorOperations<T>.Conv2D(prevHiddenNCHW, wCh_T, stride: stride, padding: padding);
        var c_preact = TensorOperations<T>.Add(c_input, c_hidden);
        var c_tilde = TensorOperations<T>.Tanh(c_preact);

        // ========== Output Gate: o_t = sigmoid(Conv2D(x_t, W_oi) + Conv2D(h_{t-1}, W_oh) + b_o) ==========
        var o_input = TensorOperations<T>.Conv2D(inputNCHW, wOi_T, bO_T, stride, padding);
        var o_hidden = TensorOperations<T>.Conv2D(prevHiddenNCHW, wOh_T, stride: stride, padding: padding);
        var o_preact = TensorOperations<T>.Add(o_input, o_hidden);
        var o_t = TensorOperations<T>.Sigmoid(o_preact);

        // ========== Cell State: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t ==========
        var forget_gated = TensorOperations<T>.ElementwiseMultiply(f_t, prevCellNCHW);
        var input_gated = TensorOperations<T>.ElementwiseMultiply(i_t, c_tilde);
        var c_t = TensorOperations<T>.Add(forget_gated, input_gated);

        // ========== Hidden State: h_t = o_t ⊙ tanh(c_t) ==========
        var c_t_activated = TensorOperations<T>.Tanh(c_t);
        var h_t_NCHW = TensorOperations<T>.ElementwiseMultiply(o_t, c_t_activated);

        // Permute output from NCHW back to NHWC for consistency with layer output format
        var h_t = TensorOperations<T>.Permute(h_t_NCHW, 0, 2, 3, 1);

        // Apply layer activation if configured (typically identity for ConvLSTM)
        var output = ApplyActivationToGraph(h_t);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. ConvLSTMLayer exports a single-step LSTM cell computation
    /// with full Conv2D operations for all gates.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for ConvLSTM exports a single timestep of the LSTM cell computation.
    /// The exported graph uses proper Conv2D operations for all gate computations, matching
    /// the behavior of the Forward method.
    /// </para>
    /// <para>
    /// For processing sequences with the JIT-compiled graph:
    /// <list type="number">
    /// <item><description>Initialize hidden and cell states to zero tensors</description></item>
    /// <item><description>For each timestep, call the compiled graph with (input, h_prev, c_prev)</description></item>
    /// <item><description>The output is the new hidden state h_t</description></item>
    /// <item><description>Track cell state c_t for the next iteration (available from intermediate computation)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}
