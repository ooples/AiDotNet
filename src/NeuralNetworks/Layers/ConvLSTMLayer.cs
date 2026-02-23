using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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

    #region GPU Training Caching Fields

    /// <summary>
    /// Cached GPU input for backward pass.
    /// </summary>
    private IGpuTensor<T>? _gpuInput;

    /// <summary>
    /// Cached GPU input shape for backward pass.
    /// </summary>
    private int[]? _gpuInputShape;

    /// <summary>
    /// Cached input slices (NCHW) for each timestep - needed for weight gradients.
    /// </summary>
    private List<IGpuBuffer>? _gpuInputSlices;

    /// <summary>
    /// Cached hidden states for each timestep - needed for hidden weight gradients.
    /// </summary>
    private List<IGpuBuffer>? _gpuHiddenStates;

    /// <summary>
    /// Cached cell states for each timestep - needed for cell gradient computation.
    /// </summary>
    private List<IGpuBuffer>? _gpuCellStates;

    /// <summary>
    /// Cached forget gate values for each timestep.
    /// </summary>
    private List<IGpuBuffer>? _gpuForgetGates;

    /// <summary>
    /// Cached input gate values for each timestep.
    /// </summary>
    private List<IGpuBuffer>? _gpuInputGates;

    /// <summary>
    /// Cached output gate values for each timestep.
    /// </summary>
    private List<IGpuBuffer>? _gpuOutputGates;

    /// <summary>
    /// Cached candidate cell values for each timestep.
    /// </summary>
    private List<IGpuBuffer>? _gpuCandidateCells;

    /// <summary>
    /// GPU state dimensions cached during forward pass.
    /// </summary>
    private int _gpuBatchSize;
    private int _gpuTimeSteps;
    private int _gpuHeight;
    private int _gpuWidth;
    private int _gpuChannels;

    #endregion

    #region GPU Weight Storage Fields

    // GPU weight tensors for GPU-resident training
    private GpuTensor<T>? _gpuWeightsFi;
    private GpuTensor<T>? _gpuWeightsIi;
    private GpuTensor<T>? _gpuWeightsCi;
    private GpuTensor<T>? _gpuWeightsOi;
    private GpuTensor<T>? _gpuWeightsFh;
    private GpuTensor<T>? _gpuWeightsIh;
    private GpuTensor<T>? _gpuWeightsCh;
    private GpuTensor<T>? _gpuWeightsOh;
    private GpuTensor<T>? _gpuBiasF;
    private GpuTensor<T>? _gpuBiasI;
    private GpuTensor<T>? _gpuBiasC;
    private GpuTensor<T>? _gpuBiasO;

    // GPU gradient tensors from BackwardGpu
    private GpuTensor<T>? _gpuWeightsFiGradient;
    private GpuTensor<T>? _gpuWeightsIiGradient;
    private GpuTensor<T>? _gpuWeightsCiGradient;
    private GpuTensor<T>? _gpuWeightsOiGradient;
    private GpuTensor<T>? _gpuWeightsFhGradient;
    private GpuTensor<T>? _gpuWeightsIhGradient;
    private GpuTensor<T>? _gpuWeightsChGradient;
    private GpuTensor<T>? _gpuWeightsOhGradient;
    private GpuTensor<T>? _gpuBiasFGradient;
    private GpuTensor<T>? _gpuBiasIGradient;
    private GpuTensor<T>? _gpuBiasCGradient;
    private GpuTensor<T>? _gpuBiasOGradient;

    // Optimizer state tensors for SGD/NAG/LARS (velocity)
    private GpuTensor<T>? _gpuWeightsFiVelocity;
    private GpuTensor<T>? _gpuWeightsIiVelocity;
    private GpuTensor<T>? _gpuWeightsCiVelocity;
    private GpuTensor<T>? _gpuWeightsOiVelocity;
    private GpuTensor<T>? _gpuWeightsFhVelocity;
    private GpuTensor<T>? _gpuWeightsIhVelocity;
    private GpuTensor<T>? _gpuWeightsChVelocity;
    private GpuTensor<T>? _gpuWeightsOhVelocity;
    private GpuTensor<T>? _gpuBiasFVelocity;
    private GpuTensor<T>? _gpuBiasIVelocity;
    private GpuTensor<T>? _gpuBiasCVelocity;
    private GpuTensor<T>? _gpuBiasOVelocity;

    // Optimizer state tensors for Adam/AdamW/LAMB (M and V)
    private GpuTensor<T>? _gpuWeightsFiM;
    private GpuTensor<T>? _gpuWeightsFiV;
    private GpuTensor<T>? _gpuWeightsIiM;
    private GpuTensor<T>? _gpuWeightsIiV;
    private GpuTensor<T>? _gpuWeightsCiM;
    private GpuTensor<T>? _gpuWeightsCiV;
    private GpuTensor<T>? _gpuWeightsOiM;
    private GpuTensor<T>? _gpuWeightsOiV;
    private GpuTensor<T>? _gpuWeightsFhM;
    private GpuTensor<T>? _gpuWeightsFhV;
    private GpuTensor<T>? _gpuWeightsIhM;
    private GpuTensor<T>? _gpuWeightsIhV;
    private GpuTensor<T>? _gpuWeightsChM;
    private GpuTensor<T>? _gpuWeightsChV;
    private GpuTensor<T>? _gpuWeightsOhM;
    private GpuTensor<T>? _gpuWeightsOhV;
    private GpuTensor<T>? _gpuBiasFM;
    private GpuTensor<T>? _gpuBiasFV;
    private GpuTensor<T>? _gpuBiasIM;
    private GpuTensor<T>? _gpuBiasIV;
    private GpuTensor<T>? _gpuBiasCM;
    private GpuTensor<T>? _gpuBiasCV;
    private GpuTensor<T>? _gpuBiasOM;
    private GpuTensor<T>? _gpuBiasOV;

    #endregion

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
    /// Gets a value indicating whether this layer supports GPU-accelerated forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ConvLSTM supports GPU execution when a DirectGpuTensorEngine is available.
    /// The GPU implementation uses FusedConv2DGpu for convolutions and GPU-native gate operations.
    /// </para>
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

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

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_weightsFi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsFh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasF, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasI, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasC, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasO, PersistentTensorRole.Biases);
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

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_weightsFi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsFh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasF, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasI, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasC, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasO, PersistentTensorRole.Biases);
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
    /// Performs a GPU-resident forward pass of the ConvLSTM layer.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor(s).</param>
    /// <returns>GPU-resident output tensor after ConvLSTM processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.
    /// The ConvLSTM gates are computed using GPU convolutions and element-wise operations.
    /// </para>
    /// <para>
    /// During training (IsTrainingMode == true), this method caches gate values and state buffers
    /// needed by BackwardGpu to perform full BPTT on GPU.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape;
        int rank = shape.Length;

        // Support any rank >= 4: last 4 dims are [T, H, W, C], earlier dims are batch-like
        if (rank < 4)
            throw new ArgumentException($"ConvLSTM layer requires at least 4D tensor [T,H,W,C]. Got rank {rank}.");

        int batchSize, timeSteps, height, width, channels;
        var originalShape = shape;

        if (rank == 4)
        {
            // 4D: [timeSteps, height, width, channels]
            batchSize = 1;
            timeSteps = shape[0];
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }
        else if (rank == 5)
        {
            // 5D: [batchSize, timeSteps, height, width, channels]
            batchSize = shape[0];
            timeSteps = shape[1];
            height = shape[2];
            width = shape[3];
            channels = shape[4];
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            batchSize = 1;
            for (int d = 0; d < rank - 4; d++)
                batchSize *= shape[d];
            timeSteps = shape[rank - 4];
            height = shape[rank - 3];
            width = shape[rank - 2];
            channels = shape[rank - 1];
        }

        // Calculate output spatial dimensions (same as input with padding)
        int outHeight = height;
        int outWidth = width;

        // Prepare NCHW-format weight buffers (transpose from [kH, kW, inC, outC] to [outC, inC, kH, kW])
        var weightsFiNCHW = _weightsFi.Transpose([3, 2, 0, 1]);
        var weightsIiNCHW = _weightsIi.Transpose([3, 2, 0, 1]);
        var weightsCiNCHW = _weightsCi.Transpose([3, 2, 0, 1]);
        var weightsOiNCHW = _weightsOi.Transpose([3, 2, 0, 1]);
        var weightsFhNCHW = _weightsFh.Transpose([3, 2, 0, 1]);
        var weightsIhNCHW = _weightsIh.Transpose([3, 2, 0, 1]);
        var weightsChNCHW = _weightsCh.Transpose([3, 2, 0, 1]);
        var weightsOhNCHW = _weightsOh.Transpose([3, 2, 0, 1]);

        // Upload weights to GPU
        using var wFiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsFiNCHW.ToArray()));
        using var wIiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsIiNCHW.ToArray()));
        using var wCiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsCiNCHW.ToArray()));
        using var wOiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsOiNCHW.ToArray()));
        using var wFhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsFhNCHW.ToArray()));
        using var wIhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsIhNCHW.ToArray()));
        using var wChBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsChNCHW.ToArray()));
        using var wOhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsOhNCHW.ToArray()));

        // Upload biases (flatten from [1,1,1,filters] to [filters])
        using var bFBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasF.ToArray()));
        using var bIBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasI.ToArray()));
        using var bCBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasC.ToArray()));
        using var bOBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasO.ToArray()));

        // State buffer sizes
        int stateSize = batchSize * _filters * outHeight * outWidth; // NCHW format
        int inputSliceSize = batchSize * channels * height * width;   // NCHW format

        // Initialize hidden and cell state buffers (NCHW format)
        var hiddenBuffer = backend.AllocateBuffer(stateSize);
        var cellBuffer = backend.AllocateBuffer(stateSize);
        backend.Fill(hiddenBuffer, 0.0f, stateSize);
        backend.Fill(cellBuffer, 0.0f, stateSize);

        // Allocate output buffer: [batch, timeSteps, outH, outW, filters] flattened
        int outputSize = batchSize * timeSteps * outHeight * outWidth * _filters;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Initialize GPU caching for backward pass if in training mode
        if (IsTrainingMode)
        {
            // Clear any previous cached buffers
            ClearGpuCache();

            // Cache dimensions
            _gpuBatchSize = batchSize;
            _gpuTimeSteps = timeSteps;
            _gpuHeight = height;
            _gpuWidth = width;
            _gpuChannels = channels;
            _gpuInput = input;
            _gpuInputShape = shape.ToArray();

            // Initialize lists to store intermediate states
            _gpuInputSlices = new List<IGpuBuffer>(timeSteps);
            _gpuHiddenStates = new List<IGpuBuffer>(timeSteps + 1);
            _gpuCellStates = new List<IGpuBuffer>(timeSteps + 1);
            _gpuForgetGates = new List<IGpuBuffer>(timeSteps);
            _gpuInputGates = new List<IGpuBuffer>(timeSteps);
            _gpuOutputGates = new List<IGpuBuffer>(timeSteps);
            _gpuCandidateCells = new List<IGpuBuffer>(timeSteps);

            // Store initial states (h0, c0)
            var h0 = backend.AllocateBuffer(stateSize);
            var c0 = backend.AllocateBuffer(stateSize);
            backend.Fill(h0, 0.0f, stateSize);
            backend.Fill(c0, 0.0f, stateSize);
            _gpuHiddenStates.Add(h0);
            _gpuCellStates.Add(c0);
        }

        // Temporary buffers for gate computations
        using var convTemp1 = backend.AllocateBuffer(stateSize);
        using var convTemp2 = backend.AllocateBuffer(stateSize);
        using var gateTemp = backend.AllocateBuffer(stateSize);
        using var forgetGate = backend.AllocateBuffer(stateSize);
        using var inputGate = backend.AllocateBuffer(stateSize);
        using var candidateCell = backend.AllocateBuffer(stateSize);
        using var outputGate = backend.AllocateBuffer(stateSize);
        var newCellBuffer = backend.AllocateBuffer(stateSize);
        var newHiddenBuffer = backend.AllocateBuffer(stateSize);

        try
        {
            for (int t = 0; t < timeSteps; t++)
            {
                // Get input slice for this timestep
                // Input is in NHWC format, need to work with NCHW for GPU convolutions
                // Slice offset in NHWC: t * batchSize * height * width * channels
                int inputOffset = t * batchSize * height * width * channels;

                // Create view of input slice (still NHWC)
                var inputSliceNHWC = input.CreateView(inputOffset, [batchSize, height, width, channels]);

                // For ConvLSTM, we need to work in NCHW format for GPU convolutions
                // Since the input is NHWC, we need to permute it
                var inputSliceNCHW = backend.AllocateBuffer(inputSliceSize);
                // Permute NHWC -> NCHW: [B, H, W, C] -> [B, C, H, W]
                backend.Permute(inputSliceNHWC.Buffer, inputSliceNCHW, [batchSize, height, width, channels], [0, 3, 1, 2]);

                // Cache input slice for backward pass if training
                if (IsTrainingMode && _gpuInputSlices != null)
                {
                    _gpuInputSlices.Add(inputSliceNCHW);
                }

                // Forget Gate: f = sigmoid(conv(x, Wfi) + conv(h, Wfh) + bf)
                backend.Conv2D(inputSliceNCHW, wFiBuffer, convTemp1,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Conv2D(hiddenBuffer, wFhBuffer, convTemp2,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(convTemp1, convTemp2, gateTemp, stateSize);
                backend.Conv2DBiasAdd(gateTemp, bFBuffer, batchSize, _filters, outHeight * outWidth);
                backend.Sigmoid(gateTemp, forgetGate, stateSize);

                // Input Gate: i = sigmoid(conv(x, Wii) + conv(h, Wih) + bi)
                backend.Conv2D(inputSliceNCHW, wIiBuffer, convTemp1,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Conv2D(hiddenBuffer, wIhBuffer, convTemp2,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(convTemp1, convTemp2, gateTemp, stateSize);
                backend.Conv2DBiasAdd(gateTemp, bIBuffer, batchSize, _filters, outHeight * outWidth);
                backend.Sigmoid(gateTemp, inputGate, stateSize);

                // Candidate Cell: c_tilde = tanh(conv(x, Wci) + conv(h, Wch) + bc)
                backend.Conv2D(inputSliceNCHW, wCiBuffer, convTemp1,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Conv2D(hiddenBuffer, wChBuffer, convTemp2,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(convTemp1, convTemp2, gateTemp, stateSize);
                backend.Conv2DBiasAdd(gateTemp, bCBuffer, batchSize, _filters, outHeight * outWidth);
                backend.Tanh(gateTemp, candidateCell, stateSize);

                // Output Gate: o = sigmoid(conv(x, Woi) + conv(h, Woh) + bo)
                backend.Conv2D(inputSliceNCHW, wOiBuffer, convTemp1,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Conv2D(hiddenBuffer, wOhBuffer, convTemp2,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(convTemp1, convTemp2, gateTemp, stateSize);
                backend.Conv2DBiasAdd(gateTemp, bOBuffer, batchSize, _filters, outHeight * outWidth);
                backend.Sigmoid(gateTemp, outputGate, stateSize);

                // Cache gate values for backward pass if training
                if (IsTrainingMode)
                {
                    if (_gpuForgetGates != null)
                    {
                        var fgCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(forgetGate, fgCache, stateSize);
                        _gpuForgetGates.Add(fgCache);
                    }
                    if (_gpuInputGates != null)
                    {
                        var igCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(inputGate, igCache, stateSize);
                        _gpuInputGates.Add(igCache);
                    }
                    if (_gpuOutputGates != null)
                    {
                        var ogCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(outputGate, ogCache, stateSize);
                        _gpuOutputGates.Add(ogCache);
                    }
                    if (_gpuCandidateCells != null)
                    {
                        var ccCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(candidateCell, ccCache, stateSize);
                        _gpuCandidateCells.Add(ccCache);
                    }
                }

                // New Cell State: c_t = f * c_{t-1} + i * c_tilde
                backend.Multiply(forgetGate, cellBuffer, convTemp1, stateSize);
                backend.Multiply(inputGate, candidateCell, convTemp2, stateSize);
                backend.Add(convTemp1, convTemp2, newCellBuffer, stateSize);

                // New Hidden State: h_t = o * tanh(c_t)
                backend.Tanh(newCellBuffer, convTemp1, stateSize);
                backend.Multiply(outputGate, convTemp1, newHiddenBuffer, stateSize);

                // Cache states for backward pass if training
                if (IsTrainingMode)
                {
                    if (_gpuCellStates != null)
                    {
                        var csCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(newCellBuffer, csCache, stateSize);
                        _gpuCellStates.Add(csCache);
                    }
                    if (_gpuHiddenStates != null)
                    {
                        var hsCache = backend.AllocateBuffer(stateSize);
                        backend.Copy(newHiddenBuffer, hsCache, stateSize);
                        _gpuHiddenStates.Add(hsCache);
                    }
                }

                // Swap state buffers
                var tempCell = cellBuffer;
                cellBuffer = newCellBuffer;
                newCellBuffer = tempCell;

                var tempHidden = hiddenBuffer;
                hiddenBuffer = newHiddenBuffer;
                newHiddenBuffer = tempHidden;

                // Store hidden state in output (convert from NCHW to NHWC)
                // Output offset in NHWC format: t * batchSize * outH * outW * filters
                int outputOffset = t * batchSize * outHeight * outWidth * _filters;
                using var outputSliceNHWC = backend.AllocateBuffer(stateSize);
                // Permute NCHW -> NHWC: [B, C, H, W] -> [B, H, W, C]
                backend.Permute(hiddenBuffer, outputSliceNHWC, [batchSize, _filters, outHeight, outWidth], [0, 2, 3, 1]);
                backend.Copy2DStrided(outputSliceNHWC, outputBuffer, 1, stateSize, outputSize, outputOffset);

                // Dispose input slice if not training (training mode caches it)
                if (!IsTrainingMode)
                {
                    inputSliceNCHW.Dispose();
                }
            }

            // Cleanup state buffers
            hiddenBuffer.Dispose();
            cellBuffer.Dispose();
            newCellBuffer.Dispose();
            newHiddenBuffer.Dispose();

            // Determine output shape — restore original leading dims for higher-rank input
            int[] outputShape;
            if (originalShape.Length > 5)
            {
                outputShape = new int[originalShape.Length];
                for (int d = 0; d < originalShape.Length - 4; d++)
                    outputShape[d] = originalShape[d];
                outputShape[originalShape.Length - 4] = timeSteps;
                outputShape[originalShape.Length - 3] = outHeight;
                outputShape[originalShape.Length - 2] = outWidth;
                outputShape[originalShape.Length - 1] = _filters;
            }
            else if (rank == 4)
            {
                outputShape = [timeSteps, outHeight, outWidth, _filters];
            }
            else
            {
                outputShape = [batchSize, timeSteps, outHeight, outWidth, _filters];
            }

            return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            // Cleanup on error
            hiddenBuffer.Dispose();
            cellBuffer.Dispose();
            newCellBuffer.Dispose();
            newHiddenBuffer.Dispose();
            outputBuffer.Dispose();
            throw;
        }
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
    /// Performs GPU-accelerated backward pass for ConvLSTM using Backpropagation Through Time (BPTT).
    /// </summary>
    /// <param name="outputGradient">GPU tensor with gradient from next layer [batch, timesteps, H, W, filters].</param>
    /// <returns>GPU tensor with input gradients [batch, timesteps, H, W, channels].</returns>
    /// <exception cref="InvalidOperationException">Thrown when ForwardGpu has not been called in training mode.</exception>
    /// <remarks>
    /// <para>
    /// This method implements full BPTT on GPU, computing gradients through all timesteps
    /// in reverse order. It uses the cached gate values, hidden states, and cell states
    /// from the forward pass.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        // Validate cached states exist
        if (_gpuInputSlices == null || _gpuHiddenStates == null || _gpuCellStates == null)
            throw new InvalidOperationException("ForwardGpu must be called in training mode before BackwardGpu.");

        if (_gpuForgetGates == null || _gpuInputGates == null || _gpuOutputGates == null || _gpuCandidateCells == null)
            throw new InvalidOperationException("Gate values not cached. Ensure ForwardGpu was called in training mode.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Retrieve cached dimensions
        int batchSize = _gpuBatchSize;
        int timeSteps = _gpuTimeSteps;
        int height = _gpuHeight;
        int width = _gpuWidth;
        int channels = _gpuChannels;
        int outHeight = height;
        int outWidth = width;

        int stateSize = batchSize * _filters * outHeight * outWidth;
        int inputSliceSize = batchSize * channels * height * width;
        int outputGradSize = outputGradient.Shape.Aggregate(1, (a, b) => a * b);

        // Prepare NCHW-format weight buffers for backward convolutions
        var weightsFiNCHW = _weightsFi.Transpose([3, 2, 0, 1]);
        var weightsIiNCHW = _weightsIi.Transpose([3, 2, 0, 1]);
        var weightsCiNCHW = _weightsCi.Transpose([3, 2, 0, 1]);
        var weightsOiNCHW = _weightsOi.Transpose([3, 2, 0, 1]);
        var weightsFhNCHW = _weightsFh.Transpose([3, 2, 0, 1]);
        var weightsIhNCHW = _weightsIh.Transpose([3, 2, 0, 1]);
        var weightsChNCHW = _weightsCh.Transpose([3, 2, 0, 1]);
        var weightsOhNCHW = _weightsOh.Transpose([3, 2, 0, 1]);

        // Upload weights to GPU
        using var wFiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsFiNCHW.ToArray()));
        using var wIiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsIiNCHW.ToArray()));
        using var wCiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsCiNCHW.ToArray()));
        using var wOiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsOiNCHW.ToArray()));
        using var wFhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsFhNCHW.ToArray()));
        using var wIhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsIhNCHW.ToArray()));
        using var wChBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsChNCHW.ToArray()));
        using var wOhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weightsOhNCHW.ToArray()));

        // Allocate gradient buffers for weights (accumulate over timesteps)
        using var gradWFi = backend.AllocateBuffer(_weightsFi.Length);
        using var gradWIi = backend.AllocateBuffer(_weightsIi.Length);
        using var gradWCi = backend.AllocateBuffer(_weightsCi.Length);
        using var gradWOi = backend.AllocateBuffer(_weightsOi.Length);
        using var gradWFh = backend.AllocateBuffer(_weightsFh.Length);
        using var gradWIh = backend.AllocateBuffer(_weightsIh.Length);
        using var gradWCh = backend.AllocateBuffer(_weightsCh.Length);
        using var gradWOh = backend.AllocateBuffer(_weightsOh.Length);

        // Allocate gradient buffers for biases
        using var gradBF = backend.AllocateBuffer(_filters);
        using var gradBI = backend.AllocateBuffer(_filters);
        using var gradBC = backend.AllocateBuffer(_filters);
        using var gradBO = backend.AllocateBuffer(_filters);

        // Initialize all gradient buffers to zero
        backend.Fill(gradWFi, 0.0f, _weightsFi.Length);
        backend.Fill(gradWIi, 0.0f, _weightsIi.Length);
        backend.Fill(gradWCi, 0.0f, _weightsCi.Length);
        backend.Fill(gradWOi, 0.0f, _weightsOi.Length);
        backend.Fill(gradWFh, 0.0f, _weightsFh.Length);
        backend.Fill(gradWIh, 0.0f, _weightsIh.Length);
        backend.Fill(gradWCh, 0.0f, _weightsCh.Length);
        backend.Fill(gradWOh, 0.0f, _weightsOh.Length);
        backend.Fill(gradBF, 0.0f, _filters);
        backend.Fill(gradBI, 0.0f, _filters);
        backend.Fill(gradBC, 0.0f, _filters);
        backend.Fill(gradBO, 0.0f, _filters);

        // Allocate input gradient buffer [batch, timesteps, H, W, channels]
        int inputGradSize = batchSize * timeSteps * height * width * channels;
        var inputGradBuffer = backend.AllocateBuffer(inputGradSize);
        backend.Fill(inputGradBuffer, 0.0f, inputGradSize);

        // Temporary buffers for BPTT
        using var dHNext = backend.AllocateBuffer(stateSize);
        using var dCNext = backend.AllocateBuffer(stateSize);
        using var dH = backend.AllocateBuffer(stateSize);
        using var dC = backend.AllocateBuffer(stateSize);
        using var temp1 = backend.AllocateBuffer(stateSize);
        using var temp2 = backend.AllocateBuffer(stateSize);
        using var dO = backend.AllocateBuffer(stateSize);
        using var dF = backend.AllocateBuffer(stateSize);
        using var dI = backend.AllocateBuffer(stateSize);
        using var dCTilde = backend.AllocateBuffer(stateSize);
        using var tanhC = backend.AllocateBuffer(stateSize);
        using var gradInputNCHW = backend.AllocateBuffer(inputSliceSize);
        using var gradTempWi = backend.AllocateBuffer(_weightsFi.Length);
        using var gradTempWh = backend.AllocateBuffer(_weightsFh.Length);

        // Initialize dHNext and dCNext to zero
        backend.Fill(dHNext, 0.0f, stateSize);
        backend.Fill(dCNext, 0.0f, stateSize);

        try
        {
            // BPTT loop - process timesteps in reverse order
            for (int t = timeSteps - 1; t >= 0; t--)
            {
                // Get output gradient slice for this timestep
                // Output is [batch, timesteps, H, W, filters] - NHWC format
                int outputOffset = t * batchSize * outHeight * outWidth * _filters;
                var outGradSlice = outputGradient.CreateView(outputOffset, [batchSize, outHeight, outWidth, _filters]);

                // Convert output gradient to NCHW
                using var outGradNCHW = backend.AllocateBuffer(stateSize);
                backend.Permute(outGradSlice.Buffer, outGradNCHW, [batchSize, outHeight, outWidth, _filters], [0, 3, 1, 2]);

                // Add incoming gradients: dH = dOut + dHNext
                backend.Add(outGradNCHW, dHNext, dH, stateSize);

                // Get cached values for this timestep
                var forgetGate = _gpuForgetGates[t];
                var inputGate = _gpuInputGates[t];
                var outputGate = _gpuOutputGates[t];
                var candidateCell = _gpuCandidateCells[t];
                var cellState = _gpuCellStates[t + 1]; // t+1 because we have h0, c0 at index 0
                var prevCellState = _gpuCellStates[t];
                var prevHiddenState = _gpuHiddenStates[t];
                var inputSlice = _gpuInputSlices[t];

                // Compute tanh(c_t) for output gate gradient
                backend.Tanh(cellState, tanhC, stateSize);

                // dO = dH * tanh(c_t) * sigmoid_deriv(o)
                // sigmoid_deriv(o) = o * (1 - o)
                backend.Multiply(dH, tanhC, temp1, stateSize);
                backend.SigmoidBackward(outputGate, temp1, dO, stateSize);

                // dC += dH * o * tanh_deriv(c_t) + dCNext
                // tanh_deriv(c) = 1 - tanh(c)^2
                backend.TanhBackward(tanhC, temp1, temp2, stateSize); // temp2 = dH * o * tanh'(c)
                backend.Multiply(dH, outputGate, temp1, stateSize);
                backend.Multiply(temp1, temp2, dC, stateSize);
                backend.Add(dC, dCNext, dC, stateSize);

                // dF = dC * c_{t-1} * sigmoid_deriv(f)
                backend.Multiply(dC, prevCellState, temp1, stateSize);
                backend.SigmoidBackward(forgetGate, temp1, dF, stateSize);

                // dI = dC * c_tilde * sigmoid_deriv(i)
                backend.Multiply(dC, candidateCell, temp1, stateSize);
                backend.SigmoidBackward(inputGate, temp1, dI, stateSize);

                // dC_tilde = dC * i * tanh_deriv(c_tilde)
                backend.Multiply(dC, inputGate, temp1, stateSize);
                backend.TanhBackward(candidateCell, temp1, dCTilde, stateSize);

                // Compute gradients for input weights using Conv2DBackwardKernel
                // dW_fi += Conv2DBackward(dF, x_t)
                backend.Conv2DBackwardKernel(inputSlice, dF, gradTempWi,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWFi, gradTempWi, gradWFi, _weightsFi.Length);

                // dW_ii += Conv2DBackward(dI, x_t)
                backend.Conv2DBackwardKernel(inputSlice, dI, gradTempWi,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWIi, gradTempWi, gradWIi, _weightsIi.Length);

                // dW_ci += Conv2DBackward(dC_tilde, x_t)
                backend.Conv2DBackwardKernel(inputSlice, dCTilde, gradTempWi,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWCi, gradTempWi, gradWCi, _weightsCi.Length);

                // dW_oi += Conv2DBackward(dO, x_t)
                backend.Conv2DBackwardKernel(inputSlice, dO, gradTempWi,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWOi, gradTempWi, gradWOi, _weightsOi.Length);

                // Compute gradients for hidden weights using Conv2DBackwardKernel
                // dW_fh += Conv2DBackward(dF, h_{t-1})
                backend.Conv2DBackwardKernel(prevHiddenState, dF, gradTempWh,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWFh, gradTempWh, gradWFh, _weightsFh.Length);

                // dW_ih += Conv2DBackward(dI, h_{t-1})
                backend.Conv2DBackwardKernel(prevHiddenState, dI, gradTempWh,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWIh, gradTempWh, gradWIh, _weightsIh.Length);

                // dW_ch += Conv2DBackward(dC_tilde, h_{t-1})
                backend.Conv2DBackwardKernel(prevHiddenState, dCTilde, gradTempWh,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWCh, gradTempWh, gradWCh, _weightsCh.Length);

                // dW_oh += Conv2DBackward(dO, h_{t-1})
                backend.Conv2DBackwardKernel(prevHiddenState, dO, gradTempWh,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradWOh, gradTempWh, gradWOh, _weightsOh.Length);

                // Compute bias gradients by summing over batch and spatial dimensions
                backend.LocallyConnectedConv2DBackwardBias(dF, gradBF, batchSize, _filters, outHeight, outWidth);
                backend.LocallyConnectedConv2DBackwardBias(dI, gradBI, batchSize, _filters, outHeight, outWidth);
                backend.LocallyConnectedConv2DBackwardBias(dCTilde, gradBC, batchSize, _filters, outHeight, outWidth);
                backend.LocallyConnectedConv2DBackwardBias(dO, gradBO, batchSize, _filters, outHeight, outWidth);

                // Compute input gradient using Conv2DBackwardInput
                backend.Fill(gradInputNCHW, 0.0f, inputSliceSize);

                // dX += Conv2DBackwardInput(dF, W_fi)
                using var tempInputGrad = backend.AllocateBuffer(inputSliceSize);
                backend.Conv2DBackwardInput(dF, wFiBuffer, tempInputGrad,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradInputNCHW, tempInputGrad, gradInputNCHW, inputSliceSize);

                // dX += Conv2DBackwardInput(dI, W_ii)
                backend.Conv2DBackwardInput(dI, wIiBuffer, tempInputGrad,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradInputNCHW, tempInputGrad, gradInputNCHW, inputSliceSize);

                // dX += Conv2DBackwardInput(dC_tilde, W_ci)
                backend.Conv2DBackwardInput(dCTilde, wCiBuffer, tempInputGrad,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradInputNCHW, tempInputGrad, gradInputNCHW, inputSliceSize);

                // dX += Conv2DBackwardInput(dO, W_oi)
                backend.Conv2DBackwardInput(dO, wOiBuffer, tempInputGrad,
                    batchSize, channels, height, width, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(gradInputNCHW, tempInputGrad, gradInputNCHW, inputSliceSize);

                // Convert input gradient from NCHW to NHWC and store
                int inputOffset = t * batchSize * height * width * channels;
                using var gradInputNHWC = backend.AllocateBuffer(inputSliceSize);
                backend.Permute(gradInputNCHW, gradInputNHWC, [batchSize, channels, height, width], [0, 2, 3, 1]);
                backend.Copy2DStrided(gradInputNHWC, inputGradBuffer, 1, inputSliceSize, inputGradSize, inputOffset);

                // Compute dHNext for previous timestep
                // dH_prev = sum of Conv2DBackwardInput for all gates with hidden weights
                backend.Fill(dHNext, 0.0f, stateSize);
                using var tempHGrad = backend.AllocateBuffer(stateSize);

                backend.Conv2DBackwardInput(dF, wFhBuffer, tempHGrad,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(dHNext, tempHGrad, dHNext, stateSize);

                backend.Conv2DBackwardInput(dI, wIhBuffer, tempHGrad,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(dHNext, tempHGrad, dHNext, stateSize);

                backend.Conv2DBackwardInput(dCTilde, wChBuffer, tempHGrad,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(dHNext, tempHGrad, dHNext, stateSize);

                backend.Conv2DBackwardInput(dO, wOhBuffer, tempHGrad,
                    batchSize, _filters, outHeight, outWidth, _filters, outHeight, outWidth,
                    _kernelSize, _kernelSize, _strides, _strides, _padding, _padding, 1, 1);
                backend.Add(dHNext, tempHGrad, dHNext, stateSize);

                // Compute dCNext for previous timestep
                // dC_prev = dC * f
                backend.Multiply(dC, forgetGate, dCNext, stateSize);
            }

            // Store gradients as GPU tensors for UpdateParametersGpu
            // Note: Gradients are in NCHW format [outC, inC, kH, kW]
            int inputWeightSize = _filters * channels * _kernelSize * _kernelSize;
            int hiddenWeightSize = _filters * _filters * _kernelSize * _kernelSize;
            _gpuWeightsFiGradient = new GpuTensor<T>(backend, gradWFi, [_filters, channels, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsIiGradient = new GpuTensor<T>(backend, gradWIi, [_filters, channels, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsCiGradient = new GpuTensor<T>(backend, gradWCi, [_filters, channels, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsOiGradient = new GpuTensor<T>(backend, gradWOi, [_filters, channels, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsFhGradient = new GpuTensor<T>(backend, gradWFh, [_filters, _filters, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsIhGradient = new GpuTensor<T>(backend, gradWIh, [_filters, _filters, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsChGradient = new GpuTensor<T>(backend, gradWCh, [_filters, _filters, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsOhGradient = new GpuTensor<T>(backend, gradWOh, [_filters, _filters, _kernelSize, _kernelSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasFGradient = new GpuTensor<T>(backend, gradBF, [_filters], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasIGradient = new GpuTensor<T>(backend, gradBI, [_filters], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasCGradient = new GpuTensor<T>(backend, gradBC, [_filters], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasOGradient = new GpuTensor<T>(backend, gradBO, [_filters], GpuTensorRole.Gradient, ownsBuffer: true);

            // Also download to CPU for _gradients dictionary compatibility
            var gradWFiData = backend.DownloadBuffer(gradWFi);
            var gradWIiData = backend.DownloadBuffer(gradWIi);
            var gradWCiData = backend.DownloadBuffer(gradWCi);
            var gradWOiData = backend.DownloadBuffer(gradWOi);
            var gradWFhData = backend.DownloadBuffer(gradWFh);
            var gradWIhData = backend.DownloadBuffer(gradWIh);
            var gradWChData = backend.DownloadBuffer(gradWCh);
            var gradWOhData = backend.DownloadBuffer(gradWOh);
            var gradBFData = backend.DownloadBuffer(gradBF);
            var gradBIData = backend.DownloadBuffer(gradBI);
            var gradBCData = backend.DownloadBuffer(gradBC);
            var gradBOData = backend.DownloadBuffer(gradBO);

            // Convert back to NHWC format for weights [kH, kW, inC, outC] from [outC, inC, kH, kW]
            // and store in gradients dictionary
            _gradients = new Dictionary<string, object>
            {
                ["weightsFi"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWFiData), _weightsFi.Shape).Transpose([2, 3, 1, 0]),
                ["weightsIi"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWIiData), _weightsIi.Shape).Transpose([2, 3, 1, 0]),
                ["weightsCi"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWCiData), _weightsCi.Shape).Transpose([2, 3, 1, 0]),
                ["weightsOi"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWOiData), _weightsOi.Shape).Transpose([2, 3, 1, 0]),
                ["weightsFh"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWFhData), _weightsFh.Shape).Transpose([2, 3, 1, 0]),
                ["weightsIh"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWIhData), _weightsIh.Shape).Transpose([2, 3, 1, 0]),
                ["weightsCh"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWChData), _weightsCh.Shape).Transpose([2, 3, 1, 0]),
                ["weightsOh"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradWOhData), _weightsOh.Shape).Transpose([2, 3, 1, 0]),
                ["biasF"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBFData), _biasF.Shape),
                ["biasI"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBIData), _biasI.Shape),
                ["biasC"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBCData), _biasC.Shape),
                ["biasO"] = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBOData), _biasO.Shape)
            };

            // Clear GPU cache
            ClearGpuCache();

            // Determine output shape
            int[] inputGradShape = _gpuInputShape ?? [batchSize, timeSteps, height, width, channels];

            return new GpuTensor<T>(backend, inputGradBuffer, inputGradShape, GpuTensorRole.Gradient, ownsBuffer: true);
        }
        catch
        {
            inputGradBuffer.Dispose();
            throw;
        }
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

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_weightsFi);
        Engine.InvalidatePersistentTensor(_weightsIi);
        Engine.InvalidatePersistentTensor(_weightsCi);
        Engine.InvalidatePersistentTensor(_weightsOi);
        Engine.InvalidatePersistentTensor(_weightsFh);
        Engine.InvalidatePersistentTensor(_weightsIh);
        Engine.InvalidatePersistentTensor(_weightsCh);
        Engine.InvalidatePersistentTensor(_weightsOh);
        Engine.InvalidatePersistentTensor(_biasF);
        Engine.InvalidatePersistentTensor(_biasI);
        Engine.InvalidatePersistentTensor(_biasC);
        Engine.InvalidatePersistentTensor(_biasO);

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

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_weightsFi);
        Engine.InvalidatePersistentTensor(_weightsIi);
        Engine.InvalidatePersistentTensor(_weightsCi);
        Engine.InvalidatePersistentTensor(_weightsOi);
        Engine.InvalidatePersistentTensor(_weightsFh);
        Engine.InvalidatePersistentTensor(_weightsIh);
        Engine.InvalidatePersistentTensor(_weightsCh);
        Engine.InvalidatePersistentTensor(_weightsOh);
        Engine.InvalidatePersistentTensor(_biasF);
        Engine.InvalidatePersistentTensor(_biasI);
        Engine.InvalidatePersistentTensor(_biasC);
        Engine.InvalidatePersistentTensor(_biasO);
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

        // Clear GPU cached tensors
        ClearGpuCache();
    }

    /// <summary>
    /// Clears all cached GPU buffers from the forward pass.
    /// </summary>
    private void ClearGpuCache()
    {
        _gpuInput = null;
        _gpuInputShape = null;

        // Dispose and clear input slices
        if (_gpuInputSlices != null)
        {
            foreach (var buffer in _gpuInputSlices)
                buffer?.Dispose();
            _gpuInputSlices = null;
        }

        // Dispose and clear hidden states
        if (_gpuHiddenStates != null)
        {
            foreach (var buffer in _gpuHiddenStates)
                buffer?.Dispose();
            _gpuHiddenStates = null;
        }

        // Dispose and clear cell states
        if (_gpuCellStates != null)
        {
            foreach (var buffer in _gpuCellStates)
                buffer?.Dispose();
            _gpuCellStates = null;
        }

        // Dispose and clear forget gates
        if (_gpuForgetGates != null)
        {
            foreach (var buffer in _gpuForgetGates)
                buffer?.Dispose();
            _gpuForgetGates = null;
        }

        // Dispose and clear input gates
        if (_gpuInputGates != null)
        {
            foreach (var buffer in _gpuInputGates)
                buffer?.Dispose();
            _gpuInputGates = null;
        }

        // Dispose and clear output gates
        if (_gpuOutputGates != null)
        {
            foreach (var buffer in _gpuOutputGates)
                buffer?.Dispose();
            _gpuOutputGates = null;
        }

        // Dispose and clear candidate cells
        if (_gpuCandidateCells != null)
        {
            foreach (var buffer in _gpuCandidateCells)
                buffer?.Dispose();
            _gpuCandidateCells = null;
        }
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

    #region GPU Parameter Update

    /// <summary>
    /// GPU-resident parameter update with polymorphic optimizer support.
    /// Updates all weight tensors directly on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWeightsFiGradient == null || _gpuWeightsIiGradient == null ||
            _gpuWeightsCiGradient == null || _gpuWeightsOiGradient == null ||
            _gpuWeightsFhGradient == null || _gpuWeightsIhGradient == null ||
            _gpuWeightsChGradient == null || _gpuWeightsOhGradient == null ||
            _gpuBiasFGradient == null || _gpuBiasIGradient == null ||
            _gpuBiasCGradient == null || _gpuBiasOGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Convert weights to NCHW format for GPU operations (same as gradients)
        var wFiNCHW = _weightsFi.Transpose([3, 2, 0, 1]);
        var wIiNCHW = _weightsIi.Transpose([3, 2, 0, 1]);
        var wCiNCHW = _weightsCi.Transpose([3, 2, 0, 1]);
        var wOiNCHW = _weightsOi.Transpose([3, 2, 0, 1]);
        var wFhNCHW = _weightsFh.Transpose([3, 2, 0, 1]);
        var wIhNCHW = _weightsIh.Transpose([3, 2, 0, 1]);
        var wChNCHW = _weightsCh.Transpose([3, 2, 0, 1]);
        var wOhNCHW = _weightsOh.Transpose([3, 2, 0, 1]);

        // Ensure GPU weight tensors exist
        _gpuWeightsFi ??= new GpuTensor<T>(backend, wFiNCHW, GpuTensorRole.Weight);
        _gpuWeightsIi ??= new GpuTensor<T>(backend, wIiNCHW, GpuTensorRole.Weight);
        _gpuWeightsCi ??= new GpuTensor<T>(backend, wCiNCHW, GpuTensorRole.Weight);
        _gpuWeightsOi ??= new GpuTensor<T>(backend, wOiNCHW, GpuTensorRole.Weight);
        _gpuWeightsFh ??= new GpuTensor<T>(backend, wFhNCHW, GpuTensorRole.Weight);
        _gpuWeightsIh ??= new GpuTensor<T>(backend, wIhNCHW, GpuTensorRole.Weight);
        _gpuWeightsCh ??= new GpuTensor<T>(backend, wChNCHW, GpuTensorRole.Weight);
        _gpuWeightsOh ??= new GpuTensor<T>(backend, wOhNCHW, GpuTensorRole.Weight);
        _gpuBiasF ??= new GpuTensor<T>(backend, _biasF, GpuTensorRole.Bias);
        _gpuBiasI ??= new GpuTensor<T>(backend, _biasI, GpuTensorRole.Bias);
        _gpuBiasC ??= new GpuTensor<T>(backend, _biasC, GpuTensorRole.Bias);
        _gpuBiasO ??= new GpuTensor<T>(backend, _biasO, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureConvLstmOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeightsFi.Buffer, _gpuWeightsFiGradient.Buffer, BuildConvLstmOptimizerState("Wfi"), _weightsFi.Length);
        config.ApplyUpdate(backend, _gpuWeightsIi.Buffer, _gpuWeightsIiGradient.Buffer, BuildConvLstmOptimizerState("Wii"), _weightsIi.Length);
        config.ApplyUpdate(backend, _gpuWeightsCi.Buffer, _gpuWeightsCiGradient.Buffer, BuildConvLstmOptimizerState("Wci"), _weightsCi.Length);
        config.ApplyUpdate(backend, _gpuWeightsOi.Buffer, _gpuWeightsOiGradient.Buffer, BuildConvLstmOptimizerState("Woi"), _weightsOi.Length);
        config.ApplyUpdate(backend, _gpuWeightsFh.Buffer, _gpuWeightsFhGradient.Buffer, BuildConvLstmOptimizerState("Wfh"), _weightsFh.Length);
        config.ApplyUpdate(backend, _gpuWeightsIh.Buffer, _gpuWeightsIhGradient.Buffer, BuildConvLstmOptimizerState("Wih"), _weightsIh.Length);
        config.ApplyUpdate(backend, _gpuWeightsCh.Buffer, _gpuWeightsChGradient.Buffer, BuildConvLstmOptimizerState("Wch"), _weightsCh.Length);
        config.ApplyUpdate(backend, _gpuWeightsOh.Buffer, _gpuWeightsOhGradient.Buffer, BuildConvLstmOptimizerState("Woh"), _weightsOh.Length);
        config.ApplyUpdate(backend, _gpuBiasF.Buffer, _gpuBiasFGradient.Buffer, BuildConvLstmOptimizerState("Bf"), _biasF.Length);
        config.ApplyUpdate(backend, _gpuBiasI.Buffer, _gpuBiasIGradient.Buffer, BuildConvLstmOptimizerState("Bi"), _biasI.Length);
        config.ApplyUpdate(backend, _gpuBiasC.Buffer, _gpuBiasCGradient.Buffer, BuildConvLstmOptimizerState("Bc"), _biasC.Length);
        config.ApplyUpdate(backend, _gpuBiasO.Buffer, _gpuBiasOGradient.Buffer, BuildConvLstmOptimizerState("Bo"), _biasO.Length);

        // Sync back to CPU tensors for compatibility (convert back to NHWC)
        _weightsFi = _gpuWeightsFi.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsIi = _gpuWeightsIi.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsCi = _gpuWeightsCi.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsOi = _gpuWeightsOi.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsFh = _gpuWeightsFh.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsIh = _gpuWeightsIh.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsCh = _gpuWeightsCh.ToTensor().Transpose([2, 3, 1, 0]);
        _weightsOh = _gpuWeightsOh.ToTensor().Transpose([2, 3, 1, 0]);
        _biasF = _gpuBiasF.ToTensor();
        _biasI = _gpuBiasI.ToTensor();
        _biasC = _gpuBiasC.ToTensor();
        _biasO = _gpuBiasO.ToTensor();
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all ConvLSTM parameters.
    /// </summary>
    private void EnsureConvLstmOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int inputWeightSize = _weightsFi.Length;
        int hiddenWeightSize = _weightsFh.Length;
        int biasSize = _biasF.Length;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Velocity buffers
                _gpuWeightsFiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightsFiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // SquaredAvg buffers for RMSprop/Adagrad
                _gpuWeightsFiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    /// <summary>
    /// Builds the optimizer state for a specific ConvLSTM parameter.
    /// </summary>
    private GpuOptimizerState BuildConvLstmOptimizerState(string paramName)
    {
        return paramName switch
        {
            "Wfi" => new GpuOptimizerState { Velocity = _gpuWeightsFiVelocity?.Buffer, M = _gpuWeightsFiM?.Buffer, V = _gpuWeightsFiV?.Buffer, SquaredAvg = _gpuWeightsFiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsFiVelocity?.Buffer },
            "Wii" => new GpuOptimizerState { Velocity = _gpuWeightsIiVelocity?.Buffer, M = _gpuWeightsIiM?.Buffer, V = _gpuWeightsIiV?.Buffer, SquaredAvg = _gpuWeightsIiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsIiVelocity?.Buffer },
            "Wci" => new GpuOptimizerState { Velocity = _gpuWeightsCiVelocity?.Buffer, M = _gpuWeightsCiM?.Buffer, V = _gpuWeightsCiV?.Buffer, SquaredAvg = _gpuWeightsCiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsCiVelocity?.Buffer },
            "Woi" => new GpuOptimizerState { Velocity = _gpuWeightsOiVelocity?.Buffer, M = _gpuWeightsOiM?.Buffer, V = _gpuWeightsOiV?.Buffer, SquaredAvg = _gpuWeightsOiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsOiVelocity?.Buffer },
            "Wfh" => new GpuOptimizerState { Velocity = _gpuWeightsFhVelocity?.Buffer, M = _gpuWeightsFhM?.Buffer, V = _gpuWeightsFhV?.Buffer, SquaredAvg = _gpuWeightsFhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsFhVelocity?.Buffer },
            "Wih" => new GpuOptimizerState { Velocity = _gpuWeightsIhVelocity?.Buffer, M = _gpuWeightsIhM?.Buffer, V = _gpuWeightsIhV?.Buffer, SquaredAvg = _gpuWeightsIhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsIhVelocity?.Buffer },
            "Wch" => new GpuOptimizerState { Velocity = _gpuWeightsChVelocity?.Buffer, M = _gpuWeightsChM?.Buffer, V = _gpuWeightsChV?.Buffer, SquaredAvg = _gpuWeightsChVelocity?.Buffer, AccumulatedGrad = _gpuWeightsChVelocity?.Buffer },
            "Woh" => new GpuOptimizerState { Velocity = _gpuWeightsOhVelocity?.Buffer, M = _gpuWeightsOhM?.Buffer, V = _gpuWeightsOhV?.Buffer, SquaredAvg = _gpuWeightsOhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsOhVelocity?.Buffer },
            "Bf" => new GpuOptimizerState { Velocity = _gpuBiasFVelocity?.Buffer, M = _gpuBiasFM?.Buffer, V = _gpuBiasFV?.Buffer, SquaredAvg = _gpuBiasFVelocity?.Buffer, AccumulatedGrad = _gpuBiasFVelocity?.Buffer },
            "Bi" => new GpuOptimizerState { Velocity = _gpuBiasIVelocity?.Buffer, M = _gpuBiasIM?.Buffer, V = _gpuBiasIV?.Buffer, SquaredAvg = _gpuBiasIVelocity?.Buffer, AccumulatedGrad = _gpuBiasIVelocity?.Buffer },
            "Bc" => new GpuOptimizerState { Velocity = _gpuBiasCVelocity?.Buffer, M = _gpuBiasCM?.Buffer, V = _gpuBiasCV?.Buffer, SquaredAvg = _gpuBiasCVelocity?.Buffer, AccumulatedGrad = _gpuBiasCVelocity?.Buffer },
            "Bo" => new GpuOptimizerState { Velocity = _gpuBiasOVelocity?.Buffer, M = _gpuBiasOM?.Buffer, V = _gpuBiasOV?.Buffer, SquaredAvg = _gpuBiasOVelocity?.Buffer, AccumulatedGrad = _gpuBiasOVelocity?.Buffer },
            _ => new GpuOptimizerState()
        };
    }

    #endregion
}
