#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Memory;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Recurrent)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, HasTrainingMode = true, ChangesShape = true, Cost = ComputeCost.High, TestInputShape = "1, 4, 4, 1", TestConstructorArgs = "new[] { 1, 4, 4, 1 }, 3, 2, 1, 1, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class ConvLSTMLayer<T> : LayerBase<T>
{
    private readonly int _kernelSize;
    private readonly int _filters;
    private readonly int _padding;
    private readonly int _strides;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]


    private Tensor<T> _weightsFi; // Forget gate input weights
    private Tensor<T> _weightsIi; // Input gate input weights
    private Tensor<T> _weightsCi; // Cell state input weights
    private Tensor<T> _weightsOi; // Output gate input weights

    private Tensor<T> _weightsFh; // Forget gate hidden weights
    private Tensor<T> _weightsIh; // Input gate hidden weights
    private Tensor<T> _weightsCh; // Cell state hidden weights
    private Tensor<T> _weightsOh; // Output gate hidden weights

    [TrainableParameter(Role = PersistentTensorRole.Biases)]


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
    private Tensor<T>? _gpuInput;

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
    private Tensor<T>? _gpuWeightsFi;
    private Tensor<T>? _gpuWeightsIi;
    private Tensor<T>? _gpuWeightsCi;
    private Tensor<T>? _gpuWeightsOi;
    private Tensor<T>? _gpuWeightsFh;
    private Tensor<T>? _gpuWeightsIh;
    private Tensor<T>? _gpuWeightsCh;
    private Tensor<T>? _gpuWeightsOh;
    private Tensor<T>? _gpuBiasF;
    private Tensor<T>? _gpuBiasI;
    private Tensor<T>? _gpuBiasC;
    private Tensor<T>? _gpuBiasO;

    // GPU gradient tensors from BackwardGpu
    private Tensor<T>? _gpuWeightsFiGradient;
    private Tensor<T>? _gpuWeightsIiGradient;
    private Tensor<T>? _gpuWeightsCiGradient;
    private Tensor<T>? _gpuWeightsOiGradient;
    private Tensor<T>? _gpuWeightsFhGradient;
    private Tensor<T>? _gpuWeightsIhGradient;
    private Tensor<T>? _gpuWeightsChGradient;
    private Tensor<T>? _gpuWeightsOhGradient;
    private Tensor<T>? _gpuBiasFGradient;
    private Tensor<T>? _gpuBiasIGradient;
    private Tensor<T>? _gpuBiasCGradient;
    private Tensor<T>? _gpuBiasOGradient;

    // Optimizer state tensors for SGD/NAG/LARS (velocity)
    private Tensor<T>? _gpuWeightsFiVelocity;
    private Tensor<T>? _gpuWeightsIiVelocity;
    private Tensor<T>? _gpuWeightsCiVelocity;
    private Tensor<T>? _gpuWeightsOiVelocity;
    private Tensor<T>? _gpuWeightsFhVelocity;
    private Tensor<T>? _gpuWeightsIhVelocity;
    private Tensor<T>? _gpuWeightsChVelocity;
    private Tensor<T>? _gpuWeightsOhVelocity;
    private Tensor<T>? _gpuBiasFVelocity;
    private Tensor<T>? _gpuBiasIVelocity;
    private Tensor<T>? _gpuBiasCVelocity;
    private Tensor<T>? _gpuBiasOVelocity;

    // Optimizer state tensors for Adam/AdamW/LAMB (M and V)
    private Tensor<T>? _gpuWeightsFiM;
    private Tensor<T>? _gpuWeightsFiV;
    private Tensor<T>? _gpuWeightsIiM;
    private Tensor<T>? _gpuWeightsIiV;
    private Tensor<T>? _gpuWeightsCiM;
    private Tensor<T>? _gpuWeightsCiV;
    private Tensor<T>? _gpuWeightsOiM;
    private Tensor<T>? _gpuWeightsOiV;
    private Tensor<T>? _gpuWeightsFhM;
    private Tensor<T>? _gpuWeightsFhV;
    private Tensor<T>? _gpuWeightsIhM;
    private Tensor<T>? _gpuWeightsIhV;
    private Tensor<T>? _gpuWeightsChM;
    private Tensor<T>? _gpuWeightsChV;
    private Tensor<T>? _gpuWeightsOhM;
    private Tensor<T>? _gpuWeightsOhV;
    private Tensor<T>? _gpuBiasFM;
    private Tensor<T>? _gpuBiasFV;
    private Tensor<T>? _gpuBiasIM;
    private Tensor<T>? _gpuBiasIV;
    private Tensor<T>? _gpuBiasCM;
    private Tensor<T>? _gpuBiasCV;
    private Tensor<T>? _gpuBiasOM;
    private Tensor<T>? _gpuBiasOV;

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
    public override int ParameterCount => _weightsFi.Length + _weightsIi.Length + _weightsCi.Length + _weightsOi.Length + _weightsFh.Length + _weightsIh.Length + _weightsCh.Length + _weightsOh.Length + _biasF.Length + _biasI.Length + _biasC.Length + _biasO.Length;
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
        InitializeLayerBiases(_biasF);
        InitializeLayerBiases(_biasI);
        InitializeLayerBiases(_biasC);
        InitializeLayerBiases(_biasO);

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
        InitializeLayerBiases(_biasF);
        InitializeLayerBiases(_biasI);
        InitializeLayerBiases(_biasC);
        InitializeLayerBiases(_biasO);

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
        double scale = Math.Sqrt(2.0 / (weights.Shape[0] * weights.Shape[1] * weights.Shape[2]));
        var span = weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble((Random.NextDouble() - 0.5) * scale);
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
    private void InitializeBiasesToZero(Tensor<T> biases)
    {
        InitializeLayerBiases(biases);
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
        _originalInputShape = input.Shape.ToArray();
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

        // Rent output (fully overwritten), states need zero init for initial timestep
        var output = TensorAllocator.Rent<T>([batchSize, timeSteps, height, width, _filters]);
        _lastHiddenState = TensorAllocator.Rent<T>([batchSize, height, width, _filters]);
        _lastCellState = TensorAllocator.Rent<T>([batchSize, height, width, _filters]);

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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
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
        var shape = input.Shape.ToArray();
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

            return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
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
        var inputNCHW = input.Transpose(new[] { 0, 3, 1, 2 }).Contiguous();

        // Transpose kernel from [kH, kW, inC, outC] to [outC, inC, kH, kW] for Engine
        var kernelNCHW = kernel.Transpose(new[] { 3, 2, 0, 1 }).Contiguous();

        var stride = new int[] { _strides, _strides };
        var padding = new int[] { _padding, _padding };
        var dilation = new int[] { 1, 1 };

        // Use GPU-accelerated Conv2D
        var outputNCHW = Engine.Conv2D(inputNCHW, kernelNCHW, stride, padding, dilation);

        // Transpose output back to [B, H, W, outC]
        return outputNCHW.Transpose(new[] { 0, 2, 3, 1 }).Contiguous();
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
            var vecActivation = VectorActivation ?? throw new InvalidOperationException("VectorActivation has not been initialized.");
            return vecActivation.Derivative(input);
        }
        else
        {
            var scalarActivation = ScalarActivation ?? throw new InvalidOperationException("ScalarActivation has not been initialized.");
            return input.Transform((x, _) => scalarActivation.Derivative(x));
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

        // Gate gradients with proper derivatives:
        // h = o * tanh(c), so dL/do = dL/dh * tanh(c)
        var do_raw = Engine.TensorMultiply(dh, ApplyActivation(newC));
        // dL/dc from output: dL/dh * o * tanh'(c) + dc (from next timestep)
        var dNewC = Engine.TensorAdd(Engine.TensorMultiply(Engine.TensorMultiply(dh, o), ApplyActivationDerivative(newC)), dc);
        // c = f*prevC + i*candidate, so gradients flow to gates
        var df_raw = Engine.TensorMultiply(dNewC, prevC);
        var di_raw = Engine.TensorMultiply(dNewC, c);
        var dc_ = Engine.TensorMultiply(dNewC, i); // candidate gate needs tanh'
        var dprevC = Engine.TensorMultiply(dNewC, f);

        // Apply gate activation derivatives:
        // f, i, o use sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // candidate uses tanh: tanh'(x) = 1 - tanh(x)^2
        var ones = new Tensor<T>(f.Shape.ToArray());
        ones.Fill(NumOps.One);

        var df = Engine.TensorMultiply(df_raw, Engine.TensorMultiply(f, Engine.TensorSubtract(ones, f)));
        var di = Engine.TensorMultiply(di_raw, Engine.TensorMultiply(i, Engine.TensorSubtract(ones, i)));
        var do_ = Engine.TensorMultiply(do_raw, Engine.TensorMultiply(o, Engine.TensorSubtract(ones, o)));
        dc_ = Engine.TensorMultiply(dc_, Engine.TensorSubtract(ones, Engine.TensorMultiply(c, c))); // tanh'

        // Convert NHWC to NCHW for engine conv backward operations
        var stride = new int[] { _strides, _strides };
        var padding = new int[] { _padding, _padding };
        var dilation = new int[] { 1, 1 };

        var xtNCHW = xt.Transpose([0, 3, 1, 2]);
        var prevHNCHW = prevH.Transpose([0, 3, 1, 2]);

        // Gate gradients NHWC → NCHW
        var dfNCHW = df.Transpose([0, 3, 1, 2]);
        var diNCHW = di.Transpose([0, 3, 1, 2]);
        var dcNCHW = dc_.Transpose([0, 3, 1, 2]);
        var doNCHW = do_.Transpose([0, 3, 1, 2]);

        // Weight gradients: dW = Conv2DBackwardKernel(gradient, input, kernelShape, stride, padding, dilation)
        var kernelShape = _weightsFi.Transpose([3, 2, 0, 1]).Shape.ToArray(); // NHWC kernel → NCHW kernel shape
        var dWfi = Engine.Conv2DBackwardKernel(dfNCHW, xtNCHW, kernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWii = Engine.Conv2DBackwardKernel(diNCHW, xtNCHW, kernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWci = Engine.Conv2DBackwardKernel(dcNCHW, xtNCHW, kernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWoi = Engine.Conv2DBackwardKernel(doNCHW, xtNCHW, kernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);

        var hKernelShape = _weightsFh.Transpose([3, 2, 0, 1]).Shape.ToArray();
        var dWfh = Engine.Conv2DBackwardKernel(dfNCHW, prevHNCHW, hKernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWih = Engine.Conv2DBackwardKernel(diNCHW, prevHNCHW, hKernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWch = Engine.Conv2DBackwardKernel(dcNCHW, prevHNCHW, hKernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);
        var dWoh = Engine.Conv2DBackwardKernel(doNCHW, prevHNCHW, hKernelShape, stride, padding, dilation).Transpose([2, 3, 1, 0]);

        // Bias gradients: sum over batch, height, width (NHWC → sum dims 0,1,2)
        var dbf = df.Sum([0, 1, 2]).Reshape(_biasF.Shape.ToArray());
        var dbi = di.Sum([0, 1, 2]).Reshape(_biasI.Shape.ToArray());
        var dbc = dc_.Sum([0, 1, 2]).Reshape(_biasC.Shape.ToArray());
        var dbo = do_.Sum([0, 1, 2]).Reshape(_biasO.Shape.ToArray());

        // Input gradient: dxt = Conv2DBackwardInput(gradient, kernel, inputShape, stride, padding, dilation)
        var inputNCHWShape = xtNCHW.Shape.ToArray();
        var wFiNCHW = _weightsFi.Transpose([3, 2, 0, 1]);
        var wIiNCHW = _weightsIi.Transpose([3, 2, 0, 1]);
        var wCiNCHW = _weightsCi.Transpose([3, 2, 0, 1]);
        var wOiNCHW = _weightsOi.Transpose([3, 2, 0, 1]);
        var dxtNCHW = Engine.Conv2DBackwardInput(dfNCHW, wFiNCHW, inputNCHWShape, stride, padding, dilation)
            .Add(Engine.Conv2DBackwardInput(diNCHW, wIiNCHW, inputNCHWShape, stride, padding, dilation))
            .Add(Engine.Conv2DBackwardInput(dcNCHW, wCiNCHW, inputNCHWShape, stride, padding, dilation))
            .Add(Engine.Conv2DBackwardInput(doNCHW, wOiNCHW, inputNCHWShape, stride, padding, dilation));
        var dxt = dxtNCHW.Transpose([0, 2, 3, 1]); // NCHW → NHWC

        var prevHNCHWShape = prevHNCHW.Shape.ToArray();
        var wFhNCHW = _weightsFh.Transpose([3, 2, 0, 1]);
        var wIhNCHW = _weightsIh.Transpose([3, 2, 0, 1]);
        var wChNCHW = _weightsCh.Transpose([3, 2, 0, 1]);
        var wOhNCHW = _weightsOh.Transpose([3, 2, 0, 1]);
        var dprevHNCHW = Engine.Conv2DBackwardInput(dfNCHW, wFhNCHW, prevHNCHWShape, stride, padding, dilation)
            .Add(Engine.Conv2DBackwardInput(diNCHW, wIhNCHW, prevHNCHWShape, stride, padding, dilation))
            .Add(Engine.Conv2DBackwardInput(dcNCHW, wChNCHW, prevHNCHWShape, stride, padding, dilation))
            .Add(Engine.Conv2DBackwardInput(doNCHW, wOhNCHW, prevHNCHWShape, stride, padding, dilation));
        var dprevH = dprevHNCHW.Transpose([0, 2, 3, 1]); // NCHW → NHWC

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
        // Bias is [1,1,1,filters] — use BroadcastAdd since Tensor.Add doesn't broadcast
        var f = Engine.Sigmoid(Engine.TensorBroadcastAdd(Convolve(xt, _weightsFi).Add(Convolve(prevH, _weightsFh)), _biasF));
        var i = Engine.Sigmoid(Engine.TensorBroadcastAdd(Convolve(xt, _weightsIi).Add(Convolve(prevH, _weightsIh)), _biasI));
        var c = ApplyActivation(Engine.TensorBroadcastAdd(Convolve(xt, _weightsCi).Add(Convolve(prevH, _weightsCh)), _biasC));
        var o = Engine.Sigmoid(Engine.TensorBroadcastAdd(Convolve(xt, _weightsOi).Add(Convolve(prevH, _weightsOh)), _biasO));

        var newC = Engine.TensorAdd(Engine.TensorMultiply(f, prevC), Engine.TensorMultiply(i, c));
        var newH = Engine.TensorMultiply(o, ApplyActivation(newC));

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
            momentum = new Tensor<T>(parameter.Shape.ToArray());
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

        // Input weights (I, C, O gates first — F gate has zero gradient for seqLen=1)
        totalParams += _weightsIi.Length;
        totalParams += _weightsCi.Length;
        totalParams += _weightsOi.Length;
        totalParams += _weightsFi.Length;

        // Hidden weights
        totalParams += _weightsIh.Length;
        totalParams += _weightsCh.Length;
        totalParams += _weightsOh.Length;
        totalParams += _weightsFh.Length;

        // Biases
        totalParams += _biasI.Length;
        totalParams += _biasC.Length;
        totalParams += _biasO.Length;
        totalParams += _biasF.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        CopyTensorToVector(_weightsIi, parameters, ref index);
        CopyTensorToVector(_weightsCi, parameters, ref index);
        CopyTensorToVector(_weightsOi, parameters, ref index);
        CopyTensorToVector(_weightsFi, parameters, ref index);

        CopyTensorToVector(_weightsIh, parameters, ref index);
        CopyTensorToVector(_weightsCh, parameters, ref index);
        CopyTensorToVector(_weightsOh, parameters, ref index);
        CopyTensorToVector(_weightsFh, parameters, ref index);

        CopyTensorToVector(_biasI, parameters, ref index);
        CopyTensorToVector(_biasC, parameters, ref index);
        CopyTensorToVector(_biasO, parameters, ref index);
        CopyTensorToVector(_biasF, parameters, ref index);

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
    public override Vector<T> GetParameterGradients()
    {
        if (_gradients == null || _gradients.Count == 0)
            return new Vector<T>(ParameterCount);

        T[] GetGrad(string key, int length)
        {
            if (_gradients.TryGetValue(key, out var obj) && obj is Tensor<T> t)
                return t.ToArray();
            return new T[length];
        }

        // Order: I, C, O, F (matching GetParameters — F last since its gradient is zero for seqLen=1)
        var result = new List<T>();
        result.AddRange(GetGrad("weightsIi", _weightsIi.Length));
        result.AddRange(GetGrad("weightsCi", _weightsCi.Length));
        result.AddRange(GetGrad("weightsOi", _weightsOi.Length));
        result.AddRange(GetGrad("weightsFi", _weightsFi.Length));
        result.AddRange(GetGrad("weightsIh", _weightsIh.Length));
        result.AddRange(GetGrad("weightsCh", _weightsCh.Length));
        result.AddRange(GetGrad("weightsOh", _weightsOh.Length));
        result.AddRange(GetGrad("weightsFh", _weightsFh.Length));
        result.AddRange(GetGrad("biasI", _biasI.Length));
        result.AddRange(GetGrad("biasC", _biasC.Length));
        result.AddRange(GetGrad("biasO", _biasO.Length));
        result.AddRange(GetGrad("biasF", _biasF.Length));

        return new Vector<T>(result.ToArray());
    }

    public override void ClearGradients()
    {
        _gradients?.Clear();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        // Set input weights (I, C, O, F order matching GetParameters)
        CopyVectorToTensor(parameters, _weightsIi, ref index);
        CopyVectorToTensor(parameters, _weightsCi, ref index);
        CopyVectorToTensor(parameters, _weightsOi, ref index);
        CopyVectorToTensor(parameters, _weightsFi, ref index);

        // Set hidden weights
        CopyVectorToTensor(parameters, _weightsIh, ref index);
        CopyVectorToTensor(parameters, _weightsCh, ref index);
        CopyVectorToTensor(parameters, _weightsOh, ref index);
        CopyVectorToTensor(parameters, _weightsFh, ref index);

        // Set biases
        CopyVectorToTensor(parameters, _biasI, ref index);
        CopyVectorToTensor(parameters, _biasC, ref index);
        CopyVectorToTensor(parameters, _biasO, ref index);
        CopyVectorToTensor(parameters, _biasF, ref index);

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
        _gpuWeightsFi ??= GpuTensorHelper.UploadToGpu<T>(backend, wFiNCHW, GpuTensorRole.Weight);
        _gpuWeightsIi ??= GpuTensorHelper.UploadToGpu<T>(backend, wIiNCHW, GpuTensorRole.Weight);
        _gpuWeightsCi ??= GpuTensorHelper.UploadToGpu<T>(backend, wCiNCHW, GpuTensorRole.Weight);
        _gpuWeightsOi ??= GpuTensorHelper.UploadToGpu<T>(backend, wOiNCHW, GpuTensorRole.Weight);
        _gpuWeightsFh ??= GpuTensorHelper.UploadToGpu<T>(backend, wFhNCHW, GpuTensorRole.Weight);
        _gpuWeightsIh ??= GpuTensorHelper.UploadToGpu<T>(backend, wIhNCHW, GpuTensorRole.Weight);
        _gpuWeightsCh ??= GpuTensorHelper.UploadToGpu<T>(backend, wChNCHW, GpuTensorRole.Weight);
        _gpuWeightsOh ??= GpuTensorHelper.UploadToGpu<T>(backend, wOhNCHW, GpuTensorRole.Weight);
        _gpuBiasF ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasF, GpuTensorRole.Bias);
        _gpuBiasI ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasI, GpuTensorRole.Bias);
        _gpuBiasC ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasC, GpuTensorRole.Bias);
        _gpuBiasO ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasO, GpuTensorRole.Bias);

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
        _weightsFi = _gpuWeightsFi.Transpose([2, 3, 1, 0]);
        _weightsIi = _gpuWeightsIi.Transpose([2, 3, 1, 0]);
        _weightsCi = _gpuWeightsCi.Transpose([2, 3, 1, 0]);
        _weightsOi = _gpuWeightsOi.Transpose([2, 3, 1, 0]);
        _weightsFh = _gpuWeightsFh.Transpose([2, 3, 1, 0]);
        _weightsIh = _gpuWeightsIh.Transpose([2, 3, 1, 0]);
        _weightsCh = _gpuWeightsCh.Transpose([2, 3, 1, 0]);
        _weightsOh = _gpuWeightsOh.Transpose([2, 3, 1, 0]);
        _biasF = _gpuBiasF;
        _biasI = _gpuBiasI;
        _biasC = _gpuBiasC;
        _biasO = _gpuBiasO;
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
                _gpuWeightsFiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightsFiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // SquaredAvg buffers for RMSprop/Adagrad
                _gpuWeightsFiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
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
