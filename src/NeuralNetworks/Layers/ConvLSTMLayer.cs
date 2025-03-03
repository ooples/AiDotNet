namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Convolutional Long Short-Term Memory (ConvLSTM) layer for processing sequential spatial data.
/// </summary>
/// <remarks>
/// ConvLSTM combines convolutional operations with LSTM (Long Short-Term Memory) to handle
/// spatial-temporal data like video sequences or weather prediction maps. Unlike traditional
/// LSTMs that process vector sequences, ConvLSTMs maintain spatial information throughout
/// the processing.
/// 
/// This layer is particularly useful for:
/// - Video prediction and analysis
/// - Weather forecasting
/// - Any data with both spatial patterns and temporal dependencies
/// 
/// The layer processes input tensors of shape [batch, time, height, width, channels]
/// and outputs tensors of the same time dimension but with the specified number of filters.
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
    private Tensor<T>? _lastHiddenState;
    private Tensor<T>? _lastCellState;
    private Dictionary<string, object> _gradients = [];
    private readonly Dictionary<string, Tensor<T>> _momentums = [];
    private const double MomentumFactor = 0.9;

    private readonly SigmoidActivation<T> _sigmoidActivation = new();

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// Returns true because ConvLSTM layers have trainable parameters (weights and biases)
    /// that need to be updated during the training process.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the ConvLSTMLayer class with a scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [batch, time, height, width, channels].</param>
    /// <param name="kernelSize">The size of the convolutional kernel (filter). A larger kernel captures more spatial context.</param>
    /// <param name="filters">The number of output filters (channels) for the layer. More filters can capture more features.</param>
    /// <param name="padding">The padding added to the input. Helps maintain spatial dimensions.</param>
    /// <param name="strides">The stride of the convolution. Controls how the filter moves across the input.</param>
    /// <param name="activationFunction">The activation function to use. Defaults to tanh if not specified.</param>
    /// <remarks>
    /// The kernel size determines how much spatial context is considered in each convolution.
    /// The number of filters determines how many different features the layer can detect.
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
    /// <param name="vectorActivationFunction">The vector activation function to use. Defaults to tanh if not specified.</param>
    /// <remarks>
    /// This constructor allows using a vector activation function that can process entire tensors at once,
    /// which may be more efficient for certain operations.
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
    private static int[] CalculateOutputShape(int[] inputShape, int kernelSize, int filters, int padding, int strides)
    {
        int outputHeight = (inputShape[1] - kernelSize + 2 * padding) / strides + 1;
        int outputWidth = (inputShape[2] - kernelSize + 2 * padding) / strides + 1;
        return [inputShape[0], outputHeight, outputWidth, filters];
    }

    /// <summary>
    /// Initializes the weights of the layer with small random values.
    /// </summary>
    /// <param name="weights">The weights tensor to initialize.</param>
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
    /// This method sets all values in the bias tensor to zero, which is a common
    /// initialization strategy for biases in neural networks.
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
    /// <param name="input">Input tensor with shape [batchSize, timeSteps, height, width, channels].</param>
    /// <returns>Output tensor with shape [batchSize, timeSteps, height, width, filters].</returns>
    /// <remarks>
    /// This method processes a sequence of spatial inputs through the ConvLSTM layer:
    /// 1. Stores the input for use in the backward pass
    /// 2. Extracts dimensions from the input tensor
    /// 3. Creates output and state tensors
    /// 4. Processes each time step sequentially through the ConvLSTM cell
    /// 5. Returns the sequence of hidden states as output
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int timeSteps = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, timeSteps, height, width, _filters]);
        _lastHiddenState = new Tensor<T>([batchSize, height, width, _filters]);
        _lastCellState = new Tensor<T>([batchSize, height, width, _filters]);

        for (int t = 0; t < timeSteps; t++)
        {
            var xt = input.GetSlice(t);
            (_lastHiddenState, _lastCellState) = ConvLSTMCell(xt, _lastHiddenState, _lastCellState);
            output.SetSlice(t, _lastHiddenState);
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
    /// This method implements the core ConvLSTM cell operations:
    /// 1. Forget gate: Controls what information to discard from the cell state
    /// 2. Input gate: Controls what new information to store in the cell state
    /// 3. Candidate cell: Creates new candidate values to add to the cell state
    /// 4. Output gate: Controls what parts of the cell state to output
    /// 5. Updates cell state and hidden state based on these gates
    /// 
    /// The ConvLSTM cell uses convolution operations instead of matrix multiplications
    /// to capture spatial dependencies in the data.
    /// </remarks>
    private (Tensor<T> hiddenState, Tensor<T> cellState) ConvLSTMCell(Tensor<T> input, Tensor<T> prevHiddenState, Tensor<T> prevCellState)
    {
        var forgetGate = Convolve(input, _weightsFi).Add(Convolve(prevHiddenState, _weightsFh)).Add(_biasF).Transform((x, _) => _sigmoidActivation.Activate(x));
        var inputGate = Convolve(input, _weightsIi).Add(Convolve(prevHiddenState, _weightsIh)).Add(_biasI).Transform((x, _) => _sigmoidActivation.Activate(x));
        var candidateCell = ApplyActivation(Convolve(input, _weightsCi).Add(Convolve(prevHiddenState, _weightsCh)).Add(_biasC));
        var outputGate = Convolve(input, _weightsOi).Add(Convolve(prevHiddenState, _weightsOh)).Add(_biasO).Transform((x, _) => _sigmoidActivation.Activate(x));

        var newCellState = forgetGate.Multiply(prevCellState).Add(inputGate.Multiply(candidateCell));
        var newHiddenState = outputGate.Multiply(ApplyActivation(newCellState));

        return (newHiddenState, newCellState);
    }

    /// <summary>
    /// Performs a 2D convolution operation between an input tensor and a kernel.
    /// </summary>
    /// <param name="input">Input tensor with shape [batchSize, height, width, channels].</param>
    /// <param name="kernel">Kernel tensor with shape [kernelHeight, kernelWidth, inputChannels, outputChannels].</param>
    /// <returns>Output tensor with shape [batchSize, outputHeight, outputWidth, outputChannels].</returns>
    /// <remarks>
    /// This method implements a basic 2D convolution operation:
    /// 1. Calculates output dimensions based on input size, kernel size, padding, and stride
    /// 2. For each position in the output, computes the dot product between the kernel and the corresponding input region
    /// 3. Handles padding by skipping positions outside the input boundaries
    /// 
    /// The convolution operation is fundamental to ConvLSTM as it allows the network to capture
    /// spatial patterns in the data.
    /// </remarks>
    private Tensor<T> Convolve(Tensor<T> input, Tensor<T> kernel)
    {
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int inputChannels = input.Shape[3];
        int kernelHeight = kernel.Shape[0];
        int kernelWidth = kernel.Shape[1];
        int outputChannels = kernel.Shape[3];

        int outputHeight = (inputHeight - kernelHeight + 2 * _padding) / _strides + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * _padding) / _strides + 1;

        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, outputChannels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        T sum = NumOps.Zero;

                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                for (int ic = 0; ic < inputChannels; ic++)
                                {
                                    int ih = oh * _strides + kh - _padding;
                                    int iw = ow * _strides + kw - _padding;

                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        T inputVal = input[b, ih, iw, ic];
                                        T kernelVal = kernel[kh, kw, ic, oc];
                                        sum = NumOps.Add(sum, NumOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }

                        output[b, oh, ow, oc] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the ConvLSTM layer, computing gradients for all parameters.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer with shape [batchSize, timeSteps, height, width, filters]</param>
    /// <returns>Gradient with respect to the input with shape [batchSize, timeSteps, height, width, channels]</returns>
    /// <remarks>
    /// This method implements backpropagation through time (BPTT) for the ConvLSTM layer:
    /// 1. Initializes gradient tensors for all parameters
    /// 2. Iterates backward through time steps
    /// 3. Computes gradients for each time step using BackwardStep
    /// 4. Accumulates gradients across all time steps
    /// 5. Stores gradients for later use in parameter updates
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
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
            var currentDh = outputGradient.GetSlice(t).Add(dNextH);
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

        return dInput;
    }

    /// <summary>
    /// Applies the derivative of the activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply the activation derivative to</param>
    /// <returns>A tensor with the activation derivative applied element-wise</returns>
    /// <remarks>
    /// This method handles both vector and scalar activation functions based on the layer configuration.
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
    /// This method implements the backpropagation through a single ConvLSTM cell:
    /// 1. First computes the forward pass to get intermediate values
    /// 2. Computes gradients for each gate (forget, input, cell, output)
    /// 3. Computes gradients for all weights and biases
    /// 4. Returns gradients needed for continued backpropagation
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
    /// <para>
    /// Each gate uses convolutional operations instead of matrix multiplications used in standard LSTM,
    /// allowing the layer to maintain spatial information throughout the processing.
    /// </para>
    /// </remarks>
    private (Tensor<T> f, Tensor<T> i, Tensor<T> c, Tensor<T> o, Tensor<T> newC, Tensor<T> newH) ForwardStep(
            Tensor<T> xt, Tensor<T> prevH, Tensor<T> prevC)
    {
        var f = Convolve(xt, _weightsFi).Add(Convolve(prevH, _weightsFh)).Add(_biasF).Transform((x, _) => _sigmoidActivation.Activate(x));
        var i = Convolve(xt, _weightsIi).Add(Convolve(prevH, _weightsIh)).Add(_biasI).Transform((x, _) => _sigmoidActivation.Activate(x));
        var c = ApplyActivation(Convolve(xt, _weightsCi).Add(Convolve(prevH, _weightsCh)).Add(_biasC));
        var o = Convolve(xt, _weightsOi).Add(Convolve(prevH, _weightsOh)).Add(_biasO).Transform((x, _) => _sigmoidActivation.Activate(x));

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
    /// <para>
    /// The naming convention follows the pattern:
    /// - First letter 'd' indicates it's a derivative/gradient
    /// - Second letter 'W' (weight) or 'b' (bias)
    /// - Third letter indicates the gate (f: forget, i: input, c: cell, o: output)
    /// - Fourth letter indicates input source (i: input, h: hidden state)
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
    /// <para>
    /// Each parameter is updated using the UpdateParameterWithMomentum helper method,
    /// which applies both the gradient and accumulated momentum.
    /// </para>
    /// <exception cref="InvalidOperationException">Thrown when gradients are not available (Backward method wasn't called)</exception>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_gradients.Count == 0)
        {
            throw new InvalidOperationException("No gradients available. Ensure backward pass is called before updating parameters.");
        }

        UpdateParameterWithMomentum(_weightsFi, "weightsFi", learningRate);
        UpdateParameterWithMomentum(_weightsIi, "weightsIi", learningRate);
        UpdateParameterWithMomentum(_weightsCi, "weightsCi", learningRate);
        UpdateParameterWithMomentum(_weightsOi, "weightsOi", learningRate);
        UpdateParameterWithMomentum(_weightsFh, "weightsFh", learningRate);
        UpdateParameterWithMomentum(_weightsIh, "weightsIh", learningRate);
        UpdateParameterWithMomentum(_weightsCh, "weightsCh", learningRate);
        UpdateParameterWithMomentum(_weightsOh, "weightsOh", learningRate);

        UpdateParameterWithMomentum(_biasF, "biasF", learningRate);
        UpdateParameterWithMomentum(_biasI, "biasI", learningRate);
        UpdateParameterWithMomentum(_biasC, "biasC", learningRate);
        UpdateParameterWithMomentum(_biasO, "biasO", learningRate);

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
    /// <para>
    /// Momentum helps the optimization process navigate flat regions and small local minima
    /// by accumulating velocity in consistent directions.
    /// </para>
    /// <exception cref="InvalidOperationException">Thrown when the gradient for the parameter is not found</exception>
    /// </remarks>
    private void UpdateParameterWithMomentum(Tensor<T> parameter, string paramName, T learningRate)
    {
        if (!_gradients.TryGetValue(paramName, out var gradientObj) || gradientObj is not Tensor<T> gradient)
        {
            throw new InvalidOperationException($"Gradient for {paramName} not found or invalid.");
        }

        if (!_momentums.TryGetValue(paramName, out var momentum))
        {
            momentum = new Tensor<T>(parameter.Shape);
            _momentums[paramName] = momentum;
        }

        for (int i = 0; i < parameter.Length; i++)
        {
            // Update momentum
            momentum[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(MomentumFactor), momentum[i]),
                NumOps.Multiply(learningRate, gradient[i])
            );

            // Update parameter
            parameter[i] = NumOps.Subtract(parameter[i], momentum[i]);
        }
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
    /// <para>
    /// The returned vector can be used for optimization algorithms that operate on all parameters
    /// at once, or for saving/loading model parameters.
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
    /// This method iterates through all elements in the tensor and copies them sequentially
    /// to the vector starting at the specified index. The startIndex parameter is updated
    /// to point to the next available position in the vector after copying.
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
    /// <para>
    /// The parameter vector must contain exactly the right number of elements in the correct order,
    /// matching the format produced by GetParameters().
    /// </para>
    /// <para>
    /// This method is useful for loading saved model parameters or applying updates from
    /// optimization algorithms that operate on all parameters at once.
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
    /// This method iterates through all elements in the tensor and sets them sequentially
    /// from the vector starting at the specified index. The startIndex parameter is updated
    /// to point to the next position in the vector after copying.
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
    /// <para>
    /// Resetting the state is important when starting a new sequence or when the layer
    /// needs to forget previous computations, such as at the beginning of a new training epoch
    /// or when processing a new batch of sequences.
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
}