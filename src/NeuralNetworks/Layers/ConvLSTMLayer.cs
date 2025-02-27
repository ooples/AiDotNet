namespace AiDotNet.NeuralNetworks.Layers;

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

    private static int[] CalculateOutputShape(int[] inputShape, int kernelSize, int filters, int padding, int strides)
    {
        int outputHeight = (inputShape[1] - kernelSize + 2 * padding) / strides + 1;
        int outputWidth = (inputShape[2] - kernelSize + 2 * padding) / strides + 1;
        return [inputShape[0], outputHeight, outputWidth, filters];
    }

    private void InitializeWeights(Tensor<T> weights)
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (weights.Shape[0] * weights.Shape[1] * weights.Shape[2])));
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    private void InitializeBiases(Tensor<T> biases)
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = NumOps.Zero;
        }
    }

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
            ["dWeightsFi"] = dWeightsFi,
            ["dWeightsIi"] = dWeightsIi,
            ["dWeightsCi"] = dWeightsCi,
            ["dWeightsOi"] = dWeightsOi,
            ["dWeightsFh"] = dWeightsFh,
            ["dWeightsIh"] = dWeightsIh,
            ["dWeightsCh"] = dWeightsCh,
            ["dWeightsOh"] = dWeightsOh,
            ["dBiasF"] = dBiasF,
            ["dBiasI"] = dBiasI,
            ["dBiasC"] = dBiasC,
            ["dBiasO"] = dBiasO
        };

        return dInput;
    }

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

    private readonly SigmoidActivation<T> _sigmoidActivation = new SigmoidActivation<T>();

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

    private struct CellGradients
    {
        public Tensor<T> dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo;

        public CellGradients(Tensor<T> dWfi, Tensor<T> dWii, Tensor<T> dWci, Tensor<T> dWoi,
            Tensor<T> dWfh, Tensor<T> dWih, Tensor<T> dWch, Tensor<T> dWoh,
            Tensor<T> dbf, Tensor<T> dbi, Tensor<T> dbc, Tensor<T> dbo)
        {
            this.dWfi = dWfi; this.dWii = dWii; this.dWci = dWci; this.dWoi = dWoi;
            this.dWfh = dWfh; this.dWih = dWih; this.dWch = dWch; this.dWoh = dWoh;
            this.dbf = dbf; this.dbi = dbi; this.dbc = dbc; this.dbo = dbo;
        }
    }

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
}