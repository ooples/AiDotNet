namespace AiDotNet.NeuralNetworks.Layers;

public class LSTMLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;

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

    private readonly IActivationFunction<T>? _sigmoidActivation;
    private readonly IActivationFunction<T>? _tanhActivation;

    private readonly IVectorActivationFunction<T>? _sigmoidVectorActivation;
    private readonly IVectorActivationFunction<T>? _tanhVectorActivation;

    private readonly bool _useVectorActivation;

    public Dictionary<string, Tensor<T>> Gradients { get; private set; }

    public override bool SupportsTraining => true;

    public LSTMLayer(int inputSize, int hiddenSize, 
    IActivationFunction<T>? activation = null, 
    IActivationFunction<T>? recurrentActivation = null)
    : base([inputSize], [hiddenSize], activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        _useVectorActivation = false;

        _sigmoidActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhActivation = activation ?? new TanhActivation<T>();

        Gradients = [];

        // Initialize weights
        _weightsFi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsIi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsCi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsOi = new Tensor<T>([_hiddenSize, _inputSize]);

        _weightsFh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsIh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsCh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsOh = new Tensor<T>([_hiddenSize, _hiddenSize]);

        // Initialize biases
        _biasF = new Tensor<T>([_hiddenSize]);
        _biasI = new Tensor<T>([_hiddenSize]);
        _biasC = new Tensor<T>([_hiddenSize]);
        _biasO = new Tensor<T>([_hiddenSize]);

        InitializeWeights();
    }

    public LSTMLayer(int inputSize, int hiddenSize, 
    IVectorActivationFunction<T>? activation = null, 
    IVectorActivationFunction<T>? recurrentActivation = null)
    : base([inputSize], [hiddenSize], activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        _useVectorActivation = false;

        _sigmoidVectorActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhVectorActivation = activation ?? new TanhActivation<T>();

        Gradients = [];

        // Initialize weights
        _weightsFi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsIi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsCi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsOi = new Tensor<T>([_hiddenSize, _inputSize]);

        _weightsFh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsIh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsCh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsOh = new Tensor<T>([_hiddenSize, _hiddenSize]);

        // Initialize biases
        _biasF = new Tensor<T>([_hiddenSize]);
        _biasI = new Tensor<T>([_hiddenSize]);
        _biasC = new Tensor<T>([_hiddenSize]);
        _biasO = new Tensor<T>([_hiddenSize]);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier/Glorot initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputSize + _hiddenSize)));

        InitializeWeight(_weightsFi, scale);
        InitializeWeight(_weightsIi, scale);
        InitializeWeight(_weightsCi, scale);
        InitializeWeight(_weightsOi, scale);
        InitializeWeight(_weightsFh, scale);
        InitializeWeight(_weightsIh, scale);
        InitializeWeight(_weightsCh, scale);
        InitializeWeight(_weightsOh, scale);

        InitializeBias(_biasF);
        InitializeBias(_biasI);
        InitializeBias(_biasC);
        InitializeBias(_biasO);
    }

    private void InitializeWeight(Tensor<T> weight, T scale)
    {
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    private void InitializeBias(Tensor<T> bias)
    {
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int timeSteps = input.Shape[1];

        var output = new Tensor<T>([batchSize, timeSteps, _hiddenSize]);
        _lastHiddenState = new Tensor<T>([batchSize, _hiddenSize]);
        _lastCellState = new Tensor<T>([batchSize, _hiddenSize]);

        for (int t = 0; t < timeSteps; t++)
        {
            var xt = input.GetSlice(t);
            (_lastHiddenState, _lastCellState) = LSTMCell(xt, _lastHiddenState, _lastCellState);
            output.SetSlice(t, _lastHiddenState);
        }

        return output;
    }

    private (Tensor<T> hiddenState, Tensor<T> cellState) LSTMCell(Tensor<T> xt, Tensor<T> prevH, Tensor<T> prevC)
    {
        var f = xt.MatrixMultiply(_weightsFi).Add(prevH.MatrixMultiply(_weightsFh)).Add(_biasF);
        var i = xt.MatrixMultiply(_weightsIi).Add(prevH.MatrixMultiply(_weightsIh)).Add(_biasI);
        var c = xt.MatrixMultiply(_weightsCi).Add(prevH.MatrixMultiply(_weightsCh)).Add(_biasC);
        var o = xt.MatrixMultiply(_weightsOi).Add(prevH.MatrixMultiply(_weightsOh)).Add(_biasO);

        if (_useVectorActivation)
        {
            f = _sigmoidVectorActivation!.Activate(f);
            i = _sigmoidVectorActivation!.Activate(i);
            c = _tanhVectorActivation!.Activate(c);
            o = _sigmoidVectorActivation!.Activate(o);
        }
        else
        {
            f = f.Transform((x, _) => _sigmoidActivation!.Activate(x));
            i = i.Transform((x, _) => _sigmoidActivation!.Activate(x));
            c = c.Transform((x, _) => _tanhActivation!.Activate(x));
            o = o.Transform((x, _) => _sigmoidActivation!.Activate(x));
        }

        var newC = f.ElementwiseMultiply(prevC).Add(i.ElementwiseMultiply(c));
        var newH = o.ElementwiseMultiply(_useVectorActivation 
            ? _tanhVectorActivation!.Activate(newC) 
            : newC.Transform((x, _) => _tanhActivation!.Activate(x)));

        return (newH, newC);
    }

   public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastCellState == null)
        {
            throw new InvalidOperationException("Backward pass called before forward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];
        var inputGradient = new Tensor<T>(_lastInput.Shape);
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

        var dNextH = new Tensor<T>([batchSize, _hiddenSize]);
        var dNextC = new Tensor<T>([batchSize, _hiddenSize]);

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            var dh = outputGradient.GetSlice(t).Add(dNextH);
            var xt = _lastInput.GetSlice(t);
            var prevH = t > 0 ? _lastHiddenState.GetSlice(t - 1) : new Tensor<T>([batchSize, _hiddenSize]);
            var prevC = t > 0 ? _lastCellState.GetSlice(t - 1) : new Tensor<T>([batchSize, _hiddenSize]);

            var (dxt, dprevH, dprevC, dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo) = 
                BackwardStep(dh, dNextC, xt, prevH, prevC);

            inputGradient.SetSlice(t, dxt);
            dNextH = dprevH;
            dNextC = dprevC;

            dWeightsFi.Add(dWfi);
            dWeightsIi.Add(dWii);
            dWeightsCi.Add(dWci);
            dWeightsOi.Add(dWoi);
            dWeightsFh.Add(dWfh);
            dWeightsIh.Add(dWih);
            dWeightsCh.Add(dWch);
            dWeightsOh.Add(dWoh);
            dBiasF.Add(dbf);
            dBiasI.Add(dbi);
            dBiasC.Add(dbc);
            dBiasO.Add(dbo);
        }

        // Store gradients for use in UpdateParameters
        Gradients = new Dictionary<string, Tensor<T>>
        {
            {"weightsFi", dWeightsFi}, {"weightsIi", dWeightsIi}, {"weightsCi", dWeightsCi}, {"weightsOi", dWeightsOi},
            {"weightsFh", dWeightsFh}, {"weightsIh", dWeightsIh}, {"weightsCh", dWeightsCh}, {"weightsOh", dWeightsOh},
            {"biasF", dBiasF}, {"biasI", dBiasI}, {"biasC", dBiasC}, {"biasO", dBiasO}
        };

        return inputGradient;
    }

    private (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) 
    BackwardStep(Tensor<T> dh, Tensor<T> dc_next, Tensor<T> x, Tensor<T> prev_h, Tensor<T> prev_c)
    {
        // Forward pass calculations (needed for backward pass)
        var concat = Tensor<T>.Concatenate([x, prev_h], 1);
        var f = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation, 
            concat.Multiply(Tensor<T>.Concatenate([_weightsFi, _weightsFh], 0)).Add(_biasF));
        var i = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsIi, _weightsIh], 0)).Add(_biasI));
        var c_bar = ActivateTensorConditional(_tanhVectorActivation, _tanhActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsCi, _weightsCh], 0)).Add(_biasC));
        var o = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsOi, _weightsOh], 0)).Add(_biasO));
        var c = f.PointwiseMultiply(prev_c).Add(i.PointwiseMultiply(c_bar));
        var h = o.PointwiseMultiply(ActivateTensor(_tanhActivation, c));

        // Backward pass
        var do_ = dh.PointwiseMultiply(ActivateTensor(_tanhActivation, c));
        var dc = dh.PointwiseMultiply(o).PointwiseMultiply(DerivativeTensor(_tanhActivation, c)).Add(dc_next);
        var dc_bar = dc.PointwiseMultiply(i);
        var di = dc.PointwiseMultiply(c_bar);
        var df = dc.PointwiseMultiply(prev_c);
        var dprev_c = dc.PointwiseMultiply(f);

        // Gate derivatives
        var di_input = DerivativeTensor(_sigmoidActivation, i).PointwiseMultiply(di);
        var df_input = DerivativeTensor(_sigmoidActivation, f).PointwiseMultiply(df);
        var do_input = DerivativeTensor(_sigmoidActivation, o).PointwiseMultiply(do_);
        var dc_bar_input = DerivativeTensor(_tanhActivation, c_bar).PointwiseMultiply(dc_bar);

        // Compute gradients for weights and biases
        var dWeights = concat.Transpose(new[] { 1, 0 }).Multiply(Tensor<T>.Concatenate(new[] { di_input, df_input, dc_bar_input, do_input }, 1));
        var dWfi = dWeights.Slice(0, 0, _inputSize).Slice(1, 0, _hiddenSize);
        var dWii = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize, _hiddenSize * 2);
        var dWci = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dWoi = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize * 3, _hiddenSize * 4);
        var dWfh = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, 0, _hiddenSize);
        var dWih = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize, _hiddenSize * 2);
        var dWch = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dWoh = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize * 3, _hiddenSize * 4);

        var dBiases = Tensor<T>.Concatenate(new[] { di_input.Sum(new[] { 0 }), df_input.Sum(new[] { 0 }), dc_bar_input.Sum(new[] { 0 }), do_input.Sum(new[] { 0 }) }, 1);
        var dbf = dBiases.Slice(1, 0, _hiddenSize);
        var dbi = dBiases.Slice(1, _hiddenSize, _hiddenSize * 2);
        var dbc = dBiases.Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dbo = dBiases.Slice(1, _hiddenSize * 3, _hiddenSize * 4);

        // Compute gradient for input
        var dInputs = di_input.Multiply(_weightsIi.Transpose(new[] { 1, 0 }))
            .Add(df_input.Multiply(_weightsFi.Transpose(new[] { 1, 0 })))
            .Add(dc_bar_input.Multiply(_weightsCi.Transpose(new[] { 1, 0 })))
            .Add(do_input.Multiply(_weightsOi.Transpose(new[] { 1, 0 })));

        var dx = dInputs.Slice(1, 0, _inputSize);
        var dprev_h = dInputs.Slice(1, _inputSize, _inputSize + _hiddenSize);

        return (dx, dprev_h, dprev_c, dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo);
    }

    private Tensor<T> ActivateTensorConditional(IVectorActivationFunction<T>? vectorActivation, IActivationFunction<T>? scalarActivation, Tensor<T> input)
    {
        if (_useVectorActivation)
        {
            return ActivateTensor(vectorActivation, input);
        }
        else
        {
            return ActivateTensor(scalarActivation, input);
        }
    }

    private Tensor<T> ActivateTensor(IActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return input;
        }

        return input.Transform((x, _) => activation.Activate(x));
    }

    private Tensor<T> ActivateTensor(IVectorActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return input;
        }

        return activation.Activate(input);
    }

    private Tensor<T> DerivativeTensor(IActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return Tensor<T>.CreateDefault(input.Shape, NumOps.One);
        }

        return input.Transform((x, _) => activation.Derivative(x));
    }

    private Tensor<T> DerivativeTensor(IVectorActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return Tensor<T>.CreateDefault(input.Shape, NumOps.One);
        }

        return activation.Derivative(input);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var kvp in Gradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            switch (paramName)
            {
                case "weightsFi":
                    _weightsFi = _weightsFi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsIi":
                    _weightsIi = _weightsIi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsCi":
                    _weightsCi = _weightsCi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsOi":
                    _weightsOi = _weightsOi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsFh":
                    _weightsFh = _weightsFh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsIh":
                    _weightsIh = _weightsIh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsCh":
                    _weightsCh = _weightsCh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsOh":
                    _weightsOh = _weightsOh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasF":
                    _biasF = _biasF.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasI":
                    _biasI = _biasI.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasC":
                    _biasC = _biasC.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasO":
                    _biasO = _biasO.Subtract(gradient.Multiply(learningRate));
                    break;
            }
        }
    }

    public override void Serialize(BinaryWriter writer)
    {
        SerializeTensor(writer, _weightsFi);
        SerializeTensor(writer, _weightsIi);
        SerializeTensor(writer, _weightsCi);
        SerializeTensor(writer, _weightsOi);
        SerializeTensor(writer, _weightsFh);
        SerializeTensor(writer, _weightsIh);
        SerializeTensor(writer, _weightsCh);
        SerializeTensor(writer, _weightsOh);
        SerializeTensor(writer, _biasF);
        SerializeTensor(writer, _biasI);
        SerializeTensor(writer, _biasC);
        SerializeTensor(writer, _biasO);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _weightsFi = DeserializeTensor(reader);
        _weightsIi = DeserializeTensor(reader);
        _weightsCi = DeserializeTensor(reader);
        _weightsOi = DeserializeTensor(reader);
        _weightsFh = DeserializeTensor(reader);
        _weightsIh = DeserializeTensor(reader);
        _weightsCh = DeserializeTensor(reader);
        _weightsOh = DeserializeTensor(reader);
        _biasF = DeserializeTensor(reader);
        _biasI = DeserializeTensor(reader);
        _biasC = DeserializeTensor(reader);
        _biasO = DeserializeTensor(reader);
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        foreach (var value in tensor)
        {
            writer.Write(Convert.ToDouble(value));
        }
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }

        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return tensor;
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weightsFi.Length + _weightsIi.Length + _weightsCi.Length + _weightsOi.Length +
                          _weightsFh.Length + _weightsIh.Length + _weightsCh.Length + _weightsOh.Length +
                          _biasF.Length + _biasI.Length + _biasC.Length + _biasO.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy weights parameters
        CopyTensorToVector(_weightsFi, parameters, ref index);
        CopyTensorToVector(_weightsIi, parameters, ref index);
        CopyTensorToVector(_weightsCi, parameters, ref index);
        CopyTensorToVector(_weightsOi, parameters, ref index);
        CopyTensorToVector(_weightsFh, parameters, ref index);
        CopyTensorToVector(_weightsIh, parameters, ref index);
        CopyTensorToVector(_weightsCh, parameters, ref index);
        CopyTensorToVector(_weightsOh, parameters, ref index);
    
        // Copy bias parameters
        CopyTensorToVector(_biasF, parameters, ref index);
        CopyTensorToVector(_biasI, parameters, ref index);
        CopyTensorToVector(_biasC, parameters, ref index);
        CopyTensorToVector(_biasO, parameters, ref index);
    
        return parameters;
    }

    private void CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[startIndex++] = tensor[i];
        }
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weightsFi.Length + _weightsIi.Length + _weightsCi.Length + _weightsOi.Length +
                          _weightsFh.Length + _weightsIh.Length + _weightsCh.Length + _weightsOh.Length +
                          _biasF.Length + _biasI.Length + _biasC.Length + _biasO.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights parameters
        CopyVectorToTensor(parameters, _weightsFi, ref index);
        CopyVectorToTensor(parameters, _weightsIi, ref index);
        CopyVectorToTensor(parameters, _weightsCi, ref index);
        CopyVectorToTensor(parameters, _weightsOi, ref index);
        CopyVectorToTensor(parameters, _weightsFh, ref index);
        CopyVectorToTensor(parameters, _weightsIh, ref index);
        CopyVectorToTensor(parameters, _weightsCh, ref index);
        CopyVectorToTensor(parameters, _weightsOh, ref index);
    
        // Set bias parameters
        CopyVectorToTensor(parameters, _biasF, ref index);
        CopyVectorToTensor(parameters, _biasI, ref index);
        CopyVectorToTensor(parameters, _biasC, ref index);
        CopyVectorToTensor(parameters, _biasO, ref index);
    }

    private void CopyVectorToTensor(Vector<T> vector, Tensor<T> tensor, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = vector[startIndex++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastHiddenState = null;
        _lastCellState = null;
        Gradients.Clear();
    }
}