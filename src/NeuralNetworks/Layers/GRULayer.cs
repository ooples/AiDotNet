namespace AiDotNet.NeuralNetworks.Layers;

public class GRULayer<T> : LayerBase<T>
{
    private Matrix<T> _Wz, _Wr, _Wh;
    private Matrix<T> _Uz, _Ur, _Uh;
    private Vector<T> _bz, _br, _bh;

    private Tensor<T>? _dWz, _dWr, _dWh, _dUz, _dUr, _dUh;
    private Tensor<T>? _dbz, _dbr, _dbh;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastHiddenState;
    private Tensor<T>? _lastZ, _lastR, _lastH;
    private List<Tensor<T>>? _allHiddenStates;

    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly bool _returnSequences;

    private readonly IActivationFunction<T>? _activation;
    private readonly IActivationFunction<T>? _recurrentActivation;
    private readonly IVectorActivationFunction<T>? _vectorActivation;
    private readonly IVectorActivationFunction<T>? _vectorRecurrentActivation;

    public override int ParameterCount => 
        _hiddenSize * _inputSize * 3 +  // Wz, Wr, Wh
        _hiddenSize * _hiddenSize * 3 + // Uz, Ur, Uh
        _hiddenSize * 3;                // bz, br, bh

    public override bool SupportsTraining => true;

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
            // Extract current time step input - fix the Slice method parameters
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
                _allHiddenStates.Add(h.Copy()); // Using Copy instead of Clone
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

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastZ == null || _lastR == null || _lastH == null)
        throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward pass needs to be modified to handle sequences if returnSequences is true
        // This is a simplified version that works for the non-sequence case
        
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

        var dh_prev = dz.Multiply(_Uz.Transpose())
                        .Add(dr.Multiply(_Ur.Transpose()))
                        .Add(dh_candidate.ElementwiseMultiply(_lastR).Multiply(_Uh.Transpose()));

        // Calculate gradients for weights and biases
        var dWz = _lastInput.Transpose(new[] { 1, 0 }).Multiply(dz);
        var dWr = _lastInput.Transpose(new[] { 1, 0 }).Multiply(dr);
        var dWh = _lastInput.Transpose(new[] { 1, 0 }).Multiply(dh_candidate);

        var dUz = _lastHiddenState.Transpose(new[] { 1, 0 }).Multiply(dz);
        var dUr = _lastHiddenState.Transpose(new[] { 1, 0 }).Multiply(dr);
        var dUh = _lastHiddenState.Transpose(new[] { 1, 0 }).Multiply(dh_candidate.ElementwiseMultiply(_lastR));

        var dbz = dz.Sum(new[] { 0 });
        var dbr = dr.Sum(new[] { 0 });
        var dbh = dh_candidate.Sum(new[] { 0 });

        // Store gradients for use in UpdateParameters
        _dWz = dWz; _dWr = dWr; _dWh = dWh;
        _dUz = dUz; _dUr = dUr; _dUh = dUh;
        _dbz = dbz; _dbr = dbr; _dbh = dbh;

        return dx;
    }

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

    public override void ResetState()
    {
        _lastInput = null;
        _lastHiddenState = null;
        _lastZ = null;
        _lastR = null;
        _lastH = null;
        _allHiddenStates = null;
    }

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