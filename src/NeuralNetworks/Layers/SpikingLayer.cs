namespace AiDotNet.NeuralNetworks.Layers;

public class SpikingLayer<T> : LayerBase<T>
{
    private SpikingNeuronType _neuronType;
    private double _tau;            // Time constant for membrane potential decay
    private double _refractoryPeriod; // Refractory period in time steps
    private Matrix<T> Weights;
    private Vector<T> Bias;
    private Matrix<T> _weightGradients;
    private Vector<T> _biasGradients;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    
    private Vector<T> _membranePotential;    // Current membrane potential
    private Vector<T> _refractoryCountdown;  // Countdown timer for refractory period
    private Vector<T> _spikes;               // Output spikes
    
    // Parameters for Izhikevich model
    private Vector<T>? _recoveryVariable;    // Recovery variable for Izhikevich model
    private double _a = 0.02;                // Time scale of recovery variable
    private double _b = 0.2;                 // Sensitivity of recovery variable
    private double _c = -65.0;               // After-spike reset value of membrane potential
    private double _d = 8.0;                 // After-spike reset of recovery variable
    
    // Parameters for Adaptive Exponential model
    private Vector<T>? _adaptationVariable;  // Adaptation variable
    private double _deltaT = 2.0;            // Sharpness of exponential
    private double _vT = -50.0;              // Threshold potential
    private double _tauw = 30.0;             // Adaptation time constant
    private double _a_adex = 4.0;            // Subthreshold adaptation
    private double _b_adex = 80.5;           // Spike-triggered adaptation
    
    // Hodgkin-Huxley model parameters
    private Vector<T>? _nGate;               // Potassium activation gating variable
    private Vector<T>? _mGate;               // Sodium activation gating variable
    private Vector<T>? _hGate;               // Sodium inactivation gating variable

    public override int ParameterCount => Weights.Rows * Weights.Columns + Bias.Length;

    public override bool SupportsTraining => true;
    
    public SpikingLayer(int inputSize, int outputSize, SpikingNeuronType neuronType = SpikingNeuronType.LeakyIntegrateAndFire, 
        double tau = 10.0, double refractoryPeriod = 2.0)
        : base([inputSize], [outputSize])
    {
        _neuronType = neuronType;
        _tau = tau;
        _refractoryPeriod = refractoryPeriod;
    
        // Initialize weights with small random values
        Weights = Matrix<T>.CreateRandom(inputSize, outputSize, -0.1, 0.1);
        Bias = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize gradient accumulators
        _weightGradients = Matrix<T>.CreateDefault(inputSize, outputSize, NumOps.Zero);
        _biasGradients = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize neuron state variables
        _membranePotential = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize model-specific variables based on neuron type
        if (neuronType == SpikingNeuronType.Izhikevich)
        {
            _a = 0.02;  // Time scale of recovery variable
            _b = 0.2;   // Sensitivity of recovery variable
            _c = -65.0; // After-spike reset value of membrane potential
            _d = 8.0;   // After-spike reset of recovery variable
            _recoveryVariable = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        }
        else if (neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            _deltaT = 2.0;  // Sharpness of exponential
            _vT = -50.0;    // Threshold potential
            _tauw = 30.0;   // Adaptation time constant
            _a_adex = 4.0;  // Subthreshold adaptation
            _b_adex = 0.5;  // Spike-triggered adaptation
            _adaptationVariable = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        }
        else if (neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            _nGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.32)); // Potassium activation
            _mGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.05)); // Sodium activation
            _hGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.60)); // Sodium inactivation
        }
    }
    
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store the input for backward pass
        _lastInput = input;
        
        // Convert input tensor to vector for processing
        Vector<T> inputVector;
        if (input.Shape.Length == 1)
        {
            inputVector = input.ToVector();
        }
        else if (input.Shape.Length == 2 && input.Shape[0] == 1)
        {
            // Handle batch size of 1
            inputVector = input.Reshape([input.Shape[1]]).ToVector();
        }
        else
        {
            throw new ArgumentException("Input tensor must be 1D or have batch size of 1");
        }
        
        // Process with the appropriate neuron model
        Vector<T> outputVector = ProcessSpikes(inputVector);
        
        // Convert back to tensor and store for backward pass
        var output = Tensor<T>.FromVector(outputVector);
        _lastOutput = output;
        
        return output;
    }

    private Vector<T> ProcessSpikes(Vector<T> input)
    {
        // Calculate input current
        Vector<T> current = Weights.Multiply(input).Add(Bias);

        // Update neuron states based on the neuron model
        return _neuronType switch
        {
            SpikingNeuronType.LeakyIntegrateAndFire => UpdateLeakyIntegrateAndFire(current),
            SpikingNeuronType.IntegrateAndFire => UpdateIntegrateAndFire(current),
            SpikingNeuronType.Izhikevich => UpdateIzhikevich(current),
            SpikingNeuronType.HodgkinHuxley => UpdateHodgkinHuxley(current),
            SpikingNeuronType.AdaptiveExponential => UpdateAdaptiveExponential(current),
            _ => throw new NotImplementedException($"Neuron type {_neuronType} not implemented."),
        };
    }
    
    private Vector<T> UpdateLeakyIntegrateAndFire(Vector<T> current)
    {
        // Decay membrane potential
        T decayFactor = NumOps.FromDouble(1.0 - 1.0/_tau);
        _membranePotential = _membranePotential.Multiply(decayFactor);
    
        // Update membrane potential for neurons not in refractory period
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_refractoryCountdown[i]) <= 0)
            {
                _membranePotential[i] = NumOps.Add(_membranePotential[i], current[i]);
            }
            else
            {
                _refractoryCountdown[i] = NumOps.Subtract(_refractoryCountdown[i], NumOps.One);
            }
        }
    
        // Generate spikes and reset
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            // Threshold for spiking (typically around 1.0)
            if (Convert.ToDouble(_membranePotential[i]) >= 1.0)
            {
                _spikes[i] = NumOps.One;
                _membranePotential[i] = NumOps.Zero; // Reset potential
                _refractoryCountdown[i] = NumOps.FromDouble(_refractoryPeriod);
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
        }
    
        return _spikes;
    }
    
    private Vector<T> UpdateIntegrateAndFire(Vector<T> current)
    {
        // Similar to LIF but without leak
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_refractoryCountdown[i]) <= 0)
            {
                _membranePotential[i] = NumOps.Add(_membranePotential[i], current[i]);
            }
            else
            {
                _refractoryCountdown[i] = NumOps.Subtract(_refractoryCountdown[i], NumOps.One);
            }
        }
    
        // Generate spikes and reset
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_membranePotential[i]) >= 1.0)
            {
                _spikes[i] = NumOps.One;
                _membranePotential[i] = NumOps.Zero;
                _refractoryCountdown[i] = NumOps.FromDouble(_refractoryPeriod);
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
        }
    
        return _spikes;
    }
    
    private Vector<T> UpdateIzhikevich(Vector<T> current)
    {
        if (_recoveryVariable == null)
            throw new InvalidOperationException("Recovery variable not initialized for Izhikevich model");
            
        // Izhikevich model update
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double u = Convert.ToDouble(_recoveryVariable[i]);
            double I = Convert.ToDouble(current[i]);
            
            // Update membrane potential and recovery variable
            double dv = 0.04 * v * v + 5 * v + 140 - u + I;
            double du = _a * (_b * v - u);
            
            v += dv;
            u += du;
            
            // Check for spike
            if (v >= 30)
            {
                _spikes[i] = NumOps.One;
                v = _c;
                u += _d;
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
            
            _membranePotential[i] = NumOps.FromDouble(v);
            _recoveryVariable[i] = NumOps.FromDouble(u);
        }
        
        return _spikes;
    }
    
    private Vector<T> UpdateAdaptiveExponential(Vector<T> current)
    {
        if (_adaptationVariable == null)
            throw new InvalidOperationException("Adaptation variable not initialized for AdEx model");
            
        // Adaptive Exponential Integrate-and-Fire model
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double w = Convert.ToDouble(_adaptationVariable[i]);
            double I = Convert.ToDouble(current[i]);
            
            // Exponential term for spike initiation
            double expTerm = _deltaT * Math.Exp((v - _vT) / _deltaT);
            
            // Update membrane potential and adaptation variable
            double dv = (-v + expTerm - w + I) / _tau;
            double dw = (_a_adex * (v - _vT) - w) / _tauw;
            
            v += dv;
            w += dw;
            
            // Check for spike
            if (v >= 0) // Spike threshold
            {
                _spikes[i] = NumOps.One;
                v = -70.0; // Reset potential
                w += _b_adex; // Spike-triggered adaptation
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
            
            _membranePotential[i] = NumOps.FromDouble(v);
            _adaptationVariable[i] = NumOps.FromDouble(w);
        }
        
        return _spikes;
    }

    private Vector<T> UpdateHodgkinHuxley(Vector<T> current)
    {
        if (_nGate == null || _mGate == null || _hGate == null)
            throw new InvalidOperationException("Gate variables not initialized for Hodgkin-Huxley model");
    
        // Ensure _spikes is initialized
        _spikes ??= Vector<T>.CreateDefault(_membranePotential.Length, NumOps.Zero);

        // Constants for Hodgkin-Huxley model
        double ENa = 50.0;   // Sodium reversal potential (mV)
        double EK = -77.0;   // Potassium reversal potential (mV)
        double EL = -54.387; // Leak reversal potential (mV)
        double gNa = 120.0;  // Maximum sodium conductance (mS/cm²)
        double gK = 36.0;    // Maximum potassium conductance (mS/cm²)
        double gL = 0.3;     // Leak conductance (mS/cm²)
        double dt = 0.01;    // Time step (ms)
    
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double n = Convert.ToDouble(_nGate[i]);
            double m = Convert.ToDouble(_mGate[i]);
            double h = Convert.ToDouble(_hGate[i]);
            double I = Convert.ToDouble(current[i]);
        
            // Calculate alpha and beta values for each gate
            double alphaM = 0.1 * (v + 40.0) / (1.0 - Math.Exp(-(v + 40.0) / 10.0));
            double betaM = 4.0 * Math.Exp(-(v + 65.0) / 18.0);
        
            double alphaN = 0.01 * (v + 55.0) / (1.0 - Math.Exp(-(v + 55.0) / 10.0));
            double betaN = 0.125 * Math.Exp(-(v + 65.0) / 80.0);
        
            double alphaH = 0.07 * Math.Exp(-(v + 65.0) / 20.0);
            double betaH = 1.0 / (1.0 + Math.Exp(-(v + 35.0) / 10.0));
        
            // Update gate variables
            double dn = alphaN * (1 - n) - betaN * n;
            double dm = alphaM * (1 - m) - betaM * m;
            double dh = alphaH * (1 - h) - betaH * h;
        
            n += dt * dn;
            m += dt * dm;
            h += dt * dh;
        
            // Calculate ionic currents
            double INa = gNa * Math.Pow(m, 3) * h * (v - ENa);
            double IK = gK * Math.Pow(n, 4) * (v - EK);
            double IL = gL * (v - EL);
        
            // Update membrane potential
            double dv = I - INa - IK - IL;
            v += dt * dv;
        
            // Check for spike (threshold crossing)
            if (v > 0 && NumOps.Equals(_spikes[i], NumOps.Zero))
            {
                _spikes[i] = NumOps.One;
            }
            else if (v < -30)
            {
                _spikes[i] = NumOps.Zero;
            }
        
            // Update state variables
            _membranePotential[i] = NumOps.FromDouble(v);
            _nGate[i] = NumOps.FromDouble(n);
            _mGate[i] = NumOps.FromDouble(m);
            _hGate[i] = NumOps.FromDouble(h);
        }
    
        return _spikes;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int weightCount = Weights.Rows * Weights.Columns;
        
        // Update weights
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                int index = i * Weights.Columns + j;
                Weights[i, j] = parameters[index];
            }
        }
        
        // Update biases
        for (int i = 0; i < Bias.Length; i++)
        {
            Bias[i] = parameters[weightCount + i];
        }
    }
    
    public override Vector<T> GetParameters()
    {
        int weightCount = Weights.Rows * Weights.Columns;
        Vector<T> parameters = Vector<T>.CreateDefault(ParameterCount, NumOps.Zero);
        
        // Flatten weights
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                int index = i * Weights.Columns + j;
                parameters[index] = Weights[i, j];
            }
        }
        
        // Add biases
        for (int i = 0; i < Bias.Length; i++)
        {
            parameters[weightCount + i] = Bias[i];
        }
        
        return parameters;
    }

    public override void ResetState()
    {
        // Reset all state variables
        _membranePotential = Vector<T>.CreateDefault(_membranePotential.Length, NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(_refractoryCountdown.Length, NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(_spikes.Length, NumOps.Zero);
    
        // Reset model-specific variables
        if (_neuronType == SpikingNeuronType.Izhikevich && _recoveryVariable != null)
        {
            _recoveryVariable = Vector<T>.CreateDefault(_recoveryVariable.Length, NumOps.Zero);
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential && _adaptationVariable != null)
        {
            _adaptationVariable = Vector<T>.CreateDefault(_adaptationVariable.Length, NumOps.Zero);
        }
        else if (_neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            if (_nGate != null) _nGate = Vector<T>.CreateDefault(_nGate.Length, NumOps.FromDouble(0.32));
            if (_mGate != null) _mGate = Vector<T>.CreateDefault(_mGate.Length, NumOps.FromDouble(0.05));
            if (_hGate != null) _hGate = Vector<T>.CreateDefault(_hGate.Length, NumOps.FromDouble(0.60));
        }
    
        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
    
        // Reset gradient accumulators
        _weightGradients = Matrix<T>.CreateDefault(Weights.Rows, Weights.Columns, NumOps.Zero);
        _biasGradients = Vector<T>.CreateDefault(Bias.Length, NumOps.Zero);
    }
    
    public override void Serialize(BinaryWriter writer)
    {
        // Write neuron type and parameters
        writer.Write((int)_neuronType);
        writer.Write(_tau);
        writer.Write(_refractoryPeriod);
        
        // Write weights and biases
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(Weights[i, j]));
            }
        }
        
        for (int i = 0; i < Bias.Length; i++)
        {
            writer.Write(Convert.ToDouble(Bias[i]));
        }
        
        // Write model-specific parameters
        if (_neuronType == SpikingNeuronType.Izhikevich)
        {
            writer.Write(_a);
            writer.Write(_b);
            writer.Write(_c);
            writer.Write(_d);
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            writer.Write(_deltaT);
            writer.Write(_vT);
            writer.Write(_tauw);
            writer.Write(_a_adex);
            writer.Write(_b_adex);
        }
    }
    
    public override void Deserialize(BinaryReader reader)
    {
        // Read neuron type and parameters
        _neuronType = (SpikingNeuronType)reader.ReadInt32();
        _tau = reader.ReadDouble();
        _refractoryPeriod = reader.ReadDouble();
        
        // Read weights and biases
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                Weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        
        for (int i = 0; i < Bias.Length; i++)
        {
            Bias[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        
        // Read model-specific parameters
        if (_neuronType == SpikingNeuronType.Izhikevich)
        {
            _a = reader.ReadDouble();
            _b = reader.ReadDouble();
            _c = reader.ReadDouble();
            _d = reader.ReadDouble();
            
            // Initialize recovery variable if needed
            if (_recoveryVariable == null)
            {
                _recoveryVariable = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
            }
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            _deltaT = reader.ReadDouble();
            _vT = reader.ReadDouble();
            _tauw = reader.ReadDouble();
            _a_adex = reader.ReadDouble();
            _b_adex = reader.ReadDouble();
            
            // Initialize adaptation variable if needed
            if (_adaptationVariable == null)
            {
                _adaptationVariable = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
            }
        }
        else if (_neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            // Initialize gate variables if needed
            int outputSize = OutputShape[0];
            if (_nGate == null) _nGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.32));
            if (_mGate == null) _mGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.05));
            if (_hGate == null) _hGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.60));
        }
        
        // Initialize state variables
        _membranePotential = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Cannot perform backward pass before forward pass");

        // Convert tensor to vector for easier processing
        Vector<T> gradientVector = outputGradient.ToVector();
    
        // Initialize input gradient
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        Vector<T> inputGradientVector = Vector<T>.CreateDefault(_lastInput.Shape[0], NumOps.Zero);
    
        // Apply surrogate gradient for the non-differentiable spike function
        // We use a sigmoid-based surrogate gradient approximation
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            // Get membrane potential
            double v = Convert.ToDouble(_membranePotential[i]);
        
            // Compute surrogate gradient using a sigmoid-based function
            // This approximates the derivative of the spike function
            double beta = 10.0; // Steepness of the surrogate function
            double surrogate = 1.0 / (beta * Math.Pow(Math.Cosh(v / beta), 2));
        
            // Apply surrogate gradient to the output gradient
            T surrogateGradient = NumOps.FromDouble(surrogate);
            T gradientValue = NumOps.Multiply(gradientVector[i], surrogateGradient);
        
            // Compute weight gradients and accumulate them
            for (int j = 0; j < _lastInput.Shape[0]; j++)
            {
                T inputValue = _lastInput.ToVector()[j];
                T weightGradient = NumOps.Multiply(gradientValue, inputValue);
            
                // Accumulate weight gradients
                _weightGradients[j, i] = NumOps.Add(_weightGradients[j, i], weightGradient);
            
                // Compute input gradients (for backpropagation to previous layer)
                T currentInputGradient = NumOps.Multiply(gradientValue, Weights[j, i]);
                inputGradientVector[j] = NumOps.Add(inputGradientVector[j], currentInputGradient);
            }
        
            // Compute bias gradients
            _biasGradients[i] = NumOps.Add(_biasGradients[i], gradientValue);
        }
    
        // Convert input gradient vector back to tensor
        for (int i = 0; i < inputGradientVector.Length; i++)
        {
            inputGradient[i] = inputGradientVector[i];
        }
    
        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Update weights using accumulated gradients
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                // Compute weight update: w = w - lr * gradient
                T update = NumOps.Multiply(_weightGradients[i, j], learningRate);
                Weights[i, j] = NumOps.Subtract(Weights[i, j], update);
            
                // Reset gradient for next batch
                _weightGradients[i, j] = NumOps.Zero;
            }
        }
    
        // Update biases using accumulated gradients
        for (int i = 0; i < Bias.Length; i++)
        {
            // Compute bias update: b = b - lr * gradient
            T update = NumOps.Multiply(_biasGradients[i], learningRate);
            Bias[i] = NumOps.Subtract(Bias[i], update);
        
            // Reset gradient for next batch
            _biasGradients[i] = NumOps.Zero;
        }
    }
}