namespace AiDotNet.NeuralNetworks.Layers;

public class SynapticPlasticityLayer<T> : LayerBase<T>
{
    private Vector<T> _lastInput;
    private Vector<T> _lastOutput;
    private Tensor<T> _weights;
    private readonly double _stdpLtpRate;   // LTP learning rate
    private readonly double _stdpLtdRate;   // LTD learning rate
    private readonly double _homeostasisRate;
    private readonly double _maxWeight = 1.0;
    private readonly double _minWeight;
    private Vector<T> _presynapticTraces;
    private Vector<T> _postsynapticTraces;
    private Vector<T> _presynapticSpikes;
    private Vector<T> _postsynapticSpikes;
    private readonly double _traceDecay;

    public override bool SupportsTraining => true;

    public SynapticPlasticityLayer(int size, double stdpLtpRate = 0.005, 
        double stdpLtdRate = 0.0025, double homeostasisRate = 0.0001, double minWeight = 0, double maxWeight = 1, double traceDecay = 0.95) : base([size], [size])
    {
        _lastInput = new Vector<T>(size);
        _lastOutput = new Vector<T>(size);
        _stdpLtpRate = stdpLtpRate;
        _stdpLtdRate = stdpLtdRate;
        _homeostasisRate = homeostasisRate;
        _minWeight = minWeight;
        _maxWeight = maxWeight;
        _traceDecay = traceDecay;

        _weights = Tensor<T>.CreateRandom([size, size]); // Initialize with small random values
        _presynapticTraces = new Vector<T>(size);
        _postsynapticTraces = new Vector<T>(size);
        _presynapticSpikes = new Vector<T>(size);
        _postsynapticSpikes = new Vector<T>(size);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var inputVector = input.ToVector();
        _lastInput = inputVector;
        _lastOutput = inputVector; // Pass-through layer

        return input;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // This is a pass-through layer, so we simply pass the gradient back
        // No weight updates are performed here as they're handled in UpdateParameters
        return outputGradient;
    }

   public override void UpdateParameters(T learningRate)
    {
        int size = GetInputShape()[0];
    
        // Update spike traces (exponential decay)
        for (int i = 0; i < size; i++)
        {
            // Decay traces over time
            _presynapticTraces[i] = NumOps.Multiply(_presynapticTraces[i], NumOps.FromDouble(_traceDecay));
            _postsynapticTraces[i] = NumOps.Multiply(_postsynapticTraces[i], NumOps.FromDouble(_traceDecay));
        
            // Record new spikes (assuming binary activation where 1.0 = spike)
            if (NumOps.GreaterThan(_lastInput[i], NumOps.FromDouble(0.5)))
            {
                _presynapticSpikes[i] = NumOps.One;
                _presynapticTraces[i] = NumOps.One; // Set trace to 1.0 when spike occurs
            }
            else
            {
                _presynapticSpikes[i] = NumOps.Zero;
            }
        
            if (NumOps.GreaterThan(_lastOutput[i], NumOps.FromDouble(0.5)))
            {
                _postsynapticSpikes[i] = NumOps.One;
                _postsynapticTraces[i] = NumOps.One; // Set trace to 1.0 when spike occurs
            }
            else
            {
                _postsynapticSpikes[i] = NumOps.Zero;
            }
        }
    
        // Apply STDP rule to update weights
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                // Skip self-connections
                if (i == j) continue;
            
                T currentWeight = _weights[i, j];
                T weightChange = NumOps.Zero;
            
                // LTP: If presynaptic neuron fired before postsynaptic neuron
                if (NumOps.Equals(_presynapticSpikes[i], NumOps.One) && 
                    NumOps.GreaterThan(_postsynapticTraces[j], NumOps.Zero))
                {
                    // The strength of potentiation depends on the postsynaptic trace
                    T potentiation = NumOps.Multiply(
                        NumOps.FromDouble(_stdpLtpRate),
                        NumOps.Multiply(_postsynapticTraces[j], 
                            NumOps.Subtract(NumOps.FromDouble(_maxWeight), currentWeight))
                    );
                    weightChange = NumOps.Add(weightChange, potentiation);
                }
            
                // LTD: If postsynaptic neuron fired before presynaptic neuron
                if (NumOps.Equals(_postsynapticSpikes[j], NumOps.One) && 
                    NumOps.GreaterThan(_presynapticTraces[i], NumOps.Zero))
                {
                    // The strength of depression depends on the presynaptic trace
                    T depression = NumOps.Multiply(
                        NumOps.FromDouble(_stdpLtdRate),
                        NumOps.Multiply(_presynapticTraces[i], 
                            NumOps.Subtract(currentWeight, NumOps.FromDouble(_minWeight)))
                    );
                    weightChange = NumOps.Subtract(weightChange, depression);
                }
            
                // Apply calcium-based metaplasticity (homeostasis)
                // If a synapse is very strong, make it harder to strengthen further
                T homeostasisFactor = NumOps.Multiply(
                    NumOps.FromDouble(_homeostasisRate),
                    NumOps.Subtract(currentWeight, NumOps.FromDouble(0.5))
                );
                weightChange = NumOps.Subtract(weightChange, homeostasisFactor);
            
                // Apply neuromodulation (using the provided learning rate as a global modulator)
                weightChange = NumOps.Multiply(weightChange, learningRate);
            
                // Update weight
                _weights[i, j] = NumOps.Add(currentWeight, weightChange);
            
                // Ensure weight stays within bounds
                if (NumOps.LessThan(_weights[i, j], NumOps.FromDouble(_minWeight)))
                    _weights[i, j] = NumOps.FromDouble(_minWeight);
                if (NumOps.GreaterThan(_weights[i, j], NumOps.FromDouble(_maxWeight)))
                    _weights[i, j] = NumOps.FromDouble(_maxWeight);
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        // This layer doesn't have traditional parameters like weights and biases
        // Instead, it uses the input and output values for plasticity rules
        // Return an empty vector to satisfy the interface
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Reset the internal state of the layer
        for (int i = 0; i < GetInputShape()[0]; i++)
        {
            _lastInput[i] = NumOps.Zero;
            _lastOutput[i] = NumOps.Zero;
        }
    }
}