namespace AiDotNet.NeuralNetworks.Layers;

public class QuantumLayer<T> : LayerBase<T>
{
    private readonly int _numQubits;
    private Tensor<Complex<T>> _quantumCircuit;
    private Tensor<T>? _lastInput;
    private Vector<T> _rotationAngles;
    private Vector<T> _angleGradients;
    private readonly INumericOperations<Complex<T>> _complexOps;

    public override bool SupportsTraining => true;

    public QuantumLayer(int inputSize, int outputSize, int numQubits) : base([inputSize], [outputSize])
    {
        _numQubits = numQubits;
        _complexOps = MathHelper.GetNumericOperations<Complex<T>>();
            
        // Initialize parameters
        _rotationAngles = new Vector<T>(_numQubits);
        _angleGradients = new Vector<T>(_numQubits);
            
        // Create quantum circuit as a tensor
        int dimension = 1 << _numQubits;
        _quantumCircuit = new Tensor<Complex<T>>([dimension, dimension]);

        InitializeQuantumCircuit();
    }

    private void InitializeQuantumCircuit()
    {
        int dimension = 1 << _numQubits;
            
        // Initialize quantum circuit as identity matrix
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                _quantumCircuit[i, j] = i == j ? 
                    new Complex<T>(NumOps.One, NumOps.Zero) : 
                    new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }

        // Initialize rotation angles randomly
        for (int i = 0; i < _numQubits; i++)
        {
            _rotationAngles[i] = NumOps.FromDouble(Random.NextDouble() * 2 * Math.PI);
            ApplyRotation(i, _rotationAngles[i]);
        }
    }

    private void ApplyRotation(int qubit, T angle)
    {
        int dimension = 1 << _numQubits;
            
        // Calculate rotation parameters
        var halfAngle = NumOps.Divide(angle, NumOps.FromDouble(2.0));
        var cos = MathHelper.Cos(halfAngle);
        var sin = MathHelper.Sin(halfAngle);
            
        // Create complex values for the rotation
        var cosComplex = new Complex<T>(cos, NumOps.Zero);
        var sinComplex = new Complex<T>(sin, NumOps.Zero);
        var imaginary = new Complex<T>(NumOps.Zero, NumOps.One);
        var negativeImaginary = new Complex<T>(NumOps.Zero, NumOps.Negate(NumOps.One));

        // Create a temporary copy of the circuit for the transformation
        var tempCircuit = _quantumCircuit.Copy();

        for (int i = 0; i < dimension; i++)
        {
            if ((i & (1 << qubit)) == 0)
            {
                int j = i | (1 << qubit);
                for (int k = 0; k < dimension; k++)
                {
                    var temp = tempCircuit[k, i];
                        
                    // Apply rotation matrix
                    _quantumCircuit[k, i] = _complexOps.Add(
                        _complexOps.Multiply(cosComplex, temp),
                        _complexOps.Multiply(negativeImaginary, _complexOps.Multiply(sinComplex, tempCircuit[k, j]))
                    );
                        
                    _quantumCircuit[k, j] = _complexOps.Add(
                        _complexOps.Multiply(imaginary, _complexOps.Multiply(sinComplex, temp)),
                        _complexOps.Multiply(cosComplex, tempCircuit[k, j])
                    );
                }
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int dimension = 1 << _numQubits;
        
        // Create output tensor
        var output = new Tensor<T>([batchSize, dimension]);
        
        for (int b = 0; b < batchSize; b++)
        {
            // Convert input to quantum state
            var quantumState = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < Math.Min(input.Shape[1], dimension); i++)
            {
                quantumState[i] = new Complex<T>(input[b, i], NumOps.Zero);
            }
            
            // Normalize the quantum state
            var normFactor = NumOps.Zero;
            for (int i = 0; i < dimension; i++)
            {
                // Calculate magnitude squared manually
                var complex = quantumState[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                normFactor = NumOps.Add(normFactor, magnitudeSquared);
            }
            
            normFactor = NumOps.Sqrt(normFactor);
            if (!NumOps.Equals(normFactor, NumOps.Zero))
            {
                for (int i = 0; i < dimension; i++)
                {
                    quantumState[i] = _complexOps.Divide(quantumState[i], 
                        new Complex<T>(normFactor, NumOps.Zero));
                }
            }

            // Apply quantum circuit
            var result = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                result[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int j = 0; j < dimension; j++)
                {
                    result[i] = _complexOps.Add(result[i], 
                        _complexOps.Multiply(_quantumCircuit[i, j], quantumState[j]));
                }
            }

            // Convert complex amplitudes to probabilities
            for (int i = 0; i < dimension; i++)
            {
                // Calculate magnitude squared manually
                var complex = result[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                output[b, i] = magnitudeSquared;
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        int batchSize = outputGradient.Shape[0];
        int dimension = 1 << _numQubits;
        int inputDimension = _lastInput.Shape[1];
        
        // Create input gradient tensor
        var inputGradient = new Tensor<T>([batchSize, inputDimension]);
        
        for (int b = 0; b < batchSize; b++)
        {
            // Convert output gradient to complex form
            var gradientState = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                gradientState[i] = new Complex<T>(outputGradient[b, i], NumOps.Zero);
            }

            // Backpropagate through quantum circuit
            var backpropGradient = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                backpropGradient[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int j = 0; j < dimension; j++)
                {
                    // Use the Conjugate method from Complex<T> directly
                    var conjugate = _quantumCircuit[j, i].Conjugate();
                    backpropGradient[i] = _complexOps.Add(backpropGradient[i], 
                        _complexOps.Multiply(conjugate, gradientState[j]));
                }
            }

            // Update parameter gradients
            UpdateAngleGradients(gradientState, b);

            // Copy gradients to output tensor
            for (int i = 0; i < Math.Min(inputDimension, dimension); i++)
            {
                // Calculate magnitude manually from the complex number
                var complex = backpropGradient[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                inputGradient[b, i] = NumOps.Sqrt(magnitudeSquared);
            }
        }

        return inputGradient;
    }

    private void UpdateAngleGradients(Tensor<Complex<T>> gradientState, int batchIndex)
    {
        int dimension = 1 << _numQubits;
        
        for (int qubit = 0; qubit < _numQubits; qubit++)
        {
            T qubitGradient = NumOps.Zero;
            
            for (int i = 0; i < dimension; i++)
            {
                if ((i & (1 << qubit)) == 0)
                {
                    int j = i | (1 << qubit);
                    
                    // Calculate gradient contribution for this qubit
                    var gradDiff = gradientState[j] * _quantumCircuit[j, i].Conjugate() -
                                   gradientState[i] * _quantumCircuit[i, j].Conjugate();
                    
                    // Extract the real part of the complex number
                    qubitGradient = NumOps.Add(qubitGradient, gradDiff.Real);
                }
            }
            
            // Accumulate gradients across batches
            _angleGradients[qubit] = NumOps.Add(_angleGradients[qubit], qubitGradient);
        }
    }

    public override void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _numQubits; i++)
        {
            // Update rotation angles using gradient descent
            _rotationAngles[i] = NumOps.Subtract(_rotationAngles[i], NumOps.Multiply(learningRate, _angleGradients[i]));

            // Ensure angles stay within [0, 2π]
            _rotationAngles[i] = MathHelper.Modulo(
                NumOps.Add(_rotationAngles[i], NumOps.FromDouble(2 * Math.PI)), 
                NumOps.FromDouble(2 * Math.PI));

            // Apply updated rotation
            ApplyRotation(i, _rotationAngles[i]);
        }

        // Reset angle gradients for the next iteration
        _angleGradients = new Vector<T>(_numQubits);
    }

    public override Vector<T> GetParameters()
    {
        // Return a copy of the rotation angles
        return _rotationAngles.Copy();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numQubits)
        {
            throw new ArgumentException($"Expected {_numQubits} parameters, but got {parameters.Length}");
        }
    
        // Reset the quantum circuit to identity
        ResetQuantumCircuit();
    
        // Set new rotation angles and apply them
        for (int i = 0; i < _numQubits; i++)
        {
            _rotationAngles[i] = parameters[i];
            ApplyRotation(i, _rotationAngles[i]);
        }
    }

    private void ResetQuantumCircuit()
    {
        int dimension = 1 << _numQubits;
    
        // Reset quantum circuit to identity matrix
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                _quantumCircuit[i, j] = i == j ? 
                    new Complex<T>(NumOps.One, NumOps.Zero) : 
                    new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    
        // Reset angle gradients
        _angleGradients = new Vector<T>(_numQubits);
    }
}