namespace AiDotNet.NeuralNetworks.Layers;

public class MeasurementLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public override bool SupportsTraining => false;

    public MeasurementLayer(int size) : base([size], [size])
    {
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Assume input is a complex-valued tensor representing quantum states
        var probabilities = new T[input.Shape[0]];
    
        // Get numeric operations for complex numbers
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < input.Shape[0]; i++)
        {
            // Get the complex value from the input tensor
            var complexValue = Tensor<T>.GetComplex(input, i);
        
            // Calculate |z|² = real² + imag²
            var realSquared = NumOps.Multiply(complexValue.Real, complexValue.Real);
            var imagSquared = NumOps.Multiply(complexValue.Imaginary, complexValue.Imaginary);
            probabilities[i] = NumOps.Add(realSquared, imagSquared);
        }

        // Normalize probabilities
        var sum = NumOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            sum = NumOps.Add(sum, probabilities[i]);
        }

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.Divide(probabilities[i], sum);
        }

        // Create a new tensor with the calculated probabilities
        _lastOutput = new Tensor<T>([input.Shape[0]], new Vector<T>(probabilities));
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        // The gradient of measurement with respect to input amplitudes
        var inputGradientData = new Vector<T>(_lastInput.Length);

        for (int i = 0; i < _lastInput.Shape[0]; i++)
        {
            // Get the complex value from the input tensor
            var complexValue = Tensor<T>.GetComplex(_lastInput, i);
            var prob = _lastOutput[i];

            // Gradient of probability with respect to real and imaginary parts
            // dP/dReal = 2 * real / prob
            // dP/dImag = 2 * imag / prob
            var two = NumOps.FromDouble(2.0);
            var dProbdReal = NumOps.Divide(NumOps.Multiply(two, complexValue.Real), prob);
            var dProbdImag = NumOps.Divide(NumOps.Multiply(two, complexValue.Imaginary), prob);

            // Combine gradients
            var gradientValue = NumOps.Multiply(outputGradient[i], 
                NumOps.Add(dProbdReal, dProbdImag));
        
            inputGradientData[i] = gradientValue;
        }

        // Create a new tensor with the calculated gradients
        return new Tensor<T>(_lastInput.Shape, inputGradientData);
    }

    public override void UpdateParameters(T learningRate)
    {
        // MeasurementLayer doesn't have trainable parameters
    }

    public override Vector<T> GetParameters()
    {
        // MeasurementLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
    }
}