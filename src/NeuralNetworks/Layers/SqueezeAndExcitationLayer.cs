namespace AiDotNet.NeuralNetworks.Layers;

public class SqueezeAndExcitationLayer<T> : LayerBase<T>
{
    private readonly int _channels;

    private Matrix<T> _weights1;
    private Vector<T> _bias1;
    private Matrix<T> _weights2;
    private Vector<T> _bias2;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private Matrix<T>? _weights1Gradient;
    private Vector<T>? _bias1Gradient;
    private Matrix<T>? _weights2Gradient;
    private Vector<T>? _bias2Gradient;
    private readonly IActivationFunction<T>? _firstActivation;
    private readonly IActivationFunction<T>? _secondActivation;
    private readonly IVectorActivationFunction<T>? _firstVectorActivation;
    private readonly IVectorActivationFunction<T>? _secondVectorActivation;

    private readonly int _reducedChannels;

    public SqueezeAndExcitationLayer(int channels, int reductionRatio, 
        IActivationFunction<T>? firstActivation = null, 
        IActivationFunction<T>? secondActivation = null)
        : base([[channels]], [channels])
    {
        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstActivation = firstActivation ?? new ReLUActivation<T>();
        _secondActivation = secondActivation ?? new SigmoidActivation<T>();

        _weights1 = new Matrix<T>(_channels, _reducedChannels);
        _bias1 = new Vector<T>(_reducedChannels);
        _weights2 = new Matrix<T>(_reducedChannels, _channels);
        _bias2 = new Vector<T>(_channels);

        InitializeWeights();
    }

    public SqueezeAndExcitationLayer(int channels, int reductionRatio, 
        IVectorActivationFunction<T>? firstVectorActivation = null, 
        IVectorActivationFunction<T>? secondVectorActivation = null)
        : base([[channels]], [channels])
    {
        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstVectorActivation = firstVectorActivation ?? new ReLUActivation<T>();
        _secondVectorActivation = secondVectorActivation ?? new SigmoidActivation<T>();

        _weights1 = new Matrix<T>(_channels, _reducedChannels);
        _bias1 = new Vector<T>(_reducedChannels);
        _weights2 = new Matrix<T>(_reducedChannels, _channels);
        _bias2 = new Vector<T>(_channels);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        InitializeMatrix(_weights1, NumOps.FromDouble(0.1));
        InitializeMatrix(_weights2, NumOps.FromDouble(0.1));
        InitializeVector(_bias1, NumOps.FromDouble(0.1));
        InitializeVector(_bias2, NumOps.FromDouble(0.1));
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    private void InitializeVector(Vector<T> vector, T scale)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int height = input.Shape[1];
        int width = input.Shape[2];

        // Squeeze: Global Average Pooling
        var squeezed = new Matrix<T>(batchSize, _channels);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, h, w, c]);
                    }
                }

                squeezed[b, c] = NumOps.Divide(sum, NumOps.FromDouble(height * width));
            }
        }

        // Excitation: Two FC layers with activation
        var excitation1 = squeezed.Multiply(_weights1).AddVectorToEachRow(_bias1);
        excitation1 = ApplyActivation(excitation1, isFirstActivation: true);
        var excitation2 = excitation1.Multiply(_weights2).AddVectorToEachRow(_bias2);
        var excitation = ApplyActivation(excitation2, isFirstActivation: false);

        // Scale the input
        var output = new Tensor<T>(input.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < _channels; c++)
                    {
                        output[b, h, w, c] = NumOps.Multiply(input[b, h, w, c], excitation[b, c]);
                    }
                }
            }
        }

        _lastOutput = output;
        return output;
    }

    private Matrix<T> ApplyActivation(Matrix<T> input, bool isFirstActivation)
    {
        if (isFirstActivation)
        {
            if (_firstVectorActivation != null)
            {
                return ApplyVectorActivation(input, _firstVectorActivation);
            }
            else if (_firstActivation != null)
            {
                return ApplyScalarActivation(input, _firstActivation);
            }
        }
        else
        {
            if (_secondVectorActivation != null)
            {
                return ApplyVectorActivation(input, _secondVectorActivation);
            }
            else if (_secondActivation != null)
            {
                return ApplyScalarActivation(input, _secondActivation);
            }
        }

        // If no activation function is set, return the input as is
        return input;
    }

    private static Matrix<T> ApplyVectorActivation(Matrix<T> input, IVectorActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            Vector<T> row = input.GetRow(i);
            Vector<T> activatedRow = activationFunction.Activate(row);
            result.SetRow(i, activatedRow);
        }

        return result;
    }

    private static Matrix<T> ApplyScalarActivation(Matrix<T> input, IActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                result[i, j] = activationFunction.Activate(input[i, j]);
            }
        }

        return result;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int height = _lastInput.Shape[1];
        int width = _lastInput.Shape[2];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _weights1Gradient = new Matrix<T>(_weights1.Rows, _weights1.Columns);
        _bias1Gradient = new Vector<T>(_bias1.Length);
        _weights2Gradient = new Matrix<T>(_weights2.Rows, _weights2.Columns);
        _bias2Gradient = new Vector<T>(_bias2.Length);

        // Calculate gradients for scaling and input
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < _channels; c++)
                    {
                        T scaleFactor = NumOps.Divide(_lastOutput[b, h, w, c], _lastInput[b, h, w, c]);
                        inputGradient[b, h, w, c] = NumOps.Multiply(outputGradient[b, h, w, c], scaleFactor);
                    }
                }
            }
        }

        // Calculate gradients for excitation
        var excitationGradient = new Matrix<T>(batchSize, _channels);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(outputGradient[b, h, w, c], _lastInput[b, h, w, c]));
                    }
                }
                excitationGradient[b, c] = sum;
            }
        }

        // Backpropagate through FC layers
        var excitation2Gradient = excitationGradient;
        if (_secondVectorActivation != null)
        {
            excitation2Gradient = ApplyVectorActivationGradient(excitationGradient, _secondVectorActivation);
        }
        else if (_secondActivation != null)
        {
            excitation2Gradient = ApplyScalarActivationGradient(excitationGradient, _secondActivation);
        }

        _weights2Gradient = excitation2Gradient.Transpose().Multiply(excitationGradient);
        _bias2Gradient = excitationGradient.SumColumns();

        var excitation1Gradient = excitation2Gradient.Multiply(_weights2.Transpose());
        if (_firstVectorActivation != null)
        {
            excitation1Gradient = ApplyVectorActivationGradient(excitation1Gradient, _firstVectorActivation);
        }
        else if (_firstActivation != null)
        {
            excitation1Gradient = ApplyScalarActivationGradient(excitation1Gradient, _firstActivation);
        }

        _weights1Gradient = excitation1Gradient.Transpose().Multiply(excitationGradient);
        _bias1Gradient = excitation1Gradient.SumColumns();

        return inputGradient;
    }

    private static Matrix<T> ApplyVectorActivationGradient(Matrix<T> input, IVectorActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            Vector<T> row = input.GetRow(i);
            Matrix<T> gradientMatrix = activationFunction.Derivative(row);
            Vector<T> gradientDiagonal = gradientMatrix.Diagonal();
        
            // Element-wise multiplication of the input row with the gradient diagonal
            Vector<T> gradientRow = row.ElementwiseMultiply(gradientDiagonal);
        
            result.SetRow(i, gradientRow);
        }

        return result;
    }

    private static Matrix<T> ApplyScalarActivationGradient(Matrix<T> input, IActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                result[i, j] = activationFunction.Derivative(input[i, j]);
            }
        }

        return result;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weights1Gradient == null || _bias1Gradient == null || _weights2Gradient == null || _bias2Gradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights1 = _weights1.Subtract(_weights1Gradient.Multiply(learningRate));
        _bias1 = _bias1.Subtract(_bias1Gradient.Multiply(learningRate));
        _weights2 = _weights2.Subtract(_weights2Gradient.Multiply(learningRate));
        _bias2 = _bias2.Subtract(_bias2Gradient.Multiply(learningRate));
    }
}