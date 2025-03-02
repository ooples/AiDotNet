namespace AiDotNet.NeuralNetworks.Layers;

public class LogVarianceLayer<T> : LayerBase<T>
{
    public int Axis { get; private set; }

    public override bool SupportsTraining => false;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _meanValues;

    public LogVarianceLayer(int[] inputShape, int axis)
        : base(inputShape, CalculateOutputShape(inputShape, axis))
    {
        Axis = axis;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int axis)
    {
        var outputShape = new int[inputShape.Length - 1];
        int outputIndex = 0;
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (i != axis)
            {
                outputShape[outputIndex++] = inputShape[i];
            }
        }

        return outputShape;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var output = new Tensor<T>(OutputShape);
        _meanValues = new Tensor<T>(OutputShape);

        int axisSize = input.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);

        // Compute mean
        var indices = new int[input.Shape.Length];
        IterateOverDimensions(input, _meanValues, indices, 0, Axis, (input, mean, indices) =>
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                sum = NumOps.Add(sum, input[indices]);
            }
            mean[indices] = NumOps.Multiply(sum, axisScale);
        });

        // Compute log variance
        IterateOverDimensions(input, output, indices, 0, Axis, (input, output, indices) =>
        {
            T sumSquaredDiff = NumOps.Zero;
            T mean = _meanValues[indices];
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                T diff = NumOps.Subtract(input[indices], mean);
                sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Square(diff));
            }
            T variance = NumOps.Multiply(sumSquaredDiff, axisScale);
            output[indices] = NumOps.Log(NumOps.Add(variance, NumOps.FromDouble(1e-8))); // Add small epsilon for numerical stability
        });

        _lastOutput = output;
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _meanValues == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        int axisSize = _lastInput.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);

        var indices = new int[_lastInput.Shape.Length];
        IterateOverDimensions(_lastInput, outputGradient, indices, 0, Axis, (input, outputGrad, indices) =>
        {
            T mean = _meanValues[indices];
            T variance = NumOps.Exp(_lastOutput[indices]);
            T gradScale = NumOps.Divide(outputGrad[indices], variance);

            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                T diff = NumOps.Subtract(input[indices], mean);
                T grad = NumOps.Multiply(NumOps.Multiply(diff, gradScale), NumOps.FromDouble(2.0 / axisSize));
                inputGradient[indices] = grad;
            }
        });

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // LogVarianceLayer has no learnable parameters, so this method is empty
    }

    private void IterateOverDimensions(Tensor<T> input, Tensor<T> output, int[] indices, int currentDim, int skipDim, Action<Tensor<T>, Tensor<T>, int[]> action)
    {
        if (currentDim == input.Shape.Length)
        {
            action(input, output, indices);
            return;
        }

        if (currentDim == skipDim)
        {
            IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
        }
        else
        {
            for (int i = 0; i < input.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        // LogVarianceLayer has no trainable parameters
        return new Vector<T>(0);
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastOutput = null;
        _meanValues = null;
    }
}