namespace AiDotNet.ActivationFunctions;

public class MaxoutActivation<T> : ActivationFunctionBase<T>
{
    private readonly int _numPieces;

    public MaxoutActivation(int numPieces)
    {
        if (numPieces < 2)
        {
            throw new ArgumentException("Number of pieces must be at least 2.", nameof(numPieces));
        }

        _numPieces = numPieces;
    }

    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        if (input.Length % _numPieces != 0)
        {
            throw new ArgumentException("Input vector length must be divisible by the number of pieces.");
        }

        int outputSize = input.Length / _numPieces;
        Vector<T> output = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            T maxValue = input[i * _numPieces];
            for (int j = 1; j < _numPieces; j++)
            {
                maxValue = MathHelper.Max(maxValue, input[i * _numPieces + j]);
            }

            output[i] = maxValue;
        }

        return output;
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        if (input.Length % _numPieces != 0)
        {
            throw new ArgumentException("Input vector length must be divisible by the number of pieces.");
        }

        int outputSize = input.Length / _numPieces;
        Matrix<T> jacobian = new Matrix<T>(outputSize, input.Length);

        for (int i = 0; i < outputSize; i++)
        {
            int maxIndex = i * _numPieces;
            T maxValue = input[maxIndex];

            for (int j = 1; j < _numPieces; j++)
            {
                int currentIndex = i * _numPieces + j;
                if (NumOps.GreaterThan(input[currentIndex], maxValue))
                {
                    maxIndex = currentIndex;
                    maxValue = input[currentIndex];
                }
            }

            jacobian[i, maxIndex] = NumOps.One;
        }

        return jacobian;
    }
}