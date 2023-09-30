namespace AiDotNet.Helpers;

internal static class ValidationHelper
{
    public static void CheckForNullItems(double[][] inputs, double[] outputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs), "Inputs can't be null");
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs), "Outputs can't be null");
        }
    }

    public static void CheckForNullItems(double[][] inputs, double[] outputs, double[] weights)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs), "Inputs can't be null");
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs), "Outputs can't be null");
        }

        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights), "Weights can't be null");
        }
    }

    public static void CheckForNullItems(double[] inputs, double[] outputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs), "Inputs can't be null");
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs), "Outputs can't be null");
        }
    }

    public static void CheckForInvalidInputSize(int inputSize, int outputsLength)
    {
        if (inputSize != outputsLength)
        {
            throw new ArgumentException("Inputs and outputs must have the same length");
        }

        if (inputSize < 2)
        {
            throw new ArgumentException("Inputs and outputs must have at least 2 values each");
        }
    }

    public static void CheckForInvalidTrainingPctSize(double trainingPctSize)
    {
        if (trainingPctSize <= 0 || trainingPctSize >= 100)
        {
            throw new ArgumentException($"{nameof(trainingPctSize)} must be greater than 0 and less than 100", nameof(trainingPctSize));
        }
    }

    public static void CheckForInvalidTrainingSizes(int trainingSize, int outOfSampleSize, int minSize, double trainingPctSize)
    {
        if (trainingSize < minSize)
        {
            throw new ArgumentException($"Training data must contain at least {minSize} values. " +
                                        $"You either need to increase your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }

        if (outOfSampleSize < 2)
        {
            throw new ArgumentException($"Out of sample data must contain at least 2 values. " +
                                        $"You either need to decrease your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }
    }
}