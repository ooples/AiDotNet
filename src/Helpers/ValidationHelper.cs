﻿namespace AiDotNet.Helpers;

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

    internal static void CheckForNullItems(double[][] inputs, double[][] outputs)
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

    internal static void CheckForNullItems(double[] inputs, double[] outputs)
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

    internal static void CheckForInvalidOrder(int order, double[] inputs)
    {
        if (order < 1)
        {
            throw new ArgumentException("Order must be greater than 0", nameof(order));
        }

        if (order > inputs.Length - 1)
        {
            throw new ArgumentException(
                $"The order amount you use can't be greater or equal to the amount of inputs. " +
                $"You currently have {inputs.Length} inputs and the order must be {inputs.Length - 1} or less",
                nameof(inputs));
        }
    }

    internal static void CheckForInvalidWeights(double[] weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights), "Weights can't be null");
        }

        if (weights.All(x => x == 0))
        {
            throw new ArgumentException("Weights can't contain all zeros", nameof(weights));
        }

        if (weights.Any(x => double.IsNaN(x) || double.IsInfinity(x)))
        {
            throw new ArgumentException("Weights can't contain invalid values such as NaN or Infinity", nameof(weights));
        }
    }

    internal static void CheckForInvalidInputSize(int inputSize, int outputsLength)
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

    internal static void CheckForInvalidTrainingPctSize(double trainingPctSize)
    {
        if (trainingPctSize <= 0 || trainingPctSize >= 100)
        {
            throw new ArgumentException($"{nameof(trainingPctSize)} must be greater than 0 and less than 100", nameof(trainingPctSize));
        }
    }

    internal static void CheckForInvalidTrainingSizes(int trainingSize, int outOfSampleSize, int minSize, double trainingPctSize)
    {
        if (trainingSize < minSize)
        {
            throw new ArgumentException($"Training data must contain at least {minSize} values. " +
                                        $"You either need to increase your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }

        if (outOfSampleSize < minSize)
        {
            throw new ArgumentException($"Out of sample data must contain at least {minSize} values. " +
                                        $"You either need to decrease your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }
    }

    internal static void CheckForNaNOrInfinity(double[] preparedValues)
    {
        if (preparedValues.Contains(double.NaN))
        {
            throw new ArgumentException("Normalized Inputs can't contain NaN values. " +
                                        "Log Normalization creates NaN values when a raw input value is negative.", nameof(preparedValues));
        }

        if (preparedValues.Contains(double.PositiveInfinity) || preparedValues.Contains(double.NegativeInfinity))
        {
            throw new ArgumentException("Normalized Inputs can't contain Infinity values. " +
                                        "Log Normalization creates Infinity values when a raw input value is 0 or infinity.", nameof(preparedValues));
        }
    }

    internal static void CheckForNaNOrInfinity(double[][] preparedValues)
    {
        for (var i = 0; i < preparedValues.Length; i++)
        {
            var preparedValuesArray = preparedValues[i];

            CheckForNaNOrInfinity(preparedValuesArray);
        }
    }

    internal static void CheckForMinimumInputSize(int inputSize, int minimumSize)
    {
        if (inputSize < minimumSize)
        {
            throw new ArgumentException($"The length of the array is too small. Please make sure your array has at least {minimumSize} values.", nameof(inputSize));
        }
    }
}