namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for neural network operations including activation functions and loss functions.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural networks are computing systems inspired by the human brain. They process information
/// through interconnected nodes (neurons) that transform input data using mathematical functions.
/// This helper class provides those mathematical functions needed to build neural networks.
/// </para>
/// </remarks>
public static class NeuralNetworkHelper<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A scalar value representing the Euclidean distance.</returns>
    /// <remarks>
    /// <para>For Beginners: Euclidean distance is the straight-line distance between two points in space.
    /// Think of it as measuring the length of a ruler placed between two points. This is used in many
    /// machine learning algorithms to measure how different two data points are.</para>
    /// </remarks>
    public static T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = _numOps.Subtract(v1[i], v2[i]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Sqrt(sum);
    }

    /// <summary>
    /// Gets the default loss function based on the task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An appropriate loss function for the task type.</returns>
    public static ILossFunction<T> GetDefaultLossFunction(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.BinaryClassification => new BinaryCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.MultiClassClassification => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.MultiLabelClassification => new BinaryCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.Regression => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.SequenceToSequence => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.SequenceClassification => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.TimeSeriesForecasting => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.ImageClassification => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.ObjectDetection => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.ImageSegmentation => new DiceLoss<T>(),
            NeuralNetworkTaskType.NaturalLanguageProcessing => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.TextGeneration => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.ReinforcementLearning => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.AnomalyDetection => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.Recommendation => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.Clustering => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.DimensionalityReduction => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.Generative => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.SpeechRecognition => new CategoricalCrossEntropyLoss<T>(),
            NeuralNetworkTaskType.AudioProcessing => new MeanSquaredErrorLoss<T>(),
            NeuralNetworkTaskType.Translation => new CategoricalCrossEntropyLoss<T>(),
            _ => new MeanSquaredErrorLoss<T>() // Default to MSE for Custom or unknown types
        };
    }

    /// <summary>
    /// Gets the default activation function based on the task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An appropriate activation function for the task type.</returns>
    public static IActivationFunction<T> GetDefaultActivationFunction(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.BinaryClassification => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.MultiClassClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.MultiLabelClassification => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.Regression => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SequenceToSequence => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SequenceClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.TimeSeriesForecasting => new IdentityActivation<T>(),
            NeuralNetworkTaskType.ImageClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.ObjectDetection => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.ImageSegmentation => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.NaturalLanguageProcessing => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.TextGeneration => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.ReinforcementLearning => new TanhActivation<T>(),
            NeuralNetworkTaskType.AnomalyDetection => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.Recommendation => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Clustering => new IdentityActivation<T>(),
            NeuralNetworkTaskType.DimensionalityReduction => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Generative => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SpeechRecognition => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.AudioProcessing => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Translation => new SoftmaxActivation<T>(),
            _ => new SigmoidActivation<T>() // Default to sigmoid for Custom or unknown types
        };
    }

    /// <summary>
    /// Gets the default vector activation function based on the task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An appropriate vector activation function for the task type.</returns>
    public static IVectorActivationFunction<T> GetDefaultVectorActivationFunction(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.BinaryClassification => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.MultiClassClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.MultiLabelClassification => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.Regression => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SequenceToSequence => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SequenceClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.TimeSeriesForecasting => new IdentityActivation<T>(),
            NeuralNetworkTaskType.ImageClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.ObjectDetection => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.ImageSegmentation => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.NaturalLanguageProcessing => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.TextGeneration => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.ReinforcementLearning => new TanhActivation<T>(),
            NeuralNetworkTaskType.AnomalyDetection => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.Recommendation => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Clustering => new IdentityActivation<T>(),
            NeuralNetworkTaskType.DimensionalityReduction => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Generative => new IdentityActivation<T>(),
            NeuralNetworkTaskType.SpeechRecognition => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.AudioProcessing => new IdentityActivation<T>(),
            NeuralNetworkTaskType.Translation => new SoftmaxActivation<T>(),
            _ => new SigmoidActivation<T>() // Default to sigmoid for Custom or unknown types
        };
    }

    /// <summary>
    /// Applies the appropriate activation function to the output tensor based on the task type.
    /// </summary>
    /// <param name="output">The output tensor to apply activation to.</param>
    public static void ApplyOutputActivation(Tensor<T> output, NeuralNetworkArchitecture<T> architecture)
    {
        // Apply appropriate activation based on task type
        switch (architecture.TaskType)
        {
            case NeuralNetworkTaskType.BinaryClassification:
                // Apply sigmoid to the output
                ApplySigmoid(output);
                break;

            case NeuralNetworkTaskType.MultiClassClassification:
            case NeuralNetworkTaskType.ImageClassification:
            case NeuralNetworkTaskType.SequenceClassification:
            case NeuralNetworkTaskType.NaturalLanguageProcessing:
            case NeuralNetworkTaskType.TextGeneration:
            case NeuralNetworkTaskType.SpeechRecognition:
            case NeuralNetworkTaskType.Translation:
                // Apply softmax to the output
                ApplySoftmax(output);
                break;

            case NeuralNetworkTaskType.MultiLabelClassification:
            case NeuralNetworkTaskType.ObjectDetection:
            case NeuralNetworkTaskType.ImageSegmentation:
            case NeuralNetworkTaskType.AnomalyDetection:
                // Apply sigmoid to each output independently
                ApplySigmoid(output);
                break;

            case NeuralNetworkTaskType.ReinforcementLearning:
                // Apply tanh activation
                ApplyTanh(output);
                break;

            // For regression tasks and others, usually no activation or linear activation
            case NeuralNetworkTaskType.Regression:
            case NeuralNetworkTaskType.TimeSeriesForecasting:
            case NeuralNetworkTaskType.Recommendation:
            case NeuralNetworkTaskType.Clustering:
            case NeuralNetworkTaskType.DimensionalityReduction:
            case NeuralNetworkTaskType.Generative:
            case NeuralNetworkTaskType.AudioProcessing:
            case NeuralNetworkTaskType.Custom:
            case NeuralNetworkTaskType.SequenceToSequence:
                // No activation (identity)
                break;

            default:
                // Default to no activation
                break;
        }
    }

    /// <summary>
    /// Applies the sigmoid activation function to all elements in a tensor.
    /// </summary>
    /// <param name="tensor">The tensor to apply sigmoid to.</param>
    private static void ApplySigmoid(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                double value = Convert.ToDouble(tensor[i, j]);
                tensor[i, j] = _numOps.FromDouble(1.0 / (1.0 + Math.Exp(-value)));
            }
        }
    }

    /// <summary>
    /// Applies the tanh activation function to all elements in a tensor.
    /// </summary>
    /// <param name="tensor">The tensor to apply tanh to.</param>
    private static void ApplyTanh(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                double value = Convert.ToDouble(tensor[i, j]);
                tensor[i, j] = _numOps.FromDouble(Math.Tanh(value));
            }
        }
    }

    /// <summary>
    /// Applies the softmax activation function to a tensor along the last dimension.
    /// </summary>
    /// <param name="tensor">The tensor to apply softmax to.</param>
    private static void ApplySoftmax(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            // Find maximum value for numerical stability
            T max = tensor[i, 0];
            for (int j = 1; j < tensor.Shape[1]; j++)
            {
                if (_numOps.GreaterThan(tensor[i, j], max))
                {
                    max = tensor[i, j];
                }
            }

            // Calculate exp of each element (shifted by max)
            T sum = _numOps.Zero;
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                T exp = _numOps.FromDouble(Math.Exp(Convert.ToDouble(_numOps.Subtract(tensor[i, j], max))));
                tensor[i, j] = exp;
                sum = _numOps.Add(sum, exp);
            }

            // Normalize by sum
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                tensor[i, j] = _numOps.Divide(tensor[i, j], sum);
            }
        }
    }

    /// <summary>
    /// Applies an activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector to apply the activation function to.</param>
    /// <param name="scalarActivation">An optional scalar activation function.</param>
    /// <param name="vectorActivation">An optional vector activation function.</param>
    /// <returns>A new vector with the activation function applied.</returns>
    /// <remarks>
    /// <para>
    /// This method applies either a vector activation function, a scalar activation function,
    /// or no activation (identity) to the input vector, based on the provided parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Activation functions are mathematical operations applied to the output
    /// of a neuron in a neural network. They introduce non-linearity, allowing the network to learn
    /// complex patterns. This method lets you apply different types of activation functions to a set of numbers:
    /// - Vector activation: applies the function to the whole set at once
    /// - Scalar activation: applies the function to each number individually
    /// - If no activation is specified, it returns the original numbers unchanged
    /// </para>
    /// </remarks>
    public static Vector<T> ApplyActivation(Vector<T> input, IActivationFunction<T>? scalarActivation = null, IVectorActivationFunction<T>? vectorActivation = null)
    {
        if (vectorActivation != null)
        {
            return vectorActivation.Activate(input);
        }
        else if (scalarActivation != null)
        {
            return input.Transform(scalarActivation.Activate);
        }
        else
        {
            return input; // Identity activation
        }
    }

    /// <summary>
    /// Applies an activation function to a tensor of values.
    /// </summary>
    /// <param name="input">The input tensor to apply the activation function to.</param>
    /// <param name="scalarActivation">An optional scalar activation function.</param>
    /// <param name="vectorActivation">An optional vector activation function.</param>
    /// <returns>A new tensor with the activation function applied.</returns>
    /// <exception cref="ArgumentException">Thrown when the input tensor is not rank-1 (vector).</exception>
    /// <remarks>
    /// <para>
    /// This method flattens the input tensor to a vector, applies the specified activation function,
    /// and then reconstructs the result as a tensor. It only works with rank-1 tensors (vectors).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is similar to the vector version above, but it works with
    /// tensors. A tensor is a more general mathematical object that can represent multi-dimensional data.
    /// In this case, we're working with a specific type of tensor that's essentially a vector (a list of numbers).
    /// The method converts the tensor to a vector, applies the activation function, and then converts it back to a tensor.
    /// </para>
    /// </remarks>
    public static Tensor<T> ApplyActivation(Tensor<T> input, IActivationFunction<T>? scalarActivation = null, IVectorActivationFunction<T>? vectorActivation = null)
    {
        if (input.Rank != 1)
            throw new ArgumentException("Input tensor must be rank-1 (vector).");

        Vector<T> inputVector = input.ToVector();
        Vector<T> outputVector = ApplyActivation(inputVector, scalarActivation, vectorActivation);

        return Tensor<T>.FromVector(outputVector);
    }
}
