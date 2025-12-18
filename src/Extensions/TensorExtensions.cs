namespace AiDotNet.Extensions;

public static class TensorExtensions
{
    /// <summary>
    /// Converts a tensor to a matrix for various calculations.
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    /// <returns>A matrix representation of the tensor.</returns>
    /// <remarks>
    /// Helper method for converting tensors to matrices for mathematical operations.
    /// </remarks>
    public static Matrix<T> ConvertToMatrix<T>(this Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // For 2D tensor, convert directly to matrix
        if (tensor.Rank == 2)
        {
            Matrix<T> matrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
            // Copy entire tensor data at once using vectorized operation
            numOps.Copy(tensor.AsSpan(), matrix.AsWritableSpan());
            return matrix;
        }
        // For vector, return as column matrix
        else if (tensor.Rank == 1)
        {
            Matrix<T> matrix = new Matrix<T>(tensor.Shape[0], 1);
            // Copy entire tensor data at once using vectorized operation
            numOps.Copy(tensor.AsSpan(), matrix.AsWritableSpan());
            return matrix;
        }
        // Handle other cases
        else
        {
            throw new ArgumentException("Tensor must be 1D or 2D for conversion to matrix.");
        }
    }

    /// <summary>
    /// Converts a flattened vector back into a tensor with the specified shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensor">The tensor to populate with values.</param>
    /// <param name="flattenedValues">The flattened vector containing values to reshape into the tensor.</param>
    /// <returns>The tensor populated with the values from the flattened vector.</returns>
    /// <remarks>
    /// <para>
    /// This method takes a flattened vector and reshapes it back into a tensor with the shape of the current tensor.
    /// It's the inverse operation of flattening a tensor into a vector.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a long line of numbers and arranging them back into a
    /// multi-dimensional structure (like rows and columns). It's similar to taking a string of text and
    /// formatting it back into paragraphs.
    /// </para>
    /// </remarks>
    public static Tensor<T> Unflatten<T>(this Tensor<T> tensor, Vector<T> flattenedValues)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (flattenedValues == null)
            throw new ArgumentNullException(nameof(flattenedValues));

        // Calculate the total size of the tensor
        int totalSize = tensor.Shape.Aggregate(1, (acc, dim) => acc * dim);

        if (flattenedValues.Length != totalSize)
            throw new ArgumentException($"The size of the flattened vector ({flattenedValues.Length}) does not match the tensor shape total size ({totalSize})");

        // Create a new tensor with the same shape
        var result = new Tensor<T>(tensor.Shape);

        // Copy the values from the flattened vector to the tensor
        int index = 0;
        result.ForEachPosition((position, _) =>
        {
            result[position] = flattenedValues[index++];
            return true;
        });

        return result;
    }

    /// <summary>
    /// Iterates through all positions in the tensor and applies a function to each position.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensor">The tensor to iterate through.</param>
    /// <param name="action">The action to apply at each position. Takes the position array and current value as parameters.
    /// Return true to continue iteration, false to stop.</param>
    /// <remarks>
    /// <para>
    /// This method provides a way to iterate through all positions in a tensor and perform an operation at each position.
    /// The action receives the current position (as an array of indices) and the current value at that position.
    /// </para>
    /// <para><b>For Beginners:</b> This is like visiting every cell in a spreadsheet one by one and performing
    /// some action at each cell. You can use this to transform, inspect, or extract data from the tensor.
    /// </para>
    /// </remarks>
    public static void ForEachPosition<T>(this Tensor<T> tensor, Func<int[], T, bool> action)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (action == null)
            throw new ArgumentNullException(nameof(action));

        // Create an array to hold the current position
        int[] position = new int[tensor.Shape.Length];

        // Start recursive iteration
        ForEachPositionRecursive(tensor, position, 0, action);
    }

    /// <summary>
    /// Helper method for recursive iteration through tensor positions.
    /// </summary>
    private static bool ForEachPositionRecursive<T>(Tensor<T> tensor, int[] position, int dimension, Func<int[], T, bool> action)
    {
        if (dimension == tensor.Shape.Length)
        {
            // We've reached a complete position, apply the action
            return action(position, tensor[position]);
        }
        else
        {
            // Iterate through the current dimension
            for (int i = 0; i < tensor.Shape[dimension]; i++)
            {
                position[dimension] = i;
                if (!ForEachPositionRecursive(tensor, position, dimension + 1, action))
                {
                    return false; // Stop iteration if the action returns false
                }
            }

            return true;
        }
    }

    public static bool TensorEquals<T>(this Tensor<T> a, Tensor<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (!a.Shape.SequenceEqual(b.Shape))
            return false;

        // Use spans for vectorized comparison
        ReadOnlySpan<T> spanA = a.AsSpan();
        ReadOnlySpan<T> spanB = b.AsSpan();

        for (int i = 0; i < spanA.Length; i++)
        {
            if (!numOps.Equals(spanA[i], spanB[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Concatenates two tensors along the last dimension.
    /// </summary>
    /// <param name="tensorA">The first tensor.</param>
    /// <param name="tensorB">The second tensor.</param>
    /// <returns>The concatenated tensor.</returns>
    public static Tensor<T> ConcatenateTensors<T>(this Tensor<T> tensorA, Tensor<T> tensorB)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Get shapes and verify they can be concatenated
        int[] shapeA = tensorA.Shape;
        int[] shapeB = tensorB.Shape;

        if (shapeA.Length != shapeB.Length)
        {
            throw new ArgumentException("Tensors must have the same number of dimensions to concatenate");
        }

        for (int i = 0; i < shapeA.Length - 1; i++)
        {
            if (shapeA[i] != shapeB[i])
            {
                throw new ArgumentException("All dimensions except the last must match for concatenation");
            }
        }

        // Calculate shape of result tensor
        int[] resultShape = new int[shapeA.Length];
        Array.Copy(shapeA, resultShape, shapeA.Length);
        resultShape[resultShape.Length - 1] = shapeA[shapeA.Length - 1] + shapeB[shapeB.Length - 1];

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Handle different dimensionality cases
        if (shapeA.Length == 2)
        {
            // 2D tensors (batch, features)
            int featuresA = shapeA[1];
            int featuresB = shapeB[1];
            ReadOnlySpan<T> spanA = tensorA.AsSpan();
            ReadOnlySpan<T> spanB = tensorB.AsSpan();
            Span<T> spanResult = result.AsWritableSpan();

            for (int b = 0; b < shapeA[0]; b++)
            {
                int offsetA = b * featuresA;
                int offsetB = b * featuresB;
                int offsetResult = b * (featuresA + featuresB);

                // Copy entire row from first tensor using vectorized operation
                numOps.Copy(spanA.Slice(offsetA, featuresA), spanResult.Slice(offsetResult, featuresA));

                // Copy entire row from second tensor using vectorized operation
                numOps.Copy(spanB.Slice(offsetB, featuresB), spanResult.Slice(offsetResult + featuresA, featuresB));
            }
        }
        else if (shapeA.Length == 3)
        {
            // 3D tensors (batch, sequence, features)
            int sequenceLen = shapeA[1];
            int featuresA = shapeA[2];
            int featuresB = shapeB[2];
            int strideA = sequenceLen * featuresA;
            int strideB = sequenceLen * featuresB;
            int strideResult = sequenceLen * (featuresA + featuresB);

            ReadOnlySpan<T> spanA = tensorA.AsSpan();
            ReadOnlySpan<T> spanB = tensorB.AsSpan();
            Span<T> spanResult = result.AsWritableSpan();

            for (int b = 0; b < shapeA[0]; b++)
            {
                for (int s = 0; s < sequenceLen; s++)
                {
                    int offsetA = b * strideA + s * featuresA;
                    int offsetB = b * strideB + s * featuresB;
                    int offsetResult = b * strideResult + s * (featuresA + featuresB);

                    // Copy entire feature vector from first tensor using vectorized operation
                    numOps.Copy(spanA.Slice(offsetA, featuresA), spanResult.Slice(offsetResult, featuresA));

                    // Copy entire feature vector from second tensor using vectorized operation
                    numOps.Copy(spanB.Slice(offsetB, featuresB), spanResult.Slice(offsetResult + featuresA, featuresB));
                }
            }
        }
        else
        {
            throw new NotSupportedException("Concatenation currently supports only 2D and 3D tensors");
        }

        return result;
    }

    /// <summary>
    /// Creates and initializes a tensor with Xavier/Glorot initialization using random values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="stddev">Standard deviation for the random initialization.</param>
    /// <param name="random">The random number generator to use.</param>
    /// <returns>A new tensor initialized with random values scaled by stddev.</returns>
    /// <remarks>
    /// <para>
    /// This method implements Xavier/Glorot initialization, which helps neural networks
    /// train more effectively by keeping the variance of activations consistent across layers.
    /// </para>
    /// <para><b>For Beginners:</b> When training neural networks, the initial values of weights
    /// matter a lot. Xavier initialization chooses random values that aren't too big or too small,
    /// helping the network learn faster and more reliably.
    /// </para>
    /// </remarks>
    public static Tensor<T> CreateXavierInitializedTensor<T>(int[] shape, double stddev, Random random)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    /// <summary>
    /// Creates a tensor initialized with all ones.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="size">The size of the 1D tensor to create.</param>
    /// <returns>A new 1D tensor filled with ones.</returns>
    /// <remarks>
    /// <para>
    /// This is commonly used for initializing layer normalization gamma parameters.
    /// </para>
    /// </remarks>
    public static Tensor<T> CreateOnesTensor<T>(int size)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            tensor[i] = numOps.One;
        }
        return tensor;
    }

    /// <summary>
    /// Calculates the He standard deviation for weight initialization with ReLU activations.
    /// </summary>
    /// <param name="fanIn">The number of input units.</param>
    /// <returns>The standard deviation to use for initialization.</returns>
    /// <remarks>
    /// <para>
    /// He initialization (Kaiming initialization) uses sqrt(2 / fanIn), which is designed
    /// for networks with ReLU activations. For sigmoid/tanh activations, consider using
    /// Xavier/Glorot initialization with sqrt(1 / fanIn) or sqrt(2 / (fanIn + fanOut)).
    /// </para>
    /// </remarks>
    public static double HeStddev(int fanIn)
    {
        return Math.Sqrt(2.0 / fanIn);
    }

    /// <summary>
    /// Calculates the Xavier/Glorot standard deviation for weight initialization.
    /// </summary>
    /// <param name="fanIn">The number of input units.</param>
    /// <param name="fanOut">The number of output units.</param>
    /// <returns>The standard deviation to use for initialization.</returns>
    /// <remarks>
    /// <para>
    /// Xavier/Glorot initialization uses sqrt(2 / (fanIn + fanOut)), which is designed
    /// to maintain gradient variance across layers with sigmoid/tanh activations.
    /// </para>
    /// </remarks>
    public static double XavierStddev(int fanIn, int fanOut)
    {
        return Math.Sqrt(2.0 / (fanIn + fanOut));
    }
}
