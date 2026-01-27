namespace AiDotNet.FeatureSelectors;

/// <summary>
/// Provides common helper methods for feature selection algorithms.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class contains shared functionality that different feature selection methods
/// can use. It helps avoid duplicating code and keeps the feature selectors more focused on their
/// specific selection strategies rather than common data handling tasks.
/// </para>
/// </remarks>
public static class FeatureSelectorHelper<T, TInput>
{
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Extracts a feature vector from the input data.
    /// </summary>
    /// <param name="input">The input data containing all features.</param>
    /// <param name="featureIndex">The index of the feature to extract.</param>
    /// <param name="numSamples">The number of samples in the input data.</param>
    /// <param name="higherDimensionStrategy">Strategy to use for higher-dimensional data.</param>
    /// <param name="dimensionWeights">Weights to use for WeightedSum strategy.</param>
    /// <returns>A vector containing the values of the specified feature across all samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single column of data from your dataset.
    /// 
    /// Imagine your data as a spreadsheet with rows representing different examples (houses, patients, etc.)
    /// and columns representing different characteristics (price, size, age, etc.). This method
    /// pulls out one entire column so you can analyze just that specific characteristic.
    /// </para>
    /// </remarks>
    public static Vector<T> ExtractFeatureVector(
        TInput input,
        int featureIndex,
        int numSamples,
        FeatureExtractionStrategy higherDimensionStrategy,
        Dictionary<int, T> dimensionWeights)
    {
        var featureVector = new Vector<T>(numSamples);
        var numOps = MathHelper.GetNumericOperations<T>();

        if (input is Matrix<T> matrix)
        {
            return matrix.GetColumn(featureIndex);
        }
        else if (input is Tensor<T> tensor)
        {
            for (int i = 0; i < numSamples; i++)
            {
                if (tensor.Rank == 2)
                {
                    featureVector[i] = tensor[i, featureIndex];
                }
                else if (tensor.Rank > 2)
                {
                    // Handle higher-dimension tensors with configurable feature extraction strategy
                    featureVector[i] = higherDimensionStrategy switch
                    {
                        FeatureExtractionStrategy.Mean => CalculateMeanFeature(tensor, i, featureIndex),
                        FeatureExtractionStrategy.Max => CalculateMaxFeature(tensor, i, featureIndex),
                        FeatureExtractionStrategy.Flatten => GetFirstElement(tensor, i, featureIndex),
                        FeatureExtractionStrategy.WeightedSum => CalculateWeightedSum(tensor, i, featureIndex, dimensionWeights),
                        _ => throw new InvalidOperationException($"Unsupported feature extraction strategy: {higherDimensionStrategy}"),
                    };
                }
                else
                {
                    throw new ArgumentException("Tensor must have at least 2 dimensions to extract features");
                }
            }
            return featureVector;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {input?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Recursively traverses tensor dimensions to find the maximum value.
    /// </summary>
    /// <param name="tensor">The tensor to traverse.</param>
    /// <param name="indices">The current indices being processed.</param>
    /// <param name="dimension">The dimension currently being processed.</param>
    /// <param name="max">Running maximum value (passed by reference).</param>
    /// <param name="found">Flag indicating if at least one value has been found (passed by reference).</param>
    private static void CalculateMaxRecursive(Tensor<T> tensor, int[] indices, int dimension, ref T max, ref bool found)
    {
        if (dimension >= tensor.Rank)
        {
            // Base case: we've filled in all indices, so we can get the actual value
            T value = tensor[indices];
            if (!found || _numOps.GreaterThan(value, max))
            {
                max = value;
            }
            found = true;
            return;
        }

        // Recursive case: iterate through all values in the current dimension
        for (int i = 0; i < tensor.Shape[dimension]; i++)
        {
            indices[dimension] = i;
            CalculateMaxRecursive(tensor, indices, dimension + 1, ref max, ref found);
        }
    }

    /// <summary>
    /// Gets the first element for a specific feature and sample.
    /// </summary>
    /// <param name="tensor">The tensor containing the data.</param>
    /// <param name="sampleIndex">The index of the sample.</param>
    /// <param name="featureIndex">The index of the feature.</param>
    /// <returns>The first element representing the feature for the specified sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method uses the first value to represent a complex feature.
    /// 
    /// When a feature has multiple values (like different pixels in an image),
    /// sometimes using the first value is sufficient. This is a simpler approach
    /// that works well when the values are similar or when the first value is particularly important.
    /// </para>
    /// </remarks>
    private static T GetFirstElement(Tensor<T> tensor, int sampleIndex, int featureIndex)
    {
        if (tensor.Rank == 3)
        {
            return tensor[sampleIndex, featureIndex, 0];
        }
        else if (tensor.Rank == 4)
        {
            return tensor[sampleIndex, featureIndex, 0, 0];
        }
        else if (tensor.Rank > 4)
        {
            // For even higher dimensions, use array indexing with all remaining indices set to 0
            int[] indices = new int[tensor.Rank];
            indices[0] = sampleIndex;
            indices[1] = featureIndex;
            // All other indices remain 0
            return tensor[indices];
        }
        else
        {
            throw new InvalidOperationException($"Invalid tensor rank: {tensor.Rank}");
        }
    }

    /// <summary>
    /// Calculates a weighted sum of values across all dimensions for a specific feature and sample.
    /// </summary>
    /// <param name="tensor">The tensor containing the data.</param>
    /// <param name="sampleIndex">The index of the sample.</param>
    /// <param name="featureIndex">The index of the feature.</param>
    /// <param name="weights">The weights to apply to each dimension.</param>
    /// <returns>The weighted sum representing the feature for the specified sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method combines values with different levels of importance.
    /// 
    /// In complex data, some values might be more important than others. This method
    /// allows you to specify weights (importance levels) for different parts of your data.
    /// For example, in an image, central pixels might be given higher weight than edge pixels.
    /// </para>
    /// </remarks>
    private static T CalculateWeightedSum(Tensor<T> tensor, int sampleIndex, int featureIndex, Dictionary<int, T> weights)
    {
        if (weights == null || weights.Count == 0)
        {
            throw new InvalidOperationException("Weights must be provided for weighted sum feature extraction");
        }

        T sum = _numOps.Zero;
        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIndex;
        indices[1] = featureIndex;

        CalculateWeightedSumRecursive(tensor, indices, 2, weights, ref sum);

        return sum;
    }

    /// <summary>
    /// Recursively traverses tensor dimensions to calculate a weighted sum of elements.
    /// </summary>
    /// <param name="tensor">The tensor to traverse.</param>
    /// <param name="indices">The current indices being processed.</param>
    /// <param name="dimension">The dimension currently being processed.</param>
    /// <param name="weights">The weights to apply to each dimension.</param>
    /// <param name="sum">Running sum of weighted elements (passed by reference).</param>
    private static void CalculateWeightedSumRecursive(Tensor<T> tensor, int[] indices, int dimension, Dictionary<int, T> weights, ref T sum)
    {
        if (dimension >= tensor.Rank)
        {
            // Base case: we've filled in all indices, so we can get the actual value
            // Calculate a weight based on position in each dimension
            T weight = _numOps.One;
            for (int i = 2; i < indices.Length; i++)
            {
                int indexInDimension = indices[i];
                if (weights.TryGetValue(indexInDimension, out T? dimensionWeight))
                {
                    weight = _numOps.Multiply(weight, dimensionWeight);
                }
            }

            sum = _numOps.Add(sum, _numOps.Multiply(tensor[indices], weight));
            return;
        }

        // Recursive case: iterate through all values in the current dimension
        for (int i = 0; i < tensor.Shape[dimension]; i++)
        {
            indices[dimension] = i;
            CalculateWeightedSumRecursive(tensor, indices, dimension + 1, weights, ref sum);
        }
    }

    /// <summary>
    /// Finds the maximum value across all dimensions for a specific feature and sample.
    /// </summary>
    /// <param name="tensor">The tensor containing the data.</param>
    /// <param name="sampleIndex">The index of the sample.</param>
    /// <param name="featureIndex">The index of the feature.</param>
    /// <returns>The maximum value representing the feature for the specified sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds the highest value across complex data.
    /// 
    /// For complex data like images, this finds the strongest signal or highest intensity.
    /// For example, in an image, it might find the brightest pixel in a certain color channel.
    /// </para>
    /// </remarks>
    private static T CalculateMaxFeature(Tensor<T> tensor, int sampleIndex, int featureIndex)
    {
        T max = _numOps.MinValue;
        bool found = false;
        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIndex;
        indices[1] = featureIndex;

        CalculateMaxRecursive(tensor, indices, 2, ref max, ref found);

        if (!found)
        {
            throw new InvalidOperationException("No elements found when calculating max feature");
        }

        return max;
    }

    /// <summary>
    /// Creates a subset of features from the original input data.
    /// </summary>
    /// <param name="originalData">The original input data.</param>
    /// <param name="featureIndices">Indices of features to include in the subset.</param>
    /// <returns>A new input object containing only the specified features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a temporary working version of your dataset
    /// that includes only specific columns (features) for analysis.
    /// 
    /// Think of it like creating a smaller spreadsheet that only has the columns
    /// you're currently interested in. This makes it easier to analyze just those
    /// features without getting distracted by other data.
    /// </para>
    /// </remarks>
    public static TInput CreateFeatureSubset(
        TInput originalData,
        List<int> featureIndices)
    {
        // Handle different input types
        if (originalData is Matrix<T> matrix)
        {
            // Extract the selected columns as vectors
            var selectedColumns = featureIndices
                .Select(i => matrix.GetColumn(i))
                .ToArray();

            // Create a new matrix from the selected columns
            return (TInput)(object)Matrix<T>.FromColumns(selectedColumns);
        }
        else if (originalData is Tensor<T> tensor)
        {
            // For tensor inputs, create a new tensor with selected features
            int[] newShape = (int[])tensor.Shape.Clone();
            newShape[1] = featureIndices.Count; // Adjust feature dimension

            var resultTensor = new Tensor<T>(newShape);
            int numSamples = tensor.Shape[0];

            // Copy the selected features to the result tensor
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < featureIndices.Count; j++)
                {
                    CopyFeature(tensor, resultTensor, i, featureIndices[j], j);
                }
            }

            return (TInput)(object)resultTensor;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {originalData?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Creates a new data structure containing only the selected features.
    /// </summary>
    /// <param name="originalData">The original input data.</param>
    /// <param name="selectedFeatureIndices">The indices of features to include.</param>
    /// <returns>A new data structure with only the selected features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a simplified version of your dataset.
    /// 
    /// After determining which features are most important and independent, this method
    /// creates a new dataset that contains only those important features. Think of it as
    /// creating a streamlined version of your spreadsheet with only the most relevant columns.
    /// </para>
    /// </remarks>
    public static TInput CreateFilteredData(TInput originalData, List<int> selectedFeatureIndices)
    {
        if (originalData is Matrix<T> matrix)
        {
            // Handle empty matrix (0 rows) - return an empty matrix with the correct column count
            if (matrix.Rows == 0)
            {
                return (TInput)(object)new Matrix<T>(0, selectedFeatureIndices.Count);
            }

            var selectedColumns = selectedFeatureIndices
                .Select(i => matrix.GetColumn(i))
                .ToArray();

            return (TInput)(object)Matrix<T>.FromColumns(selectedColumns);
        }
        else if (originalData is Tensor<T> tensor)
        {
            // Create a new tensor with only the selected features
            int[] newShape = (int[])tensor.Shape.Clone();
            newShape[1] = selectedFeatureIndices.Count;

            var resultTensor = new Tensor<T>(newShape);

            // Copy the selected features to the result tensor
            for (int i = 0; i < tensor.Shape[0]; i++) // For each sample
            {
                for (int j = 0; j < selectedFeatureIndices.Count; j++) // For each selected feature
                {
                    CopyFeature(tensor, resultTensor, i, selectedFeatureIndices[j], j);
                }
            }

            return (TInput)(object)resultTensor;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {originalData?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Copies a feature from the source tensor to the destination tensor.
    /// </summary>
    /// <param name="source">The source tensor.</param>
    /// <param name="destination">The destination tensor.</param>
    /// <param name="sampleIndex">The sample index.</param>
    /// <param name="sourceFeatureIndex">The feature index in the source tensor.</param>
    /// <param name="destFeatureIndex">The feature index in the destination tensor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method copies data from one structure to another.
    /// 
    /// When creating a simplified dataset with only selected features, we need to copy
    /// the values from the original dataset to the new one. This method handles that
    /// copying process for different types of data structures.
    /// </para>
    /// </remarks>
    public static void CopyFeature(Tensor<T> source, Tensor<T> destination, int sampleIndex, int sourceFeatureIndex, int destFeatureIndex)
    {
        if (source.Rank == 2)
        {
            destination[sampleIndex, destFeatureIndex] = source[sampleIndex, sourceFeatureIndex];
        }
        else if (source.Rank == 3)
        {
            for (int k = 0; k < source.Shape[2]; k++)
            {
                destination[sampleIndex, destFeatureIndex, k] = source[sampleIndex, sourceFeatureIndex, k];
            }
        }
        else if (source.Rank == 4)
        {
            for (int k = 0; k < source.Shape[2]; k++)
            {
                for (int l = 0; l < source.Shape[3]; l++)
                {
                    destination[sampleIndex, destFeatureIndex, k, l] = source[sampleIndex, sourceFeatureIndex, k, l];
                }
            }
        }
        else
        {
            throw new NotSupportedException($"Feature copying for tensors with rank {source.Rank} is not currently supported");
        }
    }

    /// <summary>
    /// Recursively traverses tensor dimensions to calculate the sum and count of elements.
    /// </summary>
    /// <param name="tensor">The tensor to traverse.</param>
    /// <param name="indices">The current indices being processed.</param>
    /// <param name="dimension">The dimension currently being processed.</param>
    /// <param name="sum">Running sum of elements (passed by reference).</param>
    /// <param name="count">Running count of elements (passed by reference).</param>
    private static void CalculateMeanRecursive(Tensor<T> tensor, int[] indices, int dimension, ref T sum, ref int count)
    {
        if (dimension >= tensor.Rank)
        {
            // Base case: we've filled in all indices, so we can get the actual value
            sum = _numOps.Add(sum, tensor[indices]);
            count++;
            return;
        }

        // Recursive case: iterate through all values in the current dimension
        for (int i = 0; i < tensor.Shape[dimension]; i++)
        {
            indices[dimension] = i;
            CalculateMeanRecursive(tensor, indices, dimension + 1, ref sum, ref count);
        }
    }

    /// <summary>
    /// Calculates the mean value across all dimensions for a specific feature and sample.
    /// </summary>
    /// <param name="tensor">The tensor containing the data.</param>
    /// <param name="sampleIndex">The index of the sample.</param>
    /// <param name="featureIndex">The index of the feature.</param>
    /// <returns>The mean value representing the feature for the specified sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds the average value across complex data.
    /// 
    /// For complex data like images or time series, this calculates an average value
    /// that represents the feature. For example, in an image, it might calculate the
    /// average intensity of a certain color channel across all pixels.
    /// </para>
    /// </remarks>
    private static T CalculateMeanFeature(Tensor<T> tensor, int sampleIndex, int featureIndex)
    {
        // Use recursive traversal to handle any tensor rank
        T sum = _numOps.Zero;
        int count = 0;
        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIndex;
        indices[1] = featureIndex;

        CalculateMeanRecursive(tensor, indices, 2, ref sum, ref count);

        if (count == 0)
        {
            throw new InvalidOperationException("No elements found when calculating mean feature");
        }

        return _numOps.Divide(sum, _numOps.FromDouble(count));
    }

    // The remaining methods from CorrelationFeatureSelector would be moved here:
    // - CalculateMeanFeature
    // - CalculateMeanRecursive
    // - CalculateMaxFeature
    // - CalculateMaxRecursive
    // - GetFirstElement
    // - CalculateWeightedSum
    // - CalculateWeightedSumRecursive
    // - CopyFeature

    // For brevity, I'm not showing all implementations here, but they would be 
    // the same as in the CorrelationFeatureSelector class, just moved to this helper class
}
