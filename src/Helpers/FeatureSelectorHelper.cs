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
            if (numSamples == 0)
            {
                return featureVector;
            }

            if (tensor.Rank < 2)
            {
                throw new ArgumentException("Tensor must have at least 2 dimensions to extract features");
            }

            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var engine = AiDotNetEngine.Current;

            // [numSamples, d2, d3, ...]: this feature's values for the first numSamples samples
            var slab = engine.TensorNarrow(tensor.GetSliceAlongDimension(featureIndex, 1), 0, 0, numSamples);
            if (tensor.Rank == 2)
            {
                return slab.ToVector();
            }

            // Handle higher-dimension tensors with configurable feature extraction strategy
            return ReduceFeatureSlab(engine, slab, higherDimensionStrategy, dimensionWeights).ToVector();
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {input?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Collapses each sample's trailing dimensions of a [samples, d2, d3, ...] feature slab to one value
    /// with the configured strategy: mean, max, the first element, or a weighted sum whose weight at each
    /// position is the product of the per-dimension weights for its indices (1 where no weight is given).
    /// </summary>
    private static Tensor<T> ReduceFeatureSlab(
        IEngine engine,
        Tensor<T> slab,
        FeatureExtractionStrategy strategy,
        Dictionary<int, T> weights)
    {
        int samples = slab.Shape[0];
        int trailingRank = slab.Rank - 1;
        var axes = Enumerable.Range(1, trailingRank).ToArray();
        int trailingCount = slab.Length / Math.Max(samples, 1);
        bool hasTrailing = Enumerable.Range(1, trailingRank).All(d => slab.Shape[d] > 0);

        switch (strategy)
        {
            case FeatureExtractionStrategy.Mean:
                if (!hasTrailing)
                {
                    throw new InvalidOperationException("No elements found when calculating mean feature");
                }

                return engine.ReduceMean(slab, axes, false);

            case FeatureExtractionStrategy.Max:
                if (!hasTrailing)
                {
                    throw new InvalidOperationException("No elements found when calculating max feature");
                }

                return engine.ReduceMax(slab, axes, false);

            case FeatureExtractionStrategy.Flatten:
                return slab.Reshape(new[] { samples, trailingCount }).GetSliceAlongDimension(0, 1);

            case FeatureExtractionStrategy.WeightedSum:
            {
                if (weights == null || weights.Count == 0)
                {
                    throw new InvalidOperationException("Weights must be provided for weighted sum feature extraction");
                }

                T one = _numOps.One;
                Tensor<T>? positionWeights = null;
                for (int d = 1; d <= trailingRank; d++)
                {
                    var dimensionWeights = Tensor<T>.FromVector(new Vector<T>(Enumerable.Range(0, slab.Shape[d])
                        .Select(i => weights.TryGetValue(i, out var w) && w is not null ? w : one)));
                    positionWeights = positionWeights is null
                        ? dimensionWeights
                        : engine.TensorOuter(positionWeights, dimensionWeights).Reshape(new[] { positionWeights.Length * dimensionWeights.Length });
                }

                var trailingShape = new int[slab.Rank];
                trailingShape[0] = 1;
                for (int d = 1; d < slab.Rank; d++)
                {
                    trailingShape[d] = slab.Shape[d];
                }

                var fullWeights = positionWeights is null
                    ? new Tensor<T>(trailingShape)
                    : engine.TensorBroadcastTo(positionWeights.Reshape(trailingShape), slab._shape);
                return engine.ReduceSum(engine.TensorMultiply(slab, fullWeights), axes, false);
            }

            default:
                throw new InvalidOperationException($"Unsupported feature extraction strategy: {strategy}");
        }
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
            // For tensor inputs, create a new tensor with selected features.
            // Pre-fix this took `(int[])tensor._shape` which on TensorShape's
            // operator-cast returns a reference to the underlying array, then
            // `newShape[1] = ...` mutated the SOURCE tensor's shape — every
            // subsequent `tensor[i, j]` indexing failed because the tensor's
            // declared columns had shrunk to the subset count. Allocate a
            // fresh array and copy the dims to keep source untouched.
            int rank = tensor._shape.Length;
            int[] newShape = new int[rank];
            for (int d = 0; d < rank; d++) newShape[d] = tensor._shape[d];
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
            // Create a new tensor with only the selected features.
            // See CreateFeatureSubset for the rationale on copying the
            // shape array instead of casting (cast returns a reference
            // to the source tensor's array, mutation would corrupt the
            // source's declared dimensions).
            int rank = tensor._shape.Length;
            int[] newShape = new int[rank];
            for (int d = 0; d < rank; d++) newShape[d] = tensor._shape[d];
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
        else if (source.Rank == 3 || source.Rank == 4)
        {
            // Copy the whole [1, 1, d2(, d3)] block of this sample's feature in one slice.
            int rank = source.Rank;
            var start = new int[rank];
            var length = new int[rank];
            start[0] = sampleIndex;
            start[1] = sourceFeatureIndex;
            length[0] = 1;
            length[1] = 1;
            for (int d = 2; d < rank; d++)
            {
                length[d] = source.Shape[d];
            }

            Tensor<T> block;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                block = AiDotNetEngine.Current.TensorSlice(source, start, length);
            }

            var destStart = new int[rank];
            destStart[0] = sampleIndex;
            destStart[1] = destFeatureIndex;
            destination.SetSubTensor(destStart, block);
        }
        else
        {
            throw new NotSupportedException($"Feature copying for tensors with rank {source.Rank} is not currently supported");
        }
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
