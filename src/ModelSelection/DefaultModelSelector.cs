using System.Data;
using System.Linq;

namespace AiDotNet.ModelSelection;

/// <summary>
/// Default implementation of the IModelSelector interface that analyzes data characteristics
/// to recommend appropriate machine learning models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class provides a standard way to analyze your data and recommend
/// appropriate models. It examines characteristics like whether your data is tabular or time-series,
/// whether you're trying to predict categories or numeric values, and the size and complexity
/// of your dataset to suggest models that might work well.
/// </remarks>
public class DefaultModelSelector<T, TInput, TOutput> : IModelSelector<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Analyzes input and output data to automatically select the most appropriate model.
    /// </summary>
    /// <param name="sampleX">A sample of the input data to analyze its structure.</param>
    /// <param name="sampleY">A sample of the output data to analyze its structure.</param>
    /// <returns>The selected model instance.</returns>
    public IFullModel<T, TInput, TOutput> SelectModel(TInput sampleX, TOutput sampleY)
    {
        // Analyze input structure
        bool isTimeSeries = IsTimeSeriesData(sampleX);
        bool isImageData = IsImageData(sampleX);
        bool isTabularData = IsTabularData(sampleX);

        // Analyze output structure
        bool isCategorical = IsCategoricalData(sampleY);
        bool isMultiOutput = IsMultiOutputData(sampleY);
        int outputDimension = GetOutputDimension(sampleY);

        // Select appropriate model based on data characteristics
        IFullModel<T, TInput, TOutput>? model = null;

        if (isTimeSeries)
        {
            if (isCategorical)
            {
                //model = new TimeSeriesClassification<T, TInput, TOutput>();
            }
            else
            {
                model = (IFullModel<T, TInput, TOutput>)new TimeSeriesRegression<T>();
            }
        }
        else if (isImageData)
        {
            if (isCategorical)
            {
                // Determine image dimensions and output characteristics
                int inputHeight = GetImageHeight(sampleX);
                int inputWidth = GetImageWidth(sampleX);
                int inputChannels = GetImageChannels(sampleX);
                int outputFeatures = GetOutputDimension(sampleY);

                // Create a properly configured CNN
                model = (IFullModel<T, TInput, TOutput>)new ConvolutionalNeuralNetwork<T>(new NeuralNetworkArchitecture<T>(
                    taskType: isCategorical ? NeuralNetworkTaskType.MultiClassClassification : NeuralNetworkTaskType.Regression,
                    complexity: NetworkComplexity.Medium,
                    shouldReturnFullSequence: false
                ));
            }
            else
            {
                model = (IFullModel<T, TInput, TOutput>)new NeuralNetworkRegression<T>();
            }
        }
        else if (isTabularData)
        {
            if (isCategorical)
            {
                if (outputDimension > 2) // Multi-class classification
                {
                    model = (IFullModel<T, TInput, TOutput>)new RandomForestRegression<T>();
                }
                else // Binary classification
                {
                    model = (IFullModel<T, TInput, TOutput>)new LogisticRegression<T>();
                }
            }
            else
            {
                if (isMultiOutput)
                {
                    model = (IFullModel<T, TInput, TOutput>)new MultipleRegression<T>();
                }
                else
                {
                    // Analyze feature dimensionality to choose between linear and non-linear
                    int featureDimension = GetFeatureDimension(sampleX);
                    if (featureDimension > 100) // High-dimensional data
                    {
                        model = (IFullModel<T, TInput, TOutput>)new MultipleRegression<T>(new RegressionOptions<T>(),
                            RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(new RegularizationOptions()));
                    }
                    else if (featureDimension > 10) // Medium-dimensional data
                    {
                        model = (IFullModel<T, TInput, TOutput>)new GradientBoostingRegression<T>();
                    }
                    else // Low-dimensional data
                    {
                        model = (IFullModel<T, TInput, TOutput>)new MultipleRegression<T>();
                    }
                }
            }
        }
        else
        {
            // Default to a general-purpose model if we can't categorize the data
            model = (IFullModel<T, TInput, TOutput>)new RandomForestRegression<T>();
        }

        return model ?? (IFullModel<T, TInput, TOutput>)new RandomForestRegression<T>();
    }

    /// <summary>
    /// Analyzes input and output data and provides a ranked list of recommended models with explanations.
    /// </summary>
    /// <param name="sampleX">A sample of the input data to analyze its structure.</param>
    /// <param name="sampleY">A sample of the output data to analyze its structure.</param>
    /// <returns>A ranked list of model recommendations with explanations.</returns>
    public List<ModelRecommendation<T, TInput, TOutput>> GetModelRecommendations(TInput sampleX, TOutput sampleY)
    {
        var recommendations = new List<ModelRecommendation<T, TInput, TOutput>>();

        // Analyze data characteristics
        bool isTimeSeries = IsTimeSeriesData(sampleX);
        bool isImageData = IsImageData(sampleX);
        bool isTabularData = IsTabularData(sampleX);
        bool isCategorical = IsCategoricalData(sampleY);
        bool isMultiOutput = IsMultiOutputData(sampleY);
        int featureDimension = GetFeatureDimension(sampleX);
        int sampleCount = GetSampleCount(sampleX);

        // Generate recommendations based on data characteristics
        if (isTabularData && !isCategorical)
        {
            // Regression task with tabular data
            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "Linear Regression",
                "Best for simple relationships with few features. Fast to train and easy to interpret.",
                90,
                () => (IFullModel<T, TInput, TOutput>)new MultipleRegression<T>()
            ));

            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "Gradient Boosted Trees",
                "Good for complex non-linear relationships. Can handle mixed data types and missing values.",
                85,
                () => (IFullModel<T, TInput, TOutput>)new GradientBoostingRegression<T>()
            ));

            if (featureDimension > 50)
            {
                recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                    "Ridge Regression",
                    "Good for high-dimensional data with potential multicollinearity. Helps prevent overfitting.",
                    80,
                    () => (IFullModel<T, TInput, TOutput>)new MultipleRegression<T>(new RegressionOptions<T>(), 
                    RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(new RegularizationOptions()))
                ));
            }

            if (sampleCount > 1000)
            {
                recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                    "Neural Network Regression",
                    "Good for very complex relationships with large amounts of data. Requires more tuning.",
                    75,
                    () => (IFullModel<T, TInput, TOutput>)new NeuralNetworkRegression<T>()
                ));
            }
        }
        else if (isTabularData && isCategorical)
        {
            // Classification task with tabular data
            int outputDimension = GetOutputDimension(sampleY);

            if (outputDimension <= 2) // Binary classification
            {
                recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                    "Logistic Regression",
                    "Simple and interpretable model for binary classification. Works well with linearly separable data.",
                    90,
                    () => (IFullModel<T, TInput, TOutput>)new LogisticRegression<T>()
                ));
            }

            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "Random Forest",
                "Versatile ensemble method. Handles non-linear relationships and is resistant to overfitting.",
                85,
                () => (IFullModel<T, TInput, TOutput>)new RandomForestRegression<T>()
            ));

            /*
            if (sampleCount > 1000)
            {
                recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                    "Neural Network Classification",
                    "Powerful for complex decision boundaries with sufficient data. Requires more tuning.",
                    80,
                    () => new NeuralNetworkClassification<T, TInput, TOutput>()
                ));
            }
            */
        }
        else if (isTimeSeries)
        {
            // Time series data
            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "ARIMA",
                "Classical time series model good for data with clear trends and seasonality.",
                85,
                () => (IFullModel<T, TInput, TOutput>)new ARIMAModel<T>()
            ));

            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "Long Short-Term Memory (LSTM)",
                "Neural network designed for sequential data. Good for complex patterns and long-term dependencies.",
                80,
                () => {
                    // Determine appropriate input and output sizes from the sample data
                    int inputFeatures = GetFeatureDimension(sampleX);
                    int outputFeatures = GetOutputDimension(sampleY);

                    // Create a properly configured LSTM network
                    return (IFullModel<T, TInput, TOutput>)new LSTMNeuralNetwork<T>(new NeuralNetworkArchitecture<T>(
                        taskType: isCategorical ? NeuralNetworkTaskType.MultiClassClassification : NeuralNetworkTaskType.Regression,
                        complexity: NetworkComplexity.Medium,
                        shouldReturnFullSequence: false
                    ), outputActivation: null as IActivationFunction<T>);
                }
            ));
        }
        else if (isImageData)
        {
            // Image data
            recommendations.Add(new ModelRecommendation<T, TInput, TOutput>(
                "Convolutional Neural Network",
                "Specialized neural network designed for image data. State-of-the-art for many image tasks.",
                95,
                () => {
                    // Determine image dimensions and output characteristics
                    int inputHeight = GetImageHeight(sampleX);
                    int inputWidth = GetImageWidth(sampleX);
                    int inputChannels = GetImageChannels(sampleX);
                    int outputFeatures = GetOutputDimension(sampleY);

                    // Create a properly configured CNN
                    return (IFullModel<T, TInput, TOutput>)new ConvolutionalNeuralNetwork<T>(new NeuralNetworkArchitecture<T>(
                        taskType: isCategorical ? NeuralNetworkTaskType.MultiClassClassification : NeuralNetworkTaskType.Regression,
                        complexity: NetworkComplexity.Medium,
                        shouldReturnFullSequence: false
                    ));
                }
            ));
        }

        // Sort recommendations by confidence score (descending)
        return [.. recommendations.OrderByDescending(r => r.ConfidenceScore)];
    }

    /// <summary>
    /// Gets the height of the image data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>The height of the image in pixels.</returns>
    /// <remarks>
    /// <para>
    /// This method analyzes the input data structure to determine the height dimension of image data.
    /// It handles various common image data representations including tensors, matrices, and arrays.
    /// </para>
    /// <para>
    /// For 3D tensors, it assumes the shape is [channels, height, width] or [height, width, channels].
    /// For 4D tensors, it assumes the shape is [batch, channels, height, width] or [batch, height, width, channels].
    /// </para>
    /// </remarks>
    protected virtual int GetImageHeight(TInput data)
    {
        // Handle different input types
        if (data is Tensor<T> tensor)
        {
            var shape = tensor.Shape;

            // Handle different tensor dimensions
            switch (shape.Length)
            {
                case 2: // [height, width]
                    return shape[0];

                case 3: // [channels, height, width] or [height, width, channels]
                        // Heuristic: If first dimension is small (1-4), it's likely channels, so height is second dimension
                    return shape[0] <= 4 ? shape[1] : shape[0];

                case 4: // [batch, channels, height, width] or [batch, height, width, channels]
                        // Heuristic: If second dimension is small (1-4), it's likely channels, so height is third dimension
                    return shape[1] <= 4 ? shape[2] : shape[1];

                default:
                    throw new InvalidOperationException("Cannot determine image height from tensor with unexpected dimensions");
            }
        }
        else if (data is Matrix<T>[,] matrixArray) // Multi-channel image as array of matrices
        {
            // Height is the number of rows in each matrix
            if (matrixArray.Length > 0)
            {
                // Find the first non-null matrix and return its row count
                for (int i = 0; i < matrixArray.GetLength(0); i++)
                {
                    for (int j = 0; j < matrixArray.GetLength(1); j++)
                    {
                        if (matrixArray[i, j] != null)
                        {
                            return matrixArray[i, j].Rows;
                        }
                    }
                }
                // If we've gone through the entire array and found no non-null matrices
                throw new InvalidOperationException("No valid matrices found in the array");
            }
            throw new InvalidOperationException("Empty matrix array");
        }
        else if (data is Matrix<T>[] matrixList) // Another multi-channel representation
        {
            if (matrixList.Length > 0 && matrixList[0] != null)
            {
                return matrixList[0].Rows;
            }
            throw new InvalidOperationException("Empty or null matrix list");
        }
        else if (data is Matrix<T> singleMatrix) // Single channel/grayscale
        {
            return singleMatrix.Rows;
        }
        else if (data is T[,,] array3D)
        {
            // Determine which dimension is height based on channel position
            int dim0 = array3D.GetLength(0);

            // Heuristic: If first dimension is small (1-4), it's likely channels, so height is second dimension
            return dim0 <= 4 ? array3D.GetLength(1) : array3D.GetLength(0);
        }
        else if (data is T[,,,] array4D)
        {
            // Similar logic for 4D arrays
            int dim1 = array4D.GetLength(1);

            // Heuristic: If second dimension is small (1-4), it's likely channels, so height is third dimension
            return dim1 <= 4 ? array4D.GetLength(2) : array4D.GetLength(1);
        }

        // Try to infer from data properties if available
        try
        {
            var properties = data?.GetType().GetProperties();
            var heightProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Height", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ImageHeight", StringComparison.OrdinalIgnoreCase));

            if (heightProp != null)
            {
                var value = heightProp.GetValue(data);
                if (value is int height)
                {
                    return height;
                }
            }

            // Try to infer from metadata if available
            var metadataProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Metadata", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ImageInfo", StringComparison.OrdinalIgnoreCase));

            if (metadataProp != null)
            {
                var metadata = metadataProp.GetValue(data);
                if (metadata != null)
                {
                    var metaProps = metadata.GetType().GetProperties();
                    var metaHeightProp = metaProps.FirstOrDefault(p =>
                        p.Name.Equals("Height", StringComparison.OrdinalIgnoreCase));

                    if (metaHeightProp != null)
                    {
                        var value = metaHeightProp.GetValue(metadata);
                        if (value is int height)
                        {
                            return height;
                        }
                    }
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors and fall back to default
        }

        // Default to a standard image size if we can't determine
        return 224; // Common default size for many image models
    }

    /// <summary>
    /// Gets the width of the image data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>The width of the image in pixels.</returns>
    /// <remarks>
    /// <para>
    /// This method analyzes the input data structure to determine the width dimension of image data.
    /// It handles various common image data representations including tensors, matrices, and arrays.
    /// </para>
    /// <para>
    /// For 3D tensors, it assumes the shape is [channels, height, width] or [height, width, channels].
    /// For 4D tensors, it assumes the shape is [batch, channels, height, width] or [batch, height, width, channels].
    /// </para>
    /// </remarks>
    protected virtual int GetImageWidth(TInput data)
    {
        // Handle different input types
        if (data is Tensor<T> tensor)
        {
            var shape = tensor.Shape;

            // Handle different tensor dimensions
            switch (shape.Length)
            {
                case 2: // [height, width]
                    return shape[1];

                case 3: // [channels, height, width] or [height, width, channels]
                        // Heuristic: If first dimension is small (1-4), it's likely channels, so width is third dimension
                    return shape[0] <= 4 ? shape[2] : shape[1];

                case 4: // [batch, channels, height, width] or [batch, height, width, channels]
                        // Heuristic: If second dimension is small (1-4), it's likely channels, so width is fourth dimension
                    return shape[1] <= 4 ? shape[3] : shape[2];

                default:
                    throw new InvalidOperationException("Cannot determine image width from tensor with unexpected dimensions");
            }
        }
        else if (data is Matrix<T>[,] matrixArray) // Multi-channel image as array of matrices
        {
            // Width is the number of columns in each matrix
            if (matrixArray.Length > 0)
            {
                // Find the first non-null matrix and return its column count
                for (int i = 0; i < matrixArray.GetLength(0); i++)
                {
                    for (int j = 0; j < matrixArray.GetLength(1); j++)
                    {
                        if (matrixArray[i, j] != null)
                        {
                            return matrixArray[i, j].Columns;
                        }
                    }
                }
                // If we've gone through the entire array and found no non-null matrices
                throw new InvalidOperationException("No valid matrices found in the array");
            }
            throw new InvalidOperationException("Empty matrix array");
        }
        else if (data is Matrix<T>[] matrixList) // Another multi-channel representation
        {
            if (matrixList.Length > 0 && matrixList[0] != null)
            {
                return matrixList[0].Columns;
            }
            throw new InvalidOperationException("Empty or null matrix list");
        }
        else if (data is Matrix<T> singleMatrix) // Single channel/grayscale
        {
            return singleMatrix.Columns;
        }
        else if (data is T[,,] array3D)
        {
            // Determine which dimension is width based on channel position
            int dim0 = array3D.GetLength(0);

            // Heuristic: If first dimension is small (1-4), it's likely channels, so width is third dimension
            return dim0 <= 4 ? array3D.GetLength(2) : array3D.GetLength(1);
        }
        else if (data is T[,,,] array4D)
        {
            // Similar logic for 4D arrays
            int dim1 = array4D.GetLength(1);

            // Heuristic: If second dimension is small (1-4), it's likely channels, so width is fourth dimension
            return dim1 <= 4 ? array4D.GetLength(3) : array4D.GetLength(2);
        }

        // Try to infer from data properties if available
        try
        {
            var properties = data?.GetType().GetProperties();
            var widthProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Width", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ImageWidth", StringComparison.OrdinalIgnoreCase));

            if (widthProp != null)
            {
                var value = widthProp.GetValue(data);
                if (value is int width)
                {
                    return width;
                }
            }

            // Try to infer from metadata if available
            var metadataProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Metadata", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ImageInfo", StringComparison.OrdinalIgnoreCase));

            if (metadataProp != null)
            {
                var metadata = metadataProp.GetValue(data);
                if (metadata != null)
                {
                    var metaProps = metadata.GetType().GetProperties();
                    var metaWidthProp = metaProps.FirstOrDefault(p =>
                        p.Name.Equals("Width", StringComparison.OrdinalIgnoreCase));

                    if (metaWidthProp != null)
                    {
                        var value = metaWidthProp.GetValue(metadata);
                        if (value is int width)
                        {
                            return width;
                        }
                    }
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors and fall back to default
        }

        // Default to a standard image size if we can't determine
        return 224; // Common default size for many image models
    }

    /// <summary>
    /// Gets the number of channels in the image data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>The number of channels in the image data (typically 1 for grayscale, 3 for RGB, or 4 for RGBA).</returns>
    /// <remarks>
    /// <para>
    /// This method analyzes the input data structure to determine the number of color channels.
    /// It handles various common image data representations including tensors, matrices, and arrays.
    /// </para>
    /// <para>
    /// For 3D tensors, it assumes the shape is [channels, height, width] or [height, width, channels].
    /// For 4D tensors, it assumes the shape is [batch, channels, height, width] or [batch, height, width, channels].
    /// </para>
    /// </remarks>
    protected virtual int GetImageChannels(TInput data)
    {
        // Handle different input types
        if (data is Tensor<T> tensor)
        {
            var shape = tensor.Shape;

            // Handle different tensor dimensions
            switch (shape.Length)
            {
                case 3: // [channels, height, width] or [height, width, channels]
                        // Heuristic: If first dimension is small (1-4), it's likely channels
                    return shape[0] <= 4 ? shape[0] : shape[2];

                case 4: // [batch, channels, height, width] or [batch, height, width, channels]
                        // Heuristic: If second dimension is small (1-4), it's likely channels
                    return shape[1] <= 4 ? shape[1] : shape[3];

                default:
                    // For other dimensions, try to infer from data range
                    return InferChannelsFromDataRange(tensor);
            }
        }
        else if (data is Matrix<T>[,] matrixArray) // Possible representation for multi-channel images
        {
            return matrixArray.GetLength(0);
        }
        else if (data is Matrix<T>[] matrixList) // Another possible representation
        {
            return matrixList.Length;
        }
        else if (data is Matrix<T> singleMatrix) // Likely grayscale
        {
            // Check if values are normalized (0-1) or standard (0-255)
            // If all values are <= 1.0, it's likely normalized grayscale
            return 1;
        }
        else if (data is T[,,] array3D)
        {
            // Check array dimensions to determine which is the channel dimension
            int dim0 = array3D.GetLength(0);
            int dim2 = array3D.GetLength(2);

            // Heuristic: If first dimension is small (1-4), it's likely channels
            return dim0 <= 4 ? dim0 : dim2;
        }
        else if (data is T[,,,] array4D)
        {
            // Similar logic for 4D arrays
            int dim1 = array4D.GetLength(1);
            int dim3 = array4D.GetLength(3);

            return dim1 <= 4 ? dim1 : dim3;
        }

        // If we can't determine channels, use a safe default
        try
        {
            // Try to infer from data properties if available
            var properties = data?.GetType().GetProperties();
            var channelsProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Channels", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("NumChannels", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ColorChannels", StringComparison.OrdinalIgnoreCase));

            if (channelsProp != null)
            {
                var value = channelsProp.GetValue(data);
                if (value is int channels)
                {
                    return channels;
                }
            }

            // Try to infer from metadata if available
            var metadataProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("Metadata", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ImageInfo", StringComparison.OrdinalIgnoreCase));

            if (metadataProp != null)
            {
                var metadata = metadataProp.GetValue(data);
                if (metadata != null)
                {
                    var metaProps = metadata.GetType().GetProperties();
                    var metaChannelsProp = metaProps.FirstOrDefault(p =>
                        p.Name.Equals("Channels", StringComparison.OrdinalIgnoreCase) ||
                        p.Name.Equals("NumChannels", StringComparison.OrdinalIgnoreCase));

                    if (metaChannelsProp != null)
                    {
                        var value = metaChannelsProp.GetValue(metadata);
                        if (value is int channels)
                        {
                            return channels;
                        }
                    }
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors and fall back to default
        }

        // Default to grayscale (1 channel) if we can't determine
        return 1;
    }

    /// <summary>
    /// Infers the number of channels by analyzing the data range and distribution.
    /// </summary>
    /// <param name="tensor">The tensor to analyze.</param>
    /// <returns>The inferred number of channels.</returns>
    private int InferChannelsFromDataRange(Tensor<T> tensor)
    {
        try
        {
            // Sample some values to check if they match typical image patterns
            var sample = tensor.GetSample(100);

            // Check if values are binary (0/1) - likely masks
            if (sample.All(v => MathHelper.AlmostEqual<T>(v, _numOps.Zero) || MathHelper.AlmostEqual<T>(v, _numOps.One)))
            {
                return 1; // Binary mask (grayscale)
            }

            // Check if values are in [0-255] range - standard image encoding
            if (sample.All(v => Convert.ToDouble(v) >= 0 && Convert.ToDouble(v) <= 255))
            {
                // Further analysis could distinguish between grayscale/RGB
                // For now, default to RGB as it's more common for standard encoding
                return 3;
            }

            // Check if values are normalized [0-1] range
            if (sample.All(v => Convert.ToDouble(v) >= 0 && Convert.ToDouble(v) <= 1))
            {
                // Normalized images could be either grayscale or RGB
                // Default to RGB as it's more common
                return 3;
            }
        }
        catch (Exception)
        {
            // If analysis fails, fall back to default
        }

        // Default to RGB (3 channels) as a safe assumption for unknown image data
        return 3;
    }

    /// <summary>
    /// Determines if the input data appears to be time series data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>True if the data appears to be time series data, false otherwise.</returns>
    protected virtual bool IsTimeSeriesData(TInput data)
    {
        // Check if data has time-related properties
        try
        {
            var properties = data?.GetType().GetProperties();

            // Look for time-related property names
            var timeProps = properties?.Where(p =>
                p.Name.Contains("Time") ||
                p.Name.Contains("Date") ||
                p.Name.Contains("Timestamp") ||
                p.Name.Contains("Period") ||
                p.Name.Contains("Sequence"));

            if (timeProps?.Any() == true)
            {
                return true;
            }

            // Check for sequential structure in tensors
            if (data is Tensor<T> tensor && tensor.Shape.Length >= 2)
            {
                // Time series data often has shape [batch_size, sequence_length, ...]
                return tensor.Shape.Length >= 2 && tensor.Shape[1] > 1;
            }

            // Check for array of vectors/matrices which could represent time steps
            if (data is Vector<T>[] || data is Matrix<T>[])
            {
                return true;
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        return false;
    }

    /// <summary>
    /// Determines if the input data appears to be image data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>True if the data appears to be image data, false otherwise.</returns>
    protected virtual bool IsImageData(TInput data)
    {
        // Check if data is explicitly marked as image
        try
        {
            var properties = data?.GetType().GetProperties();
            var imageProps = properties?.Where(p =>
                p.Name.Contains("Image") ||
                p.Name.Contains("Picture") ||
                p.Name.Contains("Photo") ||
                p.Name.Contains("Pixel"));

            if (imageProps?.Any() == true)
            {
                return true;
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // Check data structure
        try
        {
            // Check for tensor with appropriate dimensions for images
            if (data is Tensor<T> tensor)
            {
                // Images typically have 3-4 dimensions: [batch, height, width, channels] or [batch, channels, height, width]
                if (tensor.Shape.Length == 3 || tensor.Shape.Length == 4)
                {
                    // Check if dimensions are reasonable for images
                    // Most images have height and width between 16 and 4096 pixels
                    int height = GetImageHeight(data);
                    int width = GetImageWidth(data);
                    int channels = GetImageChannels(data);

                    bool validDimensions = height >= 16 && height <= 4096 &&
                                          width >= 16 && width <= 4096;
                    bool validChannels = channels >= 1 && channels <= 4;

                    return validDimensions && validChannels;
                }
            }

            // Check for multi-channel matrix representations
            if (data is Matrix<T>[,] || data is Matrix<T>[])
            {
                return true;
            }

            // Check for 3D or 4D arrays
            if (data is T[,,] || data is T[,,,])
            {
                return true;
            }
        }
        catch (Exception)
        {
            // If analysis fails, assume it's not image data
        }

        return false;
    }

    /// <summary>
    /// Determines if the input data appears to be tabular data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>True if the data appears to be tabular data, false otherwise.</returns>
    protected virtual bool IsTabularData(TInput data)
    {
        // Check for matrix structure which often represents tabular data
        if (data is Matrix<T>)
        {
            return true;
        }

        // Check for 2D array structure
        if (data is T[,])
        {
            return true;
        }

        // Check for array of vectors (each vector could be a row)
        if (data is Vector<T>[] vectors && vectors.Length > 0)
        {
            return true;
        }

        // Check for dictionary with string keys (column names) and vector values
        if (data is Dictionary<string, Vector<T>>)
        {
            return true;
        }

        // If it's not time series or image data, it's likely tabular
        if (!IsTimeSeriesData(data) && !IsImageData(data))
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Determines if the output data appears to be categorical.
    /// </summary>
    /// <param name="data">The output data to analyze.</param>
    /// <returns>True if the data appears to be categorical, false otherwise.</returns>
    protected virtual bool IsCategoricalData(TOutput data)
    {
        try
        {
            // Check if it's a vector with discrete values
            if (data is Vector<T> vector)
            {
                // Get a sample of values to analyze
                var distinctValues = new HashSet<T>();
                int sampleSize = Math.Min(vector.Length, 1000);

                for (int i = 0; i < sampleSize; i++)
                {
                    distinctValues.Add(vector[i]);
                }

                // If there are few distinct values relative to the sample size, likely categorical
                double distinctRatio = (double)distinctValues.Count / sampleSize;

                // Check if values are integers or close to integers
                bool allIntegers = true;
                foreach (var val in distinctValues)
                {
                    double dVal = Convert.ToDouble(val);
                    if (Math.Abs(dVal - Math.Round(dVal)) > 1e-10)
                    {
                        allIntegers = false;
                        break;
                    }
                }

                // Categorical data typically has few distinct values and they're often integers
                return (distinctRatio < 0.1 || distinctValues.Count < 10) && allIntegers;
            }

            // Check for one-hot encoded data
            if (data is Matrix<T> matrix)
            {
                // One-hot encoding typically has exactly one 1 per row and the rest 0s
                bool couldBeOneHot = true;
                int sampleRows = Math.Min(matrix.Rows, 100);

                for (int i = 0; i < sampleRows && couldBeOneHot; i++)
                {
                    int onesCount = 0;
                    for (int j = 0; j < matrix.Columns; j++)
                    {
                        if (MathHelper.AlmostEqual(matrix[i, j], _numOps.One))
                        {
                            onesCount++;
                        }
                        else if (!MathHelper.AlmostEqual(matrix[i, j], _numOps.Zero))
                        {
                            couldBeOneHot = false;
                            break;
                        }
                    }

                    if (onesCount != 1)
                    {
                        couldBeOneHot = false;
                    }
                }

                return couldBeOneHot;
            }
        }
        catch (Exception)
        {
            // Ignore analysis errors
        }

        return false;
    }

    /// <summary>
    /// Determines if the output data has multiple output variables.
    /// </summary>
    /// <param name="data">The output data to analyze.</param>
    /// <returns>True if the data appears to have multiple output variables, false otherwise.</returns>
    protected virtual bool IsMultiOutputData(TOutput data)
    {
        // Check if it's a matrix (multiple columns could represent multiple outputs)
        if (data is Matrix<T> matrix && matrix.Columns > 1)
        {
            // If it's not one-hot encoded (categorical), it's likely multi-output
            return !IsCategoricalData(data);
        }

        // Check if it's a collection of vectors
        if (data is Vector<T>[] vectors && vectors.Length > 1)
        {
            return true;
        }

        // Check if it's a dictionary with multiple entries
        if (data is Dictionary<string, Vector<T>> dict && dict.Count > 1)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Gets the dimensionality of the output data.
    /// </summary>
    /// <param name="data">The output data to analyze.</param>
    /// <returns>The number of output dimensions.</returns>
    protected virtual int GetOutputDimension(TOutput data)
    {
        // Check if data has a known dimension property
        try
        {
            var dimensionProperty = data?.GetType().GetProperty("Dimension");
            if (dimensionProperty != null)
            {
                var dimension = dimensionProperty.GetValue(data);
                if (dimension is int dimensionValue)
                {
                    return dimensionValue;
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // Calculate based on data structure
        if (data is Vector<T> vector)
        {
            // For categorical data, dimension is the number of categories
            if (IsCategoricalData(data))
            {
                var distinctValues = new HashSet<T>();
                int sampleSize = Math.Min(vector.Length, 1000);

                for (int i = 0; i < sampleSize; i++)
                {
                    distinctValues.Add(vector[i]);
                }

                return distinctValues.Count;
            }

            // For regression, dimension is 1 (single output variable)
            return 1;
        }

        if (data is Matrix<T> matrix)
        {
            // If categorical (one-hot encoded), dimension is number of classes
            if (IsCategoricalData(data))
            {
                return matrix.Columns;
            }

            // For multi-output regression, dimension is number of columns
            return matrix.Columns;
        }

        if (data is Vector<T>[] vectors)
        {
            return vectors.Length;
        }

        if (data is Dictionary<string, Vector<T>> dict)
        {
            return dict.Count;
        }

        if (data is Tensor<T> tensor)
        {
            // For tensors, use the last dimension as output dimension
            return tensor.Shape[tensor.Shape.Length - 1];
        }

        // Default to 1 if we can't determine
        return 1;
    }

    /// <summary>
    /// Gets the dimensionality of the feature space in the input data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>The number of features or dimensions in the input data.</returns>
    /// <remarks>
    /// <para>
    /// This method analyzes the input data structure to determine the number of features or dimensions.
    /// For tabular data, this is typically the number of columns. For image data, it might be the
    /// total number of pixels. For time series, it's the number of features at each time step.
    /// </para>
    /// <para>
    /// The method handles various data structures including vectors, matrices, tensors, and arrays.
    /// </para>
    /// </remarks>
    protected virtual int GetFeatureDimension(TInput data)
    {
        // Check if data has a known dimension property
        try
        {
            var dimensionProperty = data?.GetType().GetProperty("Dimension");
            if (dimensionProperty != null)
            {
                var dimension = dimensionProperty.GetValue(data);
                if (dimension is int dimensionValue)
                {
                    return dimensionValue;
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // Calculate based on data structure
        if (data is Vector<T> vector)
        {
            return vector.Length;
        }

        if (data is Matrix<T> matrix)
        {
            // For tabular data, features are columns
            return matrix.Columns;
        }

        if (data is Tensor<T> tensor)
        {
            if (IsImageData(data))
            {
                // For image data, feature dimension is height × width × channels
                int height = GetImageHeight(data);
                int width = GetImageWidth(data);
                int channels = GetImageChannels(data);
                return height * width * channels;
            }
            else if (IsTimeSeriesData(data))
            {
                // For time series, feature dimension is typically the last dimension
                return tensor.Shape[tensor.Shape.Length - 1];
            }
            else
            {
                // For general tensors, flatten all dimensions except the first (batch dimension)
                int featureDim = 1;
                for (int i = 1; i < tensor.Shape.Length; i++)
                {
                    featureDim *= tensor.Shape[i];
                }
                return featureDim;
            }
        }

        if (data is T[] array)
        {
            return array.Length;
        }

        if (data is T[,] array2D)
        {
            // For 2D arrays, features are typically columns
            return array2D.GetLength(1);
        }

        if (data is Vector<T>[] vectors && vectors.Length > 0)
        {
            // For array of vectors, feature dimension is the length of each vector
            return vectors[0].Length;
        }

        if (data is Dictionary<string, Vector<T>> dict)
        {
            // For dictionary representation, each key is a feature
            return dict.Count;
        }

        // Try to infer from data properties if available
        try
        {
            var properties = data?.GetType().GetProperties();
            var featureProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("FeatureCount", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("FeatureDimension", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("InputDimension", StringComparison.OrdinalIgnoreCase));

            if (featureProp != null)
            {
                var value = featureProp.GetValue(data);
                if (value is int featureCount)
                {
                    return featureCount;
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // If we can't determine, return a default value
        return 10; // Arbitrary default
    }

    /// <summary>
    /// Gets the number of samples in the input data.
    /// </summary>
    /// <param name="data">The input data to analyze.</param>
    /// <returns>The number of samples or observations in the input data.</returns>
    /// <remarks>
    /// <para>
    /// This method analyzes the input data structure to determine the number of samples or observations.
    /// For tabular data, this is typically the number of rows. For image data, it might be the
    /// batch size. For time series, it's the number of sequences.
    /// </para>
    /// <para>
    /// The method handles various data structures including vectors, matrices, tensors, and arrays.
    /// </para>
    /// </remarks>
    protected virtual int GetSampleCount(TInput data)
    {
        // Check if data has a known count property
        try
        {
            var countProperty = data?.GetType().GetProperty("Count");
            if (countProperty != null)
            {
                var count = countProperty.GetValue(data);
                if (count is int countValue)
                {
                    return countValue;
                }
            }

            var lengthProperty = data?.GetType().GetProperty("Length");
            if (lengthProperty != null)
            {
                var length = lengthProperty.GetValue(data);
                if (length is int lengthValue)
                {
                    return lengthValue;
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // Calculate based on data structure
        if (data is Vector<T>)
        {
            // A single vector is typically one sample
            return 1;
        }

        if (data is Matrix<T> matrix)
        {
            // For tabular data, samples are rows
            return matrix.Rows;
        }

        if (data is Tensor<T> tensor)
        {
            // First dimension is typically batch/sample dimension
            return tensor.Shape[0];
        }

        if (data is T[])
        {
            // A single array is typically one sample
            return 1;
        }

        if (data is T[,] array2D)
        {
            // For 2D arrays, samples are typically rows
            return array2D.GetLength(0);
        }

        if (data is Vector<T>[] vectors)
        {
            // For array of vectors, each vector could be a sample
            return vectors.Length;
        }

        if (data is Dictionary<string, Vector<T>> dict && dict.Count > 0)
        {
            // For dictionary representation, sample count is the length of any vector
            return dict.Values.First().Length;
        }

        // Try to infer from data properties if available
        try
        {
            var properties = data?.GetType().GetProperties();
            var sampleProp = properties?.FirstOrDefault(p =>
                p.Name.Equals("SampleCount", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("ObservationCount", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("RowCount", StringComparison.OrdinalIgnoreCase));

            if (sampleProp != null)
            {
                var value = sampleProp.GetValue(data);
                if (value is int sampleCount)
                {
                    return sampleCount;
                }
            }
        }
        catch (Exception)
        {
            // Ignore reflection errors
        }

        // If we can't determine, return a default value
        return 100; // Arbitrary default
    }

    /// <summary>
    /// Analyzes the complexity of the relationship between input and output data.
    /// </summary>
    /// <param name="sampleX">A sample of the input data.</param>
    /// <param name="sampleY">A sample of the output data.</param>
    /// <returns>A value indicating the estimated complexity of the relationship.</returns>
    /// <remarks>
    /// <para>
    /// This method attempts to estimate how complex the relationship between inputs and outputs might be.
    /// It uses various heuristics such as checking for linear correlations, examining the distribution
    /// of output values, and analyzing the dimensionality of the data.
    /// </para>
    /// <para>
    /// The returned value is on a scale where:
    /// - 0-0.3: Likely simple relationship (e.g., linear)
    /// - 0.3-0.7: Moderate complexity
    /// - 0.7-1.0: Highly complex relationship
    /// </para>
    /// </remarks>
    protected virtual double EstimateRelationshipComplexity(TInput sampleX, TOutput sampleY)
    {
        double complexity = 0.5; // Start with moderate complexity as default

        try
        {
            // Check data dimensionality - higher dimensions often mean more complex relationships
            int featureDimension = GetFeatureDimension(sampleX);
            if (featureDimension > 100)
            {
                complexity += 0.2; // High-dimensional data often has complex relationships
            }
            else if (featureDimension < 5)
            {
                complexity -= 0.1; // Low-dimensional data might have simpler relationships
            }

            // Check if data is categorical - can indicate more complex decision boundaries
            if (IsCategoricalData(sampleY))
            {
                int outputDimension = GetOutputDimension(sampleY);
                if (outputDimension > 10)
                {
                    complexity += 0.15; // Many output classes often means complex boundaries
                }
            }

            // Check if data is time series or image - often involves complex patterns
            if (IsTimeSeriesData(sampleX))
            {
                complexity += 0.15;
            }
            else if (IsImageData(sampleX))
            {
                complexity += 0.2;
            }

            // Try to check for linear correlation if data is in appropriate format
            if (sampleX is Matrix<T> xMatrix && sampleY is Vector<T> yVector && xMatrix.Rows == yVector.Length)
            {
                // Sample a subset of features to check correlation
                int featuresToCheck = Math.Min(10, xMatrix.Columns);
                double avgAbsCorrelation = 0;

                for (int i = 0; i < featuresToCheck; i++)
                {
                    // Extract feature vector
                    var featureVector = new Vector<T>(xMatrix.Rows);
                    for (int j = 0; j < xMatrix.Rows; j++)
                    {
                        featureVector[j] = xMatrix[j, i];
                    }

                    // Calculate correlation
                    double correlation = Math.Abs(Convert.ToDouble(StatisticsHelper<T>.CalculatePearsonCorrelation(featureVector, yVector)));
                    avgAbsCorrelation += correlation;
                }

                avgAbsCorrelation /= featuresToCheck;

                // Strong linear correlation suggests simpler relationship
                if (avgAbsCorrelation > 0.8)
                {
                    complexity -= 0.3;
                }
                else if (avgAbsCorrelation < 0.2)
                {
                    complexity += 0.1; // Very weak correlation might indicate complex relationship
                }
            }

            // Ensure complexity stays in [0,1] range
            complexity = Math.Max(0, Math.Min(1, complexity));
        }
        catch (Exception)
        {
            // If analysis fails, return moderate complexity
            return 0.5;
        }

        return complexity;
    }
}