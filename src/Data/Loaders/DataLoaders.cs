using AiDotNet.FederatedLearning.Benchmarks.Leaf;
using AiDotNet.LinearAlgebra;
using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Static factory class for creating data loaders with beginner-friendly methods.
/// </summary>
/// <remarks>
/// <para>
/// DataLoaders provides the easiest way to create data loaders for common scenarios.
/// It follows a factory pattern with static methods that handle type inference and
/// common configurations automatically.
/// </para>
/// <para><b>For Beginners:</b> This is your starting point for loading data into AiDotNet!
/// Choose the method that matches your data format:
///
/// **Common Patterns:**
/// ```csharp
/// // From arrays (simplest for small datasets)
/// var loader = DataLoaders.FromArrays(features, labels);
///
/// // From Matrix and Vector (most common for ML)
/// var loader = DataLoaders.FromMatrixVector(featureMatrix, labelVector);
///
/// // From Tensors (for deep learning)
/// var loader = DataLoaders.FromTensors(inputTensor, outputTensor);
/// ```
///
/// All loaders support:
/// - Batching: `loader.BatchSize = 32;`
/// - Shuffling: `loader.Shuffle();`
/// - Splitting: `var (train, val, test) = loader.Split();`
/// </para>
/// </remarks>
public static class DataLoaders
{
    #region Array-based Factory Methods

    /// <summary>
    /// Creates a data loader from 2D feature array and 1D label array.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">2D array where rows are samples and columns are features.</param>
    /// <param name="labels">1D array of labels, one per sample.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the simplest way to load tabular data.
    ///
    /// **Example - Predicting House Prices:**
    /// ```csharp
    /// // Features: [sqft, bedrooms, bathrooms]
    /// double[,] features = new double[,] {
    ///     { 1500, 3, 2 },
    ///     { 2000, 4, 3 },
    ///     { 1200, 2, 1 }
    /// };
    ///
    /// // Labels: price
    /// double[] labels = { 300000, 450000, 250000 };
    ///
    /// var loader = DataLoaders.FromArrays(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> FromArrays<T>(
        T[,] features,
        T[] labels)
    {
        if (features is null)
        {
            throw new ArgumentNullException(nameof(features), "Features array cannot be null.");
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels), "Labels array cannot be null.");
        }

        int rows = features.GetLength(0);
        int cols = features.GetLength(1);

        if (rows != labels.Length)
        {
            throw new ArgumentException(
                $"Feature rows ({rows}) must match label count ({labels.Length}).",
                nameof(labels));
        }

        // Convert arrays to Matrix and Vector
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = features[i, j];
            }
        }

        var vector = new Vector<T>(labels);

        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(matrix, vector);
    }

    /// <summary>
    /// Creates a data loader from jagged feature array and 1D label array.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Jagged array where each inner array is a sample's features.</param>
    /// <param name="labels">1D array of labels, one per sample.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match or arrays are inconsistent.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when your data is in jagged array format.
    ///
    /// **Example:**
    /// ```csharp
    /// double[][] features = {
    ///     new[] { 1.0, 2.0, 3.0 },
    ///     new[] { 4.0, 5.0, 6.0 },
    ///     new[] { 7.0, 8.0, 9.0 }
    /// };
    /// double[] labels = { 0, 1, 0 };
    ///
    /// var loader = DataLoaders.FromArrays(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> FromArrays<T>(
        T[][] features,
        T[] labels)
    {
        if (features is null)
        {
            throw new ArgumentNullException(nameof(features), "Features array cannot be null.");
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels), "Labels array cannot be null.");
        }

        if (features.Length == 0)
        {
            throw new ArgumentException("Features array cannot be empty.", nameof(features));
        }

        int rows = features.Length;
        int cols = features[0]?.Length ?? 0;

        if (rows != labels.Length)
        {
            throw new ArgumentException(
                $"Feature rows ({rows}) must match label count ({labels.Length}).",
                nameof(labels));
        }

        // Validate all rows have same length
        for (int i = 0; i < rows; i++)
        {
            if (features[i] is null)
            {
                throw new ArgumentException($"Feature row {i} cannot be null.", nameof(features));
            }

            if (features[i].Length != cols)
            {
                throw new ArgumentException(
                    $"All feature rows must have the same length. Row 0 has {cols} columns, row {i} has {features[i].Length}.",
                    nameof(features));
            }
        }

        // Convert to Matrix
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = features[i][j];
            }
        }

        var vector = new Vector<T>(labels);

        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(matrix, vector);
    }

    /// <summary>
    /// Creates a data loader from 1D feature array (single feature) and 1D label array.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">1D array of single feature values.</param>
    /// <param name="labels">1D array of labels, one per sample.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when lengths don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for simple regression with one input variable.
    ///
    /// **Example - Simple Linear Regression:**
    /// ```csharp
    /// // X: study hours
    /// double[] features = { 1, 2, 3, 4, 5 };
    /// // Y: test scores
    /// double[] labels = { 50, 60, 70, 80, 90 };
    ///
    /// var loader = DataLoaders.FromArrays(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> FromArrays<T>(
        T[] features,
        T[] labels)
    {
        if (features is null)
        {
            throw new ArgumentNullException(nameof(features), "Features array cannot be null.");
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels), "Labels array cannot be null.");
        }

        if (features.Length != labels.Length)
        {
            throw new ArgumentException(
                $"Feature count ({features.Length}) must match label count ({labels.Length}).",
                nameof(labels));
        }

        // Convert single feature to matrix with one column
        var matrix = new Matrix<T>(features.Length, 1);
        for (int i = 0; i < features.Length; i++)
        {
            matrix[i, 0] = features[i];
        }

        var vector = new Vector<T>(labels);

        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(matrix, vector);
    }

    #endregion

    #region Matrix/Vector Factory Methods

    /// <summary>
    /// Creates a data loader from a feature Matrix and label Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Matrix where rows are samples and columns are features.</param>
    /// <param name="labels">Vector of labels, one per sample.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when row count doesn't match label count.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the most common format for machine learning.
    /// Use this when you already have Matrix and Vector objects.
    ///
    /// **Example:**
    /// ```csharp
    /// var features = new Matrix&lt;double&gt;(100, 5);  // 100 samples, 5 features
    /// var labels = new Vector&lt;double&gt;(100);       // 100 labels
    ///
    /// // Fill your data...
    ///
    /// var loader = DataLoaders.FromMatrixVector(features, labels);
    ///
    /// // Use with PredictionModelBuilder
    /// var result = await builder
    ///     .ConfigureDataLoader(loader)
    ///     .ConfigureModel(model)
    ///     .BuildAsync();
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> FromMatrixVector<T>(
        Matrix<T> features,
        Vector<T> labels)
    {
        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(features, labels);
    }

    /// <summary>
    /// Creates a data loader from a feature Matrix and label Matrix (for multi-output regression).
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Matrix where rows are samples and columns are features.</param>
    /// <param name="labels">Matrix where rows are samples and columns are output dimensions.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when row counts don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when predicting multiple outputs simultaneously.
    ///
    /// **Example - Predicting Multiple Properties:**
    /// ```csharp
    /// // Input: molecule features
    /// var features = new Matrix&lt;double&gt;(100, 10);
    ///
    /// // Output: multiple properties (e.g., toxicity, solubility, binding affinity)
    /// var labels = new Matrix&lt;double&gt;(100, 3);
    ///
    /// var loader = DataLoaders.FromMatrices(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Matrix<T>> FromMatrices<T>(
        Matrix<T> features,
        Matrix<T> labels)
    {
        return new InMemoryDataLoader<T, Matrix<T>, Matrix<T>>(features, labels);
    }

    #endregion

    #region Tensor Factory Methods

    /// <summary>
    /// Creates a data loader from input and output Tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Input tensor where first dimension is batch/samples.</param>
    /// <param name="labels">Output tensor where first dimension is batch/samples.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when sample counts don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use tensors for deep learning with multi-dimensional data.
    ///
    /// **Example - Image Classification:**
    /// ```csharp
    /// // Input: 1000 images, 28x28 pixels, 1 channel (grayscale)
    /// var features = new Tensor&lt;float&gt;([1000, 28, 28, 1]);
    ///
    /// // Output: 1000 labels, 10 classes (one-hot encoded)
    /// var labels = new Tensor&lt;float&gt;([1000, 10]);
    ///
    /// var loader = DataLoaders.FromTensors(features, labels);
    /// ```
    ///
    /// **Example - Sequence Data:**
    /// ```csharp
    /// // Input: 500 sequences, 100 time steps, 32 features per step
    /// var features = new Tensor&lt;double&gt;([500, 100, 32]);
    ///
    /// // Output: 500 predictions
    /// var labels = new Tensor&lt;double&gt;([500, 1]);
    ///
    /// var loader = DataLoaders.FromTensors(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Tensor<T>, Tensor<T>> FromTensors<T>(
        Tensor<T> features,
        Tensor<T> labels)
    {
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(features, labels);
    }

    /// <summary>
    /// Creates a data loader from a Tensor of features and a Vector of labels.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Input tensor where first dimension is batch/samples.</param>
    /// <param name="labels">Vector of labels, one per sample.</param>
    /// <returns>A configured InMemoryDataLoader ready for training.</returns>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when sample counts don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Common pattern for classification with complex inputs.
    ///
    /// **Example - Image Classification with Class Labels:**
    /// ```csharp
    /// // Input: images as tensor
    /// var features = new Tensor&lt;float&gt;([1000, 28, 28, 1]);
    ///
    /// // Output: class indices (0-9)
    /// var labels = new Vector&lt;float&gt;(1000);  // Contains values 0-9
    ///
    /// var loader = DataLoaders.FromTensorVector(features, labels);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Tensor<T>, Vector<T>> FromTensorVector<T>(
        Tensor<T> features,
        Vector<T> labels)
    {
        return new InMemoryDataLoader<T, Tensor<T>, Vector<T>>(features, labels);
    }

    #endregion

    #region Federated Benchmark Methods

    /// <summary>
    /// Creates a LEAF federated data loader from LEAF benchmark JSON files.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="trainFilePath">Path to the LEAF train split JSON file.</param>
    /// <param name="testFilePath">Optional path to the LEAF test split JSON file.</param>
    /// <param name="options">Optional LEAF load options (subset, validation).</param>
    /// <returns>A configured LEAF data loader ready for federated learning.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> LEAF is a standard federated learning benchmark suite where each "user" is treated as one client.
    /// This loader keeps that per-client split intact so federated learning simulations match the benchmark.
    /// </para>
    /// </remarks>
    public static LeafFederatedDataLoader<T> FromLeafFederatedJsonFiles<T>(
        string trainFilePath,
        string? testFilePath = null,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        return new LeafFederatedDataLoader<T>(trainFilePath, testFilePath, options);
    }

    #endregion

    #region Convenience Methods

    /// <summary>
    /// Creates an empty data loader placeholder (useful for meta-learning or custom scenarios).
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <returns>A data loader with empty data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> You typically won't need this method.
    /// It's used for advanced scenarios where data is loaded dynamically or for meta-learning tasks
    /// that don't use traditional supervised learning data.
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> Empty<T>()
    {
        var emptyFeatures = new Matrix<T>(0, 0);
        var emptyLabels = new Vector<T>(0);
        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(emptyFeatures, emptyLabels);
    }

    /// <summary>
    /// Creates a data loader with pre-configured batch size.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="features">Matrix where rows are samples and columns are features.</param>
    /// <param name="labels">Vector of labels, one per sample.</param>
    /// <param name="batchSize">The batch size for iteration.</param>
    /// <returns>A configured InMemoryDataLoader with the specified batch size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch size determines how many samples are processed together.
    /// Common values:
    /// - 32: Good default for most cases
    /// - 16-64: Standard range for GPU training
    /// - 1: Stochastic gradient descent (slowest but most updates)
    /// - Full dataset: Batch gradient descent (fewer updates but more stable)
    ///
    /// **Example:**
    /// ```csharp
    /// var loader = DataLoaders.WithBatchSize(features, labels, batchSize: 64);
    /// ```
    /// </para>
    /// </remarks>
    public static InMemoryDataLoader<T, Matrix<T>, Vector<T>> WithBatchSize<T>(
        Matrix<T> features,
        Vector<T> labels,
        int batchSize)
    {
        var loader = new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(features, labels)
        {
            BatchSize = batchSize
        };
        return loader;
    }

    #region Geometry Dataset Methods

    /// <summary>
    /// Creates a ModelNet40 classification data loader.
    /// </summary>
    public static ModelNet40ClassificationDataLoader<T> ModelNet40Classification<T>(
        ModelNet40ClassificationDataLoaderOptions? options = null)
    {
        return new ModelNet40ClassificationDataLoader<T>(options);
    }

    /// <summary>
    /// Creates a ShapeNetCore part segmentation data loader.
    /// </summary>
    public static ShapeNetCorePartSegmentationDataLoader<T> ShapeNetCorePartSegmentation<T>(
        ShapeNetCorePartSegmentationDataLoaderOptions? options = null)
    {
        return new ShapeNetCorePartSegmentationDataLoader<T>(options);
    }

    /// <summary>
    /// Creates a ScanNet semantic segmentation data loader.
    /// </summary>
    public static ScanNetSemanticSegmentationDataLoader<T> ScanNetSemanticSegmentation<T>(
        ScanNetSemanticSegmentationDataLoaderOptions? options = null)
    {
        return new ScanNetSemanticSegmentationDataLoader<T>(options);
    }

    #endregion
    #endregion

    #region Streaming Factory Methods

    /// <summary>
    /// Creates a streaming data loader that reads samples on-demand.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <typeparam name="TInput">The input data type for each sample.</typeparam>
    /// <typeparam name="TOutput">The output/label data type for each sample.</typeparam>
    /// <param name="sampleCount">Total number of samples in the dataset.</param>
    /// <param name="sampleReader">Async function that reads a single sample by index.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers. Default is 4.</param>
    /// <returns>A streaming data loader.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when your dataset is too large to fit in memory.
    /// The sampleReader function is called on-demand to load individual samples.
    ///
    /// **Example - Loading Images:**
    /// ```csharp
    /// var loader = DataLoaders.Streaming&lt;float, float[], int&gt;(
    ///     sampleCount: 1000000,
    ///     sampleReader: async (index, ct) =&gt;
    ///     {
    ///         var image = await LoadImageAsync($"images/{index}.png", ct);
    ///         var label = GetLabel(index);
    ///         return (image, label);
    ///     },
    ///     batchSize: 32
    /// );
    ///
    /// await foreach (var batch in loader.GetBatchesAsync())
    /// {
    ///     await model.TrainOnBatchAsync(batch.Inputs, batch.Outputs);
    /// }
    /// ```
    /// </para>
    /// </remarks>
    public static StreamingDataLoader<T, TInput, TOutput> Streaming<T, TInput, TOutput>(
        int sampleCount,
        Func<int, CancellationToken, Task<(TInput, TOutput)>> sampleReader,
        int batchSize,
        int prefetchCount = 2,
        int numWorkers = 4)
    {
        if (sampleReader is null)
        {
            throw new ArgumentNullException(nameof(sampleReader), "Sample reader function cannot be null.");
        }

        if (sampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be greater than 0.");
        }

        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than 0.");
        }

        if (prefetchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be greater than 0.");
        }

        if (numWorkers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be greater than 0.");
        }

        return new StreamingDataLoader<T, TInput, TOutput>(
            sampleCount, sampleReader, batchSize, null, prefetchCount, numWorkers);
    }

    /// <summary>
    /// Creates a streaming data loader from a directory of files.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <typeparam name="TInput">The input data type for each sample.</typeparam>
    /// <typeparam name="TOutput">The output/label data type for each sample.</typeparam>
    /// <param name="directory">The directory containing data files.</param>
    /// <param name="filePattern">The file pattern to match (e.g., "*.png", "*.csv").</param>
    /// <param name="fileProcessor">Function that processes a file and returns (input, output).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="searchOption">Whether to search subdirectories. Default is TopDirectoryOnly.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers. Default is 4.</param>
    /// <returns>A file streaming data loader.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you have a folder of data files (images, audio, etc.)
    /// that you want to stream during training.
    ///
    /// **Example - Image Dataset:**
    /// ```csharp
    /// var loader = DataLoaders.FromDirectory&lt;float, float[], int&gt;(
    ///     directory: "data/images",
    ///     filePattern: "*.png",
    ///     fileProcessor: async (filePath, ct) =&gt;
    ///     {
    ///         var pixels = await LoadImagePixelsAsync(filePath, ct);
    ///         var label = ParseLabelFromFilename(filePath);
    ///         return (pixels, label);
    ///     },
    ///     batchSize: 64
    /// );
    /// ```
    /// </para>
    /// </remarks>
    public static FileStreamingDataLoader<T, TInput, TOutput> FromDirectory<T, TInput, TOutput>(
        string directory,
        string filePattern,
        Func<string, CancellationToken, Task<(TInput, TOutput)>> fileProcessor,
        int batchSize,
        SearchOption searchOption = SearchOption.TopDirectoryOnly,
        int prefetchCount = 2,
        int numWorkers = 4)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentNullException(nameof(directory), "Directory path cannot be null or empty.");
        }

        if (string.IsNullOrWhiteSpace(filePattern))
        {
            throw new ArgumentNullException(nameof(filePattern), "File pattern cannot be null or empty.");
        }

        if (fileProcessor is null)
        {
            throw new ArgumentNullException(nameof(fileProcessor), "File processor function cannot be null.");
        }

        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than 0.");
        }

        if (prefetchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be greater than 0.");
        }

        if (numWorkers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be greater than 0.");
        }

        return new FileStreamingDataLoader<T, TInput, TOutput>(
            directory, filePattern, fileProcessor, batchSize, searchOption, prefetchCount, numWorkers);
    }

    /// <summary>
    /// Creates a streaming data loader from a CSV file.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <typeparam name="TInput">The input data type for each row.</typeparam>
    /// <typeparam name="TOutput">The output/label data type for each row.</typeparam>
    /// <param name="filePath">Path to the CSV file.</param>
    /// <param name="lineParser">Function that parses a CSV line into (input, output).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="hasHeader">Whether the CSV has a header row to skip. Default is true.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <returns>A CSV streaming data loader.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for large CSV files that don't fit in memory.
    /// The file is read line by line during training.
    ///
    /// **Example - Large Tabular Dataset:**
    /// ```csharp
    /// var loader = DataLoaders.FromCsv&lt;double, double[], double&gt;(
    ///     filePath: "data/huge_dataset.csv",
    ///     lineParser: (line, lineNumber) =&gt;
    ///     {
    ///         var parts = line.Split(',');
    ///         var features = parts.Take(10).Select(double.Parse).ToArray();
    ///         var label = double.Parse(parts[10]);
    ///         return (features, label);
    ///     },
    ///     batchSize: 256,
    ///     hasHeader: true
    /// );
    /// ```
    /// </para>
    /// </remarks>
    public static CsvStreamingDataLoader<T, TInput, TOutput> FromCsv<T, TInput, TOutput>(
        string filePath,
        Func<string, int, (TInput, TOutput)> lineParser,
        int batchSize,
        bool hasHeader = true,
        int prefetchCount = 2)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentNullException(nameof(filePath), "File path cannot be null or empty.");
        }

        if (lineParser is null)
        {
            throw new ArgumentNullException(nameof(lineParser), "Line parser function cannot be null.");
        }

        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than 0.");
        }

        if (prefetchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be greater than 0.");
        }

        return new CsvStreamingDataLoader<T, TInput, TOutput>(
            filePath, lineParser, batchSize, hasHeader, prefetchCount);
    }

    #endregion
}
