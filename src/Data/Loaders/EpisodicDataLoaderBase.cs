using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides a base implementation for episodic data loaders with common functionality for N-way K-shot meta-learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for tasks (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type for tasks (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IEpisodicDataLoader interface and provides common functionality
/// for episodic task sampling. It handles dataset validation, class-to-indices preprocessing, and
/// parameter validation while allowing derived classes to focus on implementing specific sampling strategies.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all episodic data loaders build upon.
///
/// Think of it like a template for creating meta-learning tasks:
/// - It handles common tasks (validating inputs, organizing data by class, checking requirements)
/// - Specific loaders just implement how they sample tasks
/// - This ensures all episodic loaders work consistently and follow SOLID principles
///
/// The base class takes care of:
/// - Validating that your dataset has enough classes and examples
/// - Building an efficient index (class → example indices) for fast sampling
/// - Storing configuration (N-way, K-shot, query shots)
/// - Providing protected access to the dataset and configuration
/// </para>
/// </remarks>
public abstract class EpisodicDataLoaderBase<T, TInput, TOutput> : IEpisodicDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// The feature matrix containing all examples.
    /// </summary>
    protected readonly Matrix<T> DatasetX;

    /// <summary>
    /// The label vector containing class labels for all examples.
    /// </summary>
    protected readonly Vector<T> DatasetY;

    /// <summary>
    /// The number of classes per task (N in N-way).
    /// </summary>
    protected readonly int NWay;

    /// <summary>
    /// The number of support examples per class (K in K-shot).
    /// </summary>
    protected readonly int KShot;

    /// <summary>
    /// The number of query examples per class.
    /// </summary>
    protected readonly int QueryShots;

    /// <summary>
    /// Random number generator for task sampling.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Mapping from class label to list of example indices for that class.
    /// </summary>
    protected readonly Dictionary<int, List<int>> ClassToIndices;

    /// <summary>
    /// Array of all available class labels in the dataset.
    /// </summary>
    protected readonly int[] AvailableClasses;

    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the EpisodicDataLoaderBase class with industry-standard defaults.
    /// </summary>
    /// <param name="datasetX">The feature matrix where each row is an example. Shape: [num_examples, num_features].</param>
    /// <param name="datasetY">The label vector containing class labels for each example. Length: num_examples.</param>
    /// <param name="nWay">The number of unique classes per task. Default is 5 (standard in meta-learning).</param>
    /// <param name="kShot">The number of support examples per class. Default is 5 (balanced difficulty).</param>
    /// <param name="queryShots">The number of query examples per class. Default is 15 (3x kShot).</param>
    /// <param name="seed">Optional random seed for reproducible task sampling. If null, uses a time-based seed.</param>
    /// <exception cref="ArgumentNullException">Thrown when datasetX or datasetY is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions are invalid or dataset is too small.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the base infrastructure for episodic sampling.
    ///
    /// It performs several important tasks:
    /// 1. Validates all inputs to catch configuration errors early
    /// 2. Builds an index mapping each class to its example indices for fast lookup
    /// 3. Verifies the dataset has enough classes and examples per class
    /// 4. Stores all configuration for use by derived classes
    ///
    /// After construction, the derived class can use the protected fields to implement
    /// its specific sampling strategy.
    /// </para>
    /// </remarks>
    protected EpisodicDataLoaderBase(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay = 5,        // Default: 5-way (standard in meta-learning literature)
        int kShot = 5,       // Default: 5-shot (balanced between 1-shot and 10-shot)
        int queryShots = 15, // Default: 15 queries (3x kShot is common practice)
        int? seed = null)
    {
        // Validate inputs
        ValidateDataset(datasetX, datasetY);
        ValidateConfiguration(nWay, kShot, queryShots);

        DatasetX = datasetX;
        DatasetY = datasetY;
        NWay = nWay;
        KShot = kShot;
        QueryShots = queryShots;
        Random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Preprocess: Build class-to-indices mapping
        ClassToIndices = BuildClassIndex(datasetY);
        AvailableClasses = ClassToIndices.Keys.ToArray();

        // Validate dataset has sufficient classes and examples
        ValidateDatasetSize(ClassToIndices, nWay, kShot, queryShots);
    }

    /// <summary>
    /// Samples and returns the next N-way K-shot meta-learning task.
    /// </summary>
    /// <returns>
    /// A MetaLearningTask containing support and query sets with the configured N-way K-shot specification.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method delegates to the derived class's implementation of GetNextTaskCore,
    /// which contains the specific sampling strategy.
    /// </para>
    /// </remarks>
    public MetaLearningTask<T, TInput, TOutput> GetNextTask()
    {
        return GetNextTaskCore();
    }

    /// <summary>
    /// Core task sampling logic to be implemented by derived classes.
    /// </summary>
    /// <returns>A MetaLearningTask with sampled support and query sets.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific task sampling strategy.
    ///
    /// You have access to:
    /// - DatasetX and DatasetY: The full dataset
    /// - ClassToIndices: Fast lookup of examples by class
    /// - AvailableClasses: Array of all class labels
    /// - NWay, KShot, QueryShots: Task configuration
    /// - Random: For randomized sampling
    /// - NumOps: For numeric operations
    ///
    /// Your implementation should:
    /// 1. Select NWay classes
    /// 2. Sample KShot + QueryShots examples per class
    /// 3. Split into support and query sets
    /// 4. Build and return a MetaLearningTask
    /// </para>
    /// </remarks>
    protected abstract MetaLearningTask<T, TInput, TOutput> GetNextTaskCore();

    /// <summary>
    /// Validates the dataset inputs.
    /// </summary>
    private void ValidateDataset(Matrix<T> datasetX, Vector<T> datasetY)
    {
        if (datasetX == null)
        {
            throw new ArgumentNullException(nameof(datasetX), "Dataset features cannot be null");
        }

        if (datasetY == null)
        {
            throw new ArgumentNullException(nameof(datasetY), "Dataset labels cannot be null");
        }

        if (datasetX.Rows != datasetY.Length)
        {
            throw new ArgumentException(
                $"Number of examples in features ({datasetX.Rows}) must match number of labels ({datasetY.Length})",
                nameof(datasetX));
        }
    }

    /// <summary>
    /// Validates the N-way K-shot configuration.
    /// </summary>
    private void ValidateConfiguration(int nWay, int kShot, int queryShots)
    {
        if (nWay < 2)
        {
            throw new ArgumentException("nWay must be at least 2 (need at least 2 classes per task)", nameof(nWay));
        }

        if (kShot < 1)
        {
            throw new ArgumentException("kShot must be at least 1 (need at least 1 support example per class)", nameof(kShot));
        }

        if (queryShots < 1)
        {
            throw new ArgumentException("queryShots must be at least 1 (need at least 1 query example per class)", nameof(queryShots));
        }
    }

    /// <summary>
    /// Builds a dictionary mapping each class label to its example indices.
    /// </summary>
    private Dictionary<int, List<int>> BuildClassIndex(Vector<T> datasetY)
    {
        var classToIndices = new Dictionary<int, List<int>>();

        for (int i = 0; i < datasetY.Length; i++)
        {
            int classLabel = NumOps.ToInt32(datasetY[i]);

            if (!classToIndices.ContainsKey(classLabel))
            {
                classToIndices[classLabel] = new List<int>();
            }

            classToIndices[classLabel].Add(i);
        }

        return classToIndices;
    }

    /// <summary>
    /// Validates that the dataset has sufficient classes and examples per class.
    /// </summary>
    private void ValidateDatasetSize(Dictionary<int, List<int>> classToIndices, int nWay, int kShot, int queryShots)
    {
        var availableClasses = classToIndices.Keys.ToArray();

        // Validate dataset has enough classes
        if (availableClasses.Length < nWay)
        {
            throw new ArgumentException(
                $"Dataset has only {availableClasses.Length} classes, but nWay={nWay} requires at least {nWay} classes",
                nameof(nWay));
        }

        // Validate each class has enough examples
        int requiredExamplesPerClass = kShot + queryShots;
        var insufficientClasses = classToIndices
            .Where(kvp => kvp.Value.Count < requiredExamplesPerClass)
            .Select(kvp => $"Class {kvp.Key}: {kvp.Value.Count} examples")
            .ToList();

        if (insufficientClasses.Count > 0)
        {
            throw new ArgumentException(
                $"Some classes have insufficient examples. Need at least {requiredExamplesPerClass} " +
                $"(kShot={kShot} + queryShots={queryShots}) per class. Insufficient classes: " +
                string.Join(", ", insufficientClasses),
                nameof(nWay));
        }
    }

    /// <summary>
    /// Builds a MetaLearningTask from lists of examples and labels.
    /// </summary>
    protected MetaLearningTask<T, TInput, TOutput> BuildMetaLearningTask(
        List<Vector<T>> supportExamples,
        List<T> supportLabels,
        List<Vector<T>> queryExamples,
        List<T> queryLabels)
    {
        // Build matrices from examples
        int numFeatures = DatasetX.Columns;
        var supportMatrix = new Matrix<T>(supportExamples.Count, numFeatures);
        var supportVector = new Vector<T>(supportLabels.Count);

        for (int i = 0; i < supportExamples.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                supportMatrix[i, j] = supportExamples[i][j];
            }
            supportVector[i] = supportLabels[i];
        }

        var queryMatrix = new Matrix<T>(queryExamples.Count, numFeatures);
        var queryVector = new Vector<T>(queryLabels.Count);

        for (int i = 0; i < queryExamples.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                queryMatrix[i, j] = queryExamples[i][j];
            }
            queryVector[i] = queryLabels[i];
        }

        // Convert to TInput/TOutput using type checking
        TInput supportX = ConvertMatrixToInput(supportMatrix);
        TOutput supportY = ConvertVectorToOutput(supportVector);
        TInput queryX = ConvertMatrixToInput(queryMatrix);
        TOutput queryY = ConvertVectorToOutput(queryVector);

        return new MetaLearningTask<T, TInput, TOutput>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY
        };
    }

    /// <summary>
    /// Converts a Matrix to TInput type using the same pattern as ConversionsHelper.
    /// </summary>
    private static TInput ConvertMatrixToInput(Matrix<T> matrix)
    {
        if (typeof(TInput) == typeof(Matrix<T>))
        {
            return (TInput)(object)matrix;
        }

        if (typeof(TInput) == typeof(Tensor<T>))
        {
            // Use Tensor.FromMatrix for efficient conversion
            var tensor = Tensor<T>.FromRowMatrix(matrix);
            return (TInput)(object)tensor;
        }

        throw new NotSupportedException(
            $"Conversion from Matrix<T> to {typeof(TInput).Name} is not supported. " +
            $"Supported types: Matrix<T>, Tensor<T>");
    }

    /// <summary>
    /// Converts a Vector to TOutput type using the same pattern as ConversionsHelper.
    /// </summary>
    private static TOutput ConvertVectorToOutput(Vector<T> vector)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)vector;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            // Use Tensor.FromVector for efficient conversion
            return (TOutput)(object)Tensor<T>.FromVector(vector);
        }

        throw new NotSupportedException(
            $"Conversion from Vector<T> to {typeof(TOutput).Name} is not supported. " +
            $"Supported types: Vector<T>, Tensor<T>");
    }
}
