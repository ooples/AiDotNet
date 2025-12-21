namespace AiDotNet.DataProcessor;

/// <summary>
/// Default implementation of a data preprocessor that prepares data for machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Data preprocessing is like preparing ingredients before cooking a meal. 
/// Raw data often needs to be cleaned and transformed before a machine learning algorithm can use it effectively.
/// 
/// This class handles three important preprocessing steps:
/// 1. Removing outliers - Getting rid of unusual data points that might confuse the algorithm
/// 2. Normalizing data - Adjusting values so they're on similar scales (like converting inches and feet to all inches)
/// 3. Selecting features - Choosing which information is most relevant (like deciding which ingredients to include in a recipe)
/// 
/// These steps help machine learning algorithms work better and learn more efficiently from your data.
/// </para>
/// </remarks>
public class DefaultDataPreprocessor<T, TInput, TOutput> : IDataPreprocessor<T, TInput, TOutput>
{
    /// <summary>
    /// Component responsible for scaling data to a standard range.
    /// </summary>
    private readonly INormalizer<T, TInput, TOutput> _normalizer;

    /// <summary>
    /// Component responsible for selecting the most relevant features from the dataset.
    /// </summary>
    private readonly IFeatureSelector<T, TInput> _featureSelector;

    /// <summary>
    /// Component responsible for identifying and removing outliers from the dataset.
    /// </summary>
    private readonly IOutlierRemoval<T, TInput, TOutput> _outlierRemoval;

    /// <summary>
    /// Configuration options for the data preprocessing operations.
    /// </summary>
    private readonly DataProcessorOptions _options;

    /// <summary>
    /// Initializes a new instance of the DefaultDataPreprocessor class with specified components.
    /// </summary>
    /// <param name="normalizer">Component for normalizing data values.</param>
    /// <param name="featureSelector">Component for selecting relevant features.</param>
    /// <param name="outlierRemoval">Component for removing outliers.</param>
    /// <param name="options">Optional configuration settings for data processing.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the data preprocessor with the tools it needs.
    /// Think of it like assembling a toolkit before starting a project:
    /// - The normalizer adjusts the scale of your data
    /// - The feature selector helps choose which parts of your data are most important
    /// - The outlier removal tool identifies and removes unusual data points
    /// - The options parameter lets you customize how these tools work together
    /// </para>
    /// </remarks>
    public DefaultDataPreprocessor(INormalizer<T, TInput, TOutput> normalizer, IFeatureSelector<T, TInput> featureSelector,
        IOutlierRemoval<T, TInput, TOutput> outlierRemoval, DataProcessorOptions? options = null)
    {
        _normalizer = normalizer;
        _featureSelector = featureSelector;
        _outlierRemoval = outlierRemoval;
        _options = options ?? new();
    }

    /// <summary>
    /// Preprocesses the input data by removing outliers, normalizing values, and selecting features.
    /// </summary>
    /// <param name="X">The feature matrix where rows represent samples and columns represent features.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <returns>A tuple containing the processed feature matrix, target vector, and normalization information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method cleans and prepares your data in three steps:
    /// 
    /// 1. First, it removes outliers (unusual data points that might confuse the learning algorithm)
    /// 
    /// 2. Then, depending on your settings, it either:
    ///    - Normalizes the data first (adjusts all values to a similar scale) and then selects the most important features, or
    ///    - Selects the most important features first and then normalizes the data
    /// 
    /// 3. Finally, it returns your cleaned data along with information about how the normalization was done
    ///    (which you'll need later if you want to use your model with new data)
    /// 
    /// The "X" matrix contains your input data (like house sizes, number of bedrooms, etc.),
    /// while the "y" vector contains what you're trying to predict (like house prices).
    /// </para>
    /// </remarks>
    public (TInput X, TOutput y, NormalizationInfo<T, TInput, TOutput> normInfo) PreprocessData(TInput X, TOutput y)
    {
        NormalizationInfo<T, TInput, TOutput> normInfo = new();

        (X, y) = _outlierRemoval.RemoveOutliers(X, y);

        if (_options.NormalizeBeforeFeatureSelection)
        {
            (X, normInfo.XParams) = _normalizer.NormalizeInput(X);
            (y, normInfo.YParams) = _normalizer.NormalizeOutput(y);
            X = _featureSelector.SelectFeatures(X);
        }
        else
        {
            X = _featureSelector.SelectFeatures(X);
            (X, normInfo.XParams) = _normalizer.NormalizeInput(X);
            (y, normInfo.YParams) = _normalizer.NormalizeOutput(y);
        }

        return (X, y, normInfo);
    }

    /// <summary>
    /// Splits the dataset into training, validation, and test sets.
    /// </summary>
    /// <param name="X">The features where rows represent samples and columns represent features.</param>
    /// <param name="y">The target values to predict.</param>
    /// <returns>A tuple containing the training, validation, and test datasets.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method divides your data into three separate groups:
    /// 
    /// 1. Training data - The largest portion, used to teach the model patterns
    ///    (like studying examples to learn a new skill)
    /// 
    /// 2. Validation data - A smaller portion, used to fine-tune the model
    ///    (like practicing the skill and making adjustments)
    /// 
    /// 3. Test data - Another small portion, used to evaluate how well the model works
    ///    (like taking a final test to see how well you learned)
    /// 
    /// The data is shuffled randomly before splitting to ensure each set contains a good mix of examples.
    /// The size of each portion is determined by the settings in the options.
    /// 
    /// This separation is crucial because it helps ensure your model can work well with new data it hasn't seen before.
    /// </para>
    /// </remarks>
    public (TInput XTrain, TOutput yTrain, TInput XValidation, TOutput yValidation, TInput XTest, TOutput yTest)
        SplitData(TInput X, TOutput y)
    {
        int totalSamples = 0;

        // Handle different input types appropriately
        if (X is Matrix<T> xMatrix && y is Vector<T> yVector)
        {
            totalSamples = xMatrix.Rows;
            int trainSize = (int)(totalSamples * _options.TrainingSplitPercentage);
            int validationSize = (int)(totalSamples * _options.ValidationSplitPercentage);
            int testSize = totalSamples - trainSize - validationSize;

            var indices = Enumerable.Range(0, totalSamples).ToList();
            if (_options.ShuffleBeforeSplit)
            {
                // Shuffle the data
                var random = RandomHelper.CreateSeededRandom(_options.RandomSeed);
                indices = [.. indices.OrderBy(x => random.Next())];
            }

            // Create matrices and vectors for the split data
            var XTrain = new Matrix<T>(trainSize, xMatrix.Columns);
            var yTrain = new Vector<T>(trainSize);
            var XValidation = new Matrix<T>(validationSize, xMatrix.Columns);
            var yValidation = new Vector<T>(validationSize);
            var XTest = new Matrix<T>(testSize, xMatrix.Columns);
            var yTest = new Vector<T>(testSize);

            // Copy data to the training sets
            for (int i = 0; i < trainSize; i++)
            {
                XTrain.SetRow(i, xMatrix.GetRow(indices[i]));
                yTrain[i] = yVector[indices[i]];
            }

            // Copy data to the validation sets
            for (int i = 0; i < validationSize; i++)
            {
                XValidation.SetRow(i, xMatrix.GetRow(indices[i + trainSize]));
                yValidation[i] = yVector[indices[i + trainSize]];
            }

            // Copy data to the test sets
            for (int i = 0; i < testSize; i++)
            {
                XTest.SetRow(i, xMatrix.GetRow(indices[i + trainSize + validationSize]));
                yTest[i] = yVector[indices[i + trainSize + validationSize]];
            }

            if (XTrain is not TInput xTrainOut ||
                yTrain is not TOutput yTrainOut ||
                XValidation is not TInput xValidationOut ||
                yValidation is not TOutput yValidationOut ||
                XTest is not TInput xTestOut ||
                yTest is not TOutput yTestOut)
            {
                throw new InvalidOperationException(
                    $"SplitData produced data of unexpected runtime types. " +
                    $"Expected X to be {typeof(TInput).Name} and y to be {typeof(TOutput).Name}.");
            }

            return (xTrainOut, yTrainOut, xValidationOut, yValidationOut, xTestOut, yTestOut);
        }
        else if (X is Tensor<T> xTensor && y is Tensor<T> yTensor)
        {
            totalSamples = xTensor.Shape[0];
            int trainSize = (int)(totalSamples * _options.TrainingSplitPercentage);
            int validationSize = (int)(totalSamples * _options.ValidationSplitPercentage);
            int testSize = totalSamples - trainSize - validationSize;

            var indices = Enumerable.Range(0, totalSamples).ToList();
            if (_options.ShuffleBeforeSplit)
            {
                // Shuffle the data
                var random = RandomHelper.CreateSeededRandom(_options.RandomSeed);
                indices = [.. indices.OrderBy(x => random.Next())];
            }

            // Create new tensors for the split data
            // Clone the shape for X but change the first dimension (sample count)
            int[] xTrainShape = (int[])xTensor.Shape.Clone();
            xTrainShape[0] = trainSize;
            var XTrain = new Tensor<T>(xTrainShape);

            int[] xValidationShape = (int[])xTensor.Shape.Clone();
            xValidationShape[0] = validationSize;
            var XValidation = new Tensor<T>(xValidationShape);

            int[] xTestShape = (int[])xTensor.Shape.Clone();
            xTestShape[0] = testSize;
            var XTest = new Tensor<T>(xTestShape);

            // Similarly for y
            int[] yTrainShape = (int[])yTensor.Shape.Clone();
            yTrainShape[0] = trainSize;
            var yTrain = new Tensor<T>(yTrainShape);

            int[] yValidationShape = (int[])yTensor.Shape.Clone();
            yValidationShape[0] = validationSize;
            var yValidation = new Tensor<T>(yValidationShape);

            int[] yTestShape = (int[])yTensor.Shape.Clone();
            yTestShape[0] = testSize;
            var yTest = new Tensor<T>(yTestShape);

            // Copy data to training sets
            for (int i = 0; i < trainSize; i++)
            {
                CopySample(xTensor, XTrain, indices[i], i);
                CopySample(yTensor, yTrain, indices[i], i);
            }

            // Copy data to validation sets
            for (int i = 0; i < validationSize; i++)
            {
                CopySample(xTensor, XValidation, indices[i + trainSize], i);
                CopySample(yTensor, yValidation, indices[i + trainSize], i);
            }

            // Copy data to test sets
            for (int i = 0; i < testSize; i++)
            {
                CopySample(xTensor, XTest, indices[i + trainSize + validationSize], i);
                CopySample(yTensor, yTest, indices[i + trainSize + validationSize], i);
            }

            if (XTrain is not TInput xTrainOut ||
                yTrain is not TOutput yTrainOut ||
                XValidation is not TInput xValidationOut ||
                yValidation is not TOutput yValidationOut ||
                XTest is not TInput xTestOut ||
                yTest is not TOutput yTestOut)
            {
                throw new InvalidOperationException(
                    $"SplitData produced data of unexpected runtime types. " +
                    $"Expected X to be {typeof(TInput).Name} and y to be {typeof(TOutput).Name}.");
            }

            return (xTrainOut, yTrainOut, xValidationOut, yValidationOut, xTestOut, yTestOut);
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported combination of input type {typeof(TInput).Name} and output type {typeof(TOutput).Name}. " +
                "Currently supported combinations are: " +
                $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) and " +
                $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>).");
        }
    }

    /// <summary>
    /// Copies a single sample from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="destination">The destination tensor where the data will be copied to.</param>
    /// <param name="sourceIndex">The index of the sample in the source tensor.</param>
    /// <param name="destIndex">The index where the sample should be placed in the destination tensor.</param>
    /// <remarks>
    /// <para>
    /// This method efficiently copies a sample (the first dimension of a tensor) from one tensor to another,
    /// handling tensors of arbitrary rank. It uses recursive traversal of tensor dimensions to copy all values
    /// associated with the specified sample index.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as copying a complete "record" from one dataset to another.
    /// 
    /// In machine learning, data is often organized where:
    /// - The first dimension represents different examples or samples (like different patients, houses, or images)
    /// - The remaining dimensions contain the features or characteristics of each sample
    /// 
    /// This method copies one complete sample with all its features from the source to the destination,
    /// preserving the structure of the data. It works regardless of how complex the data structure is
    /// (whether it's a simple table or a complex multi-dimensional array like a color image).
    /// </para>
    /// </remarks>
    private void CopySample(Tensor<T> source, Tensor<T> destination, int sourceIndex, int destIndex)
    {
        // Validate parameters
        if (source.Rank != destination.Rank)
        {
            throw new ArgumentException("Source and destination tensors must have the same rank");
        }

        for (int i = 1; i < source.Shape.Length; i++)
        {
            if (source.Shape[i] != destination.Shape[i])
            {
                throw new ArgumentException(
                    $"Source and destination tensors must have the same shape in all dimensions except the first. " +
                    $"Mismatch at dimension {i}: source={source.Shape[i]}, destination={destination.Shape[i]}");
            }
        }

        if (sourceIndex < 0 || sourceIndex >= source.Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(sourceIndex),
                $"Source index {sourceIndex} is out of range [0, {source.Shape[0] - 1}]");
        }

        if (destIndex < 0 || destIndex >= destination.Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(destIndex),
                $"Destination index {destIndex} is out of range [0, {destination.Shape[0] - 1}]");
        }

        // Recursive copy function to handle tensors of any dimension
        CopySampleRecursive(source, destination, sourceIndex, destIndex, 1, new int[source.Rank]);
    }

    /// <summary>
    /// Recursively copies tensor values across multiple dimensions.
    /// </summary>
    /// <param name="source">The source tensor.</param>
    /// <param name="destination">The destination tensor.</param>
    /// <param name="sourceIndex">The sample index in the source tensor.</param>
    /// <param name="destIndex">The sample index in the destination tensor.</param>
    /// <param name="currentDim">The current dimension being processed.</param>
    /// <param name="indices">Array of indices for tracking position in the tensor.</param>
    /// <remarks>
    /// <para>
    /// This recursive helper method traverses all dimensions of the tensor, building up the
    /// multi-dimensional indices needed to copy values from the source to the destination tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This method works like a specialized GPS that navigates through
    /// all points in a multi-dimensional space to copy values from one location to another.
    /// 
    /// Imagine you have a multi-floor building (3D space) and need to copy the contents of
    /// one apartment to another apartment with the same layout but on a different floor.
    /// This method systematically goes through each room and copies everything over.
    /// </para>
    /// </remarks>
    private void CopySampleRecursive(
        Tensor<T> source,
        Tensor<T> destination,
        int sourceIndex,
        int destIndex,
        int currentDim,
        int[] indices)
    {
        if (currentDim == source.Rank)
        {
            // We've built complete indices for all dimensions, now copy the value
            indices[0] = sourceIndex; // Set the sample index for source
            T value = source[indices];

            indices[0] = destIndex; // Set the sample index for destination
            destination[indices] = value;
        }
        else
        {
            // We're still building indices, continue recursion for the next dimension
            for (int i = 0; i < source.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                CopySampleRecursive(source, destination, sourceIndex, destIndex, currentDim + 1, indices);
            }
        }
    }
}
