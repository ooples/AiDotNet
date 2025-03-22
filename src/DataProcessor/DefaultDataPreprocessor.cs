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
public class DefaultDataPreprocessor<T> : IDataPreprocessor<T>
{
    /// <summary>
    /// Component responsible for scaling data to a standard range.
    /// </summary>
    private readonly INormalizer<T> _normalizer;
    
    /// <summary>
    /// Component responsible for selecting the most relevant features from the dataset.
    /// </summary>
    private readonly IFeatureSelector<T> _featureSelector;
    
    /// <summary>
    /// Component responsible for identifying and removing outliers from the dataset.
    /// </summary>
    private readonly IOutlierRemoval<T> _outlierRemoval;
    
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
    public DefaultDataPreprocessor(INormalizer<T> normalizer, IFeatureSelector<T> featureSelector, IOutlierRemoval<T> outlierRemoval, DataProcessorOptions? options = null)
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
    public (Matrix<T> X, Vector<T> y, NormalizationInfo<T> normInfo) PreprocessData(Matrix<T> X, Vector<T> y)
    {
        NormalizationInfo<T> normInfo = new();

        (X, y) = _outlierRemoval.RemoveOutliers(X, y);

        if (_options.NormalizeBeforeFeatureSelection)
        {
            (X, normInfo.XParams) = _normalizer.NormalizeMatrix(X);
            (y, normInfo.YParams) = _normalizer.NormalizeVector(y);
            X = _featureSelector.SelectFeatures(X);
        }
        else
        {
            X = _featureSelector.SelectFeatures(X);
            (X, normInfo.XParams) = _normalizer.NormalizeMatrix(X);
            (y, normInfo.YParams) = _normalizer.NormalizeVector(y);
        }

        return (X, y, normInfo);
    }

    /// <summary>
    /// Splits the dataset into training, validation, and test sets.
    /// </summary>
    /// <param name="X">The feature matrix where rows represent samples and columns represent features.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
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
    public (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XValidation, Vector<T> yValidation, Matrix<T> XTest, Vector<T> yTest) SplitData(Matrix<T> X, Vector<T> y)
    {
        int totalSamples = X.Rows;
        int trainSize = (int)(totalSamples * _options.TrainingSplitPercentage);
        int validationSize = (int)(totalSamples * _options.ValidationSplitPercentage);
        int testSize = totalSamples - trainSize - validationSize;

        // Shuffle the data
        var random = new Random(_options.RandomSeed);
        var indices = Enumerable.Range(0, totalSamples).ToList();
        indices = [.. indices.OrderBy(x => random.Next())];

        // Split the data
        var XTrain = new Matrix<T>(trainSize, X.Columns);
        var yTrain = new Vector<T>(trainSize);
        var XValidation = new Matrix<T>(validationSize, X.Columns);
        var yValidation = new Vector<T>(validationSize);
        var XTest = new Matrix<T>(testSize, X.Columns);
        var yTest = new Vector<T>(testSize);

        for (int i = 0; i < trainSize; i++)
        {
            XTrain.SetRow(i, X.GetRow(indices[i]));
            yTrain[i] = y[indices[i]];
        }

        for (int i = 0; i < validationSize; i++)
        {
            XValidation.SetRow(i, X.GetRow(indices[i + trainSize]));
            yValidation[i] = y[indices[i + trainSize]];
        }

        for (int i = 0; i < testSize; i++)
        {
            XTest.SetRow(i, X.GetRow(indices[i + trainSize + validationSize]));
            yTest[i] = y[indices[i + trainSize + validationSize]];
        }

        return (XTrain, yTrain, XValidation, yValidation, XTest, yTest);
    }
}