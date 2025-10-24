global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.DataProcessor;
global using AiDotNet.FitDetectors;

namespace AiDotNet;

/// <summary>
/// A builder class that helps create and configure machine learning prediction models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class uses the builder pattern to configure various components of a machine learning model
/// before building and using it for predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this class as a recipe builder for creating AI models.
/// You add different ingredients (like data normalization, feature selection, etc.) 
/// and then "cook" (build) the final model. This approach makes it easy to customize
/// your model without having to understand all the complex details at once.
/// </para>
/// </remarks>
public class PredictionModelBuilder<T, TInput, TOutput> : IPredictionModelBuilder<T, TInput, TOutput>
{
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
    private IRegularization<T, TInput, TOutput>? _regularization;
    private IFitnessCalculator<T, TInput, TOutput>? _fitnessCalculator;
    private IFitDetector<T, TInput, TOutput>? _fitDetector;
    private IFullModel<T, TInput, TOutput>? _model;
    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataPreprocessor<T, TInput, TOutput>? _dataPreprocessor;
    private IOutlierRemoval<T, TInput, TOutput>? _outlierRemoval;

    /// <summary>
    /// Configures which features (input variables) should be used in the model.
    /// </summary>
    /// <param name="selector">The feature selection strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Sometimes, not all of your data is useful for making predictions.
    /// Feature selection helps pick out which parts of your data are most important.
    /// For example, when predicting house prices, the number of bedrooms might be important,
    /// but the house's street number probably isn't.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFeatureSelector(IFeatureSelector<T, TInput> selector)
    {
        _featureSelector = selector;
        return this;
    }

    /// <summary>
    /// Configures how the input data should be normalized (scaled).
    /// </summary>
    /// <param name="normalizer">The normalization strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Normalization makes sure all your data is on a similar scale.
    /// For example, if you have data about people's ages (0-100) and incomes ($0-$1,000,000),
    /// normalization might scale both to ranges like 0-1 so the model doesn't think
    /// income is 10,000 times more important than age just because the numbers are bigger.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureNormalizer(INormalizer<T, TInput, TOutput> normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    /// <summary>
    /// Configures regularization to prevent overfitting in the model.
    /// </summary>
    /// <param name="regularization">The regularization strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Regularization helps prevent your model from "memorizing" the training data
    /// instead of learning general patterns. It's like teaching a student to understand the concepts
    /// rather than just memorizing answers to specific questions. This helps the model perform better
    /// on new, unseen data.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization)
    {
        _regularization = regularization;
        return this;
    }

    /// <summary>
    /// Configures how to measure the model's performance.
    /// </summary>
    /// <param name="calculator">The fitness calculation strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how we score how well our model is doing.
    /// Different problems might need different scoring methods. For example, when predicting house prices,
    /// we might care about the average error in dollars, but when predicting if an email is spam,
    /// we might care more about the percentage of emails correctly classified.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator)
    {
        _fitnessCalculator = calculator;
        return this;
    }

    /// <summary>
    /// Configures how to detect if the model is overfitting or underfitting.
    /// </summary>
    /// <param name="detector">The fit detection strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This helps detect if your model is learning too much from the training data
    /// (overfitting) or not learning enough (underfitting). It's like having a teacher who can tell
    /// if a student is just memorizing answers or not studying enough.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector)
    {
        _fitDetector = detector;
        return this;
    }

    /// <summary>
    /// Configures the core algorithm to use for predictions.
    /// </summary>
    /// <param name="model">The prediction algorithm to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main "brain" of your AI model - the algorithm that will
    /// learn patterns from your data and make predictions. Different algorithms work better for
    /// different types of problems, so you can choose the one that fits your needs.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModel(IFullModel<T, TInput, TOutput> model)
    {
        _model = model;
        return this;
    }

    /// <summary>
    /// Configures the optimization algorithm to find the best model parameters.
    /// </summary>
    /// <param name="optimizationAlgorithm">The optimization algorithm to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The optimizer helps find the best settings for your model.
    /// It's like having someone adjust the knobs on a radio to get the clearest signal.
    /// The optimizer tries different settings and keeps the ones that work best.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm)
    {
        _optimizer = optimizationAlgorithm;
        return this;
    }

    /// <summary>
    /// Configures how the data should be preprocessed before training.
    /// </summary>
    /// <param name="dataPreprocessor">The data preprocessing strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Data preprocessing cleans and prepares your raw data before feeding it to the model.
    /// It's like washing and cutting vegetables before cooking. This might include handling missing values,
    /// converting text to numbers, or combining related features.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataPreprocessor(IDataPreprocessor<T, TInput, TOutput> dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    /// <summary>
    /// Configures how to detect and handle outliers in the data.
    /// </summary>
    /// <param name="outlierRemoval">The outlier removal strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Outliers are unusual data points that are very different from the rest of your data.
    /// For example, if you're analyzing house prices and most are between $100,000-$500,000,
    /// a $10,000,000 mansion would be an outlier. These unusual points can sometimes confuse the model,
    /// so we might want to handle them specially.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureOutlierRemoval(IOutlierRemoval<T, TInput, TOutput> outlierRemoval)
    {
        _outlierRemoval = outlierRemoval;
        return this;
    }

        /// <summary>
    /// Builds a predictive model using the provided input features and output values.
    /// </summary>
    /// <param name="x">The matrix of input features where each row is a data point and each column is a feature.</param>
    /// <param name="y">The vector of output values corresponding to each row in the input matrix.</param>
    /// <returns>A trained predictive model that can be used to make predictions.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input features or output values are null.</exception>
    /// <exception cref="ArgumentException">Thrown when the number of rows in the features matrix doesn't match the length of the output vector.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no regression method has been specified.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes your data (inputs and known outputs) and creates a trained AI model.
    /// Think of it like teaching a student: you provide examples (your data) and the student (the model) learns
    /// patterns from these examples. After building, your model is ready to make predictions on new data.
    /// 
    /// The input matrix 'x' contains your features (like house size, number of bedrooms, etc. if predicting house prices),
    /// and the vector 'y' contains the known answers (actual house prices) for those examples.
    /// </remarks>
    public IPredictiveModel<T, TInput, TOutput> Build(TInput x, TOutput y)
    {
        var convertedX = ConversionsHelper.ConvertToMatrix<T, TInput>(x);
        var convertedY = ConversionsHelper.ConvertToVector<T, TOutput>(y);

        // Validate inputs
        if (x == null)
            throw new ArgumentNullException(nameof(x), "Input features matrix can't be null");
        if (y == null)
            throw new ArgumentNullException(nameof(y), "Output vector can't be null");
        if (convertedX.Rows != convertedY.Length)
            throw new ArgumentException("Number of rows in features must match length of actual values", nameof(x));
        if (_model == null)
            throw new InvalidOperationException("Model implementation must be specified");

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
        var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>();
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();
        var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(normalizer, featureSelector, outlierRemoval);

        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Optimize the model
        var optimizationResult = optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));

        return new PredictionModelResult<T, TInput, TOutput>(optimizationResult, normInfo);
    }

    /// <summary>
    /// Uses a trained model to make predictions on new data.
    /// </summary>
    /// <param name="newData">The matrix of new input features to predict outcomes for.</param>
    /// <param name="modelResult">The trained predictive model to use for making predictions.</param>
    /// <returns>A vector containing the predicted output values for each row in the input matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> After training your model with the Build method, you can use this method to get
    /// predictions for new data. For example, if you trained a model to predict house prices based on features
    /// like size and location, you can now give it details of houses currently for sale (without knowing their prices)
    /// and the model will predict what their prices should be.
    /// 
    /// The input matrix should have the same number of columns (features) as the data you used to train the model.
    /// </remarks>
    public TOutput Predict(TInput newData, IPredictiveModel<T, TInput, TOutput> modelResult)
    {
        return modelResult.Predict(newData);
    }

    /// <summary>
    /// Saves a trained model to a file so it can be used later without retraining.
    /// </summary>
    /// <param name="modelResult">The trained predictive model to save.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Training a model can take time, so once you have a good model,
    /// you'll want to save it. This method lets you store your trained model in a file on your computer.
    /// Later, you can load this saved model and use it to make predictions without having to train it again.
    /// 
    /// Think of it like saving a document in a word processor - you can close the program and come back later
    /// to continue where you left off.
    /// </remarks>
    public void SaveModel(IPredictiveModel<T, TInput, TOutput> modelResult, string filePath)
    {
        File.WriteAllBytes(filePath, SerializeModel(modelResult));
    }

    /// <summary>
    /// Loads a previously saved model from a file.
    /// </summary>
    /// <param name="filePath">The file path where the model was saved.</param>
    /// <returns>The loaded predictive model that can be used to make predictions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method lets you load a model that you previously saved using the SaveModel method.
    /// Once loaded, you can immediately use the model to make predictions without having to train it again.
    /// 
    /// This is useful when you want to use your model in different applications or at different times
    /// without the time and computational cost of retraining.
    /// </remarks>
    public IPredictiveModel<T, TInput, TOutput> LoadModel(string filePath)
    {
        byte[] modelData = File.ReadAllBytes(filePath);
        return DeserializeModel(modelData);
    }

    /// <summary>
    /// Converts a trained model into a byte array for storage or transmission.
    /// </summary>
    /// <param name="modelResult">The trained predictive model to serialize.</param>
    /// <returns>A byte array representing the serialized model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Serialization converts your model into a format (a series of bytes) that can be
    /// easily stored or sent over a network. This is the underlying mechanism that makes saving models possible.
    /// 
    /// You might use this directly if you want to store the model in a database or send it over a network,
    /// rather than saving it to a file.
    /// </remarks>
    public byte[] SerializeModel(IPredictiveModel<T, TInput, TOutput> modelResult)
    {
        return modelResult.Serialize();
    }

    /// <summary>
    /// Converts a byte array back into a usable predictive model.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <returns>The deserialized predictive model that can be used to make predictions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the byte array
    /// representation of your model and converts it back into a usable model object. This is what happens
    /// behind the scenes when you load a model from a file.
    /// 
    /// You might use this directly if you retrieved a serialized model from a database or received it over a network.
    /// </remarks>
    public IPredictiveModel<T, TInput, TOutput> DeserializeModel(byte[] modelData)
    {
        var result = new PredictionModelResult<T, TInput, TOutput>();
        result.Deserialize(modelData);

        return result;
    }
}