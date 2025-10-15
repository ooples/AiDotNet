global using AiDotNet.CrossValidators;

namespace AiDotNet.Evaluation;

/// <summary>
/// Evaluates machine learning models by calculating various performance metrics.
/// </summary>
/// <typeparam name="T">The numeric data type used in the model (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> The ModelEvaluator helps you understand how well your AI model is performing.
/// It calculates metrics like accuracy and error rates that tell you if your model is making
/// good predictions or if it needs improvement.
/// 
/// Think of it like a report card for your AI model that shows its strengths and weaknesses.
/// </remarks>
public class DefaultModelEvaluator<T, TInput, TOutput> : IModelEvaluator<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for prediction statistics calculations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These options control how detailed the evaluation of your model will be,
    /// such as how confident we want to be in our results and how many steps to use when
    /// analyzing how the model learns.
    /// </remarks>
    protected readonly PredictionStatsOptions _predictionOptions;

    /// <summary>
    /// Initializes a new instance of the ModelEvaluator class.
    /// </summary>
    /// <param name="predictionOptions">Optional configuration for prediction statistics. If not provided, default options will be used.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new evaluator that will test how well your AI model works.
    /// You can customize how it evaluates by providing options, or just use the default settings.
    /// </remarks>
    public DefaultModelEvaluator(PredictionStatsOptions? predictionOptions = null)
    {
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
    }

    /// <summary>
    /// Evaluates a model using training, validation, and test datasets.
    /// </summary>
    /// <param name="input">The input data containing the model and datasets to evaluate.</param>
    /// <returns>A comprehensive evaluation of the model's performance across all datasets.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method tests your AI model on three different sets of data:
    /// 1. Training data - the data your model learned from
    /// 2. Validation data - data used to fine-tune your model
    /// 3. Test data - completely new data to see how well your model generalizes
    /// 
    /// It returns detailed statistics about how well your model performs on each dataset.
    /// </remarks>
    public ModelEvaluationData<T, TInput, TOutput> EvaluateModel(ModelEvaluationInput<T, TInput, TOutput> input)
    {
        var evaluationData = new ModelEvaluationData<T, TInput, TOutput>
        {
            TrainingSet = CalculateDataSetStats(input.InputData.XTrain, input.InputData.YTrain, input.Model),
            ValidationSet = CalculateDataSetStats(input.InputData.XValidation, input.InputData.YValidation, input.Model),
            TestSet = CalculateDataSetStats(input.InputData.XTest, input.InputData.YTest, input.Model),
            ModelStats = CalculateModelStats(input.Model, input.InputData.XTrain, input.NormInfo)
        };

        return evaluationData;
    }

    /// <summary>
    /// Calculates comprehensive statistics for a dataset using the provided model.
    /// </summary>
    /// <param name="X">The feature matrix containing input data.</param>
    /// <param name="y">The target vector containing actual values.</param>
    /// <param name="model">The model used to make predictions.</param>
    /// <returns>A collection of statistics about the dataset and model performance.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes how well your model performs on a specific dataset.
    /// It calculates:
    /// - How accurate the predictions are
    /// - Basic statistics about the actual values
    /// - Basic statistics about the predicted values
    /// - Detailed information about the prediction quality
    /// 
    /// This gives you a complete picture of your model's performance on this dataset.
    /// </remarks>
    private DataSetStats<T, TInput, TOutput> CalculateDataSetStats(TInput X, TOutput y, IFullModel<T, TInput, TOutput>? model)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model), "Cannot evaluate a null model.");
        }

        var predictions = model.Predict(X);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(predictions);
        var inputSize = InputHelper<T, TInput>.GetInputSize(X);
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(y);
        var metaData = model.GetModelMetadata();

        return new DataSetStats<T, TInput, TOutput>(metaData.ModelType)
        {
            ErrorStats = CalculateErrorStats(actual, predicted, inputSize, metaData.ModelType),
            ActualBasicStats = CalculateBasicStats(actual, metaData.ModelType),
            PredictedBasicStats = CalculateBasicStats(predicted, metaData.ModelType),
            PredictionStats = CalculatePredictionStats(actual, predicted, inputSize, metaData.ModelType),
            Predicted = predictions,
            Features = X,
            Actual = y
        };
    }

    /// <summary>
    /// Calculates error statistics by comparing actual values to predicted values.
    /// </summary>
    /// <param name="actual">The vector of actual target values.</param>
    /// <param name="predicted">The vector of values predicted by the model.</param>
    /// <param name="featureCount">The number of features used in the model.</param>
    /// <returns>Statistics about prediction errors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method measures how far off your model's predictions are from the actual values.
    /// It calculates metrics like:
    /// - Mean Squared Error (MSE): The average of the squared differences between predictions and actual values
    /// - Root Mean Squared Error (RMSE): The square root of MSE, which gives an error measure in the same units as your data
    /// - Mean Absolute Error (MAE): The average of the absolute differences between predictions and actual values
    /// 
    /// Lower values for these metrics indicate better model performance.
    /// </remarks>
    private static ErrorStats<T> CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int featureCount, ModelType modelType, NeuralNetworkTaskType? taskType = null)
    {
        return new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = actual, Predicted = predicted, FeatureCount = featureCount }, modelType, taskType);
    }

    /// <summary>
    /// Calculates basic statistical measures for a set of values.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <returns>Basic statistical measures such as mean, median, and standard deviation.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates common statistics about a set of numbers, such as:
    /// - Mean (average): The sum of all values divided by the count
    /// - Median: The middle value when all values are sorted
    /// - Standard Deviation: A measure of how spread out the values are
    /// - Min/Max: The smallest and largest values
    /// 
    /// These statistics help you understand the distribution of your data.
    /// </remarks>
    private static BasicStats<T> CalculateBasicStats(Vector<T> values, ModelType modelType)
    {
        return new BasicStats<T>(values, modelType);
    }

    /// <summary>
    /// Calculates advanced prediction statistics by comparing actual values to predicted values.
    /// </summary>
    /// <param name="actual">The vector of actual target values.</param>
    /// <param name="predicted">The vector of values predicted by the model.</param>
    /// <param name="featureCount">The number of features used in the model.</param>
    /// <returns>Advanced statistics about prediction quality.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates more sophisticated metrics about your model's performance:
    /// - R-squared: A measure between 0 and 1 that indicates how well your model explains the variation in the data
    /// - Adjusted R-squared: R-squared adjusted for the number of features in your model
    /// - Confidence intervals: Ranges that likely contain the true values
    /// - Learning curves: Show how your model's performance changes with more training data
    /// 
    /// These metrics help you understand not just how accurate your model is, but also how reliable and robust it is.
    /// </remarks>
    private PredictionStats<T> CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int featureCount, ModelType modelType)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T> 
        { 
            Actual = actual, 
            Predicted = predicted, 
            NumberOfParameters = featureCount, 
            ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
            LearningCurveSteps = _predictionOptions.LearningCurveSteps 
        }, modelType);
    }

    /// <summary>
    /// Calculates statistics about the model itself, independent of specific predictions.
    /// </summary>
    /// <param name="model">The model to analyze.</param>
    /// <param name="xTrain">The training feature matrix used to train the model.</param>
    /// <param name="normInfo">Information about how the data was normalized.</param>
    /// <returns>Statistics about the model's structure and characteristics.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes the model itself rather than its predictions.
    /// It examines properties like:
    /// - Model complexity: How many parameters or features the model uses
    /// - Feature importance: Which input features have the most influence on predictions
    /// - Model structure: Information about how the model is organized
    /// 
    /// This helps you understand what makes your model tick and which inputs matter most.
    /// </remarks>
    private static ModelStats<T, TInput, TOutput> CalculateModelStats(IFullModel<T, TInput, TOutput>? model, TInput xTrain, NormalizationInfo<T, TInput, TOutput> normInfo)
    {
        var predictionModelResult = new PredictionModelResult<T, TInput, TOutput>(new OptimizationResult<T, TInput, TOutput>() { BestSolution = model }, normInfo);

        return new ModelStats<T, TInput, TOutput>(new ModelStatsInputs<T, TInput, TOutput>
        {
            XMatrix = xTrain,
            FeatureCount = InputHelper<T, TInput>.GetInputSize(xTrain),
            Model = predictionModelResult?.Model
        }, modelType: model?.GetModelMetadata().ModelType ?? ModelType.None);
    }

    /// <summary>
    /// Performs cross-validation on the given model using the provided data and options.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <param name="crossValidator">Optional custom cross-validator implementation.</param>
    /// <returns>A CrossValidationResult containing the evaluation metrics for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method performs cross-validation to assess how well the model generalizes to unseen data. It splits the data into 
    /// multiple subsets (folds), trains the model on a portion of the data, and evaluates it on the held-out portion. This process 
    /// is repeated multiple times to get a robust estimate of the model's performance. The method allows for customization of the 
    /// cross-validation process through options and even allows for a custom cross-validator implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This method tests how well your model performs on different subsets of your data.
    /// 
    /// Cross-validation:
    /// - Splits your data into several parts (called folds)
    /// - Trains the model multiple times, each time using a different part as a test set
    /// - Helps understand how well the model will work on new, unseen data
    /// 
    /// This is useful for:
    /// - Getting a more reliable estimate of model performance
    /// - Detecting overfitting (when a model works well on training data but poorly on new data)
    /// - Comparing different models to see which one generalizes better
    /// 
    /// For example, in 5-fold cross-validation, your data is split into 5 parts. The model is trained 5 times, 
    /// each time using 4 parts for training and 1 for testing. The results are then averaged to get an overall performance score.
    /// </para>
    /// </remarks>
    public CrossValidationResult<T> PerformCrossValidation(
        IFullModel<T, Matrix<T>, Vector<T>> model,
        Matrix<T> X,
        Vector<T> y,
        ICrossValidator<T>? crossValidator = null)
    {
        crossValidator ??= new StandardCrossValidator<T>(model.GetModelMetadata().ModelType);

        return crossValidator.Validate(model, X, y);
    }
}