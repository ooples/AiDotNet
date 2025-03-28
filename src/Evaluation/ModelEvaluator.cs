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
public class ModelEvaluator<T> : IModelEvaluator<T>
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
    public ModelEvaluator(PredictionStatsOptions? predictionOptions = null)
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
    public ModelEvaluationData<T> EvaluateModel(ModelEvaluationInput<T> input)
    {
        var model = input.Model ?? new VectorModel<T>(Vector<T>.Empty());

        var evaluationData = new ModelEvaluationData<T>
        {
            TrainingSet = CalculateDataSetStats(input.InputData.XTrain, input.InputData.YTrain, model),
            ValidationSet = CalculateDataSetStats(input.InputData.XVal, input.InputData.YVal, model),
            TestSet = CalculateDataSetStats(input.InputData.XTest, input.InputData.YTest, model),
            ModelStats = CalculateModelStats(model, input.InputData.XTrain, input.NormInfo)
        };

        return evaluationData;
    }

    /// <summary>
    /// Uses a symbolic model to make predictions on a set of input features.
    /// </summary>
    /// <param name="model">The model used to make predictions.</param>
    /// <param name="X">The input feature matrix where each row represents a data point and each column represents a feature.</param>
    /// <returns>A vector of predictions, one for each row in the input matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes your AI model and a set of input data,
    /// then generates predictions for each data point. It processes each row of your data
    /// one by one and returns all the predictions as a collection.
    /// </remarks>
    private static Vector<T> PredictWithSymbolicModel(ISymbolicModel<T> model, Matrix<T> X)
    {
        var predictions = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = model.Evaluate(X.GetRow(i));
        }

        return predictions;
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
    private DataSetStats<T> CalculateDataSetStats(Matrix<T> X, Vector<T> y, ISymbolicModel<T> model)
    {
        var predictions = PredictWithSymbolicModel(model, X);

        return new DataSetStats<T>
        {
            ErrorStats = CalculateErrorStats(y, predictions, X.Columns),
            ActualBasicStats = CalculateBasicStats(y),
            PredictedBasicStats = CalculateBasicStats(predictions),
            PredictionStats = CalculatePredictionStats(y, predictions, X.Columns),
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
    private static ErrorStats<T> CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int featureCount)
    {
        return new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = actual, Predicted = predicted, FeatureCount = featureCount });
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
    private static BasicStats<T> CalculateBasicStats(Vector<T> values)
    {
        return new BasicStats<T>(new BasicStatsInputs<T> { Values = values });
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
    private PredictionStats<T> CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int featureCount)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T> 
        { 
            Actual = actual, 
            Predicted = predicted, 
            NumberOfParameters = featureCount, 
            ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
            LearningCurveSteps = _predictionOptions.LearningCurveSteps 
        });
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
    private static ModelStats<T> CalculateModelStats(ISymbolicModel<T> model, Matrix<T> xTrain, NormalizationInfo<T> normInfo)
    {
        var predictionModelResult = new PredictionModelResult<T>(model, new OptimizationResult<T>(), normInfo);

        return new ModelStats<T>(new ModelStatsInputs<T>
        {
            XMatrix = xTrain,
            FeatureCount = xTrain.Columns,
            Model = predictionModelResult
        });
    }
}