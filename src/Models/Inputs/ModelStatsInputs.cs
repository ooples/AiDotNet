namespace AiDotNet.Models.Inputs;

/// <summary>
/// Represents a container for inputs needed to calculate various statistics and metrics for a model.
/// </summary>
/// <remarks>
/// <para>
/// This class holds the data and parameters necessary for evaluating model performance and calculating
/// statistics. It includes actual and predicted values, feature information, the model itself, and
/// optional functions for fitting and prediction.
/// </para>
/// <para><b>For Beginners:</b> This class collects all the data needed to evaluate how well a model performs.
/// 
/// Think of it like gathering all the ingredients before baking:
/// - Actual values (what really happened)
/// - Predicted values (what the model thought would happen)
/// - Feature data (the information used to make predictions)
/// - The model itself and its parameters
/// 
/// This organized collection makes it easier to calculate accuracy metrics, perform 
/// diagnostic tests, and visualize model performance without having to pass many
/// separate parameters around.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data the model accepts.</typeparam>
/// <typeparam name="TOutput">The type of output data the model produces.</typeparam>
public class ModelStatsInputs<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the actual observed values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This contains the true values that were observed in reality. These are compared with
    /// the model's predictions to evaluate performance and calculate error metrics.
    /// </para>
    /// <para><b>For Beginners:</b> These are the real values that actually occurred.
    /// 
    /// For example:
    /// - In sales prediction: The actual sales figures that occurred
    /// - In temperature forecasting: The temperatures that were actually recorded
    /// - In classification: The true categories that items belong to
    /// 
    /// These values serve as the "ground truth" against which we measure how accurate
    /// the model's predictions were.
    /// </para>
    /// </remarks>
    public TOutput Actual { get; set; }

    /// <summary>
    /// Gets or sets the values predicted by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This contains the values predicted by the model. These are compared with
    /// the actual values to evaluate the model's performance and accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> These are the values that your model predicted would occur.
    /// 
    /// For example:
    /// - In sales prediction: The sales figures the model predicted
    /// - In temperature forecasting: The temperatures the model forecasted
    /// - In classification: The categories the model assigned to items
    /// 
    /// By comparing these predictions to what actually happened, we can measure
    /// how accurate the model is.
    /// </para>
    /// </remarks>
    public TOutput Predicted { get; set; }

    /// <summary>
    /// Gets or sets the number of features or predictor variables in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates how many features or predictor variables are used in the model.
    /// It's important for correctly interpreting the coefficient vector and for certain statistical tests.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of input variables your model uses to make predictions.
    /// 
    /// For example:
    /// - A house price model might use 5 features (size, location, age, bedrooms, bathrooms)
    /// - A weather model might use 10 features (temperature, pressure, humidity, etc.)
    /// 
    /// Knowing the feature count helps interpret model statistics and is needed for
    /// certain calculations like degrees of freedom in statistical tests.
    /// </para>
    /// </remarks>
    public int FeatureCount { get; set; }

    /// <summary>
    /// Gets or sets the input data used for predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This contains the feature values used as inputs to the model. The exact structure depends
    /// on the model type, but typically each row or element represents an observation with its features.
    /// </para>
    /// <para><b>For Beginners:</b> This is the input data your model used to make predictions.
    /// 
    /// For example, in a house price prediction model, this might be a table where:
    /// - Each row is a different house
    /// - Each column is a feature (size, location, age, bedrooms, bathrooms)
    /// 
    /// This data is essential for many advanced statistical calculations,
    /// such as confidence intervals and influence measures.
    /// </para>
    /// </remarks>
    public TInput XMatrix { get; set; }

    /// <summary>
    /// Gets or sets a function that fits a model to data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optional function takes input data and target values and returns predicted values.
    /// It's used for certain statistical calculations that require refitting the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is a function that can train a model on data.
    /// 
    /// Think of it as the recipe for how your model learns from data:
    /// - It takes in features (inputs) and actual values
    /// - It figures out the best way to predict the actual values from the features
    /// - It returns predictions based on what it learned
    /// 
    /// This is used for advanced techniques like cross-validation or bootstrapping
    /// where we need to retrain the model multiple times.
    /// </para>
    /// </remarks>
    public Func<TInput, TOutput>? FitFunction { get; set; }

    /// <summary>
    /// Gets or sets the coefficient values for the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the coefficients or parameters learned by the model during training.
    /// These coefficients determine how the features are used to generate predictions.
    /// </para>
    /// <para><b>For Beginners:</b> These are the weights your model assigned to each feature.
    /// 
    /// For example, in a house price model:
    /// - A coefficient of 100 for house size means each additional square foot adds $100 to the predicted price
    /// - A coefficient of -5000 for age means each additional year reduces the predicted price by $5000
    /// 
    /// These values tell you how important each feature is to the model and in what direction
    /// (positive or negative) they influence the prediction.
    /// </para>
    /// </remarks>
    public Vector<T> Coefficients { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Gets or sets the predictive model used to generate the statistics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property holds a reference to the actual model object that generated the predictions.
    /// Having the model available allows for calculating additional statistics or making new predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual model object that made the predictions.
    /// 
    /// Having the model itself (not just its predictions) allows us to:
    /// - Make new predictions with different data
    /// - Extract information about how the model works
    /// - Calculate advanced statistics specific to this type of model
    /// 
    /// It's like having the full recipe book, not just the meal that was cooked.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Gets or sets the names of the features used in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains the names of the features or predictor variables used in the model.
    /// Having descriptive names makes it easier to interpret coefficients and other statistics.
    /// </para>
    /// <para><b>For Beginners:</b> These are the labels for each input variable your model uses.
    /// 
    /// For example, in a house price model, feature names might be:
    /// - "SquareFootage"
    /// - "Neighborhood"
    /// - "YearBuilt"
    /// - "Bedrooms"
    /// - "Bathrooms"
    /// 
    /// Having names instead of just "Feature1", "Feature2", etc. makes it much easier
    /// to understand what the model is actually using to make predictions.
    /// </para>
    /// </remarks>
    public List<string> FeatureNames { get; set; } = [];

    /// <summary>
    /// Gets or sets the values for each feature organized by feature name.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary maps feature names to their values across all observations.
    /// The exact structure of the values depends on the model type and data representation.
    /// This format can be more convenient than a single input structure for certain operations and visualizations.
    /// </para>
    /// <para><b>For Beginners:</b> This stores all values for each feature separately.
    /// 
    /// Instead of having one big table, this breaks out each feature into its own collection:
    /// - "SquareFootage": [1500, 2200, 1800, ...]
    /// - "Bedrooms": [3, 4, 2, ...]
    /// 
    /// This format makes it easier to:
    /// - Calculate statistics for specific features
    /// - Create plots showing individual feature distributions
    /// - Perform feature-specific transformations
    /// </para>
    /// </remarks>
    public Dictionary<string, TOutput> FeatureValues { get; set; } = [];

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelStatsInputs{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new empty container for model statistics inputs.
    /// 
    /// It initializes all the properties with appropriate empty values based on the types
    /// of inputs and outputs your model uses.
    /// </para>
    /// </remarks>
    public ModelStatsInputs()
    {
        // Initialize with appropriate empty instances based on the generic types
        if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Vector<T>))
        {
            XMatrix = (TInput)(object)Matrix<T>.Empty();
            Actual = (TOutput)(object)Vector<T>.Empty();
            Predicted = (TOutput)(object)Vector<T>.Empty();
        }
        else if (typeof(TInput) == typeof(Tensor<T>) && typeof(TOutput) == typeof(Tensor<T>))
        {
            XMatrix = (TInput)(object)Tensor<T>.Empty();
            Actual = (TOutput)(object)Tensor<T>.Empty();
            Predicted = (TOutput)(object)Tensor<T>.Empty();
        }
        else
        {
            // For other combinations, provide a clear error message
            throw new InvalidOperationException(
                $"Unsupported combination of input type {typeof(TInput).Name} and output type {typeof(TOutput).Name}. " +
                "Currently supported combinations are: " +
                $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) for linear models and " +
                $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>) for neural network models.");
        }
    }
}
