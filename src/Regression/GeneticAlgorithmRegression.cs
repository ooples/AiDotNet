namespace AiDotNet.Regression;

/// <summary>
/// Implements a regression model that uses genetic algorithms to optimize model parameters,
/// mimicking the process of natural selection to find the best solution.
/// </summary>
/// <remarks>
/// <para>
/// Genetic Algorithm Regression uses evolutionary principles to find optimal model coefficients.
/// It maintains a population of potential solutions (models) that evolve over generations through
/// selection, crossover, and mutation operations. This approach is particularly useful for complex
/// problems where traditional optimization methods might struggle, as it can effectively explore
/// large solution spaces and avoid local optima.
/// </para>
/// <para><b>For Beginners:</b> This model uses a technique inspired by natural evolution to find the best solution.
/// 
/// Think of it like breeding the best solution:
/// - Start with a random "population" of potential solutions (different sets of coefficients)
/// - Test how well each solution performs on your data (fitness evaluation)
/// - Keep the best solutions and let them "reproduce" to create new solutions
/// - Occasionally introduce random changes (mutations) to explore new possibilities
/// - Repeat this process over multiple "generations" until you find an excellent solution
/// 
/// The benefit of this approach is that it can find good solutions to complex problems
/// without getting stuck in suboptimal answers. It's similar to how nature evolves
/// successful organisms over time, but applied to finding the best mathematical model.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GeneticAlgorithmRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the genetic algorithm optimizer.
    /// </summary>
    private readonly GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>> _gaOptions;

    /// <summary>
    /// The genetic algorithm optimizer that finds optimal model parameters.
    /// Created during training when input dimensions are known.
    /// </summary>
    private GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>? _optimizer;

    /// <summary>
    /// Component responsible for normalizing feature values to a common scale.
    /// </summary>
    private readonly INormalizer<T, Matrix<T>, Vector<T>> _normalizer;

    /// <summary>
    /// Component that selects the most relevant features for the model.
    /// </summary>
    private readonly IFeatureSelector<T, Matrix<T>> _featureSelector;

    /// <summary>
    /// Component that identifies and removes outliers from the training data.
    /// </summary>
    private readonly IOutlierRemoval<T, Matrix<T>, Vector<T>> _outlierRemoval;

    /// <summary>
    /// Component that handles all data preprocessing steps before training.
    /// </summary>
    private readonly IDataPreprocessor<T, Matrix<T>, Vector<T>> _dataPreprocessor;

    /// <summary>
    /// The best model found by the genetic algorithm.
    /// </summary>
    private IFullModel<T, Matrix<T>, Vector<T>>? _bestModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="GeneticAlgorithmRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional regression options for the model.</param>
    /// <param name="gaOptions">Optional configuration options for the genetic algorithm optimizer.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <param name="normalizer">Optional component for normalizing feature values.</param>
    /// <param name="featureSelector">Optional component for selecting relevant features.</param>
    /// <param name="outlierRemoval">Optional component for removing outliers.</param>
    /// <param name="dataPreprocessor">Optional component for preprocessing data.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Genetic Algorithm Regression model with the specified components and configuration
    /// options. If components are not provided, default implementations are used. The constructor sets up all the
    /// necessary infrastructure for the genetic algorithm to optimize model parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Genetic Algorithm Regression model.
    /// 
    /// The constructor allows you to customize many aspects of the model:
    /// - General regression settings (like whether to include an intercept term)
    /// - Genetic algorithm settings (like population size and mutation rate)
    /// - How to measure how well solutions perform (fitness calculation)
    /// - How to prepare your data before training (normalization, feature selection, outlier removal)
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a basic model with default settings
    /// var gaRegression = new GeneticAlgorithmRegression&lt;double&gt;();
    /// 
    /// // Create a model with custom genetic algorithm settings
    /// var gaOptions = new GeneticAlgorithmOptimizerOptions {
    ///     PopulationSize = 200,
    ///     MaxGenerations = 100,
    ///     MutationRate = 0.05
    /// };
    /// var customGaRegression = new GeneticAlgorithmRegression&lt;double&gt;(gaOptions: gaOptions);
    /// ```
    /// </para>
    /// </remarks>
    public GeneticAlgorithmRegression(
        RegressionOptions<T>? options = null,
        GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>? gaOptions = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        INormalizer<T, Matrix<T>, Vector<T>>? normalizer = null,
        IFeatureSelector<T, Matrix<T>>? featureSelector = null,
        IOutlierRemoval<T, Matrix<T>, Vector<T>>? outlierRemoval = null,
        IDataPreprocessor<T, Matrix<T>, Vector<T>>? dataPreprocessor = null)
        : base(options, regularization)
    {
        _gaOptions = gaOptions ?? new GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>();
        var dummyModel = new VectorModel<T>(Vector<T>.Empty());
        _optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(dummyModel, _gaOptions);
        _normalizer = normalizer ?? new NoNormalizer<T, Matrix<T>, Vector<T>>();
        _featureSelector = featureSelector ?? new NoFeatureSelector<T, Matrix<T>>();
        _outlierRemoval = outlierRemoval ?? new NoOutlierRemoval<T, Matrix<T>, Vector<T>>();
        // Note: outlierRemoval is kept for backward compatibility but is no longer passed to DefaultDataPreprocessor
        // Outlier removal should now be configured via DataPreparationPipeline through AiModelBuilder.ConfigureDataPreparation()
        _dataPreprocessor = dataPreprocessor ?? new DefaultDataPreprocessor<T, Matrix<T>, Vector<T>>(_normalizer, _featureSelector);
    }

    /// <summary>
    /// Trains the Genetic Algorithm Regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Genetic Algorithm Regression model by first preprocessing the data, then splitting it
    /// into training, validation, and test sets, and finally using a genetic algorithm to find the optimal model
    /// parameters. The genetic algorithm evolves a population of potential solutions over multiple generations,
    /// gradually improving the model's fit to the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// The training process involves several steps:
    /// 1. Preprocessing the data (normalizing features, removing outliers, etc.)
    /// 2. Splitting the data into separate sets for training and testing
    /// 3. Running the genetic algorithm, which:
    ///    - Creates a starting population of random solutions
    ///    - Evaluates how well each solution performs
    ///    - Selects the best solutions to "reproduce"
    ///    - Creates new solutions through crossover and mutation
    ///    - Repeats this process over multiple generations
    /// 4. Selects the best performing solution as the final model
    /// 
    /// After training, the model will be ready to make predictions on new data.
    /// 
    /// Example:
    /// ```csharp
    /// // Train the model
    /// gaRegression.Train(features, targets);
    /// ```
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Preprocess the data
        var (preprocessedX, preprocessedY, _) = _dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (xTrain, yTrain, xVal, yVal, xTest, yTest) = _dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // If HasIntercept is true, prepend a column of 1s to each matrix for the intercept term
        if (HasIntercept)
        {
            xTrain = PrependInterceptColumn(xTrain);
            xVal = PrependInterceptColumn(xVal);
            xTest = PrependInterceptColumn(xTest);
        }

        // Initialize optimizer with proper dimensions based on input data
        int featureCount = xTrain.Columns;
        _bestModel = new VectorModel<T>(new Vector<T>(featureCount));
        _optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(_bestModel, _gaOptions);

        var result = _optimizer.Optimize(OptimizerHelper<T, Matrix<T>, Vector<T>>.CreateOptimizationInputData(xTrain, yTrain, xVal, yVal, xTest, yTest));

        _bestModel = result.BestSolution;
        UpdateCoefficientsAndIntercept();
    }

    /// <summary>
    /// Prepends a column of 1s to the matrix for the intercept term.
    /// </summary>
    private Matrix<T> PrependInterceptColumn(Matrix<T> matrix)
    {
        var result = new Matrix<T>(matrix.Rows, matrix.Columns + 1);
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i, 0] = NumOps.One;
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j + 1] = matrix[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained Genetic Algorithm Regression model.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data using the best model found during the genetic algorithm
    /// optimization process. It applies the learned coefficients to the input features to compute the predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// Once your model has been trained, you can use it to predict values for new data points.
    /// The model applies the best set of coefficients discovered by the genetic algorithm
    /// to calculate predicted values for each input sample.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = gaRegression.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        if (_bestModel == null)
            return Vector<T>.Empty();

        // If HasIntercept is true, prepend a column of 1s to match the model's expected input
        var input = HasIntercept ? PrependInterceptColumn(x) : x;
        return _bestModel.Predict(input);
    }

    /// <summary>
    /// Gets the model type of the Genetic Algorithm Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType() => ModelType.GeneticAlgorithmRegression;

    /// <summary>
    /// Updates the model coefficients and intercept based on the best solution found by the genetic algorithm.
    /// </summary>
    private void UpdateCoefficientsAndIntercept()
    {
        Coefficients = _bestModel?.GetParameters() ?? Vector<T>.Empty();

        if (HasIntercept && Coefficients.Length > 0)
        {
            Intercept = Coefficients[0];
            Coefficients = Coefficients.Length > 1
                ? Coefficients.Slice(1, Coefficients.Length - 1)
                : Vector<T>.Empty();
        }
        else
        {
            Intercept = NumOps.Zero;
        }
    }

    /// <summary>
    /// Serializes the Genetic Algorithm Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Genetic Algorithm Regression model into a byte array that can be stored in a file,
    /// database, or transmitted over a network. The serialized data includes the base regression model data,
    /// the best model coefficients found by the genetic algorithm, and the genetic algorithm configuration options.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - The model coefficients discovered by the genetic algorithm
    /// - Settings like population size and mutation rate
    /// - Other information needed to recreate the exact same model
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = gaRegression.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("gaRegression.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GeneticAlgorithmRegression specific data
        var parameters = _bestModel?.GetParameters() ?? Vector<T>.Empty();
        writer.Write(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            writer.Write(Convert.ToDouble(parameters[i]));
        }

        // Serialize GeneticAlgorithmOptions
        var gaOptions = _gaOptions;
        writer.Write(gaOptions.MaxGenerations);
        writer.Write(gaOptions.PopulationSize);
        writer.Write(gaOptions.MutationRate);
        writer.Write(gaOptions.CrossoverRate);

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Genetic Algorithm Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a Genetic Algorithm Regression model from a byte array that was previously created
    /// using the Serialize method. It restores the base regression model data, the best model coefficients found
    /// by the genetic algorithm, and the genetic algorithm configuration options, allowing the model to be used
    /// for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - All settings are restored
    /// - The best solution found by the genetic algorithm is recovered
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("gaRegression.model");
    /// 
    /// // Deserialize the model
    /// var gaRegression = new GeneticAlgorithmRegression&lt;double&gt;();
    /// gaRegression.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = gaRegression.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GeneticAlgorithmRegression specific data
        int coefficientsLength = reader.ReadInt32();
        var coefficients = new T[coefficientsLength];
        for (int i = 0; i < coefficientsLength; i++)
        {
            coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _bestModel = new VectorModel<T>(new Vector<T>(coefficients));

        // Deserialize GeneticAlgorithmOptions
        var gaOptions = new GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            MaxGenerations = reader.ReadInt32(),
            PopulationSize = reader.ReadInt32(),
            MutationRate = reader.ReadDouble(),
            CrossoverRate = reader.ReadDouble()
        };

        // Recreate the optimizer with the deserialized options
        if (_bestModel == null)
        {
            throw new InvalidOperationException("Deserialization failed: _bestModel is null. Model coefficients may be missing or corrupted.");
        }
        _optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(_bestModel, gaOptions);

        // Update coefficients and intercept
        UpdateCoefficientsAndIntercept();
    }

    /// <summary>
    /// Creates a new instance of the GeneticAlgorithmRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new GeneticAlgorithmRegression instance with the same options and components as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the GeneticAlgorithmRegression model with the same configuration options,
    /// regularization settings, and preprocessing components as the current instance. This is useful for model cloning,
    /// ensemble methods, or cross-validation scenarios where multiple instances of the same model with identical
    /// configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GeneticAlgorithmRegression<T>(
            Options,
            _gaOptions,
            Regularization,
            _normalizer,
            _featureSelector,
            _outlierRemoval,
            _dataPreprocessor);
    }
}
