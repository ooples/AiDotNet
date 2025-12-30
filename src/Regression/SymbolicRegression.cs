namespace AiDotNet.Regression;

/// <summary>
/// Implements symbolic regression, which discovers mathematical expressions that best describe the relationship
/// between input features and target values. Unlike traditional regression methods, symbolic regression
/// can discover both the form of the equation and its parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Symbolic regression works by:
/// - Creating a population of mathematical expressions (typically as expression trees)
/// - Evolving these expressions using genetic programming techniques
/// - Evaluating expressions based on how well they fit the data
/// - Selecting the best expressions to create new generations
/// - Eventually converging on an optimal or near-optimal mathematical model
/// 
/// This approach can discover complex, nonlinear relationships without requiring the user
/// to specify the form of the equation in advance.
/// </para>
/// <para><b>For Beginners:</b> Symbolic regression is like having an AI mathematician that invents formulas.
/// 
/// Think of it like this:
/// - Instead of you telling the computer what equation to use (like y = mx + b)
/// - The computer tries thousands of different formulas (like y = x², y = sin(x), etc.)
/// - It tests each formula to see how well it predicts your data
/// - It combines good formulas to make even better ones
/// - Eventually, it finds a formula that best explains your data
/// 
/// For example, when modeling how a plant grows, instead of assuming it follows a linear or
/// exponential pattern, symbolic regression might discover it follows a pattern like
/// "growth = sunlight² × water / (1 + temperature)".
/// </para>
/// </remarks>
public class SymbolicRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the symbolic regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control key aspects of the symbolic regression algorithm, including:
    /// - Population size: The number of mathematical expressions in each generation
    /// - Max generations: How many iterations of evolution to perform
    /// - Mutation rate: The probability of random changes to expressions
    /// - Crossover rate: The probability of combining parts of different expressions
    /// </para>
    /// <para><b>For Beginners:</b> These settings control how your AI mathematician works:
    /// 
    /// - Population size: How many different formulas to test at once
    /// - Max generations: How many rounds of improvement to perform
    /// - Mutation rate: How often to try random changes to formulas
    /// - Crossover rate: How often to combine parts of promising formulas
    /// 
    /// Higher values can lead to better results but take longer to run. Think of it
    /// like giving the AI more time and resources to experiment with different formulas.
    /// </para>
    /// </remarks>
    private readonly SymbolicRegressionOptions _options;

    /// <summary>
    /// The calculator used to evaluate the fitness or quality of symbolic models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This component calculates a score for each mathematical expression based on how well
    /// it fits the training data. Common fitness metrics include R-squared, mean squared error,
    /// or adjusted R-squared.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a judge in a contest of formulas.
    /// 
    /// The fitness calculator:
    /// - Tests each formula against your actual data
    /// - Gives it a score based on how accurately it predicts the results
    /// - Helps decide which formulas to keep and which to discard
    /// 
    /// Higher scores mean the formula better explains your data pattern.
    /// </para>
    /// </remarks>
    private readonly IFitnessCalculator<T, Matrix<T>, Vector<T>> _fitnessCalculator;

    /// <summary>
    /// The component responsible for normalizing input and output data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Data normalization transforms values to a standard range, typically between 0 and 1
    /// or -1 and 1. This improves the stability and performance of the symbolic regression process.
    /// </para>
    /// <para><b>For Beginners:</b> This is like converting all measurements to the same scale.
    /// 
    /// Think of it like this:
    /// - If you have measurements in inches, feet, and miles
    /// - The normalizer converts them all to a common unit
    /// - This makes it easier for the AI to find patterns
    /// 
    /// Without normalization, the AI might think larger numbers are more important
    /// simply because of their scale, not their actual significance.
    /// </para>
    /// </remarks>
    private readonly INormalizer<T, Matrix<T>, Vector<T>> _normalizer;

    /// <summary>
    /// The component responsible for selecting relevant features from the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Feature selection identifies which input variables are most relevant to predicting
    /// the target variable. This can improve model performance by removing irrelevant or
    /// redundant features.
    /// </para>
    /// <para><b>For Beginners:</b> This helps focus on what actually matters in your data.
    /// 
    /// It's like a detective who:
    /// - Examines all the possible clues (features)
    /// - Figures out which ones are actually relevant to solving the case
    /// - Ignores the red herrings that would just confuse the investigation
    /// 
    /// For example, when predicting house prices, it might determine that square footage
    /// and location are important, but the house's street number isn't.
    /// </para>
    /// </remarks>
    private readonly IFeatureSelector<T, Matrix<T>> _featureSelector;

    /// <summary>
    /// The component that detects when a satisfactory model has been found.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The fit detector monitors the evolution process and can terminate it early
    /// if a model with sufficient accuracy has been found, saving computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This is like knowing when to stop looking for a better solution.
    /// 
    /// Think of it as:
    /// - A quality inspector who checks each formula
    /// - When a formula is good enough (meets your accuracy needs)
    /// - It signals that we can stop searching and use this formula
    /// 
    /// This saves time by avoiding unnecessary additional calculations when
    /// you've already found a formula that works well enough.
    /// </para>
    /// </remarks>
    private readonly IFitDetector<T, Matrix<T>, Vector<T>> _fitDetector;

    /// <summary>
    /// The component responsible for identifying and removing outliers from the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Outlier removal identifies and handles data points that deviate significantly from
    /// the norm. These outliers could otherwise disproportionately influence the model.
    /// </para>
    /// <para><b>For Beginners:</b> This removes unusual data points that might throw off the results.
    /// 
    /// It's like:
    /// - Picking out the wildly incorrect measurements in your data
    /// - Setting them aside so they don't confuse the formula-finding process
    /// - Ensuring your formula fits the typical pattern, not the exceptions
    /// 
    /// For example, if measuring typical household income and your data includes a billionaire,
    /// the outlier removal might identify this as an extreme value that shouldn't influence
    /// the general pattern.
    /// </para>
    /// </remarks>
    private readonly IOutlierRemoval<T, Matrix<T>, Vector<T>> _outlierRemoval;

    /// <summary>
    /// The component that handles data preprocessing tasks before model training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The data preprocessor coordinates various data preparation steps including normalization,
    /// feature selection, and outlier removal. It ensures that the data is in optimal form
    /// for the symbolic regression process.
    /// </para>
    /// <para><b>For Beginners:</b> This prepares your raw data for analysis.
    /// 
    /// Think of it as a chef who:
    /// - Takes your raw ingredients (data)
    /// - Washes, peels, and chops them (normalizes values)
    /// - Removes spoiled items (outliers)
    /// - Selects the ingredients that work well together (feature selection)
    /// 
    /// This preparation ensures the AI gets high-quality data that's ready for modeling.
    /// </para>
    /// </remarks>
    private readonly IDataPreprocessor<T, Matrix<T>, Vector<T>> _dataPreprocessor;

    /// <summary>
    /// The optimizer used to evolve and improve symbolic models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The optimizer implements the evolutionary algorithm that generates, selects, and evolves
    /// symbolic expressions. It typically uses genetic programming techniques like mutation,
    /// crossover, and selection to find optimal expressions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine that powers the formula discovery process.
    /// 
    /// The optimizer:
    /// - Creates generations of candidate formulas
    /// - Tests how well each formula performs
    /// - Keeps the best formulas for the next generation
    /// - Combines successful formulas to create new ones
    /// - Occasionally introduces random changes to explore new possibilities
    /// 
    /// It mimics natural evolution, but for mathematical formulas instead of organisms.
    /// </para>
    /// </remarks>
    private readonly IOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    /// <summary>
    /// The best symbolic model found during the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This model represents the mathematical expression that best fits the training data
    /// according to the fitness calculator. It is used to make predictions on new data.
    /// </para>
    /// <para><b>For Beginners:</b> This is your winning formula.
    /// 
    /// After trying thousands of formulas:
    /// - This is the champion that performed best on your data
    /// - It captures the mathematical relationship between your inputs and outputs
    /// - It's used to make predictions for new data points
    /// 
    /// This formula is the actual "product" of symbolic regression - a mathematical
    /// expression that describes your data pattern.
    /// </para>
    /// </remarks>
    private IFullModel<T, Matrix<T>, Vector<T>>? _bestModel;

    /// <summary>
    /// The fitness score of the best model found.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents how well the best model fits the training data according to
    /// the fitness calculator. It is useful for comparing different models or tracking
    /// improvement over time.
    /// </para>
    /// <para><b>For Beginners:</b> This is the score of your winning formula.
    /// 
    /// Think of it as:
    /// - A rating from 0 to 1 (for many fitness measures)
    /// - Higher numbers usually mean better performance
    /// - It tells you how confident you can be in the model's predictions
    /// 
    /// For example, an R-squared value of 0.95 means your formula explains about 95%
    /// of the variation in your data, which is typically very good.
    /// </para>
    /// </remarks>
    private T _bestFitness;

    /// <summary>
    /// Gets the best symbolic model discovered during optimization.
    /// </summary>
    public IFullModel<T, Matrix<T>, Vector<T>>? BestModel => _bestModel;

    /// <summary>
    /// Gets the fitness score of the best model discovered during optimization.
    /// </summary>
    public T BestFitness => _bestFitness;

    /// <summary>
    /// Creates a new symbolic regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the symbolic regression model. These settings control aspects like:
    /// - The population size and number of generations for the genetic algorithm
    /// - The mutation and crossover rates for evolving expressions
    /// - The complexity penalty to prefer simpler expressions
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting.
    /// If not provided, no additional regularization will be applied.
    /// </param>
    /// <param name="fitnessCalculator">
    /// Optional calculator for evaluating model fitness.
    /// If not provided, R-squared will be used as the fitness metric.
    /// </param>
    /// <param name="normalizer">
    /// Optional component for normalizing input and output data.
    /// If not provided, no normalization will be applied.
    /// </param>
    /// <param name="featureSelector">
    /// Optional component for selecting relevant features.
    /// If not provided, all features will be used.
    /// </param>
    /// <param name="fitDetector">
    /// Optional component for detecting when a satisfactory model has been found.
    /// If not provided, the default fit detector will be used.
    /// </param>
    /// <param name="outlierRemoval">
    /// Optional component for identifying and removing outliers.
    /// If not provided, no outlier removal will be performed.
    /// </param>
    /// <param name="dataPreprocessor">
    /// Optional component for preprocessing data before model training.
    /// If not provided, a default preprocessor will be used with the specified normalizer,
    /// feature selector, and outlier removal components.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new symbolic regression model with the specified components.
    /// If components are not provided, default implementations are used. It initializes the
    /// genetic algorithm optimizer with the configured population size, generations, and rates.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up your AI formula discoverer.
    /// 
    /// Think of it like assembling a team of specialists:
    /// - The options define the overall strategy
    /// - The fitness calculator evaluates each formula
    /// - The normalizer, feature selector, and outlier remover prepare your data
    /// - The fit detector knows when to stop searching
    /// - The optimizer manages the evolution of formulas
    /// 
    /// You can use the default team members (by not specifying them) or bring in your own
    /// specialists with different approaches to each task.
    /// </para>
    /// </remarks>
    public SymbolicRegression(
        SymbolicRegressionOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        IFitnessCalculator<T, Matrix<T>, Vector<T>>? fitnessCalculator = null,
        INormalizer<T, Matrix<T>, Vector<T>>? normalizer = null,
        IFeatureSelector<T, Matrix<T>>? featureSelector = null,
        IFitDetector<T, Matrix<T>, Vector<T>>? fitDetector = null,
        IOutlierRemoval<T, Matrix<T>, Vector<T>>? outlierRemoval = null,
        IDataPreprocessor<T, Matrix<T>, Vector<T>>? dataPreprocessor = null)
        : base(options, regularization)
    {
        _options = options ?? new SymbolicRegressionOptions();
        var dummyModel = new VectorModel<T>(Vector<T>.Empty());
        _optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(
            dummyModel,
            new GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>
            {
                PopulationSize = _options.PopulationSize,
                MaxGenerations = _options.MaxGenerations,
                MutationRate = _options.MutationRate,
                CrossoverRate = _options.CrossoverRate
            });
        _fitnessCalculator = fitnessCalculator ?? new RSquaredFitnessCalculator<T, Matrix<T>, Vector<T>>();
        _normalizer = normalizer ?? new NoNormalizer<T, Matrix<T>, Vector<T>>();
        _featureSelector = featureSelector ?? new NoFeatureSelector<T, Matrix<T>>();
        _fitDetector = fitDetector ?? new DefaultFitDetector<T, Matrix<T>, Vector<T>>();
        _outlierRemoval = outlierRemoval ?? new NoOutlierRemoval<T, Matrix<T>, Vector<T>>();
        _dataPreprocessor = dataPreprocessor ?? new DefaultDataPreprocessor<T, Matrix<T>, Vector<T>>(_normalizer, _featureSelector, _outlierRemoval);
        _bestFitness = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue;
    }

    /// <summary>
    /// Optimizes the symbolic regression model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input feature matrix, where rows represent observations and columns represent features.</param>
    /// <param name="y">The target values vector containing the actual output values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core optimization for symbolic regression. It:
    /// 1. Preprocesses the data (normalization, feature selection, outlier removal)
    /// 2. Splits the data into training, validation, and test sets
    /// 3. Uses the genetic algorithm optimizer to evolve symbolic models
    /// 4. Stores the best model found during the optimization process
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best formula to describe your data.
    /// 
    /// The process works like this:
    /// 
    /// 1. First, it cleans and prepares your data
    /// 2. Then it divides your data into portions for different purposes:
    ///    - Training data: Used to create and improve formulas
    ///    - Validation data: Used to check formulas during development
    ///    - Test data: Used for a final check of the best formula
    /// 3. Next, it runs the genetic algorithm to evolve better and better formulas
    /// 4. Finally, it saves the best formula it found
    /// 
    /// This is where the magic happens - the AI explores thousands of possible
    /// mathematical relationships to find the one that best describes your data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Preprocess the data
        var (preprocessedX, preprocessedY, _) = _dataPreprocessor.PreprocessData(x, y);
        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = _dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Recreate the optimizer with proper dimensions based on actual input data
        // The optimizer was initially created with an empty model in the constructor,
        // but now we know the actual feature count from the training data
        int featureCount = XTrain.Columns;
        var properlyDimensionedModel = new VectorModel<T>(new Vector<T>(featureCount));
        var optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(
            properlyDimensionedModel,
            new GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>
            {
                PopulationSize = _options.PopulationSize,
                MaxGenerations = _options.MaxGenerations,
                MutationRate = _options.MutationRate,
                CrossoverRate = _options.CrossoverRate
            });

        // Optimize the model
        var optimizationResult = optimizer.Optimize(OptimizerHelper<T, Matrix<T>, Vector<T>>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));
        _bestFitness = optimizationResult.BestFitnessScore;
        _bestModel = optimizationResult.BestSolution ?? throw new InvalidOperationException("Optimization result does not contain a valid symbolic model.");
    }

    /// <summary>
    /// Predicts target values for a matrix of input features.
    /// </summary>
    /// <param name="X">The input feature matrix for which to make predictions.</param>
    /// <returns>A vector of predicted values, one for each row in the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions for multiple input samples by:
    /// 1. Applying the best symbolic model to each row of the input matrix
    /// 2. Returning a vector of predicted values
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your discovered formula to make predictions.
    /// 
    /// Once the system has found the best formula:
    /// - This method takes new data points
    /// - It plugs each data point into the formula
    /// - It calculates and returns the predicted results
    /// 
    /// For example, if your formula determined plant growth based on sunlight and water,
    /// you could use this method to predict how much a plant would grow with specific
    /// amounts of sunlight and water.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        return _bestModel?.Predict(X) ?? Vector<T>.Empty();
    }

    /// <summary>
    /// Predicts a target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The input feature vector for which to make a prediction.</param>
    /// <returns>The predicted value for the input vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements prediction for a single input sample. It:
    /// 1. Applies regularization to the input vector
    /// 2. Evaluates the best symbolic model with the regularized input
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts a value for a single data point.
    /// 
    /// Think of it like this:
    /// 1. It first applies regularization to your input (which helps ensure stable predictions)
    /// 2. It then plugs the values into your discovered formula
    /// 3. It calculates and returns the result
    /// 
    /// This is useful when you want to make a prediction for just one specific case,
    /// rather than a whole batch of data.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        if (_bestModel == null)
        {
            throw new InvalidOperationException("The model has not been optimized yet. Please call OptimizeModel first.");
        }

        Vector<T> regularizedInput = Regularization.Regularize(input);
        return _bestModel.Predict(Matrix<T>.FromVector(regularizedInput))[0];
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for symbolic regression.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the enum value that identifies this model as a symbolic regression model.
    /// This is used for model identification in serialization/deserialization and for logging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a symbolic regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// 
    /// You generally won't need to call this method directly in your code.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.SymbolicRegression;

    /// <summary>
    /// Creates a new instance of the Symbolic Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Symbolic Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Symbolic Regression model, including its discovered 
    /// formula, configuration options, fitness calculator, data preprocessors, and optimization components.
    /// The new instance is completely independent of the original, allowing modifications without 
    /// affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of the current symbolic regression model.
    /// 
    /// Think of it like making a perfect duplicate of your AI mathematician:
    /// - It copies the winning formula that was discovered
    /// - It maintains the same configuration settings (population size, mutation rates, etc.)
    /// - It preserves all the specialty components (fitness calculator, normalizer, etc.)
    /// - It remembers how good the best formula was (the fitness score)
    /// 
    /// This is useful when you want to:
    /// - Create a backup before making changes
    /// - Create variations of the same model for different purposes
    /// - Share the model while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        var newModel = new SymbolicRegression<T>(
            options: _options,
            regularization: Regularization,
            fitnessCalculator: _fitnessCalculator,
            normalizer: _normalizer,
            featureSelector: _featureSelector,
            fitDetector: _fitDetector,
            outlierRemoval: _outlierRemoval,
            dataPreprocessor: _dataPreprocessor);

        // Copy the best model found (if any)
        if (_bestModel != null)
        {
            newModel._bestModel = _bestModel.Clone();
        }

        // Copy the best fitness score
        newModel._bestFitness = _bestFitness;

        return newModel;
    }
}
