using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Preprocessing;

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
/// <example>
/// <code>
/// // Create a regression model optimized with genetic algorithms
/// var options = new GeneticAlgorithmRegressionOptions&lt;double&gt;();
/// var model = new GeneticAlgorithmRegression&lt;double&gt;(options);
///
/// // Prepare training data: 5 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(5, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10 });
/// var targets = new Vector&lt;double&gt;(new double[] { 2.5, 5.3, 8.1, 10.9, 13.7 });
///
/// // Train with evolutionary optimization (selection, crossover, mutation)
/// model.Train(features, targets);
///
/// // Predict for a new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 11, 12 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Adaptation in Natural and Artificial Systems", "https://doi.org/10.7551/mitpress/1090.001.0001")]
public partial class GeneticAlgorithmRegression<T> : RegressionBase<T>
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
    /// Component that identifies and removes outliers from the training data.
    /// </summary>
    private readonly IOutlierRemoval<T, Matrix<T>, Vector<T>> _outlierRemoval;

    /// <summary>
    /// Component that handles all data preprocessing steps before training.
    /// </summary>
    private readonly PreprocessingPipeline<T, Matrix<T>, Matrix<T>>? _preprocessingPipeline;

    /// <summary>
    /// The best model found by the genetic algorithm.
    /// </summary>
    [Scratch]
    private IFullModel<T, Matrix<T>, Vector<T>>? _bestModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="GeneticAlgorithmRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional regression options for the model.</param>
    /// <param name="gaOptions">Optional configuration options for the genetic algorithm optimizer.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <param name="outlierRemoval">Optional component for removing outliers.</param>
    /// <param name="preprocessingPipeline">Optional preprocessing pipeline for data transformation.</param>
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
    /// - How to prepare your data before training (preprocessing pipeline)
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
        IOutlierRemoval<T, Matrix<T>, Vector<T>>? outlierRemoval = null,
        PreprocessingPipeline<T, Matrix<T>, Matrix<T>>? preprocessingPipeline = null)
        : base(options, regularization)
    {
        _gaOptions = gaOptions ?? new GeneticAlgorithmOptimizerOptions<T, Matrix<T>, Vector<T>>();
        var dummyModel = new VectorModel<T>(Vector<T>.Empty());
        _optimizer = new GeneticAlgorithmOptimizer<T, Matrix<T>, Vector<T>>(dummyModel, _gaOptions);
        _outlierRemoval = outlierRemoval ?? new NoOutlierRemoval<T, Matrix<T>, Vector<T>>();
        _preprocessingPipeline = preprocessingPipeline;
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
    /// <summary>GA regression doesn't benefit from optimizer parameter injection.</summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        TrainingFeatureCount = x.Columns;

        // Preprocess the data if pipeline is configured
        var preprocessedX = _preprocessingPipeline is not null
            ? _preprocessingPipeline.FitTransform(x)
            : x;
        var preprocessedY = y;

        // Split the data using the base class options. Each split needs at least one row: a
        // proportional split alone empties the validation or test set on small inputs.
        int totalSamples = preprocessedX.Rows;
        if (totalSamples < 3)
        {
            throw new ArgumentException(
                $"Genetic algorithm regression needs at least 3 samples to form " +
                $"train/validation/test splits, but received {totalSamples}.", nameof(x));
        }

        int trainSize = Math.Max(1, (int)(totalSamples * 0.7));  // 70% training
        int valSize = Math.Max(1, (int)(totalSamples * 0.15));   // 15% validation
        int testSize = totalSamples - trainSize - valSize;

        if (testSize < 1)
        {
            testSize = 1;
            trainSize = totalSamples - valSize - testSize;
        }

        // GetSubMatrix takes (startRow, startColumn, rowCount, columnCount). These calls previously
        // passed the split size as the START COLUMN, so every split came back with zero rows while
        // its matching target vector kept the full length. The argument order was wrong from the
        // day it was written and went unnoticed because the OLS short-circuit above meant this code
        // never executed.
        var xTrain = preprocessedX.GetSubMatrix(0, 0, trainSize, preprocessedX.Columns);
        var yTrain = preprocessedY.SubVector(0, trainSize);
        var xVal = preprocessedX.GetSubMatrix(trainSize, 0, valSize, preprocessedX.Columns);
        var yVal = preprocessedY.SubVector(trainSize, valSize);
        var xTest = preprocessedX.GetSubMatrix(trainSize + valSize, 0, testSize, preprocessedX.Columns);
        var yTest = preprocessedY.SubVector(trainSize + valSize, testSize);

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

    /// <summary>
    /// Updates the model coefficients and intercept based on the best solution found by the genetic algorithm.
    /// </summary>
    private void UpdateCoefficientsAndIntercept()
    {
        Coefficients = (_bestModel as IParameterizable<T, Matrix<T>, Vector<T>>)?.GetParameters() ?? Vector<T>.Empty();

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
}
