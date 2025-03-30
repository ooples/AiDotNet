namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a base class for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers use the gradient of the loss function to update the model parameters
/// in a direction that minimizes the loss. This base class provides common functionality for
/// various gradient-based optimization techniques.
/// </para>
/// <para><b>For Beginners:</b> Think of gradient-based optimization like finding the bottom of a valley:
/// 
/// - You start at a random point on a hilly landscape (your initial model parameters)
/// - You look around to see which way is steepest downhill (calculate the gradient)
/// - You take a step in that direction (update the parameters)
/// - You repeat this process until you reach the bottom of the valley (optimize the model)
/// 
/// This approach helps the model learn by gradually adjusting its parameters to minimize errors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class GradientBasedOptimizerBase<T> : OptimizerBase<T>, IGradientBasedOptimizer<T>
{
    /// <summary>
    /// Options specific to gradient-based optimization algorithms.
    /// </summary>
    protected GradientBasedOptimizerOptions GradientOptions;

    /// <summary>
    /// The current learning rate used in the optimization process.
    /// </summary>
    private double _currentLearningRate;

    /// <summary>
    /// The current momentum factor used in the optimization process.
    /// </summary>
    private double _currentMomentum;

    /// <summary>
    /// The gradient from the previous optimization step, used for momentum calculations.
    /// </summary>
    protected Vector<T> _previousGradient;

    /// <summary>
    /// A cache for storing and retrieving gradients to improve performance.
    /// </summary>
    protected IGradientCache<T> GradientCache;

    /// <summary>
    /// Initializes a new instance of the GradientBasedOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the gradient-based optimizer with its initial settings.
    /// It's like preparing for your hike by choosing your starting point, deciding how big your steps
    /// will be, and how much you'll consider your previous direction when choosing your next step.
    /// </para>
    /// </remarks>
    /// <param name="options">Options for the gradient-based optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    protected GradientBasedOptimizerBase(
        GradientBasedOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null) : 
        base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        GradientOptions = options ?? new();
        _currentLearningRate = GradientOptions.InitialLearningRate;
        _currentMomentum = GradientOptions.InitialMomentum;
        _previousGradient = Vector<T>.Empty();
        GradientCache = gradientCache ?? new DefaultGradientCache<T>();
    }

    /// <summary>
    /// Calculates the gradient for the given model and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    protected Vector<T> CalculateGradient(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        string cacheKey = GenerateGradientCacheKey(model, X, y);
        var cachedGradient = GradientCache.GetCachedGradient(cacheKey);
        if (cachedGradient != null)
        {
            return cachedGradient.Coefficients;
        }

        var predictions = model.Predict(X);
        var errors = predictions.Subtract(y);
        var gradient = X.Transpose().Multiply(errors);
        var result = gradient.Divide(NumOps.FromDouble(X.Rows));

        var gradientModel = result.ToSymbolicModel();
        GradientCache.CacheGradient(cacheKey, gradientModel);

        return result;
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for each gradient calculation.
    /// It's like labeling each spot on the hill so you can remember what the gradient was there.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>A string key for caching the gradient.</returns>
    protected virtual string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        return $"{model.GetType().Name}_{X.Rows}_{X.Columns}_{GradientOptions.GetType().Name}";
    }

    /// <summary>
    /// Resets the optimizer to its initial state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears all the remembered information and starts fresh.
    /// It's like wiping your map clean and starting your hike from the beginning.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        GradientCache.ClearCache();
    }

    /// <summary>
    /// Applies momentum to the gradient calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method considers the direction you were moving in previously
    /// when deciding which way to go next. It's like considering your momentum when hiking -
    /// you might keep going in roughly the same direction rather than abruptly changing course.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The gradient adjusted for momentum.</returns>
    protected virtual Vector<T> ApplyMomentum(Vector<T> gradient)
    {
        if (_previousGradient == null)
        {
            _previousGradient = gradient;
            return gradient;
        }

        var momentumGradient = _previousGradient.Add(gradient.Multiply(NumOps.FromDouble(_currentMomentum)));
        _previousGradient = momentumGradient;
        return momentumGradient;
    }

    /// <summary>
    /// Updates a matrix of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve its performance.
    /// It's like taking a step in the direction you've determined will lead you downhill.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates a vector of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateMatrix, but for when the parameters
    /// are in a vector format instead of a matrix. It's another way of taking a step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates the options for the gradient-based optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer
    /// while it's running. It's like adjusting your hiking strategy mid-journey based on the terrain you encounter.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is GradientBasedOptimizerOptions gradientOptions)
        {
            GradientOptions = gradientOptions;
        }
    }
}