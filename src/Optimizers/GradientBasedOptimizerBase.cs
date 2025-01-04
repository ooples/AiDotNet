namespace AiDotNet.Optimizers;

public abstract class GradientBasedOptimizerBase<T> : OptimizerBase<T>, IGradientBasedOptimizer<T>
{
    protected GradientBasedOptimizerOptions GradientOptions;
    private double _currentLearningRate;
    private double _currentMomentum;
    protected Vector<T> _previousGradient;

    protected GradientBasedOptimizerBase(
        GradientBasedOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null) : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        GradientOptions = options ?? new();
        _currentLearningRate = GradientOptions.InitialLearningRate;
        _currentMomentum = GradientOptions.InitialMomentum;
        _previousGradient = Vector<T>.Empty();
    }

    protected Vector<T> CalculateGradient(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        string cacheKey = GenerateGradientCacheKey(model, X, y);
        if (ModelCache.GetCachedModel(cacheKey) is ISymbolicModel<T> cachedGradient)
        {
            return cachedGradient.ToVector();
        }

        var predictions = model.Predict(X);
        var errors = predictions.Subtract(y);
        var gradient = X.Transpose().Multiply(errors);
        var result = gradient.Divide(NumOps.FromDouble(X.Rows));

        ModelCache.CacheModel(cacheKey, result.ToSymbolicModel());
        return result;
    }

    protected virtual string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        return $"{model.GetType().Name}_{X.Rows}_{X.Columns}_{GradientOptions.GetType().Name}";
    }

    public virtual void Reset()
    {
        ModelCache.ClearCache();
    }

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

    public virtual Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    public virtual Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is GradientBasedOptimizerOptions gradientOptions)
        {
            GradientOptions = gradientOptions;
        }
    }
}