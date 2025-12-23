using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Difficulty estimator based on training loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator uses the model's prediction loss as a
/// measure of difficulty. Samples with high loss are considered harder because the
/// model struggles to predict them correctly.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>For each sample, the model makes a prediction</description></item>
/// <item><description>The loss (error) between prediction and true value is calculated</description></item>
/// <item><description>Higher loss = harder sample</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Bengio et al. "Curriculum Learning" (ICML 2009)</description></item>
/// </list>
/// </remarks>
public class LossBasedDifficultyEstimator<T, TInput, TOutput> : DifficultyEstimatorBase<T, TInput, TOutput>
{
    private readonly ILossFunction<T>? _lossFunction;
    private readonly bool _useModelLoss;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => "LossBased";

    /// <summary>
    /// Gets whether this estimator requires the model.
    /// </summary>
    public override bool RequiresModel => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="LossBasedDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="lossFunction">Optional loss function. If null, uses model's internal loss.</param>
    /// <param name="normalize">Whether to normalize difficulties to [0, 1].</param>
    public LossBasedDifficultyEstimator(
        ILossFunction<T>? lossFunction = null,
        bool normalize = true)
    {
        _lossFunction = lossFunction;
        _useModelLoss = lossFunction == null;
        _normalize = normalize;
    }

    /// <summary>
    /// Estimates the difficulty of a single sample based on prediction loss.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model),
                "LossBasedDifficultyEstimator requires a model to compute losses.");
        }

        // Get model prediction
        var prediction = model.Predict(input);

        // Calculate loss
        if (_useModelLoss)
        {
            // Use model's default loss function
            var predictionVec = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
            var expectedVec = ConversionsHelper.ConvertToVector<T, TOutput>(expectedOutput);
            return model.DefaultLossFunction.CalculateLoss(predictionVec, expectedVec);
        }
        else
        {
            // Use provided loss function - convert TOutput to Vector<T>
            var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
            var expectedVector = ConversionsHelper.ConvertToVector<T, TOutput>(expectedOutput);
            return _lossFunction!.CalculateLoss(predictionVector, expectedVector);
        }
    }

    /// <summary>
    /// Estimates difficulty scores for all samples (batch optimized).
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));
        if (model is null) throw new ArgumentNullException(nameof(model));

        // Return cached scores if available
        if (CacheScores && HasCachedScores && CachedScores!.Length == dataset.Count)
        {
            return CachedScores!;
        }

        var difficulties = new T[dataset.Count];

        // Calculate loss for each sample
        for (int i = 0; i < dataset.Count; i++)
        {
            var sample = dataset.GetSample(i);
            difficulties[i] = EstimateDifficulty(sample.Input, sample.Output, model);
        }

        var result = new Vector<T>(difficulties);

        // Normalize if requested
        if (_normalize)
        {
            result = NormalizeDifficulties(result);
        }

        if (CacheScores)
        {
            CachedScores = result;
        }

        return result;
    }
}

/// <summary>
/// Loss-based difficulty estimator with moving average smoothing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This variant uses a moving average of losses across
/// training epochs. This helps stabilize difficulty estimates and prevents sudden
/// changes in curriculum ordering due to random fluctuations.</para>
/// </remarks>
public class SmoothedLossDifficultyEstimator<T, TInput, TOutput> : LossBasedDifficultyEstimator<T, TInput, TOutput>
{
    private readonly T _smoothingFactor;
    private Vector<T>? _smoothedLosses;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => "SmoothedLossBased";

    /// <summary>
    /// Initializes a new instance of the <see cref="SmoothedLossDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="smoothingFactor">Exponential smoothing factor (0-1). Higher = more smoothing.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="normalize">Whether to normalize difficulties.</param>
    public SmoothedLossDifficultyEstimator(
        T smoothingFactor,
        ILossFunction<T>? lossFunction = null,
        bool normalize = true)
        : base(lossFunction, normalize)
    {
        _smoothingFactor = smoothingFactor;
    }

    /// <summary>
    /// Estimates difficulty with exponential moving average smoothing.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        var currentLosses = base.EstimateDifficulties(dataset, model);

        if (_smoothedLosses == null || _smoothedLosses.Length != currentLosses.Length)
        {
            // First time: initialize with current losses
            _smoothedLosses = currentLosses;
        }
        else
        {
            // Apply exponential moving average
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, _smoothingFactor);
            for (int i = 0; i < _smoothedLosses.Length; i++)
            {
                // smoothed = alpha * previous + (1 - alpha) * current
                _smoothedLosses[i] = NumOps.Add(
                    NumOps.Multiply(_smoothingFactor, _smoothedLosses[i]),
                    NumOps.Multiply(oneMinusAlpha, currentLosses[i]));
            }
        }

        return _smoothedLosses;
    }

    /// <summary>
    /// Resets the estimator including smoothed values.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _smoothedLosses = null;
    }
}
