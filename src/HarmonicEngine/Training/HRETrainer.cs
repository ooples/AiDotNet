using AiDotNet.HarmonicEngine.Models;

namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// Orchestrates training of an <see cref="HRELanguageModel{T}"/> using a
/// pluggable <see cref="ITrainingStrategy{T}"/>. Iterates over mini-batches,
/// delegates the actual update rule to the strategy, and collects metrics.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
public class HRETrainer<T>
{
    private readonly HRELanguageModel<T> _model;
    private readonly ITrainingStrategy<T> _strategy;
    private int _stepCount;

    /// <summary>
    /// Gets the number of training steps taken so far.
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Gets the wrapped language model.
    /// </summary>
    public HRELanguageModel<T> Model => _model;

    /// <summary>
    /// Gets the training strategy being applied.
    /// </summary>
    public ITrainingStrategy<T> Strategy => _strategy;

    /// <summary>
    /// Creates a new trainer.
    /// </summary>
    public HRETrainer(HRELanguageModel<T> model, ITrainingStrategy<T> strategy)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
    }

    /// <summary>
    /// Runs a single training step on the given batch, invoking the strategy.
    /// </summary>
    public void Step(TrainingBatch<T> batch)
    {
        _model.SetTrainingMode(true);
        _strategy.TrainStep(_model, batch);
        _stepCount++;
    }

    /// <summary>
    /// Runs multiple training steps by iterating over the provided batches.
    /// </summary>
    public void TrainEpoch(IEnumerable<TrainingBatch<T>> batches)
    {
        foreach (var batch in batches)
        {
            Step(batch);
        }
    }

    /// <summary>
    /// Gets the current metrics from the strategy plus the global step count.
    /// Useful for logging and producing training curves.
    /// </summary>
    public Dictionary<string, double> GetMetrics()
    {
        var metrics = new Dictionary<string, double>(_strategy.GetMetrics())
        {
            ["step"] = _stepCount
        };
        return metrics;
    }

    /// <summary>
    /// Resets the trainer's step count and strategy metrics.
    /// </summary>
    public void Reset()
    {
        _stepCount = 0;
        _strategy.Reset();
    }
}
