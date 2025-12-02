namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Factory for creating learning rate schedulers with common configurations.
/// </summary>
/// <remarks>
/// <para>
/// This factory provides convenient methods for creating pre-configured learning rate
/// schedulers for common use cases. It simplifies scheduler creation and provides
/// sensible defaults for various training scenarios.
/// </para>
/// <para><b>For Beginners:</b> Instead of manually configuring schedulers with many parameters,
/// you can use this factory to create schedulers optimized for specific scenarios. For example,
/// CreateForTransformer() creates a scheduler tuned for transformer model training, with
/// warmup and linear decay that works well for attention-based models.
/// </para>
/// </remarks>
public static class LearningRateSchedulerFactory
{
    /// <summary>
    /// Creates a learning rate scheduler for typical CNN training.
    /// Uses StepLR with decay every 30 epochs.
    /// </summary>
    /// <param name="baseLearningRate">Initial learning rate. Default: 0.1</param>
    /// <param name="stepSize">Steps between LR reductions. Default: 30</param>
    /// <param name="gamma">Reduction factor. Default: 0.1</param>
    /// <returns>A configured StepLRScheduler.</returns>
    public static ILearningRateScheduler CreateForCNN(
        double baseLearningRate = 0.1,
        int stepSize = 30,
        double gamma = 0.1)
    {
        return new StepLRScheduler(baseLearningRate, stepSize, gamma);
    }

    /// <summary>
    /// Creates a learning rate scheduler for transformer training.
    /// Uses linear warmup followed by linear decay.
    /// </summary>
    /// <param name="baseLearningRate">Peak learning rate after warmup. Default: 1e-4</param>
    /// <param name="warmupSteps">Number of warmup steps. Default: 10000</param>
    /// <param name="totalSteps">Total training steps. Default: 100000</param>
    /// <returns>A configured LinearWarmupScheduler.</returns>
    public static ILearningRateScheduler CreateForTransformer(
        double baseLearningRate = 1e-4,
        int warmupSteps = 10000,
        int totalSteps = 100000)
    {
        return new LinearWarmupScheduler(
            baseLearningRate,
            warmupSteps,
            totalSteps,
            warmupInitLr: 0,
            decayMode: LinearWarmupScheduler.DecayMode.Linear,
            endLr: 0);
    }

    /// <summary>
    /// Creates a learning rate scheduler for fine-tuning pre-trained models.
    /// Uses constant low learning rate.
    /// </summary>
    /// <param name="baseLearningRate">Learning rate for fine-tuning. Default: 2e-5</param>
    /// <returns>A configured ConstantLRScheduler.</returns>
    public static ILearningRateScheduler CreateForFineTuning(
        double baseLearningRate = 2e-5)
    {
        return new ConstantLRScheduler(baseLearningRate);
    }

    /// <summary>
    /// Creates a learning rate scheduler for super-convergence training.
    /// Uses OneCycle policy for fast training with higher learning rates.
    /// </summary>
    /// <param name="maxLearningRate">Maximum learning rate. Default: 0.1</param>
    /// <param name="totalSteps">Total training steps. Default: 10000</param>
    /// <param name="pctStart">Percentage of warmup phase. Default: 0.3</param>
    /// <returns>A configured OneCycleLRScheduler.</returns>
    public static ILearningRateScheduler CreateForSuperConvergence(
        double maxLearningRate = 0.1,
        int totalSteps = 10000,
        double pctStart = 0.3)
    {
        return new OneCycleLRScheduler(
            maxLearningRate,
            totalSteps,
            pctStart,
            divFactor: 25,
            finalDivFactor: 10000);
    }

    /// <summary>
    /// Creates a learning rate scheduler for long training runs.
    /// Uses cosine annealing which works well for extended training.
    /// </summary>
    /// <param name="baseLearningRate">Initial learning rate. Default: 0.1</param>
    /// <param name="totalEpochs">Total training epochs. Default: 200</param>
    /// <param name="etaMin">Minimum learning rate. Default: 1e-6</param>
    /// <returns>A configured CosineAnnealingLRScheduler.</returns>
    public static ILearningRateScheduler CreateForLongTraining(
        double baseLearningRate = 0.1,
        int totalEpochs = 200,
        double etaMin = 1e-6)
    {
        return new CosineAnnealingLRScheduler(baseLearningRate, totalEpochs, etaMin);
    }

    /// <summary>
    /// Creates a learning rate scheduler with warm restarts.
    /// Good for escaping local minima in challenging optimization landscapes.
    /// </summary>
    /// <param name="baseLearningRate">Initial learning rate. Default: 0.1</param>
    /// <param name="t0">Initial restart period. Default: 10</param>
    /// <param name="tMult">Period multiplier after each restart. Default: 2</param>
    /// <param name="etaMin">Minimum learning rate. Default: 1e-6</param>
    /// <returns>A configured CosineAnnealingWarmRestartsScheduler.</returns>
    public static ILearningRateScheduler CreateWithWarmRestarts(
        double baseLearningRate = 0.1,
        int t0 = 10,
        int tMult = 2,
        double etaMin = 1e-6)
    {
        return new CosineAnnealingWarmRestartsScheduler(baseLearningRate, t0, tMult, etaMin);
    }

    /// <summary>
    /// Creates a learning rate scheduler that adapts based on validation loss.
    /// Good when you don't know the optimal schedule in advance.
    /// </summary>
    /// <param name="baseLearningRate">Initial learning rate. Default: 0.1</param>
    /// <param name="factor">Reduction factor. Default: 0.1</param>
    /// <param name="patience">Epochs to wait before reducing. Default: 10</param>
    /// <param name="minLr">Minimum learning rate. Default: 1e-7</param>
    /// <returns>A configured ReduceOnPlateauScheduler.</returns>
    public static ILearningRateScheduler CreateAdaptive(
        double baseLearningRate = 0.1,
        double factor = 0.1,
        int patience = 10,
        double minLr = 1e-7)
    {
        return new ReduceOnPlateauScheduler(
            baseLearningRate,
            factor,
            patience,
            minLearningRate: minLr);
    }

    /// <summary>
    /// Creates a scheduler based on type enum.
    /// </summary>
    /// <param name="type">The type of scheduler to create.</param>
    /// <param name="baseLearningRate">The base learning rate.</param>
    /// <param name="totalSteps">Total training steps (used by some schedulers).</param>
    /// <returns>A learning rate scheduler of the specified type.</returns>
    public static ILearningRateScheduler Create(
        LearningRateSchedulerType type,
        double baseLearningRate,
        int totalSteps = 100)
    {
        return type switch
        {
            LearningRateSchedulerType.Constant => new ConstantLRScheduler(baseLearningRate),
            LearningRateSchedulerType.Step => new StepLRScheduler(baseLearningRate, Math.Max(1, totalSteps / 3)),
            LearningRateSchedulerType.Exponential => new ExponentialLRScheduler(baseLearningRate),
            LearningRateSchedulerType.Polynomial => new PolynomialLRScheduler(baseLearningRate, totalSteps),
            LearningRateSchedulerType.CosineAnnealing => new CosineAnnealingLRScheduler(baseLearningRate, totalSteps),
            LearningRateSchedulerType.CosineAnnealingWarmRestarts => new CosineAnnealingWarmRestartsScheduler(baseLearningRate, Math.Max(1, totalSteps / 10)),
            LearningRateSchedulerType.OneCycle => new OneCycleLRScheduler(baseLearningRate, totalSteps),
            LearningRateSchedulerType.LinearWarmup => new LinearWarmupScheduler(baseLearningRate, Math.Max(1, totalSteps / 10), totalSteps),
            LearningRateSchedulerType.Cyclic => new CyclicLRScheduler(baseLearningRate / 10, baseLearningRate, Math.Max(1, totalSteps / 4)),
            LearningRateSchedulerType.ReduceOnPlateau => new ReduceOnPlateauScheduler(baseLearningRate),
            _ => throw new ArgumentException($"Unsupported scheduler type: {type}", nameof(type))
        };
    }
}
