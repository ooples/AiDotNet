using AiDotNet.LinearAlgebra;

namespace AiDotNet.CurriculumLearning.Results;

/// <summary>
/// Result of curriculum learning training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class contains all the information about how
/// curriculum learning training went - the final model performance, how long it took,
/// and detailed information about each curriculum phase.</para>
///
/// <para><b>Key Metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>FinalLoss:</b> The loss value at the end of training</description></item>
/// <item><description><b>BestLoss:</b> The lowest loss achieved during training</description></item>
/// <item><description><b>PhaseResults:</b> Detailed results for each curriculum phase</description></item>
/// <item><description><b>CurriculumEfficiency:</b> How much curriculum learning helped</description></item>
/// </list>
/// </remarks>
public class CurriculumLearningResult<T>
{
    /// <summary>
    /// Gets or sets whether training was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the error message if training failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the final training loss.
    /// </summary>
    public T FinalTrainingLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the final validation loss (if validation data provided).
    /// </summary>
    public T? FinalValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the best training loss achieved during training.
    /// </summary>
    public T BestTrainingLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the best validation loss achieved during training.
    /// </summary>
    public T? BestValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the epoch at which best validation loss was achieved.
    /// </summary>
    public int BestEpoch { get; set; }

    /// <summary>
    /// Gets or sets the total number of epochs trained.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets the number of curriculum phases completed.
    /// </summary>
    public int PhasesCompleted { get; set; }

    /// <summary>
    /// Gets or sets the total training time in milliseconds.
    /// </summary>
    public long TrainingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the training history (loss per epoch).
    /// </summary>
    public List<T> TrainingLossHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the validation loss history.
    /// </summary>
    public List<T>? ValidationLossHistory { get; set; }

    /// <summary>
    /// Gets or sets the results for each curriculum phase.
    /// </summary>
    public List<CurriculumPhaseResult<T>> PhaseResults { get; set; } = new List<CurriculumPhaseResult<T>>();

    /// <summary>
    /// Gets or sets the curriculum progression history.
    /// </summary>
    public List<CurriculumProgressionEntry<T>> CurriculumProgression { get; set; } = new List<CurriculumProgressionEntry<T>>();

    /// <summary>
    /// Gets or sets whether early stopping was triggered.
    /// </summary>
    public bool EarlyStopTriggered { get; set; }

    /// <summary>
    /// Gets or sets the epoch at which early stopping was triggered.
    /// </summary>
    public int? EarlyStopEpoch { get; set; }

    /// <summary>
    /// Gets or sets the total number of samples used during training.
    /// </summary>
    public int TotalSamplesUsed { get; set; }

    /// <summary>
    /// Gets or sets the curriculum efficiency metric.
    /// </summary>
    /// <remarks>
    /// <para>Measures how effective curriculum learning was compared to
    /// standard training. Values > 1 indicate curriculum learning helped.</para>
    /// </remarks>
    public T? CurriculumEfficiency { get; set; }

    /// <summary>
    /// Gets or sets the average improvement per phase.
    /// </summary>
    public T? AveragePhaseImprovement { get; set; }

    /// <summary>
    /// Gets or sets the total number of training samples.
    /// </summary>
    public int? TrainingSamples { get; set; }

    /// <summary>
    /// Gets or sets the total number of validation samples (if validation data provided).
    /// </summary>
    public int? ValidationSamples { get; set; }

    /// <summary>
    /// Gets or sets the final difficulty scores for all samples.
    /// </summary>
    public Vector<T>? FinalDifficulties { get; set; }

    /// <summary>
    /// Gets or sets the scheduler statistics at the end of training.
    /// </summary>
    public Dictionary<string, object>? SchedulerStatistics { get; set; }

    /// <summary>
    /// Gets the training time in human-readable format.
    /// </summary>
    public string TrainingTimeFormatted => FormatTime(TrainingTimeMs);

    private static string FormatTime(long milliseconds)
    {
        var timeSpan = TimeSpan.FromMilliseconds(milliseconds);
        if (timeSpan.TotalHours >= 1)
            return $"{timeSpan.Hours}h {timeSpan.Minutes}m {timeSpan.Seconds}s";
        if (timeSpan.TotalMinutes >= 1)
            return $"{timeSpan.Minutes}m {timeSpan.Seconds}s";
        return $"{timeSpan.Seconds}.{timeSpan.Milliseconds:D3}s";
    }
}

/// <summary>
/// Result of a single curriculum phase.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumPhaseResult<T>
{
    /// <summary>
    /// Gets or sets the phase number (0-indexed).
    /// </summary>
    public int PhaseNumber { get; set; }

    /// <summary>
    /// Gets or sets the data fraction used in this phase.
    /// </summary>
    public T DataFraction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the number of samples available in this phase.
    /// </summary>
    public int SampleCount { get; set; }

    /// <summary>
    /// Gets or sets the starting epoch of this phase.
    /// </summary>
    public int StartEpoch { get; set; }

    /// <summary>
    /// Gets or sets the ending epoch of this phase.
    /// </summary>
    public int EndEpoch { get; set; }

    /// <summary>
    /// Gets or sets the training loss at the start of the phase.
    /// </summary>
    public T StartLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the training loss at the end of the phase.
    /// </summary>
    public T EndLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the best loss achieved during this phase.
    /// </summary>
    public T BestLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the validation loss at the start of the phase.
    /// </summary>
    public T? StartValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the validation loss at the end of the phase.
    /// </summary>
    public T? EndValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the loss improvement in this phase.
    /// </summary>
    public T LossImprovement { get; set; } = default!;

    /// <summary>
    /// Gets or sets the improvement rate (improvement per epoch).
    /// </summary>
    public T ImprovementRate { get; set; } = default!;

    /// <summary>
    /// Gets or sets the training time for this phase in milliseconds.
    /// </summary>
    public long PhaseTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the average loss for newly introduced samples.
    /// </summary>
    public T? AverageNewSampleLoss { get; set; }

    /// <summary>
    /// Gets or sets the difficulty range of samples in this phase.
    /// </summary>
    public (T Min, T Max)? DifficultyRange { get; set; }

    /// <summary>
    /// Gets or sets the training loss history for this phase.
    /// </summary>
    public List<T> TrainingLosses { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the validation loss history for this phase.
    /// </summary>
    public List<T> ValidationLosses { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the final training loss at end of this phase.
    /// </summary>
    public T FinalTrainingLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the final validation loss at end of this phase (if available).
    /// </summary>
    public T? FinalValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of samples used in this phase.
    /// </summary>
    public int NumSamples { get; set; }

    /// <summary>
    /// Gets the number of epochs in this phase.
    /// </summary>
    public int EpochCount => EndEpoch - StartEpoch + 1;
}

/// <summary>
/// Entry in the curriculum progression history.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumProgressionEntry<T>
{
    /// <summary>
    /// Gets or sets the epoch number.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Gets or sets the phase number.
    /// </summary>
    public int Phase { get; set; }

    /// <summary>
    /// Gets or sets the current data fraction.
    /// </summary>
    public T DataFraction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the training loss.
    /// </summary>
    public T TrainingLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the validation loss (if available).
    /// </summary>
    public T? ValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of samples used.
    /// </summary>
    public int SamplesUsed { get; set; }

    /// <summary>
    /// Gets or sets the difficulty threshold for this epoch.
    /// </summary>
    public T DifficultyThreshold { get; set; } = default!;

    /// <summary>
    /// Gets or sets the timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
