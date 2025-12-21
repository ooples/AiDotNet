namespace AiDotNet.Models;

/// <summary>
/// Represents a single trial in hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A trial is one attempt at training with a specific set of hyperparameters.
/// The optimizer tries many trials to find the best hyperparameter values.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class HyperparameterTrial<T>
{
    /// <summary>
    /// Gets the unique identifier for this trial.
    /// </summary>
    public string TrialId { get; private set; }

    /// <summary>
    /// Gets the trial number (sequential).
    /// </summary>
    public int TrialNumber { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameter values for this trial.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; }

    /// <summary>
    /// Gets or sets the objective value achieved (e.g., validation accuracy).
    /// </summary>
    public T? ObjectiveValue { get; set; }

    /// <summary>
    /// Gets or sets intermediate values logged during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Intermediate values are measurements taken during training,
    /// used for pruning (stopping unpromising trials early).
    /// </remarks>
    public Dictionary<int, T> IntermediateValues { get; set; }

    /// <summary>
    /// Gets or sets the start time of the trial.
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time of the trial.
    /// </summary>
    public DateTime? EndTime { get; set; }

    /// <summary>
    /// Gets or sets the status of the trial.
    /// </summary>
    public TrialStatus Status { get; set; }

    /// <summary>
    /// Gets or sets user attributes for the trial.
    /// </summary>
    public Dictionary<string, object> UserAttributes { get; set; }

    /// <summary>
    /// Gets or sets system attributes for the trial.
    /// </summary>
    public Dictionary<string, object> SystemAttributes { get; set; }

    /// <summary>
    /// Initializes a new instance of the HyperparameterTrial class.
    /// </summary>
    public HyperparameterTrial(int trialNumber)
    {
        TrialId = Guid.NewGuid().ToString();
        TrialNumber = trialNumber;
        Parameters = new Dictionary<string, object>();
        IntermediateValues = new Dictionary<int, T>();
        StartTime = DateTime.UtcNow;
        Status = TrialStatus.Running;
        UserAttributes = new Dictionary<string, object>();
        SystemAttributes = new Dictionary<string, object>();
    }

    /// <summary>
    /// Gets the duration of the trial.
    /// </summary>
    public TimeSpan? GetDuration()
    {
        if (EndTime.HasValue)
            return EndTime.Value - StartTime;

        if (Status == TrialStatus.Running)
            return DateTime.UtcNow - StartTime;

        return null;
    }

    /// <summary>
    /// Marks the trial as complete with the final objective value.
    /// </summary>
    public void Complete(T objectiveValue)
    {
        ObjectiveValue = objectiveValue;
        Status = TrialStatus.Complete;
        EndTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Marks the trial as pruned (stopped early).
    /// </summary>
    public void Prune()
    {
        Status = TrialStatus.Pruned;
        EndTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Marks the trial as failed.
    /// </summary>
    public void Fail()
    {
        Status = TrialStatus.Failed;
        EndTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Reports an intermediate value for a training step.
    /// </summary>
    public void ReportIntermediateValue(int step, T value)
    {
        IntermediateValues[step] = value;
    }
}

/// <summary>
/// Represents the status of a hyperparameter trial.
/// </summary>
public enum TrialStatus
{
    /// <summary>Trial is currently running.</summary>
    Running,
    /// <summary>Trial completed successfully.</summary>
    Complete,
    /// <summary>Trial was pruned (stopped early).</summary>
    Pruned,
    /// <summary>Trial failed with an error.</summary>
    Failed
}
