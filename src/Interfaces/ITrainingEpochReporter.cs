namespace AiDotNet.Interfaces;

/// <summary>
/// Implemented by models that run their own epoch loop internally and can report each epoch back
/// to the caller.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <remarks>
/// <para>
/// The <c>AiModelBuilder</c> facade drives an epoch loop itself for models it trains step-by-step,
/// and reports every epoch to the callbacks registered via <c>ConfigureTrainingCallback</c>. Models
/// that instead own their loop (the TimeSeries family, whose <c>Train</c> the facade calls once)
/// are invisible to that machinery, so their users get no progress reporting and no early-stop
/// veto. Implementing this interface lets the facade hand its per-epoch hook to the model so those
/// features behave identically across both kinds of model.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some models do all their training inside one call, so the library can't
/// see the individual passes ("epochs") happening. This interface is how such a model tells the
/// library about each pass, which is what makes progress reporting and early stopping work.
/// </para>
/// </remarks>
public interface ITrainingEpochReporter<T>
{
    /// <summary>
    /// Gets or sets the per-epoch hook invoked once at the end of every training epoch.
    /// </summary>
    /// <value>
    /// A function receiving the epoch's progress snapshot and returning <c>true</c> to continue
    /// training or <c>false</c> to stop early; <c>null</c> when nothing is observing.
    /// </value>
    /// <remarks>
    /// Set by the facade during the build. A model must treat a <c>false</c> return as a request
    /// to stop after the current epoch and leave itself in a usable, trained state. Custom models that own
    /// their epoch loop implement this so the facade can drive their epochs like any other model.
    /// </remarks>
    Func<TrainingProgress<T>, bool>? TrainingEpochCallback { get; set; }
}
