namespace AiDotNet.Interfaces;

/// <summary>
/// A user-registerable hook that observes supervised training epoch-by-epoch and can
/// request that training stop early.
/// </summary>
/// <remarks>
/// <para>
/// Register callbacks on the facade via
/// <c>AiModelBuilder.ConfigureTrainingCallback(...)</c>. The training loop invokes
/// <see cref="OnTrainBegin"/> once before the first epoch, <see cref="OnEpochEnd"/> after
/// every completed epoch, and <see cref="OnTrainEnd"/> once after training finishes
/// (whether it ran to completion or was aborted). Multiple callbacks may be registered;
/// they are all invoked, and training aborts if <em>any</em> of them returns
/// <c>false</c> from <see cref="OnEpochEnd"/>.
/// </para>
/// <para><b>For Beginners:</b> This is your "listen in on training" hook. Implement it (or
/// use the one-line <c>Func</c> overload of <c>ConfigureTrainingCallback</c>) to:
/// - print or plot the loss after each epoch,
/// - stream progress to a UI,
/// - or pull the plug early — return <c>false</c> from <see cref="OnEpochEnd"/> and the
///   trainer stops cleanly at that epoch and returns the model trained so far.
/// The framework calls these methods for you; you never call them yourself.
/// </para>
/// <para><b>Threading:</b> callbacks are invoked synchronously from the training loop on the
/// training thread. Keep the work light; offload anything heavy (network, disk) so you do
/// not stall training.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ITrainingCallback<T>
{
    /// <summary>
    /// Called once, just before the first epoch runs.
    /// </summary>
    /// <param name="progress">
    /// The initial progress snapshot. <see cref="TrainingProgress{T}.Epoch"/> is <c>0</c> and
    /// <see cref="TrainingProgress{T}.Loss"/> is the pre-training baseline (typically the
    /// numeric zero), while <see cref="TrainingProgress{T}.TotalEpochs"/> reflects the planned
    /// epoch count.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> A good place to record a start time, open a log file, or draw the
    /// empty axes of a chart you will fill in as epochs complete.
    /// </remarks>
    void OnTrainBegin(TrainingProgress<T> progress);

    /// <summary>
    /// Called after each epoch completes. Return <c>false</c> to request that training abort.
    /// </summary>
    /// <param name="progress">The progress snapshot for the epoch that just finished.</param>
    /// <returns>
    /// <c>true</c> to continue training; <c>false</c> to request an early, graceful stop.
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This runs once per epoch. Inspect
    /// <see cref="TrainingProgress{T}.Loss"/> (and any <see cref="TrainingProgress{T}.Metrics"/>)
    /// and decide whether it is worth continuing. Returning <c>false</c> is the standard way to
    /// implement custom stopping rules (e.g. "stop if loss stopped improving" or "stop if the
    /// loss became NaN"). When any registered callback returns <c>false</c>, the trainer stops
    /// at the current epoch and returns the model as trained so far.
    /// </remarks>
    bool OnEpochEnd(TrainingProgress<T> progress);

    /// <summary>
    /// Called once, after training finishes — whether it completed all epochs or was aborted.
    /// </summary>
    /// <param name="progress">The final progress snapshot (the last completed epoch).</param>
    /// <remarks>
    /// <b>For Beginners:</b> A good place to flush logs, close files, or render the final chart.
    /// This is always called, so it is safe to release resources you acquired in
    /// <see cref="OnTrainBegin"/> here.
    /// </remarks>
    void OnTrainEnd(TrainingProgress<T> progress);
}
