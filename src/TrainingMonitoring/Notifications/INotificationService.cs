namespace AiDotNet.TrainingMonitoring.Notifications;

/// <summary>
/// Severity level for training notifications.
/// </summary>
public enum NotificationSeverity
{
    /// <summary>Informational message.</summary>
    Info,
    /// <summary>Warning that may require attention.</summary>
    Warning,
    /// <summary>Error that needs immediate attention.</summary>
    Error,
    /// <summary>Critical failure requiring immediate action.</summary>
    Critical,
    /// <summary>Success message indicating completion.</summary>
    Success
}

/// <summary>
/// Type of training notification.
/// </summary>
public enum NotificationType
{
    /// <summary>Training started.</summary>
    TrainingStarted,
    /// <summary>Training completed successfully.</summary>
    TrainingCompleted,
    /// <summary>Training failed with an error.</summary>
    TrainingFailed,
    /// <summary>Epoch completed.</summary>
    EpochCompleted,
    /// <summary>New best model achieved.</summary>
    NewBestModel,
    /// <summary>Checkpoint saved.</summary>
    CheckpointSaved,
    /// <summary>Early stopping triggered.</summary>
    EarlyStopping,
    /// <summary>Learning rate changed.</summary>
    LearningRateChanged,
    /// <summary>Resource warning (GPU/CPU/memory).</summary>
    ResourceWarning,
    /// <summary>Hyperparameter tuning progress.</summary>
    HyperparameterProgress,
    /// <summary>Custom notification.</summary>
    Custom
}

/// <summary>
/// Represents a training notification.
/// </summary>
public class TrainingNotification
{
    /// <summary>
    /// Gets or sets the notification title.
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the notification message.
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the notification severity.
    /// </summary>
    public NotificationSeverity Severity { get; set; } = NotificationSeverity.Info;

    /// <summary>
    /// Gets or sets the notification type.
    /// </summary>
    public NotificationType Type { get; set; } = NotificationType.Custom;

    /// <summary>
    /// Gets or sets the experiment name.
    /// </summary>
    public string? ExperimentName { get; set; }

    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string? RunId { get; set; }

    /// <summary>
    /// Gets or sets when the notification was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Creates a training started notification.
    /// </summary>
    public static TrainingNotification TrainingStarted(string experimentName, string? runId = null, Dictionary<string, object>? metadata = null)
    {
        return new TrainingNotification
        {
            Title = "Training Started",
            Message = $"Training has started for experiment '{experimentName}'.",
            Severity = NotificationSeverity.Info,
            Type = NotificationType.TrainingStarted,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = metadata ?? new Dictionary<string, object>()
        };
    }

    /// <summary>
    /// Creates a training completed notification.
    /// </summary>
    public static TrainingNotification TrainingCompleted(
        string experimentName,
        int totalEpochs,
        double finalLoss,
        double? finalAccuracy = null,
        TimeSpan? duration = null,
        string? runId = null)
    {
        var metadata = new Dictionary<string, object>
        {
            ["total_epochs"] = totalEpochs,
            ["final_loss"] = finalLoss
        };
        if (finalAccuracy.HasValue)
            metadata["final_accuracy"] = finalAccuracy.Value;
        if (duration.HasValue)
            metadata["duration_seconds"] = duration.Value.TotalSeconds;

        return new TrainingNotification
        {
            Title = "Training Completed",
            Message = $"Training completed for '{experimentName}' after {totalEpochs} epochs. Final loss: {finalLoss:F4}" +
                     (finalAccuracy.HasValue ? $", Accuracy: {finalAccuracy.Value:P2}" : string.Empty),
            Severity = NotificationSeverity.Success,
            Type = NotificationType.TrainingCompleted,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = metadata
        };
    }

    /// <summary>
    /// Creates a training failed notification.
    /// </summary>
    public static TrainingNotification TrainingFailed(
        string experimentName,
        string errorMessage,
        int? epochNumber = null,
        string? runId = null)
    {
        var metadata = new Dictionary<string, object>
        {
            ["error_message"] = errorMessage
        };
        if (epochNumber.HasValue)
            metadata["failed_at_epoch"] = epochNumber.Value;

        return new TrainingNotification
        {
            Title = "Training Failed",
            Message = $"Training failed for '{experimentName}': {errorMessage}",
            Severity = NotificationSeverity.Error,
            Type = NotificationType.TrainingFailed,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = metadata
        };
    }

    /// <summary>
    /// Creates a new best model notification.
    /// </summary>
    public static TrainingNotification NewBestModel(
        string experimentName,
        int epoch,
        double metricValue,
        string metricName = "validation_loss",
        string? runId = null)
    {
        return new TrainingNotification
        {
            Title = "New Best Model",
            Message = $"New best model for '{experimentName}' at epoch {epoch}! {metricName}: {metricValue:F4}",
            Severity = NotificationSeverity.Success,
            Type = NotificationType.NewBestModel,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = new Dictionary<string, object>
            {
                ["epoch"] = epoch,
                ["metric_name"] = metricName,
                ["metric_value"] = metricValue
            }
        };
    }

    /// <summary>
    /// Creates a checkpoint saved notification.
    /// </summary>
    public static TrainingNotification CheckpointSaved(
        string experimentName,
        string checkpointPath,
        int epoch,
        string? runId = null)
    {
        return new TrainingNotification
        {
            Title = "Checkpoint Saved",
            Message = $"Checkpoint saved for '{experimentName}' at epoch {epoch}.",
            Severity = NotificationSeverity.Info,
            Type = NotificationType.CheckpointSaved,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = new Dictionary<string, object>
            {
                ["checkpoint_path"] = checkpointPath,
                ["epoch"] = epoch
            }
        };
    }

    /// <summary>
    /// Creates an early stopping notification.
    /// </summary>
    public static TrainingNotification EarlyStopping(
        string experimentName,
        int epoch,
        int patience,
        double bestValue,
        string metricName = "validation_loss",
        string? runId = null)
    {
        return new TrainingNotification
        {
            Title = "Early Stopping Triggered",
            Message = $"Early stopping triggered for '{experimentName}' at epoch {epoch}. " +
                     $"No improvement in {metricName} for {patience} epochs. Best: {bestValue:F4}",
            Severity = NotificationSeverity.Warning,
            Type = NotificationType.EarlyStopping,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = new Dictionary<string, object>
            {
                ["epoch"] = epoch,
                ["patience"] = patience,
                ["best_value"] = bestValue,
                ["metric_name"] = metricName
            }
        };
    }

    /// <summary>
    /// Creates a resource warning notification.
    /// </summary>
    public static TrainingNotification ResourceWarning(
        string resourceType,
        double currentUsage,
        double threshold,
        string? experimentName = null,
        string? runId = null)
    {
        return new TrainingNotification
        {
            Title = "Resource Warning",
            Message = $"{resourceType} usage at {currentUsage:P1}, exceeding threshold of {threshold:P1}.",
            Severity = NotificationSeverity.Warning,
            Type = NotificationType.ResourceWarning,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = new Dictionary<string, object>
            {
                ["resource_type"] = resourceType,
                ["current_usage"] = currentUsage,
                ["threshold"] = threshold
            }
        };
    }

    /// <summary>
    /// Creates a hyperparameter progress notification.
    /// </summary>
    public static TrainingNotification HyperparameterProgress(
        string experimentName,
        int trialNumber,
        int totalTrials,
        double bestValue,
        Dictionary<string, object>? bestParams = null,
        string? runId = null)
    {
        var metadata = new Dictionary<string, object>
        {
            ["trial_number"] = trialNumber,
            ["total_trials"] = totalTrials,
            ["best_value"] = bestValue,
            ["progress_percent"] = (double)trialNumber / totalTrials * 100
        };
        if (bestParams is not null)
            metadata["best_params"] = bestParams;

        return new TrainingNotification
        {
            Title = "Hyperparameter Tuning Progress",
            Message = $"Trial {trialNumber}/{totalTrials} completed for '{experimentName}'. Best value: {bestValue:F4}",
            Severity = NotificationSeverity.Info,
            Type = NotificationType.HyperparameterProgress,
            ExperimentName = experimentName,
            RunId = runId,
            Metadata = metadata
        };
    }
}

/// <summary>
/// Interface for notification services.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Notification services allow you to receive alerts
/// about your training progress via email, Slack, or other channels.
/// This is especially useful for long-running training jobs where you
/// want to be notified of important events like completion, failures,
/// or when a new best model is found.
/// </remarks>
public interface INotificationService : IDisposable
{
    /// <summary>
    /// Gets the name of this notification service.
    /// </summary>
    string ServiceName { get; }

    /// <summary>
    /// Gets whether the service is properly configured and ready to send.
    /// </summary>
    bool IsConfigured { get; }

    /// <summary>
    /// Sends a notification asynchronously.
    /// </summary>
    /// <param name="notification">The notification to send.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if sent successfully, false otherwise.</returns>
    Task<bool> SendAsync(TrainingNotification notification, CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends a notification synchronously.
    /// </summary>
    /// <param name="notification">The notification to send.</param>
    /// <returns>True if sent successfully, false otherwise.</returns>
    bool Send(TrainingNotification notification);

    /// <summary>
    /// Tests the connection to the notification service.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if connection is successful, false otherwise.</returns>
    Task<bool> TestConnectionAsync(CancellationToken cancellationToken = default);
}
