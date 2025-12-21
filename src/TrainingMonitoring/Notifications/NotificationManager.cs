using System.Collections.Concurrent;

namespace AiDotNet.TrainingMonitoring.Notifications;

/// <summary>
/// Event arguments for notification events.
/// </summary>
public class NotificationEventArgs : EventArgs
{
    /// <summary>
    /// Gets the notification that was sent.
    /// </summary>
    public TrainingNotification Notification { get; }

    /// <summary>
    /// Gets the service name that handled the notification.
    /// </summary>
    public string ServiceName { get; }

    /// <summary>
    /// Gets whether the notification was sent successfully.
    /// </summary>
    public bool Success { get; }

    /// <summary>
    /// Gets any exception that occurred during sending.
    /// </summary>
    public Exception? Exception { get; }

    /// <summary>
    /// Creates notification event arguments.
    /// </summary>
    public NotificationEventArgs(TrainingNotification notification, string serviceName, bool success, Exception? exception = null)
    {
        Notification = notification;
        ServiceName = serviceName;
        Success = success;
        Exception = exception;
    }
}

/// <summary>
/// Manages multiple notification services and provides unified notification delivery.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The NotificationManager allows you to send notifications
/// through multiple channels (email, Slack, etc.) at once. It supports:
/// - Sending to all registered services
/// - Filtering by severity (only send errors via email, info via Slack)
/// - Buffering notifications to avoid spam
/// - Background sending for non-blocking operation
///
/// Example usage:
/// <code>
/// var manager = new NotificationManager();
/// manager.AddService(new EmailNotificationService(emailConfig));
/// manager.AddService(new SlackNotificationService(slackConfig));
///
/// // Configure severity filtering
/// manager.SetMinimumSeverity("Email", NotificationSeverity.Error);
/// manager.SetMinimumSeverity("Slack", NotificationSeverity.Info);
///
/// // Send notification (goes to both services based on severity)
/// await manager.SendAsync(TrainingNotification.TrainingCompleted("MyExp", 100, 0.05));
/// </code>
/// </remarks>
public class NotificationManager : IDisposable
{
    private readonly ConcurrentDictionary<string, INotificationService> _services = new();
    private readonly ConcurrentDictionary<string, NotificationSeverity> _minSeverity = new();
    private readonly ConcurrentDictionary<NotificationType, bool> _enabledTypes = new();
    private readonly ConcurrentQueue<TrainingNotification> _buffer = new();
    private readonly object _bufferLock = new object();
    private readonly Timer? _flushTimer;
    private bool _disposed;

    /// <summary>
    /// Gets whether buffering is enabled.
    /// </summary>
    public bool BufferingEnabled { get; private set; }

    /// <summary>
    /// Gets the buffer flush interval.
    /// </summary>
    public TimeSpan BufferFlushInterval { get; private set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Gets the maximum buffer size before automatic flush.
    /// </summary>
    public int MaxBufferSize { get; set; } = 100;

    /// <summary>
    /// Gets the default minimum severity for all services.
    /// </summary>
    public NotificationSeverity DefaultMinimumSeverity { get; set; } = NotificationSeverity.Info;

    /// <summary>
    /// Event raised when a notification is sent.
    /// </summary>
    public event EventHandler<NotificationEventArgs>? NotificationSent;

    /// <summary>
    /// Event raised when a notification fails to send.
    /// </summary>
    public event EventHandler<NotificationEventArgs>? NotificationFailed;

    /// <summary>
    /// Creates a new notification manager.
    /// </summary>
    /// <param name="enableBuffering">Whether to enable buffering.</param>
    /// <param name="flushInterval">Buffer flush interval if buffering is enabled.</param>
    public NotificationManager(bool enableBuffering = false, TimeSpan? flushInterval = null)
    {
        BufferingEnabled = enableBuffering;
        if (flushInterval.HasValue)
        {
            BufferFlushInterval = flushInterval.Value;
        }

        if (BufferingEnabled)
        {
            _flushTimer = new Timer(FlushBuffer, null, BufferFlushInterval, BufferFlushInterval);
        }

        // Enable all notification types by default
        foreach (NotificationType type in Enum.GetValues(typeof(NotificationType)))
        {
            _enabledTypes[type] = true;
        }
    }

    /// <summary>
    /// Adds a notification service.
    /// </summary>
    /// <param name="service">The service to add.</param>
    /// <returns>True if added, false if a service with the same name exists.</returns>
    public bool AddService(INotificationService service)
    {
        if (service is null)
            throw new ArgumentNullException(nameof(service));

        return _services.TryAdd(service.ServiceName, service);
    }

    /// <summary>
    /// Removes a notification service.
    /// </summary>
    /// <param name="serviceName">The name of the service to remove.</param>
    /// <returns>True if removed, false if not found.</returns>
    public bool RemoveService(string serviceName)
    {
        if (_services.TryRemove(serviceName, out var service))
        {
            service.Dispose();
            return true;
        }
        return false;
    }

    /// <summary>
    /// Gets a registered service by name.
    /// </summary>
    public INotificationService? GetService(string serviceName)
    {
        _services.TryGetValue(serviceName, out var service);
        return service;
    }

    /// <summary>
    /// Gets all registered service names.
    /// </summary>
    public IEnumerable<string> GetServiceNames()
    {
        return _services.Keys.ToList();
    }

    /// <summary>
    /// Sets the minimum severity for a specific service.
    /// </summary>
    /// <param name="serviceName">The service name.</param>
    /// <param name="severity">The minimum severity to send.</param>
    public void SetMinimumSeverity(string serviceName, NotificationSeverity severity)
    {
        _minSeverity[serviceName] = severity;
    }

    /// <summary>
    /// Enables or disables a notification type.
    /// </summary>
    /// <param name="type">The notification type.</param>
    /// <param name="enabled">Whether to enable or disable.</param>
    public void SetTypeEnabled(NotificationType type, bool enabled)
    {
        _enabledTypes[type] = enabled;
    }

    /// <summary>
    /// Checks if a notification type is enabled.
    /// </summary>
    public bool IsTypeEnabled(NotificationType type)
    {
        return _enabledTypes.GetValueOrDefault(type, true);
    }

    /// <summary>
    /// Sends a notification to all applicable services.
    /// </summary>
    /// <param name="notification">The notification to send.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Dictionary of service names and their send results.</returns>
    public async Task<Dictionary<string, bool>> SendAsync(
        TrainingNotification notification,
        CancellationToken cancellationToken = default)
    {
        if (notification is null)
            throw new ArgumentNullException(nameof(notification));

        // Check if type is enabled
        if (!IsTypeEnabled(notification.Type))
        {
            return new Dictionary<string, bool>();
        }

        if (BufferingEnabled)
        {
            EnqueueNotification(notification);
            return new Dictionary<string, bool> { ["buffered"] = true };
        }

        return await SendImmediateAsync(notification, cancellationToken);
    }

    /// <summary>
    /// Sends a notification synchronously.
    /// </summary>
    public Dictionary<string, bool> Send(TrainingNotification notification)
    {
        if (notification is null)
            throw new ArgumentNullException(nameof(notification));

        if (!IsTypeEnabled(notification.Type))
        {
            return new Dictionary<string, bool>();
        }

        if (BufferingEnabled)
        {
            EnqueueNotification(notification);
            return new Dictionary<string, bool> { ["buffered"] = true };
        }

        var results = new Dictionary<string, bool>();

        foreach (var kvp in _services)
        {
            var serviceName = kvp.Key;
            var service = kvp.Value;

            if (!ShouldSend(serviceName, notification))
            {
                continue;
            }

            try
            {
                var success = service.Send(notification);
                results[serviceName] = success;

                if (success)
                {
                    OnNotificationSent(notification, serviceName);
                }
                else
                {
                    OnNotificationFailed(notification, serviceName, null);
                }
            }
            catch (Exception ex)
            {
                results[serviceName] = false;
                OnNotificationFailed(notification, serviceName, ex);
            }
        }

        return results;
    }

    /// <summary>
    /// Sends notifications in the background (fire-and-forget).
    /// </summary>
    /// <param name="notification">The notification to send.</param>
    public void SendBackground(TrainingNotification notification)
    {
        if (notification is null)
            throw new ArgumentNullException(nameof(notification));

        if (!IsTypeEnabled(notification.Type))
        {
            return;
        }

        if (BufferingEnabled)
        {
            EnqueueNotification(notification);
            return;
        }

        Task.Run(async () =>
        {
            try
            {
                await SendImmediateAsync(notification, CancellationToken.None);
            }
            catch (Exception)
            {
                // Swallow exceptions in background send
            }
        });
    }

    /// <summary>
    /// Tests all registered services.
    /// </summary>
    /// <returns>Dictionary of service names and their connection test results.</returns>
    public async Task<Dictionary<string, bool>> TestAllServicesAsync(CancellationToken cancellationToken = default)
    {
        var results = new Dictionary<string, bool>();

        foreach (var kvp in _services)
        {
            try
            {
                results[kvp.Key] = await kvp.Value.TestConnectionAsync(cancellationToken);
            }
            catch (Exception)
            {
                results[kvp.Key] = false;
            }
        }

        return results;
    }

    /// <summary>
    /// Flushes the notification buffer immediately.
    /// </summary>
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        var notifications = new List<TrainingNotification>();

        while (_buffer.TryDequeue(out var notification))
        {
            notifications.Add(notification);
        }

        if (notifications.Count == 0)
            return;

        // Send a digest notification
        var digest = CreateDigestNotification(notifications);
        await SendImmediateAsync(digest, cancellationToken);
    }

    private async Task<Dictionary<string, bool>> SendImmediateAsync(
        TrainingNotification notification,
        CancellationToken cancellationToken)
    {
        var results = new Dictionary<string, bool>();
        var tasks = new List<Task<(string serviceName, bool success, Exception? ex)>>();

        foreach (var kvp in _services)
        {
            var serviceName = kvp.Key;
            var service = kvp.Value;

            if (!ShouldSend(serviceName, notification))
            {
                continue;
            }

            tasks.Add(SendToServiceAsync(serviceName, service, notification, cancellationToken));
        }

        var sendResults = await Task.WhenAll(tasks);

        foreach (var (serviceName, success, ex) in sendResults)
        {
            results[serviceName] = success;

            if (success)
            {
                OnNotificationSent(notification, serviceName);
            }
            else
            {
                OnNotificationFailed(notification, serviceName, ex);
            }
        }

        return results;
    }

    private async Task<(string serviceName, bool success, Exception? ex)> SendToServiceAsync(
        string serviceName,
        INotificationService service,
        TrainingNotification notification,
        CancellationToken cancellationToken)
    {
        try
        {
            var success = await service.SendAsync(notification, cancellationToken);
            return (serviceName, success, null);
        }
        catch (Exception ex)
        {
            return (serviceName, false, ex);
        }
    }

    private bool ShouldSend(string serviceName, TrainingNotification notification)
    {
        // Check if service is configured
        if (!_services.TryGetValue(serviceName, out var service) || !service.IsConfigured)
        {
            return false;
        }

        // Check minimum severity
        var minSeverity = _minSeverity.GetValueOrDefault(serviceName, DefaultMinimumSeverity);
        return notification.Severity >= minSeverity;
    }

    private void EnqueueNotification(TrainingNotification notification)
    {
        _buffer.Enqueue(notification);

        // Check if buffer is full
        if (_buffer.Count >= MaxBufferSize)
        {
            lock (_bufferLock)
            {
                if (_buffer.Count >= MaxBufferSize)
                {
                    _ = FlushAsync(CancellationToken.None);
                }
            }
        }
    }

    private void FlushBuffer(object? state)
    {
        _ = FlushAsync(CancellationToken.None);
    }

    private TrainingNotification CreateDigestNotification(List<TrainingNotification> notifications)
    {
        var grouped = notifications.GroupBy(n => n.Severity).OrderByDescending(g => g.Key);
        var highestSeverity = notifications.Max(n => n.Severity);

        var summary = string.Join("\n", grouped.Select(g =>
            $"{g.Key}: {g.Count()} notification(s)"));

        return new TrainingNotification
        {
            Title = $"Training Digest ({notifications.Count} notifications)",
            Message = summary,
            Severity = highestSeverity,
            Type = NotificationType.Custom,
            Metadata = new Dictionary<string, object>
            {
                ["total_count"] = notifications.Count,
                ["by_type"] = notifications.GroupBy(n => n.Type).ToDictionary(g => g.Key.ToString(), g => g.Count()),
                ["by_severity"] = notifications.GroupBy(n => n.Severity).ToDictionary(g => g.Key.ToString(), g => g.Count()),
                ["time_range_start"] = notifications.Min(n => n.CreatedAt),
                ["time_range_end"] = notifications.Max(n => n.CreatedAt)
            }
        };
    }

    private void OnNotificationSent(TrainingNotification notification, string serviceName)
    {
        NotificationSent?.Invoke(this, new NotificationEventArgs(notification, serviceName, true));
    }

    private void OnNotificationFailed(TrainingNotification notification, string serviceName, Exception? ex)
    {
        NotificationFailed?.Invoke(this, new NotificationEventArgs(notification, serviceName, false, ex));
    }

    /// <summary>
    /// Disposes the notification manager and all services.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the notification manager and all services.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _flushTimer?.Dispose();

            foreach (var service in _services.Values)
            {
                service.Dispose();
            }

            _services.Clear();
            _disposed = true;
        }
    }
}
