using System.Net;
using System.Net.Mail;
using System.Text;
using AiDotNet.Validation;

namespace AiDotNet.TrainingMonitoring.Notifications;

/// <summary>
/// Configuration for email notification service.
/// </summary>
public class EmailConfiguration
{
    /// <summary>
    /// Gets or sets the SMTP server host.
    /// </summary>
    public string SmtpHost { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the SMTP server port.
    /// </summary>
    public int SmtpPort { get; set; } = 587;

    /// <summary>
    /// Gets or sets whether to use SSL/TLS.
    /// </summary>
    public bool EnableSsl { get; set; } = true;

    /// <summary>
    /// Gets or sets the username for SMTP authentication.
    /// </summary>
    public string Username { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the password for SMTP authentication.
    /// </summary>
    public string Password { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the sender email address.
    /// </summary>
    public string FromAddress { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the sender display name.
    /// </summary>
    public string FromName { get; set; } = "AiDotNet Training";

    /// <summary>
    /// Gets or sets the recipient email addresses.
    /// </summary>
    public List<string> ToAddresses { get; set; } = new();

    /// <summary>
    /// Gets or sets CC email addresses.
    /// </summary>
    public List<string> CcAddresses { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to send HTML emails.
    /// </summary>
    public bool UseHtml { get; set; } = true;

    /// <summary>
    /// Gets or sets the connection timeout in milliseconds.
    /// </summary>
    public int TimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Creates a Gmail configuration.
    /// </summary>
    /// <param name="email">Gmail address.</param>
    /// <param name="appPassword">Gmail app password (not regular password).</param>
    /// <param name="toAddresses">Recipient addresses.</param>
    public static EmailConfiguration ForGmail(string email, string appPassword, params string[] toAddresses)
    {
        return new EmailConfiguration
        {
            SmtpHost = "smtp.gmail.com",
            SmtpPort = 587,
            EnableSsl = true,
            Username = email,
            Password = appPassword,
            FromAddress = email,
            FromName = "AiDotNet Training",
            ToAddresses = toAddresses.ToList()
        };
    }

    /// <summary>
    /// Creates an Outlook/Office365 configuration.
    /// </summary>
    /// <param name="email">Outlook email address.</param>
    /// <param name="password">Password or app password.</param>
    /// <param name="toAddresses">Recipient addresses.</param>
    public static EmailConfiguration ForOutlook(string email, string password, params string[] toAddresses)
    {
        return new EmailConfiguration
        {
            SmtpHost = "smtp.office365.com",
            SmtpPort = 587,
            EnableSsl = true,
            Username = email,
            Password = password,
            FromAddress = email,
            FromName = "AiDotNet Training",
            ToAddresses = toAddresses.ToList()
        };
    }

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    public bool IsValid()
    {
        return !string.IsNullOrWhiteSpace(SmtpHost) &&
               SmtpPort > 0 &&
               !string.IsNullOrWhiteSpace(Username) &&
               !string.IsNullOrWhiteSpace(Password) &&
               !string.IsNullOrWhiteSpace(FromAddress) &&
               ToAddresses.Count > 0;
    }
}

/// <summary>
/// Email notification service using SMTP.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This service sends training notifications via email.
/// It supports various email providers like Gmail, Outlook, and custom SMTP servers.
///
/// Example usage:
/// <code>
/// var config = EmailConfiguration.ForGmail("your@gmail.com", "app-password", "recipient@email.com");
/// var emailService = new EmailNotificationService(config);
///
/// var notification = TrainingNotification.TrainingCompleted("MyExperiment", 100, 0.05);
/// await emailService.SendAsync(notification);
/// </code>
///
/// Note: For Gmail, you need to use an "App Password" (not your regular password).
/// Enable 2FA and generate an app password in your Google Account settings.
/// </remarks>
public class EmailNotificationService : INotificationService
{
    private readonly EmailConfiguration _config;
    private readonly SmtpClient _smtpClient;
    private bool _disposed;

    /// <inheritdoc />
    public string ServiceName => "Email";

    /// <inheritdoc />
    public bool IsConfigured => _config.IsValid();

    /// <summary>
    /// Creates a new email notification service.
    /// </summary>
    /// <param name="configuration">Email configuration.</param>
    public EmailNotificationService(EmailConfiguration configuration)
    {
        Guard.NotNull(configuration);
        _config = configuration;

        _smtpClient = new SmtpClient(_config.SmtpHost, _config.SmtpPort)
        {
            EnableSsl = _config.EnableSsl,
            Credentials = new NetworkCredential(_config.Username, _config.Password),
            Timeout = _config.TimeoutMs,
            DeliveryMethod = SmtpDeliveryMethod.Network
        };
    }

    /// <inheritdoc />
    public async Task<bool> SendAsync(TrainingNotification notification, CancellationToken cancellationToken = default)
    {
        if (!IsConfigured)
        {
            return false;
        }

        try
        {
            using var message = CreateMessage(notification);

            // SmtpClient.SendMailAsync doesn't support CancellationToken in older .NET
            // Use Task.Run to allow cancellation
            await Task.Run(async () => await _smtpClient.SendMailAsync(message), cancellationToken);
            return true;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Log the exception for debugging - callers can check return value for success/failure
            System.Diagnostics.Debug.WriteLine($"[EmailNotificationService] Failed to send async email: {ex.Message}");
            return false;
        }
    }

    /// <inheritdoc />
    public bool Send(TrainingNotification notification)
    {
        if (!IsConfigured)
        {
            return false;
        }

        try
        {
            using var message = CreateMessage(notification);
            _smtpClient.Send(message);
            return true;
        }
        catch (Exception ex)
        {
            // Log the exception for debugging - callers can check return value for success/failure
            System.Diagnostics.Debug.WriteLine($"[EmailNotificationService] Failed to send email: {ex.Message}");
            return false;
        }
    }

    /// <inheritdoc />
    public async Task<bool> TestConnectionAsync(CancellationToken cancellationToken = default)
    {
        if (!IsConfigured)
        {
            return false;
        }

        try
        {
            var testNotification = new TrainingNotification
            {
                Title = "Test Notification",
                Message = "This is a test notification from AiDotNet Training Infrastructure.",
                Severity = NotificationSeverity.Info,
                Type = NotificationType.Custom
            };

            return await SendAsync(testNotification, cancellationToken);
        }
        catch (Exception)
        {
            return false;
        }
    }

    private MailMessage CreateMessage(TrainingNotification notification)
    {
        var message = new MailMessage
        {
            From = new MailAddress(_config.FromAddress, _config.FromName),
            Subject = FormatSubject(notification),
            IsBodyHtml = _config.UseHtml
        };

        if (_config.UseHtml)
        {
            message.Body = FormatHtmlBody(notification);
            message.AlternateViews.Add(
                AlternateView.CreateAlternateViewFromString(
                    FormatPlainTextBody(notification),
                    Encoding.UTF8,
                    "text/plain"));
        }
        else
        {
            message.Body = FormatPlainTextBody(notification);
        }

        foreach (var to in _config.ToAddresses)
        {
            message.To.Add(new MailAddress(to));
        }

        foreach (var cc in _config.CcAddresses)
        {
            message.CC.Add(new MailAddress(cc));
        }

        // Set priority based on severity
        message.Priority = notification.Severity switch
        {
            NotificationSeverity.Critical => MailPriority.High,
            NotificationSeverity.Error => MailPriority.High,
            NotificationSeverity.Warning => MailPriority.Normal,
            _ => MailPriority.Normal
        };

        return message;
    }

    private static string FormatSubject(TrainingNotification notification)
    {
        var severityEmoji = notification.Severity switch
        {
            NotificationSeverity.Success => "[SUCCESS]",
            NotificationSeverity.Error => "[ERROR]",
            NotificationSeverity.Critical => "[CRITICAL]",
            NotificationSeverity.Warning => "[WARNING]",
            _ => "[INFO]"
        };

        var experimentPart = string.IsNullOrEmpty(notification.ExperimentName)
            ? string.Empty
            : $" - {notification.ExperimentName}";

        return $"{severityEmoji} {notification.Title}{experimentPart}";
    }

    private static string FormatHtmlBody(TrainingNotification notification)
    {
        var severityColor = notification.Severity switch
        {
            NotificationSeverity.Success => "#28a745",
            NotificationSeverity.Error => "#dc3545",
            NotificationSeverity.Critical => "#721c24",
            NotificationSeverity.Warning => "#ffc107",
            _ => "#17a2b8"
        };

        var sb = new StringBuilder();
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html><head><style>");
        sb.AppendLine("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }");
        sb.AppendLine(".header { padding: 15px; border-radius: 8px 8px 0 0; color: white; }");
        sb.AppendLine(".content { background: #f8f9fa; padding: 20px; border-radius: 0 0 8px 8px; }");
        sb.AppendLine(".metadata { background: white; border-radius: 4px; padding: 15px; margin-top: 15px; }");
        sb.AppendLine(".metadata-item { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }");
        sb.AppendLine(".metadata-item:last-child { border-bottom: none; }");
        sb.AppendLine(".label { color: #666; font-weight: 500; }");
        sb.AppendLine(".value { color: #333; }");
        sb.AppendLine(".footer { margin-top: 20px; font-size: 12px; color: #666; text-align: center; }");
        sb.AppendLine("</style></head><body>");

        // Header
        sb.AppendLine($"<div class=\"header\" style=\"background-color: {severityColor};\">");
        sb.AppendLine($"<h2 style=\"margin: 0;\">{notification.Title}</h2>");
        sb.AppendLine("</div>");

        // Content
        sb.AppendLine("<div class=\"content\">");
        sb.AppendLine($"<p>{notification.Message}</p>");

        // Experiment info
        if (!string.IsNullOrEmpty(notification.ExperimentName))
        {
            sb.AppendLine("<div class=\"metadata\">");
            sb.AppendLine($"<div class=\"metadata-item\"><span class=\"label\">Experiment:</span><span class=\"value\">{notification.ExperimentName}</span></div>");

            if (!string.IsNullOrEmpty(notification.RunId))
            {
                sb.AppendLine($"<div class=\"metadata-item\"><span class=\"label\">Run ID:</span><span class=\"value\">{notification.RunId}</span></div>");
            }

            sb.AppendLine($"<div class=\"metadata-item\"><span class=\"label\">Time:</span><span class=\"value\">{notification.CreatedAt:yyyy-MM-dd HH:mm:ss} UTC</span></div>");
            sb.AppendLine("</div>");
        }

        // Additional metadata
        if (notification.Metadata.Count > 0)
        {
            sb.AppendLine("<div class=\"metadata\" style=\"margin-top: 10px;\">");
            sb.AppendLine("<p style=\"margin: 0 0 10px 0; font-weight: bold; color: #666;\">Additional Details:</p>");

            foreach (var kvp in notification.Metadata)
            {
                var value = FormatMetadataValue(kvp.Value);
                sb.AppendLine($"<div class=\"metadata-item\"><span class=\"label\">{FormatKey(kvp.Key)}:</span><span class=\"value\">{value}</span></div>");
            }

            sb.AppendLine("</div>");
        }

        sb.AppendLine("</div>");

        // Footer
        sb.AppendLine("<div class=\"footer\">");
        sb.AppendLine("<p>This notification was sent by AiDotNet Training Infrastructure.</p>");
        sb.AppendLine("</div>");

        sb.AppendLine("</body></html>");

        return sb.ToString();
    }

    private static string FormatPlainTextBody(TrainingNotification notification)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"=== {notification.Title.ToUpperInvariant()} ===");
        sb.AppendLine();
        sb.AppendLine(notification.Message);
        sb.AppendLine();

        if (!string.IsNullOrEmpty(notification.ExperimentName))
        {
            sb.AppendLine("--- Experiment Info ---");
            sb.AppendLine($"Experiment: {notification.ExperimentName}");

            if (!string.IsNullOrEmpty(notification.RunId))
            {
                sb.AppendLine($"Run ID: {notification.RunId}");
            }

            sb.AppendLine($"Time: {notification.CreatedAt:yyyy-MM-dd HH:mm:ss} UTC");
            sb.AppendLine();
        }

        if (notification.Metadata.Count > 0)
        {
            sb.AppendLine("--- Additional Details ---");

            foreach (var kvp in notification.Metadata)
            {
                var value = FormatMetadataValue(kvp.Value);
                sb.AppendLine($"{FormatKey(kvp.Key)}: {value}");
            }

            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine("This notification was sent by AiDotNet Training Infrastructure.");

        return sb.ToString();
    }

    private static string FormatKey(string key)
    {
        // Convert snake_case to Title Case
        return string.Join(" ", key.Split('_').Select(s =>
            string.IsNullOrEmpty(s) ? s : char.ToUpperInvariant(s[0]) + s.Substring(1)));
    }

    private static string FormatMetadataValue(object value)
    {
        return value switch
        {
            double d => d.ToString("F4"),
            float f => f.ToString("F4"),
            DateTime dt => dt.ToString("yyyy-MM-dd HH:mm:ss") + " UTC",
            TimeSpan ts => ts.ToString(@"hh\:mm\:ss"),
            Dictionary<string, object> dict => string.Join(", ", dict.Select(kv => $"{kv.Key}={kv.Value}")),
            _ => value?.ToString() ?? "null"
        };
    }

    /// <summary>
    /// Disposes the email service resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the email service resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _smtpClient.Dispose();
            _disposed = true;
        }
    }
}
