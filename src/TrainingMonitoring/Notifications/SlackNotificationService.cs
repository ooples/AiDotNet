#if !NET6_0_OR_GREATER
#pragma warning disable CS8600, CS8601, CS8602, CS8603, CS8604
#endif
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.TrainingMonitoring.Notifications;

/// <summary>
/// Configuration for Slack notification service.
/// </summary>
public class SlackConfiguration
{
    /// <summary>
    /// Gets or sets the webhook URL.
    /// </summary>
    public string? WebhookUrl { get; set; }

    /// <summary>
    /// Gets or sets the bot token (alternative to webhook).
    /// </summary>
    public string? BotToken { get; set; }

    /// <summary>
    /// Gets or sets the channel to post to (required when using bot token).
    /// </summary>
    public string? Channel { get; set; }

    /// <summary>
    /// Gets or sets the bot username.
    /// </summary>
    public string Username { get; set; } = "AiDotNet Training";

    /// <summary>
    /// Gets or sets the bot icon emoji.
    /// </summary>
    public string IconEmoji { get; set; } = ":robot_face:";

    /// <summary>
    /// Gets or sets the request timeout in milliseconds.
    /// </summary>
    public int TimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Creates a configuration using a webhook URL.
    /// </summary>
    /// <param name="webhookUrl">The Slack incoming webhook URL.</param>
    public static SlackConfiguration ForWebhook(string webhookUrl)
    {
        return new SlackConfiguration
        {
            WebhookUrl = webhookUrl
        };
    }

    /// <summary>
    /// Creates a configuration using a bot token.
    /// </summary>
    /// <param name="botToken">The Slack bot OAuth token.</param>
    /// <param name="channel">The channel ID or name to post to.</param>
    public static SlackConfiguration ForBotToken(string botToken, string channel)
    {
        return new SlackConfiguration
        {
            BotToken = botToken,
            Channel = channel
        };
    }

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    public bool IsValid()
    {
        // Either webhook URL or (bot token + channel) must be provided
        if (!string.IsNullOrWhiteSpace(WebhookUrl))
        {
            return true;
        }

        return !string.IsNullOrWhiteSpace(BotToken) && !string.IsNullOrWhiteSpace(Channel);
    }
}

/// <summary>
/// Slack notification service using webhooks or bot tokens.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This service sends training notifications to Slack.
/// You can use either an incoming webhook (simpler) or a bot token (more flexible).
///
/// Webhook Example:
/// <code>
/// var config = SlackConfiguration.ForWebhook("https://hooks.slack.com/services/XXX/YYY/ZZZ");
/// var slackService = new SlackNotificationService(config);
///
/// var notification = TrainingNotification.TrainingCompleted("MyExperiment", 100, 0.05);
/// await slackService.SendAsync(notification);
/// </code>
///
/// Bot Token Example:
/// <code>
/// var config = SlackConfiguration.ForBotToken("xoxb-...", "#ml-training");
/// var slackService = new SlackNotificationService(config);
/// </code>
///
/// To create a webhook:
/// 1. Go to https://api.slack.com/apps
/// 2. Create a new app or select existing
/// 3. Go to Incoming Webhooks
/// 4. Add a webhook to your workspace
/// 5. Copy the webhook URL
/// </remarks>
public class SlackNotificationService : INotificationService
{
    private readonly SlackConfiguration _config;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private bool _disposed;

    private const string SlackApiUrl = "https://slack.com/api/chat.postMessage";

    /// <inheritdoc />
    public string ServiceName => "Slack";

    /// <inheritdoc />
    public bool IsConfigured => _config.IsValid();

    /// <summary>
    /// Creates a new Slack notification service.
    /// </summary>
    /// <param name="configuration">Slack configuration.</param>
    /// <param name="httpClient">Optional HTTP client to use.</param>
    public SlackNotificationService(SlackConfiguration configuration, HttpClient? httpClient = null)
    {
        Guard.NotNull(configuration);
        _config = configuration;

        if (httpClient is not null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromMilliseconds(_config.TimeoutMs)
            };
            _ownsHttpClient = true;
        }
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
            var payload = CreatePayload(notification);
            var json = JsonConvert.SerializeObject(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage response;

            if (!string.IsNullOrWhiteSpace(_config.WebhookUrl))
            {
                // Use webhook - just check HTTP status
                response = await _httpClient.PostAsync(_config.WebhookUrl, content, cancellationToken);
                return response.IsSuccessStatusCode;
            }
            else
            {
                // Use bot token - Slack API returns 200 even on failure, so check response body
                using var request = new HttpRequestMessage(HttpMethod.Post, SlackApiUrl);
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _config.BotToken);
                request.Content = content;
                response = await _httpClient.SendAsync(request, cancellationToken);

                if (!response.IsSuccessStatusCode)
                    return false;

                // Parse response to check "ok" field
                var responseContent = await response.Content.ReadAsStringAsync();
                var responseData = JsonConvert.DeserializeObject<Dictionary<string, object>>(responseContent);
                if (responseData != null && responseData.TryGetValue("ok", out var okValue))
                {
                    return okValue is bool ok && ok;
                }

                // If response format is unexpected, assume failure
                return false;
            }
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <inheritdoc />
    public bool Send(TrainingNotification notification)
    {
        try
        {
            return SendAsync(notification).GetAwaiter().GetResult();
        }
        catch (Exception)
        {
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

        var testNotification = new TrainingNotification
        {
            Title = "Test Notification",
            Message = "This is a test notification from AiDotNet Training Infrastructure.",
            Severity = NotificationSeverity.Info,
            Type = NotificationType.Custom
        };

        return await SendAsync(testNotification, cancellationToken);
    }

    private object CreatePayload(TrainingNotification notification)
    {
        var color = GetColorForSeverity(notification.Severity);
        var emoji = GetEmojiForSeverity(notification.Severity);

        var fields = new List<object>();

        // Add experiment info
        if (!string.IsNullOrEmpty(notification.ExperimentName))
        {
            fields.Add(new { title = "Experiment", value = notification.ExperimentName, @short = true });
        }

        if (!string.IsNullOrEmpty(notification.RunId))
        {
            fields.Add(new { title = "Run ID", value = notification.RunId, @short = true });
        }

        fields.Add(new { title = "Type", value = notification.Type.ToString(), @short = true });
        fields.Add(new { title = "Time", value = notification.CreatedAt.ToString("yyyy-MM-dd HH:mm:ss") + " UTC", @short = true });

        // Add metadata fields
        foreach (var kvp in notification.Metadata)
        {
            var value = FormatMetadataValue(kvp.Value);
            fields.Add(new { title = FormatKey(kvp.Key), value, @short = true });
        }

        var attachment = new
        {
            color,
            fallback = $"{emoji} {notification.Title}: {notification.Message}",
            pretext = $"{emoji} *{notification.Title}*",
            text = notification.Message,
            fields,
            footer = "AiDotNet Training Infrastructure",
            ts = ((DateTimeOffset)notification.CreatedAt).ToUnixTimeSeconds()
        };

        var payload = new Dictionary<string, object>
        {
            ["username"] = _config.Username,
            ["icon_emoji"] = _config.IconEmoji,
            ["attachments"] = new[] { attachment }
        };

        // Add channel if using bot token
        var channel = _config.Channel;
        if (!string.IsNullOrWhiteSpace(channel))
        {
            payload["channel"] = channel;
        }

        return payload;
    }

    private static string GetColorForSeverity(NotificationSeverity severity)
    {
        return severity switch
        {
            NotificationSeverity.Success => "#28a745",  // Green
            NotificationSeverity.Error => "#dc3545",    // Red
            NotificationSeverity.Critical => "#721c24", // Dark red
            NotificationSeverity.Warning => "#ffc107",  // Yellow
            _ => "#17a2b8"                              // Blue (info)
        };
    }

    private static string GetEmojiForSeverity(NotificationSeverity severity)
    {
        return severity switch
        {
            NotificationSeverity.Success => ":white_check_mark:",
            NotificationSeverity.Error => ":x:",
            NotificationSeverity.Critical => ":rotating_light:",
            NotificationSeverity.Warning => ":warning:",
            _ => ":information_source:"
        };
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
    /// Disposes the Slack service resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the Slack service resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            if (_ownsHttpClient)
            {
                _httpClient.Dispose();
            }
            _disposed = true;
        }
    }
}
