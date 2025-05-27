using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configures how logging should be performed in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class lets you customize how logging works. You can set where log files are saved,
/// how detailed they should be, how large each file can get, and how many are kept before the oldest is deleted.
/// These settings help manage disk space while ensuring you have the logs you need for troubleshooting.
/// </para>
/// </remarks>
public class LoggingOptions
{
    /// <summary>
    /// Gets or sets whether logging is enabled.
    /// </summary>
    public bool IsEnabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum level of log entries that will be recorded.
    /// </summary>
    public LoggingLevel MinimumLevel { get; set; } = LoggingLevel.Information;

    /// <summary>
    /// Gets or sets the directory where log files will be stored.
    /// </summary>
    public string LogDirectory { get; set; } = "Logs";

    /// <summary>
    /// Gets or sets the template for log file names.
    /// </summary>
    /// <remarks>
    /// The template can include placeholders:
    /// - {Date} will be replaced with the current date in yyyy-MM-dd format
    /// </remarks>
    public string FileNameTemplate { get; set; } = "aidotnet-{Date}.log";

    /// <summary>
    /// Gets or sets the maximum size in bytes for each log file before a new one is created.
    /// </summary>
    /// <remarks>
    /// Default is 10MB (10 * 1024 * 1024 bytes)
    /// </remarks>
    public long FileSizeLimitBytes { get; set; } = 10 * 1024 * 1024;

    /// <summary>
    /// Gets or sets the maximum number of log files to retain.
    /// </summary>
    /// <remarks>
    /// When this limit is reached, the oldest log files will be deleted as new ones are created.
    /// </remarks>
    public int RetainedFileCountLimit { get; set; } = 31;

    /// <summary>
    /// Gets or sets whether to log to the console in addition to files.
    /// </summary>
    public bool LogToConsole { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to include contextual information specific to machine learning operations.
    /// </summary>
    /// <remarks>
    /// This includes information such as feature counts, data shapes, performance metrics, etc.
    /// </remarks>
    public bool IncludeMlContext { get; set; } = true;
}