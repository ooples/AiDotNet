global using AiDotNet.Logging;
global using Serilog;
global using ILogging = AiDotNet.Interfaces.ILogging;
global using Serilog.Sinks.SystemConsole;
global using Serilog.Sinks.File;

namespace AiDotNet.Factories;

/// <summary>
/// Factory for creating and configuring the logger used throughout the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This static class creates and manages the main logger object
/// used by the entire AiDotNet library. It handles initializing the logger with your
/// configuration settings and ensures all parts of the library use the same logging system.
/// </para>
/// </remarks>
public static class LoggingFactory
{
    private static ILogging _defaultLogger = CreateDefaultLogger();
    private static LoggingOptions _options = new LoggingOptions();

    /// <summary>
    /// Gets the current logging options.
    /// </summary>
    public static LoggingOptions Options => _options;

    /// <summary>
    /// Gets the current logger instance.
    /// </summary>
    public static ILogging Current => _defaultLogger;

    /// <summary>
    /// Configures the logger with the specified options.
    /// </summary>
    /// <param name="options">The logging options to use.</param>
    public static void Configure(LoggingOptions options)
    {
        _options = options ?? new LoggingOptions();
        _defaultLogger = CreateConfiguredLogger(_options);
    }

    /// <summary>
    /// Gets a logger for the specified type.
    /// </summary>
    /// <typeparam name="T">The type to associate with the logger.</typeparam>
    /// <returns>A logger instance that includes the type name in its context.</returns>
    public static ILogging GetLogger<T>()
    {
        return _defaultLogger.ForContext<T>();
    }

    /// <summary>
    /// Gets a logger for the specified type.
    /// </summary>
    /// <param name="type">The type to associate with the logger.</param>
    /// <returns>A logger instance that includes the type name in its context.</returns>
    public static ILogging GetLogger(Type type)
    {
        return _defaultLogger.ForContext(type);
    }

    /// <summary>
    /// Gets a logger with added context.
    /// </summary>
    /// <param name="name">The context property name.</param>
    /// <param name="value">The context property value.</param>
    /// <returns>A logger instance that includes the specified context property.</returns>
    public static ILogging GetContextualLogger(string name, object value)
    {
        return _defaultLogger.ForContext(name, value);
    }

    private static ILogging CreateDefaultLogger()
    {
        var serilogLogger = new Serilog.LoggerConfiguration()
            .MinimumLevel.Information()
            .CreateLogger();

        return new AiDotNetLogger(serilogLogger);
    }

    private static ILogging CreateConfiguredLogger(LoggingOptions options)
    {
        if (!options.IsEnabled)
        {
            // Return a logger that doesn't log anything
            var nullLogger = new Serilog.LoggerConfiguration()
                .MinimumLevel.Fatal()
                .CreateLogger();

            return new AiDotNetLogger(nullLogger);
        }

        var loggerConfiguration = new Serilog.LoggerConfiguration();

        // Set minimum level
        loggerConfiguration = options.MinimumLevel switch
        {
            LoggingLevel.Trace => loggerConfiguration.MinimumLevel.Verbose(),
            LoggingLevel.Debug => loggerConfiguration.MinimumLevel.Debug(),
            LoggingLevel.Information => loggerConfiguration.MinimumLevel.Information(),
            LoggingLevel.Warning => loggerConfiguration.MinimumLevel.Warning(),
            LoggingLevel.Error => loggerConfiguration.MinimumLevel.Error(),
            LoggingLevel.Critical => loggerConfiguration.MinimumLevel.Fatal(),
            _ => loggerConfiguration.MinimumLevel.Information()
        };

        // Ensure log directory exists
        Directory.CreateDirectory(options.LogDirectory);

        // Configure file logging
        loggerConfiguration = loggerConfiguration.WriteTo.File(
            Path.Combine(options.LogDirectory, options.FileNameTemplate),
            rollingInterval: RollingInterval.Day,
            fileSizeLimitBytes: options.FileSizeLimitBytes,
            retainedFileCountLimit: options.RetainedFileCountLimit,
            outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}{Properties:j}{NewLine}"
        );

        // Add console logging if enabled
        if (options.LogToConsole)
        {
            loggerConfiguration = loggerConfiguration.WriteTo.Console(
                outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}"
            );
        }

        var serilogLogger = loggerConfiguration.CreateLogger();
        return new AiDotNetLogger(serilogLogger);
    }

    /// <summary>
    /// Creates a zip file containing all current log files for support purposes.
    /// </summary>
    /// <param name="destinationPath">The path where the zip file should be created. If null, creates in the current directory.</param>
    /// <returns>The full path to the created zip file.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a single compressed file containing
    /// all your log files. This makes it easy to send logs to customer support when
    /// you need help troubleshooting an issue.
    /// </para>
    /// </remarks>
    public static string CreateLogArchive(string? destinationPath = null)
    {
        try
        {
            var logDir = _options.LogDirectory;
            if (!Directory.Exists(logDir))
            {
                return string.Empty;
            }

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var zipFileName = $"AiDotNet_Logs_{timestamp}.zip";

            destinationPath = destinationPath ?? Directory.GetCurrentDirectory();
            var zipFilePath = Path.Combine(destinationPath, zipFileName);

            if (File.Exists(zipFilePath))
            {
                File.Delete(zipFilePath);
            }

            System.IO.Compression.ZipFile.CreateFromDirectory(logDir, zipFilePath);
            return zipFilePath;
        }
        catch (Exception ex)
        {
            _defaultLogger.Error(ex, "Failed to create log archive");
            return string.Empty;
        }
    }
}