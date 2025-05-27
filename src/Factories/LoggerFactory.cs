using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Logging;
using AiDotNet.Models.Options;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using System.IO;

namespace AiDotNet.Factories;

/// <summary>
/// Factory for creating and configuring loggers for the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory creates loggers with the right configuration 
/// based on the options you provide. It helps set up file paths, log levels, and 
/// formats consistently throughout the application.
/// </para>
/// </remarks>
public static class LoggerFactory
{
    /// <summary>
    /// Creates a new logger configured according to the specified options.
    /// </summary>
    /// <param name="options">The logging configuration options.</param>
    /// <returns>An instance of ILogging configured according to the options.</returns>
    public static ILogging CreateLogger(LoggingOptions options)
    {
        if (!options.IsEnabled)
        {
            return new NullLogger();
        }

        var loggerConfiguration = new LoggerConfiguration()
            .MinimumLevel.Is(ConvertToSerilogLevel(options.MinimumLevel));

        // Configure console logging if enabled
        if (options.LogToConsole)
        {
            loggerConfiguration.WriteTo.Console();
        }

        // Configure file logging
        if (!string.IsNullOrEmpty(options.LogDirectory))
        {
            // Ensure the log directory exists
            Directory.CreateDirectory(options.LogDirectory);

            loggerConfiguration.WriteTo.File(
                Path.Combine(options.LogDirectory, options.FileNameTemplate),
                rollingInterval: RollingInterval.Day,
                fileSizeLimitBytes: options.FileSizeLimitBytes,
                retainedFileCountLimit: options.RetainedFileCountLimit);
        }

        // Add ML-specific enrichers if enabled
        if (options.IncludeMlContext)
        {
            loggerConfiguration.Enrich.WithProperty("Application", "AiDotNet");
        }

        return new AiDotNetLogger(loggerConfiguration.CreateLogger());
    }

    /// <summary>
    /// Creates a new logger with contextual information for model compression.
    /// </summary>
    /// <param name="options">The logging configuration options.</param>
    /// <returns>An instance of ILogging configured for model compression.</returns>
    public static ILogging CreateCompressionLogger(LoggingOptions options)
    {
        var logger = CreateLogger(options);
        return logger.ForContext("Component", "ModelCompression");
    }

    private static LogEventLevel ConvertToSerilogLevel(LoggingLevel level)
    {
        return level switch
        {
            LoggingLevel.Trace => LogEventLevel.Verbose,
            LoggingLevel.Debug => LogEventLevel.Debug,
            LoggingLevel.Information => LogEventLevel.Information,
            LoggingLevel.Warning => LogEventLevel.Warning,
            LoggingLevel.Error => LogEventLevel.Error,
            LoggingLevel.Critical => LogEventLevel.Fatal,
            _ => LogEventLevel.Information
        };
    }

    /// <summary>
    /// A logger implementation that doesn't perform any logging.
    /// </summary>
    private class NullLogger : ILogging
    {
        public void Critical(string message, params object[] args) { }
        public void Critical(Exception exception, string message, params object[] args) { }
        public void Debug(string message, params object[] args) { }
        public void Error(string message, params object[] args) { }
        public void Error(Exception exception, string message, params object[] args) { }
        public void Information(string message, params object[] args) { }
        public void Trace(string message, params object[] args) { }
        public void Warning(string message, params object[] args) { }
        public bool IsEnabled(LoggingLevel level) => false;
        public ILogging ForContext(string name, object value) => this;
        public ILogging ForContext<T>() => this;
        public ILogging ForContext(Type type) => this;
    }
}