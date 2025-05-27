using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// Provides logging functionality for the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This interface defines the methods available for logging in AiDotNet.
/// It allows you to log messages at different levels of importance, from detailed debugging
/// information to critical errors. You can also include additional context that helps
/// understand what was happening when the log was created.
/// </para>
/// </remarks>
public interface ILogging
{
    /// <summary>
    /// Logs a trace message for detailed debugging and investigation.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Trace(string message, params object[] args);

    /// <summary>
    /// Logs a debug message for interactive investigation during development.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Debug(string message, params object[] args);

    /// <summary>
    /// Logs an informational message about the general flow of the application.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Information(string message, params object[] args);

    /// <summary>
    /// Logs a warning message about an abnormal or unexpected event.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Warning(string message, params object[] args);

    /// <summary>
    /// Logs an error message about a failure in the current operation.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Error(string message, params object[] args);

    /// <summary>
    /// Logs an error message with an associated exception.
    /// </summary>
    /// <param name="exception">The exception that caused the error.</param>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Error(Exception exception, string message, params object[] args);

    /// <summary>
    /// Logs a critical message about an unrecoverable application or system crash.
    /// </summary>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Critical(string message, params object[] args);

    /// <summary>
    /// Logs a critical message with an associated exception.
    /// </summary>
    /// <param name="exception">The exception that caused the critical error.</param>
    /// <param name="message">The message to log.</param>
    /// <param name="args">Optional format parameters for the message.</param>
    void Critical(Exception exception, string message, params object[] args);

    /// <summary>
    /// Gets whether a log level is enabled in the current configuration.
    /// </summary>
    /// <param name="level">The log level to check.</param>
    /// <returns>True if the log level is enabled; otherwise, false.</returns>
    bool IsEnabled(LoggingLevel level);

    /// <summary>
    /// Creates a new logger that includes a specific property with each log entry.
    /// </summary>
    /// <param name="name">The property name.</param>
    /// <param name="value">The property value.</param>
    /// <returns>A new logger instance that will include the specified property.</returns>
    ILogging ForContext(string name, object value);

    /// <summary>
    /// Creates a new logger for a specific type that will include the type name with each log entry.
    /// </summary>
    /// <typeparam name="T">The type to associate with the logger.</typeparam>
    /// <returns>A new logger instance that includes the type name.</returns>
    ILogging ForContext<T>();

    /// <summary>
    /// Creates a new logger for a specific type that will include the type name with each log entry.
    /// </summary>
    /// <param name="type">The type to associate with the logger.</param>
    /// <returns>A new logger instance that includes the type name.</returns>
    ILogging ForContext(Type type);
}