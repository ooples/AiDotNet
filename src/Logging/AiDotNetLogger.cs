using AiDotNet.Interfaces;
using AiDotNet.Enums;

namespace AiDotNet.Logging;

/// <summary>
/// Implementation of the ML logger that uses Serilog underneath.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class handles the actual logging work behind the scenes.
/// It takes care of formatting log messages, adding contextual information, and writing
/// to the appropriate destinations according to your configuration.
/// </para>
/// </remarks>
public class AiDotNetLogger : ILogging
{
    private readonly Serilog.ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the AiDotNetLogger class with a Serilog logger.
    /// </summary>
    /// <param name="logger">The Serilog logger to use.</param>
    public AiDotNetLogger(Serilog.ILogger logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public void Trace(string message, params object[] args)
    {
        _logger.Verbose(message, args);
    }

    /// <inheritdoc/>
    public void Debug(string message, params object[] args)
    {
        _logger.Debug(message, args);
    }

    /// <inheritdoc/>
    public void Information(string message, params object[] args)
    {
        _logger.Information(message, args);
    }

    /// <inheritdoc/>
    public void Warning(string message, params object[] args)
    {
        _logger.Warning(message, args);
    }

    /// <inheritdoc/>
    public void Error(string message, params object[] args)
    {
        _logger.Error(message, args);
    }

    /// <inheritdoc/>
    public void Error(Exception exception, string message, params object[] args)
    {
        _logger.Error(exception, message, args);
    }

    /// <inheritdoc/>
    public void Critical(string message, params object[] args)
    {
        _logger.Fatal(message, args);
    }

    /// <inheritdoc/>
    public void Critical(Exception exception, string message, params object[] args)
    {
        _logger.Fatal(exception, message, args);
    }

    /// <inheritdoc/>
    public bool IsEnabled(LoggingLevel level)
    {
        return _logger.IsEnabled(ConvertLogLevel(level));
    }

    /// <inheritdoc/>
    public ILogging ForContext(string name, object value)
    {
        return new AiDotNetLogger(_logger.ForContext(name, value));
    }

    /// <inheritdoc/>
    public ILogging ForContext<T>()
    {
        return new AiDotNetLogger(_logger.ForContext<T>());
    }

    /// <inheritdoc/>
    public ILogging ForContext(Type type)
    {
        return new AiDotNetLogger(_logger.ForContext(type));
    }

    private static Serilog.Events.LogEventLevel ConvertLogLevel(LoggingLevel level)
    {
        return level switch
        {
            LoggingLevel.Trace => Serilog.Events.LogEventLevel.Verbose,
            LoggingLevel.Debug => Serilog.Events.LogEventLevel.Debug,
            LoggingLevel.Information => Serilog.Events.LogEventLevel.Information,
            LoggingLevel.Warning => Serilog.Events.LogEventLevel.Warning,
            LoggingLevel.Error => Serilog.Events.LogEventLevel.Error,
            LoggingLevel.Critical => Serilog.Events.LogEventLevel.Fatal,
            _ => Serilog.Events.LogEventLevel.Information
        };
    }
}