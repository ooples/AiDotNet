using System;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using Serilog;
using Serilog.Events;

namespace AiDotNet.Logging
{
    /// <summary>
    /// Adapter that wraps a Serilog logger to implement the ILogging interface.
    /// </summary>
    internal class SerilogAdapter : ILogging
    {
        private readonly ILogger _serilogLogger;

        public SerilogAdapter(ILogger serilogLogger)
        {
            _serilogLogger = serilogLogger ?? throw new ArgumentNullException(nameof(serilogLogger));
        }

        public void Trace(string message, params object[] args)
        {
            _serilogLogger.Verbose(message, args);
        }

        public void Debug(string message, params object[] args)
        {
            _serilogLogger.Debug(message, args);
        }

        public void Information(string message, params object[] args)
        {
            _serilogLogger.Information(message, args);
        }

        public void Warning(string message, params object[] args)
        {
            _serilogLogger.Warning(message, args);
        }

        public void Error(string message, params object[] args)
        {
            _serilogLogger.Error(message, args);
        }

        public void Error(Exception exception, string message, params object[] args)
        {
            _serilogLogger.Error(exception, message, args);
        }

        public void Critical(string message, params object[] args)
        {
            _serilogLogger.Fatal(message, args);
        }

        public void Critical(Exception exception, string message, params object[] args)
        {
            _serilogLogger.Fatal(exception, message, args);
        }

        public bool IsEnabled(LoggingLevel level)
        {
            var serilogLevel = level switch
            {
                LoggingLevel.Trace => LogEventLevel.Verbose,
                LoggingLevel.Debug => LogEventLevel.Debug,
                LoggingLevel.Information => LogEventLevel.Information,
                LoggingLevel.Warning => LogEventLevel.Warning,
                LoggingLevel.Error => LogEventLevel.Error,
                LoggingLevel.Critical => LogEventLevel.Fatal,
                _ => LogEventLevel.Information
            };
            
            return _serilogLogger.IsEnabled(serilogLevel);
        }

        public ILogging ForContext(string name, object value)
        {
            return new SerilogAdapter(_serilogLogger.ForContext(name, value));
        }

        public ILogging ForContext<T>()
        {
            return new SerilogAdapter(_serilogLogger.ForContext<T>());
        }

        public ILogging ForContext(Type type)
        {
            return new SerilogAdapter(_serilogLogger.ForContext(type));
        }
    }
}