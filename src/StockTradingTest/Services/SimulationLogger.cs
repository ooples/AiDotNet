using StockTradingTest.Configuration;
using StockTradingTest.Models;

namespace StockTradingTest.Services
{
    public class SimulationLogger
    {
        private readonly LoggingConfig _config;
        private readonly string _logFilePath;

        public SimulationLogger(LoggingConfig config)
        {
            _config = config;

            if (_config.LogToFile)
            {
                var directory = Path.GetDirectoryName(_config.LogFilePath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                _logFilePath = _config.LogFilePath;
            }
            else
            {
                _logFilePath = string.Empty;
            }
        }

        public void Log(StockTradingLogLevel level, string message)
        {
            if (level < _config.LogLevel)
                return;

            var logMessage = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [{level}] {message}";
            
            // Console output
            Console.WriteLine(logMessage);
            
            // File output if enabled
            if (_config.LogToFile)
            {
                try
                {
                    File.AppendAllText(_logFilePath, logMessage + Environment.NewLine);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error writing to log file: {ex.Message}");
                }
            }
        }
    }
}