using StockTradingTest.Models;

namespace StockTradingTest.Configuration
{
    public class LoggingConfig
    {
        public StockTradingLogLevel LogLevel { get; set; } = StockTradingLogLevel.Information;
        public bool LogToFile { get; set; } = true;
        public string LogFilePath { get; set; } = "Logs/trading_simulation.log";
    }
}