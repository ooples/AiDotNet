namespace StockTradingTest.Configuration
{
    public class DataSourceConfig
    {
        public string StockDataPath { get; set; } = "Data/StockData";
        public List<string> DefaultSymbols { get; set; } = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "META" };
        public DateTime StartDate { get; set; } = new DateTime(2020, 1, 1);
        public DateTime EndDate { get; set; } = new DateTime(2023, 12, 31);
        public List<string> FeaturesToInclude { get; set; } = new List<string> { "Open", "High", "Low", "Close", "Volume" };
    }
}