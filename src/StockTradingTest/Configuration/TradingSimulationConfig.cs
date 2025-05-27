namespace StockTradingTest.Configuration
{
    public class TradingSimulationConfig
    {
        public decimal InitialBalance { get; set; } = 10000m;
        public int MaxPositions { get; set; } = 5;
        public int SimulationDays { get; set; } = 30;
        public decimal CommissionPerTrade { get; set; } = 0.001m;
        public decimal MaxPositionSizePercent { get; set; } = 0.2m;
        public decimal StopLossPercent { get; set; } = 0.05m;
        public decimal TakeProfitPercent { get; set; } = 0.1m;
        public decimal RiskPerTradePercent { get; set; } = 0.02m;
    }
}