namespace StockTradingTest.Models
{
    public class TournamentResult
    {
        public string ModelId { get; set; } = string.Empty;
        public string ModelName { get; set; } = string.Empty;
        public int Round { get; set; }
        public decimal InitialBalance { get; set; }
        public decimal FinalBalance { get; set; }
        public decimal FinalProfitLoss => FinalBalance - InitialBalance;
        public decimal FinalProfitLossPercent => (FinalBalance / InitialBalance) - 1.0m;
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public int LosingTrades { get; set; }
        public decimal WinRate => TotalTrades > 0 ? (decimal)WinningTrades / TotalTrades : 0;
        public decimal MaxDrawdown { get; set; }
        public double Sharpe { get; set; }
        public double Sortino { get; set; }
        public double CalmarRatio { get; set; }
        public List<TradeAction> Trades { get; set; } = new List<TradeAction>();
        public List<PortfolioSnapshot> DailySnapshots { get; set; } = new List<PortfolioSnapshot>();
    }

    public class PortfolioSnapshot
    {
        public DateTime Date { get; set; }
        public decimal TotalValue { get; set; }
        public decimal Cash { get; set; }
        public int NumPositions { get; set; }
        public decimal ProfitLoss { get; set; }
        public decimal ProfitLossPercent { get; set; }
    }
}