namespace StockTradingTest.Models
{
    public class Portfolio
    {
        public decimal Cash { get; set; }
        public List<Position> Positions { get; set; } = new List<Position>();
        public decimal InitialBalance { get; set; }
        public DateTime StartDate { get; set; }
        public int MaxPositions { get; set; }

        public Portfolio(decimal initialBalance, int maxPositions)
        {
            Cash = initialBalance;
            InitialBalance = initialBalance;
            MaxPositions = maxPositions;
            StartDate = DateTime.Now;
        }

        public decimal TotalValue
        {
            get
            {
                return Cash + Positions.Sum(p => p.CurrentValue);
            }
        }

        public decimal TotalProfitLoss
        {
            get
            {
                return TotalValue - InitialBalance;
            }
        }

        public decimal TotalProfitLossPercent
        {
            get
            {
                return (TotalValue / InitialBalance) - 1.0m;
            }
        }

        public bool HasPosition(string symbol)
        {
            return Positions.Any(p => p.Symbol == symbol);
        }

        public Position? GetPosition(string symbol)
        {
            return Positions.FirstOrDefault(p => p.Symbol == symbol);
        }

        public bool CanOpenPosition
        {
            get { return Positions.Count < MaxPositions; }
        }
    }
}