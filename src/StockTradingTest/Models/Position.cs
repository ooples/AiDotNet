namespace StockTradingTest.Models
{
    public class Position
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal EntryPrice { get; set; }
        public decimal CurrentPrice { get; set; }
        public decimal Quantity { get; set; }
        public DateTime EntryDate { get; set; }
        public PositionType Type { get; set; } = PositionType.Long;
        public decimal StopLoss { get; set; }
        public decimal TakeProfit { get; set; }

        public decimal CurrentValue => CurrentPrice * Quantity;
        public decimal ProfitLoss => Type == PositionType.Long 
            ? (CurrentPrice - EntryPrice) * Quantity 
            : (EntryPrice - CurrentPrice) * Quantity;
        
        public decimal ProfitLossPercent => Type == PositionType.Long
            ? (CurrentPrice / EntryPrice) - 1.0m
            : 1.0m - (CurrentPrice / EntryPrice);
    }

    public enum PositionType
    {
        Long,
        Short
    }
}