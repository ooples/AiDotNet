namespace StockTradingTest.Models
{
    public class TradeAction
    {
        public TradeType Type { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public decimal Quantity { get; set; }
        public decimal Price { get; set; }
        public DateTime Date { get; set; }
        public decimal Commission { get; set; }
        public string ModelId { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
    }

    public enum TradeType
    {
        Buy,
        Sell,
        ShortSell,
        ShortCover
    }
}