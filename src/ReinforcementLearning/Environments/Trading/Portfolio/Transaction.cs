using System;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio
{
    /// <summary>
    /// Represents a trading transaction.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class Transaction<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the symbol that was traded.
        /// </summary>
        public string Symbol { get; }
        
        /// <summary>
        /// Gets the quantity that was traded (positive for buy, negative for sell).
        /// </summary>
        public T Quantity { get; }
        
        /// <summary>
        /// Gets the price at which the trade was executed.
        /// </summary>
        public T Price { get; }
        
        /// <summary>
        /// Gets the value of the trade (Quantity * Price).
        /// </summary>
        public T Value => NumOps.Multiply(Quantity, Price);
        
        /// <summary>
        /// Gets the fee paid for the transaction.
        /// </summary>
        public T Fee { get; }
        
        /// <summary>
        /// Gets the timestamp when the transaction occurred.
        /// </summary>
        public DateTime Timestamp { get; }
        
        /// <summary>
        /// Gets the cash balance after the transaction.
        /// </summary>
        public T CashAfter { get; }
        
        /// <summary>
        /// Gets the portfolio value after the transaction.
        /// </summary>
        public T PortfolioValueAfter { get; }
        
        /// <summary>
        /// Gets a value indicating whether this transaction is a buy (positive quantity).
        /// </summary>
        public bool IsBuy => NumOps.GreaterThan(Quantity, NumOps.Zero);
        
        /// <summary>
        /// Gets a value indicating whether this transaction is a sell (negative quantity).
        /// </summary>
        public bool IsSell => NumOps.LessThan(Quantity, NumOps.Zero);

        /// <summary>
        /// Initializes a new instance of the <see cref="Transaction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol that was traded.</param>
        /// <param name="quantity">The quantity that was traded (positive for buy, negative for sell).</param>
        /// <param name="price">The price at which the trade was executed.</param>
        /// <param name="fee">The fee paid for the transaction.</param>
        /// <param name="timestamp">The timestamp when the transaction occurred.</param>
        /// <param name="cashAfter">The cash balance after the transaction.</param>
        /// <param name="portfolioValueAfter">The portfolio value after the transaction.</param>
        public Transaction(
            string symbol, 
            T quantity, 
            T price, 
            T fee, 
            DateTime timestamp, 
            T cashAfter, 
            T portfolioValueAfter)
        {
            Symbol = symbol ?? throw new ArgumentNullException(nameof(symbol));
            Quantity = quantity;
            Price = price;
            Fee = fee;
            Timestamp = timestamp;
            CashAfter = cashAfter;
            PortfolioValueAfter = portfolioValueAfter;
        }

        /// <summary>
        /// Returns a string representation of the transaction.
        /// </summary>
        /// <returns>A string representation of the transaction.</returns>
        public override string ToString()
        {
            string direction = IsBuy ? "BUY" : "SELL";
            double quantityDouble = Convert.ToDouble(NumOps.Abs(Quantity));
            double priceDouble = Convert.ToDouble(Price);
            double valueDouble = Convert.ToDouble(Value);
            double feeDouble = Convert.ToDouble(Fee);
            
            return $"{direction} {quantityDouble} {Symbol} @ {priceDouble} " +
                   $"(Value: {valueDouble}, Fee: {feeDouble}, " +
                   $"Time: {Timestamp})";
        }
    }
}