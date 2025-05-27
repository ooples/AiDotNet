using System;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio
{
    /// <summary>
    /// Represents a position in a financial instrument.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class Position<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the symbol of the financial instrument.
        /// </summary>
        public string Symbol { get; }
        
        /// <summary>
        /// Gets the quantity of the position.
        /// </summary>
        public T Quantity { get; private set; }
        
        /// <summary>
        /// Gets the average entry price of the position.
        /// </summary>
        public T AverageEntryPrice { get; private set; }
        
        /// <summary>
        /// Gets the current price of the instrument.
        /// </summary>
        public T CurrentPrice { get; private set; }
        
        /// <summary>
        /// Gets the current value of the position (Quantity * CurrentPrice).
        /// </summary>
        public T CurrentValue => NumOps.Multiply(Quantity, CurrentPrice);
        
        /// <summary>
        /// Gets the cost basis of the position (Quantity * AverageEntryPrice).
        /// </summary>
        public T CostBasis => NumOps.Multiply(Quantity, AverageEntryPrice);
        
        /// <summary>
        /// Gets the unrealized profit/loss of the position (CurrentValue - CostBasis).
        /// </summary>
        public T UnrealizedPnL => NumOps.Subtract(CurrentValue, CostBasis);
        
        /// <summary>
        /// Gets the unrealized profit/loss percentage ((CurrentPrice - AverageEntryPrice) / AverageEntryPrice).
        /// </summary>
        public T UnrealizedPnLPercent
        {
            get
            {
                if (NumOps.Equals(AverageEntryPrice, NumOps.Zero)) return NumOps.Zero;
                return NumOps.Divide(
                    NumOps.Subtract(CurrentPrice, AverageEntryPrice), 
                    AverageEntryPrice);
            }
        }
        
        /// <summary>
        /// Gets the timestamp of the last update.
        /// </summary>
        public DateTime LastUpdateTime { get; private set; }
        
        /// <summary>
        /// Gets a value indicating whether this position is long (positive quantity).
        /// </summary>
        public bool IsLong => NumOps.GreaterThan(Quantity, NumOps.Zero);
        
        /// <summary>
        /// Gets a value indicating whether this position is short (negative quantity).
        /// </summary>
        public bool IsShort => NumOps.LessThan(Quantity, NumOps.Zero);
        
        /// <summary>
        /// Gets a value indicating whether this position is flat (zero quantity).
        /// </summary>
        public bool IsFlat => NumOps.Equals(Quantity, NumOps.Zero);

        /// <summary>
        /// Gets the realized profit/loss from all closed trades.
        /// </summary>
        public T RealizedPnL { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Position{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol of the financial instrument.</param>
        /// <param name="quantity">The initial quantity.</param>
        /// <param name="averageEntryPrice">The initial average entry price.</param>
        /// <param name="currentPrice">The current price.</param>
        /// <param name="timestamp">The timestamp of creation.</param>
        public Position(
            string symbol, 
            T quantity,
            T averageEntryPrice,
            T currentPrice,
            DateTime? timestamp = null)
        {
            Symbol = symbol ?? throw new ArgumentNullException(nameof(symbol));
            Quantity = quantity;
            AverageEntryPrice = averageEntryPrice;
            CurrentPrice = currentPrice;
            RealizedPnL = NumOps.Zero;
            LastUpdateTime = timestamp ?? DateTime.UtcNow;
        }

        /// <summary>
        /// Updates the position with a new trade.
        /// </summary>
        /// <param name="quantity">The quantity to add (positive) or subtract (negative).</param>
        /// <param name="price">The price of the trade.</param>
        /// <param name="timestamp">The timestamp of the trade.</param>
        /// <returns>The realized profit/loss from this trade, if any.</returns>
        public T UpdatePosition(T quantity, T price, DateTime timestamp)
        {
            if (NumOps.Equals(quantity, NumOps.Zero)) return NumOps.Zero;
            
            T realizedPnL = NumOps.Zero;
            
            // If the trade is in the opposite direction of the current position,
            // calculate realized P&L for the portion that closes the position
            if (!NumOps.Equals(Quantity, NumOps.Zero) && 
                NumOps.LessThan(NumOps.Multiply(Quantity, quantity), NumOps.Zero))
            {
                // Different sign means closing or reducing the position
                T closingQuantity;
                if (NumOps.LessThan(quantity, NumOps.Zero))
                {
                    closingQuantity = NumOps.LessThan(NumOps.Negate(quantity), Quantity) 
                        ? NumOps.Negate(quantity) 
                        : Quantity;
                }
                else
                {
                    closingQuantity = NumOps.GreaterThan(NumOps.Negate(quantity), NumOps.Negate(Quantity)) 
                        ? NumOps.Negate(Quantity) 
                        : quantity;
                }
                
                // Calculate P&L for the closed portion
                realizedPnL = NumOps.Multiply(closingQuantity, NumOps.Subtract(price, AverageEntryPrice));
                
                // Add to the running total of realized P&L
                RealizedPnL = NumOps.Add(RealizedPnL, realizedPnL);
            }
            
            // Update the position
            T newQuantity = NumOps.Add(Quantity, quantity);
            
            // If position direction is changing, reset the average price
            if ((NumOps.GreaterThan(Quantity, NumOps.Zero) && NumOps.LessThanOrEquals(newQuantity, NumOps.Zero)) ||
                (NumOps.LessThan(Quantity, NumOps.Zero) && NumOps.GreaterThanOrEquals(newQuantity, NumOps.Zero)))
            {
                AverageEntryPrice = price;
            }
            // If adding to the position, update the average price
            else if (!NumOps.Equals(newQuantity, NumOps.Zero))
            {
                // Calculate the new average entry price
                T totalCost = NumOps.Add(
                    NumOps.Multiply(Quantity, AverageEntryPrice), 
                    NumOps.Multiply(quantity, price));
                AverageEntryPrice = NumOps.Divide(totalCost, newQuantity);
            }
            
            Quantity = newQuantity;
            CurrentPrice = price;
            LastUpdateTime = timestamp;
            
            return realizedPnL;
        }

        /// <summary>
        /// Updates the current price without changing the position.
        /// </summary>
        /// <param name="price">The new current price.</param>
        /// <param name="timestamp">The timestamp of the update.</param>
        public void UpdatePrice(T price, DateTime timestamp)
        {
            CurrentPrice = price;
            LastUpdateTime = timestamp;
        }

        /// <summary>
        /// Closes the position entirely at the specified price.
        /// </summary>
        /// <param name="price">The price to close the position at.</param>
        /// <param name="timestamp">The timestamp of the close.</param>
        /// <returns>The realized profit/loss from closing the position.</returns>
        public T ClosePosition(T price, DateTime timestamp)
        {
            if (IsFlat) return NumOps.Zero;
            
            // Close the position by trading the opposite of current quantity
            T closeQuantity = NumOps.Negate(Quantity);
            return UpdatePosition(closeQuantity, price, timestamp);
        }

        /// <summary>
        /// Returns a string representation of the position.
        /// </summary>
        /// <returns>A string that represents the position.</returns>
        public override string ToString()
        {
            string positionType = IsLong ? "LONG" : IsShort ? "SHORT" : "FLAT";
            double quantityDouble = Convert.ToDouble(NumOps.Abs(Quantity));
            double entryPriceDouble = Convert.ToDouble(AverageEntryPrice);
            double currentPriceDouble = Convert.ToDouble(CurrentPrice);
            double pnlDouble = Convert.ToDouble(UnrealizedPnL);
            double pnlPercentDouble = Convert.ToDouble(UnrealizedPnLPercent);
            
            return $"{positionType} {quantityDouble} {Symbol} @ {entryPriceDouble} " +
                   $"(Current: {currentPriceDouble}, P&L: {pnlDouble}, " +
                   $"{pnlPercentDouble:P2})";
        }
    }
}