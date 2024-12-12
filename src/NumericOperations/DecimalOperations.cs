namespace AiDotNet.NumericOperations;

public class DecimalOperations : INumericOperations<decimal>
{
    public decimal Add(decimal a, decimal b) => a + b;
    public decimal Subtract(decimal a, decimal b) => a - b;
    public decimal Multiply(decimal a, decimal b) => a * b;
    public decimal Divide(decimal a, decimal b) => a / b;
    public decimal Negate(decimal a) => -a;
    public decimal Zero => 0m;
    public decimal One => 1m;
    public decimal Sqrt(decimal value) => (decimal)Math.Sqrt((double)value);
    public decimal FromDouble(double value) => (decimal)value;
    public bool GreaterThan(decimal a, decimal b) => a > b;
    public bool LessThan(decimal a, decimal b) => a < b;
    public decimal Abs(decimal value) => Math.Abs(value);
    public decimal Square(decimal value) => Multiply(value, value);
    public decimal Exp(decimal value) => (decimal)Math.Exp((double)value);
    public bool Equals(decimal a, decimal b) => a == b;
    public decimal Power(decimal baseValue, decimal exponent) => (decimal)Math.Pow((double)baseValue, (double)exponent);
    public decimal Log(decimal value) => (decimal)Math.Log((double)value);
    public bool GreaterThanOrEquals(decimal a, decimal b) => a >= b;
    public bool LessThanOrEquals(decimal a, decimal b) => a <= b;
    public int ToInt32(decimal value) => (int)Math.Round(value);
    public decimal Round(decimal value) => Math.Round(value);
}