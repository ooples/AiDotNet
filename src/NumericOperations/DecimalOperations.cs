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
}