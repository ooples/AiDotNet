namespace AiDotNet.NumericOperations;

public class Int64Operations : INumericOperations<long>
{
    public long Add(long a, long b) => a + b;
    public long Subtract(long a, long b) => a - b;
    public long Multiply(long a, long b) => a * b;
    public long Divide(long a, long b) => a / b;
    public long Negate(long a) => -a;
    public long Zero => 0L;
    public long One => 1L;
    public long Sqrt(long value) => (long)Math.Sqrt(value);
    public long FromDouble(double value) => (long)value;
    public bool GreaterThan(long a, long b) => a > b;
    public bool LessThan(long a, long b) => a < b;
    public long Abs(long value) => Math.Abs(value);
    public long Square(long value) => Multiply(value, value);
    public long Exp(long value) => (long)Math.Round(Math.Exp(value));
    public bool Equals(long a, long b) => a == b;
    public long Power(long baseValue, long exponent) => (long)Math.Pow(baseValue, exponent);
    public long Log(long value) => (long)Math.Log(value);
    public bool GreaterThanOrEquals(long a, long b) => a >= b;
    public bool LessThanOrEquals(long a, long b) => a <= b;
    public int ToInt32(long value) => (int)value;
    public long Round(long value) => value;
}