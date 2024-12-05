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
}