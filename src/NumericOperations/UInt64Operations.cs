namespace AiDotNet.NumericOperations;

public class UInt64Operations : INumericOperations<ulong>
{
    public ulong Add(ulong a, ulong b) => a + b;
    public ulong Subtract(ulong a, ulong b) => a - b;
    public ulong Multiply(ulong a, ulong b) => a * b;
    public ulong Divide(ulong a, ulong b) => a / b;
    public ulong Negate(ulong a) => ulong.MaxValue - a + 1;
    public ulong Zero => 0;
    public ulong One => 1;
    public ulong Sqrt(ulong value) => (ulong)Math.Sqrt(value);
    public ulong FromDouble(double value) => (ulong)value;
    public bool GreaterThan(ulong a, ulong b) => a > b;
    public bool LessThan(ulong a, ulong b) => a < b;
    public ulong Abs(ulong value) => value;
    public ulong Square(ulong value) => Multiply(value, value);
    public ulong Exp(ulong value) => (ulong)Math.Min(ulong.MaxValue, Math.Round(Math.Exp((double)value)));
    public bool Equals(ulong a, ulong b) => a == b;
    public ulong Power(ulong baseValue, ulong exponent) => (ulong)Math.Pow(baseValue, exponent);
    public ulong Log(ulong value) => (ulong)Math.Log(value);
    public bool GreaterThanOrEquals(ulong a, ulong b) => a >= b;
    public bool LessThanOrEquals(ulong a, ulong b) => a <= b;
    public int ToInt32(ulong value) => (int)value;
    public ulong Round(ulong value) => value;
    public ulong MinValue => ulong.MinValue;
    public ulong MaxValue => ulong.MaxValue;
    public bool IsNaN(ulong value) => false;
    public bool IsInfinity(ulong value) => false;
    public ulong SignOrZero(ulong value) => value == 0 ? 0ul : 1ul;
}