namespace AiDotNet.NumericOperations;

public class UInt16Operations : INumericOperations<ushort>
{
    public ushort Add(ushort a, ushort b) => (ushort)(a + b);
    public ushort Subtract(ushort a, ushort b) => (ushort)(a - b);
    public ushort Multiply(ushort a, ushort b) => (ushort)(a * b);
    public ushort Divide(ushort a, ushort b) => (ushort)(a / b);
    public ushort Negate(ushort a) => (ushort)(ushort.MaxValue - a + 1);
    public ushort Zero => 0;
    public ushort One => 1;
    public ushort Sqrt(ushort value) => (ushort)Math.Sqrt(value);
    public ushort FromDouble(double value) => (ushort)value;
    public bool GreaterThan(ushort a, ushort b) => a > b;
    public bool LessThan(ushort a, ushort b) => a < b;
    public ushort Abs(ushort value) => value;
    public ushort Square(ushort value) => Multiply(value, value);
    public ushort Exp(ushort value) => (ushort)Math.Min(ushort.MaxValue, Math.Round(Math.Exp(value)));
    public bool Equals(ushort a, ushort b) => a == b;
    public ushort Power(ushort baseValue, ushort exponent) => (ushort)Math.Pow(baseValue, exponent);
    public ushort Log(ushort value) => (ushort)Math.Log(value);
    public bool GreaterThanOrEquals(ushort a, ushort b) => a >= b;
    public bool LessThanOrEquals(ushort a, ushort b) => a <= b;
    public int ToInt32(ushort value) => value;
    public ushort Round(ushort value) => value;
    public ushort MinValue => ushort.MinValue;
    public ushort MaxValue => ushort.MaxValue;
    public bool IsNaN(ushort value) => false;
    public bool IsInfinity(ushort value) => false;
    public ushort SignOrZero(ushort value) => value == 0 ? (ushort)0 : (ushort)1;
}