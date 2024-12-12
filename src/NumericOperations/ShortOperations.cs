namespace AiDotNet.NumericOperations;

public class ShortOperations : INumericOperations<short>
{
    public short Add(short a, short b) => (short)(a + b);
    public short Subtract(short a, short b) => (short)(a - b);
    public short Multiply(short a, short b) => (short)(a * b);
    public short Divide(short a, short b) => (short)(a / b);
    public short Negate(short a) => (short)-a;
    public short Zero => 0;
    public short One => 1;
    public short Sqrt(short value) => (short)Math.Sqrt(value);
    public short FromDouble(double value) => (short)value;
    public bool GreaterThan(short a, short b) => a > b;
    public bool LessThan(short a, short b) => a < b;
    public short Abs(short value) => Math.Abs(value);
    public short Square(short value) => Multiply(value, value);
    public short Exp(short value) => (short)Math.Round(Math.Exp(value));
    public bool Equals(short a, short b) => a == b;
    public short Power(short baseValue, short exponent) => (short)Math.Pow(baseValue, exponent);
    public short Log(short value) => (short)Math.Log(value);
    public bool GreaterThanOrEquals(short a, short b) => a >= b;
    public bool LessThanOrEquals(short a, short b) => a <= b;
    public int ToInt32(short value) => value;
    public short Round(short value) => value;
    public short MinValue => short.MinValue;
    public short MaxValue => short.MaxValue;
    public bool IsNaN(short value) => false;
    public bool IsInfinity(short value) => false;
}