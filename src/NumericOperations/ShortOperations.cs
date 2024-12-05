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
}