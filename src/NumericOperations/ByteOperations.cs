namespace AiDotNet.NumericOperations;

public class ByteOperations : INumericOperations<byte>
{
    public byte Add(byte a, byte b) => (byte)(a + b);
    public byte Subtract(byte a, byte b) => (byte)(a - b);
    public byte Multiply(byte a, byte b) => (byte)(a * b);
    public byte Divide(byte a, byte b) => (byte)(a / b);
    public byte Negate(byte a) => (byte)-a;
    public byte Zero => 0;
    public byte One => 1;
    public byte Sqrt(byte value) => (byte)Math.Sqrt(value);
    public byte FromDouble(double value) => (byte)value;
    public bool GreaterThan(byte a, byte b) => a > b;
    public bool LessThan(byte a, byte b) => a < b;
    public byte Abs(byte value) => value;
    public byte Square(byte value) => Multiply(value, value);
    public byte Exp(byte value) => (byte)Math.Min(255, Math.Round(Math.Exp(value)));
    public bool Equals(byte a, byte b) => a == b;
    public byte Power(byte baseValue, byte exponent) => (byte)Math.Pow(baseValue, exponent);
    public byte Log(byte value) => (byte)Math.Log(value);
    public bool GreaterThanOrEquals(byte a, byte b) => a >= b;
    public bool LessThanOrEquals(byte a, byte b) => a <= b;
    public int ToInt32(byte value) => value;
    public byte Round(byte value) => value;
}