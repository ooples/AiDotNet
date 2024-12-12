namespace AiDotNet.NumericOperations;

public class UIntOperations : INumericOperations<uint>
{
    public uint Add(uint a, uint b) => a + b;
    public uint Subtract(uint a, uint b) => a - b;
    public uint Multiply(uint a, uint b) => a * b;
    public uint Divide(uint a, uint b) => a / b;
    public uint Negate(uint a) => throw new NotSupportedException("Cannot negate unsigned integer");
    public uint Zero => 0U;
    public uint One => 1U;
    public uint Sqrt(uint value) => (uint)Math.Sqrt(value);
    public uint FromDouble(double value) => (uint)value;
    public bool GreaterThan(uint a, uint b) => a > b;
    public bool LessThan(uint a, uint b) => a < b;
    public uint Abs(uint value) => value;
    public uint Square(uint value) => Multiply(value, value);
    public uint Exp(uint value) => (uint)Math.Min(uint.MaxValue, Math.Round(Math.Exp(value)));
    public bool Equals(uint a, uint b) => a == b;
    public uint Power(uint baseValue, uint exponent) => (uint)Math.Pow(baseValue, exponent);
    public uint Log(uint value) => (uint)Math.Log(value);
    public bool GreaterThanOrEquals(uint a, uint b) => a >= b;
    public bool LessThanOrEquals(uint a, uint b) => a <= b;
    public int ToInt32(uint value) => (int)value;
    public uint Round(uint value) => value;
}