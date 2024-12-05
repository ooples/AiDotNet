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
}