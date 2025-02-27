namespace AiDotNet.NumericOperations;

public class UInt32Operations : INumericOperations<uint>
{
    public uint Add(uint a, uint b) => a + b;
    public uint Subtract(uint a, uint b) => a - b;
    public uint Multiply(uint a, uint b) => a * b;
    public uint Divide(uint a, uint b) => a / b;
    public uint Negate(uint a) => uint.MaxValue - a + 1;
    public uint Zero => 0;
    public uint One => 1;
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
    public uint MinValue => uint.MinValue;
    public uint MaxValue => uint.MaxValue;
    public bool IsNaN(uint value) => false;
    public bool IsInfinity(uint value) => false;
    public uint SignOrZero(uint value) => value == 0 ? 0u : 1u;
}