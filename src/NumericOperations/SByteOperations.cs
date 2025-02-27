namespace AiDotNet.NumericOperations;

public class SByteOperations : INumericOperations<sbyte>
{
    public sbyte Add(sbyte a, sbyte b) => (sbyte)(a + b);
    public sbyte Subtract(sbyte a, sbyte b) => (sbyte)(a - b);
    public sbyte Multiply(sbyte a, sbyte b) => (sbyte)(a * b);
    public sbyte Divide(sbyte a, sbyte b) => (sbyte)(a / b);
    public sbyte Negate(sbyte a) => (sbyte)-a;
    public sbyte Zero => 0;
    public sbyte One => 1;
    public sbyte Sqrt(sbyte value) => (sbyte)Math.Sqrt(value);
    public sbyte FromDouble(double value) => (sbyte)value;
    public bool GreaterThan(sbyte a, sbyte b) => a > b;
    public bool LessThan(sbyte a, sbyte b) => a < b;
    public sbyte Abs(sbyte value) => Math.Abs(value);
    public sbyte Square(sbyte value) => Multiply(value, value);
    public sbyte Exp(sbyte value) => (sbyte)Math.Min(sbyte.MaxValue, Math.Round(Math.Exp(value)));
    public bool Equals(sbyte a, sbyte b) => a == b;
    public sbyte Power(sbyte baseValue, sbyte exponent) => (sbyte)Math.Pow(baseValue, exponent);
    public sbyte Log(sbyte value) => (sbyte)Math.Log(value);
    public bool GreaterThanOrEquals(sbyte a, sbyte b) => a >= b;
    public bool LessThanOrEquals(sbyte a, sbyte b) => a <= b;
    public int ToInt32(sbyte value) => value;
    public sbyte Round(sbyte value) => value;
    public sbyte MinValue => sbyte.MinValue;
    public sbyte MaxValue => sbyte.MaxValue;
    public bool IsNaN(sbyte value) => false;
    public bool IsInfinity(sbyte value) => false;
    public sbyte SignOrZero(sbyte value) => value == 0 ? (sbyte)0 : value > 0 ? (sbyte)1 : (sbyte)-1;
}