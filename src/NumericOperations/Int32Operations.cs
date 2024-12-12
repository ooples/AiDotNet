namespace AiDotNet.NumericOperations;

public class Int32Operations : INumericOperations<int>
{
    public int Add(int a, int b) => a + b;
    public int Subtract(int a, int b) => a - b;
    public int Multiply(int a, int b) => a * b;
    public int Divide(int a, int b) => a / b;
    public int Negate(int a) => -a;
    public int Zero => 0;
    public int One => 1;
    public int Sqrt(int value) => (int)Math.Sqrt(value);
    public int FromDouble(double value) => (int)value;
    public bool GreaterThan(int a, int b) => a > b;
    public bool LessThan(int a, int b) => a < b;
    public int Abs(int value) => Math.Abs(value);
    public int Square(int value) => Multiply(value, value);
    public int Exp(int value) => (int)Math.Round(Math.Exp(value));
    public bool Equals(int a, int b) => a == b;
    public int Power(int baseValue, int exponent) => (int)Math.Pow(baseValue, exponent);
    public int Log(int value) => (int)Math.Log(value);
    public bool GreaterThanOrEquals(int a, int b) => a >= b;
    public bool LessThanOrEquals(int a, int b) => a <= b;
    public int ToInt32(int value) => value;
    public int Round(int value) => value;
    public int MinValue => int.MinValue;
    public int MaxValue => int.MaxValue;
    public bool IsNaN(int value) => false;
    public bool IsInfinity(int value) => false;
}