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
}