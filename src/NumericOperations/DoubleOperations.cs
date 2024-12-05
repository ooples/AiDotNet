namespace AiDotNet.NumericOperations;

public class DoubleOperations : INumericOperations<double>
{
    public double Add(double a, double b) => a + b;
    public double Subtract(double a, double b) => a - b;
    public double Multiply(double a, double b) => a * b;
    public double Divide(double a, double b) => a / b;
    public double Negate(double a) => -a;
    public double Zero => 0;
    public double One => 1;
    public double Sqrt(double value) => Math.Sqrt(value);
    public double FromDouble(double value) => value;
    public bool GreaterThan(double a, double b) => a > b;
    public bool LessThan(double a, double b) => a < b;
    public double Abs(double value) => Math.Abs(value);
}