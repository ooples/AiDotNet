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
    public double Square(double value) => Multiply(value, value);
    public double Exp(double value) => Math.Exp(value);
    public bool Equals(double a, double b) => a == b;
    public double Power(double baseValue, double exponent) => Math.Pow(baseValue, exponent);
    public double Log(double value) => Math.Log(value);
    public bool GreaterThanOrEquals(double a, double b) => a >= b;
    public bool LessThanOrEquals(double a, double b) => a <= b;
    public int ToInt32(double value) => (int)Math.Round(value);
    public double Round(double value) => Math.Round(value);
}