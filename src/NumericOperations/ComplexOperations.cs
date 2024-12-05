namespace AiDotNet.NumericOperations;

public class ComplexOperations : INumericOperations<Complex>
{
    public Complex Add(Complex a, Complex b) => a + b;
    public Complex Subtract(Complex a, Complex b) => a - b;
    public Complex Multiply(Complex a, Complex b) => a * b;
    public Complex Divide(Complex a, Complex b) => a / b;
    public Complex Negate(Complex a) => -a;
    public Complex Zero => new(0, 0);
    public Complex One => new(1, 0);
    public Complex Sqrt(Complex value)
    {
        double r = Math.Sqrt(value.Magnitude);
        double theta = value.Phase / 2;
        return new Complex(r * Math.Cos(theta), r * Math.Sin(theta));
    }
    public Complex FromDouble(double value) => new(value, 0);
    public bool GreaterThan(Complex a, Complex b) => a.Magnitude > b.Magnitude;
    public bool LessThan(Complex a, Complex b) => a.Magnitude < b.Magnitude;
    public Complex Abs(Complex value) => new(value.Magnitude, 0);
}