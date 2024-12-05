namespace AiDotNet.NumericOperations;

public class FloatOperations : INumericOperations<float>
{
    public float Add(float a, float b) => a + b;
    public float Subtract(float a, float b) => a - b;
    public float Multiply(float a, float b) => a * b;
    public float Divide(float a, float b) => a / b;
    public float Negate(float a) => -a;
    public float Zero => 0f;
    public float One => 1f;
    public float Sqrt(float value) => (float)Math.Sqrt(value);
    public float FromDouble(double value) => (float)value;
    public bool GreaterThan(float a, float b) => a > b;
    public bool LessThan(float a, float b) => a < b;
    public float Abs(float value) => Math.Abs(value);
}