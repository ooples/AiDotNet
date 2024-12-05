namespace AiDotNet.Interfaces;

public interface INumericOperations<T>
{
    T Add(T a, T b);
    T Subtract(T a, T b);
    T Multiply(T a, T b);
    T Divide(T a, T b);
    T Negate(T a);
    T Zero { get; }
    T One { get; }
    T Sqrt(T value);
    T FromDouble(double value);
    bool GreaterThan(T a, T b);
    bool LessThan(T a, T b);
    T Abs(T value);
}