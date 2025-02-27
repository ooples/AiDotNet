namespace AiDotNet.Interfaces;

public interface ILinearModel<T> : IFullModel<T>
{
    Vector<T> Coefficients { get; }
    T Intercept { get; }
    bool HasIntercept { get; }
}