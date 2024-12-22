namespace AiDotNet.Interfaces;

public interface IRegression<T> : IFullModel<T>
{
    Vector<T> Coefficients { get; }
    T Intercept { get; }
    bool HasIntercept { get; }
}