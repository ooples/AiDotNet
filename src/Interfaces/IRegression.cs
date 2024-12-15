namespace AiDotNet.Interfaces;

public interface IRegression<T>
{
    void Fit(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> input);

    Vector<T> Coefficients { get; }
    T Intercept { get; }
    bool HasIntercept { get; }
}