namespace AiDotNet.Interfaces;

public interface IRegression
{
    void Fit(Matrix<double> x, Vector<double> y, IRegularization regularization);

    Vector<double> Coefficients { get; }
    double Intercept { get; }
    bool HasIntercept { get; }
}