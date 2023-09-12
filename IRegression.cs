namespace AiDotNet;

public interface IRegression
{
    double Predict(double[] x);

    void Fit(double[][] x, double[] y);

    double Score(double[][] x, double[] y);

    double[] Normalize(double[] x);
}