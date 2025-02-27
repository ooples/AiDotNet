namespace AiDotNet.Interfaces;

public interface IGradientBasedOptimizer<T> : IOptimizer<T>
{
    Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient);
    Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient);
    void Reset();
}