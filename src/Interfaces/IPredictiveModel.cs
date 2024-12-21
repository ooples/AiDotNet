namespace AiDotNet.Interfaces;

public interface IPredictiveModel<T>
{
    Vector<T> Predict(Matrix<T> input);
}