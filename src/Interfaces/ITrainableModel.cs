namespace AiDotNet.Interfaces;

public interface ITrainableModel<T>
{
    void Train(Matrix<T> x, Vector<T> y);
}