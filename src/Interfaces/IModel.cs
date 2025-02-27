namespace AiDotNet.Interfaces;

public interface IModel<T>
{
    void Train(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> input);
    ModelMetadata<T> GetModelMetadata();
}