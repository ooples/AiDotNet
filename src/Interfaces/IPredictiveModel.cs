namespace AiDotNet.Interfaces;

public interface IPredictiveModel<T> : IModelSerializer<T>
{
    Vector<T> Predict(Matrix<T> input);
    ModelMetadata<T> GetModelMetadata();
}