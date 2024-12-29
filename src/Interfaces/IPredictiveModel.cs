namespace AiDotNet.Interfaces;

public interface IPredictiveModel<T> : IModelSerializer
{
    Vector<T> Predict(Matrix<T> input);
    ModelMetadata<T> GetModelMetadata();
}