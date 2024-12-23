namespace AiDotNet.Interfaces;

public interface IAsyncTreeBasedModel<T> : ITreeBasedModel<T>
{
    Task TrainAsync(Matrix<T> x, Vector<T> y);
    Task<Vector<T>> PredictAsync(Matrix<T> input);
}