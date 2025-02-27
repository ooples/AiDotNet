namespace AiDotNet.Interfaces;

public interface ITimeSeriesModel<T> : IModelSerializer
{
    void Train(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> input);
    Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest);
}