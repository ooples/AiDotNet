namespace AiDotNet.Interfaces;

public interface IModelEvaluator<T>
{
    ModelEvaluationResult<T> EvaluateModel(
        Vector<T> actualTrain, Vector<T> predictedTrain,
        Vector<T> actualVal, Vector<T> predictedVal,
        Vector<T> actualTest, Vector<T> predictedTest);

    Dictionary<string, T> CalculateMetrics(Vector<T> actual, Vector<T> predicted);
}