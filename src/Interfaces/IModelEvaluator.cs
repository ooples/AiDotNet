namespace AiDotNet.Interfaces;

public interface IModelEvaluator
{
    ModelEvaluationResult EvaluateModel(
        Vector<double> actualTrain, Vector<double> predictedTrain,
        Vector<double> actualVal, Vector<double> predictedVal,
        Vector<double> actualTest, Vector<double> predictedTest);

    Dictionary<string, double> CalculateMetrics(Vector<double> actual, Vector<double> predicted);
}