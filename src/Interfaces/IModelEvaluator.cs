namespace AiDotNet.Interfaces;

public interface IModelEvaluator<T>
{
    ModelEvaluationData<T> EvaluateModel(ModelEvaluationInput<T> input);
}