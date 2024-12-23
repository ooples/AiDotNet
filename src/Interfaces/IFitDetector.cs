using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces;

public interface IFitDetector<T>
{
    FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData);
}