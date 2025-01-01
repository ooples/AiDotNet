namespace AiDotNet.Models.Inputs;

public class ModelEvaluationInput<T>
{
    public ISymbolicModel<T>? Model { get; set; }
    public OptimizationInputData<T> InputData { get; set; } = new();
    public NormalizationInfo<T> NormInfo { get; set; } = new();
}