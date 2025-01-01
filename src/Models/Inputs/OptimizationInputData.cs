namespace AiDotNet.Models.Inputs;

public class OptimizationInputData<T>
{
    public Matrix<T> XTrain { get; set; } = Matrix<T>.Empty();
    public Vector<T> YTrain { get; set; } = Vector<T>.Empty();
    public Matrix<T> XVal { get; set; } = Matrix<T>.Empty();
    public Vector<T> YVal { get; set; } = Vector<T>.Empty();
    public Matrix<T> XTest { get; set; } = Matrix<T>.Empty();
    public Vector<T> YTest { get; set; } = Vector<T>.Empty();
}