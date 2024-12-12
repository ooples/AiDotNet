namespace AiDotNet.Interfaces;

public interface IOutlierRemoval<T>
{
    (Matrix<T> CleanedInputs, Vector<T> CleanedOutputs) RemoveOutliers(Matrix<T> inputs, Vector<T> outputs);
}