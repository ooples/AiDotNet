
namespace AiDotNet.OutlierRemoval;

public class NoOutlierRemoval<T> : IOutlierRemoval<T>
{
    public (Matrix<T> CleanedInputs, Vector<T> CleanedOutputs) RemoveOutliers(Matrix<T> inputs, Vector<T> outputs)
    {
        return (inputs, outputs);
    }
}