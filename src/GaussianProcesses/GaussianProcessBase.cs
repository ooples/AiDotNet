using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Abstract base class for Gaussian Process models that provides IFullModel compliance.
/// Maps the GP Fit/Predict(single) API to the standard Train/Predict(batch) API.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class GaussianProcessBase<T> : ModelBase<T, Matrix<T>, Vector<T>>, IGaussianProcess<T>
{
    // NumOps and Engine inherited from ModelBase

    /// <inheritdoc/>
    public abstract void Fit(Matrix<T> X, Vector<T> y);

    /// <inheritdoc/>
    public abstract (T mean, T variance) Predict(Vector<T> x);

    /// <inheritdoc/>
    public abstract void UpdateKernel(IKernelFunction<T> kernel);

    /// <summary>
    /// Trains the GP. Delegates to Fit.
    /// </summary>
    public override void Train(Matrix<T> input, Vector<T> expectedOutput)
    {
        Fit(input, expectedOutput);
    }

    /// <summary>
    /// Predicts mean values for each row of the input matrix.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            var row = input.GetRow(i);
            var (mean, _) = Predict(row);
            predictions[i] = mean;
        }
        return predictions;
    }

    /// <inheritdoc/>
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        return (GaussianProcessBase<T>)MemberwiseClone();
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = DeepCopy();
        clone.SetParameters(parameters);
        return clone;
    }
}
