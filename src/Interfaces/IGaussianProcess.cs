namespace AiDotNet.Interfaces;

public interface IGaussianProcess<T>
{
    void Fit(Matrix<T> X, Vector<T> y);
    (T mean, T variance) Predict(Vector<T> x);
    void UpdateKernel(IKernelFunction<T> kernel);
}