namespace AiDotNet.Regularization;

public abstract class RegularizationBase<T> : IRegularization<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly RegularizationOptions Options;

    public RegularizationBase(RegularizationOptions? regularizationOptions = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = regularizationOptions ?? new();
    }

    public abstract Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix);
    public abstract Vector<T> RegularizeCoefficients(Vector<T> coefficients);
    public abstract Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients);

    public RegularizationOptions GetOptions()
    {
        return Options;
    }
}