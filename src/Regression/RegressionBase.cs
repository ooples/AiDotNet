public abstract class BaseRegression<T> : IRegression<T>
{
    protected INumericOperations<T> NumOps;
    protected RegressionOptions Options;

    public Vector<T> Coefficients { get; protected set; }
    public T Intercept { get; protected set; }
    public bool HasIntercept => Options.UseIntercept;

    protected BaseRegression(INumericOperations<T> numOps, RegressionOptions options)
    {
        NumOps = numOps;
        Options = options;
        Coefficients = new Vector<T>(0, NumOps);
        Intercept = NumOps.Zero;
    }

    public abstract void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization);

    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = input.Multiply(Coefficients);

        if (Options.UseIntercept)
        {
            predictions = predictions.Add(Intercept);
        }

        return predictions;
    }

    protected Vector<T> SolveSystem(Matrix<T> a, Vector<T> b)
    {
        return Options.DecompositionMethod switch
        {
            MatrixDecomposition.Svd => a.SvdSolve(b),
            MatrixDecomposition.Qr => a.QrSolve(b),
            MatrixDecomposition.Cholesky => a.CholeskySolve(b),
            MatrixDecomposition.Lu => a.LuSolve(b),
            _ => a.Solve(b),// Default to normal equation
        };
    }
}