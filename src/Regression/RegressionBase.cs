public abstract class RegressionBase<T> : IRegression<T>
{
    protected INumericOperations<T> NumOps { get; private set; }
    protected RegressionOptions<T> Options { get; private set; }
    protected IRegularization<T> Regularization { get; private set; }

    public Vector<T> Coefficients { get; protected set; }
    public T Intercept { get; protected set; }

    public bool HasIntercept => Options.UseIntercept;

    protected RegressionBase(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        Regularization = regularization ?? new NoRegularization<T>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new RegressionOptions<T>();
        Coefficients = new Vector<T>(0, NumOps);
        Intercept = NumOps.Zero;
    }

    public abstract void Fit(Matrix<T> x, Vector<T> y);

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
        var decomposition = Options.DecompositionMethod;

        if (decomposition != null)
        {
            return decomposition.Solve(b);
        }
        else
        {
            // Use normal equation if specifically selected or as a fallback
            return SolveNormalEquation(a, b);
        }
    }

    private Vector<T> SolveNormalEquation(Matrix<T> a, Vector<T> b)
    {
        var aTa = a.Transpose().Multiply(a);
        var aTb = a.Transpose().Multiply(b);

        // Use LU decomposition for solving the normal equation
        var normalDecomposition = new NormalDecomposition<T>(aTa);
        return normalDecomposition.Solve(aTb);
    }
}