namespace AiDotNet.Regression;

public class MultipleRegression<T> : BaseRegression<T>
{
    public MultipleRegression(INumericOperations<T> numOps, RegressionOptions options)
        : base(numOps, options) { }

    public override void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization)
    {
        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(regularization.GetRegularizationMatrix(xTx.Rows, NumOps.FromDouble(Options.RegularizationStrength)));
        var xTy = x.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([.. solution.Skip(1)], NumOps);
        }
        else
        {
            Coefficients = new Vector<T>(solution, NumOps);
        }
    }
}