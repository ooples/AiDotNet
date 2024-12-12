namespace AiDotNet.Regression;

public class SimpleRegression<T> : BaseRegression<T>
{
    public SimpleRegression(INumericOperations<T> numOps, RegressionOptions options)
        : base(numOps, options) { }

    public override void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization)
    {
        if (x.Columns != 1)
            throw new ArgumentException("Simple regression expects only one feature column.");

        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(regularization.GetRegularizationMatrix(xTx.Rows, NumOps.FromDouble(Options.RegularizationStrength)));
        var xTy = x.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([solution[1]], NumOps);
        }
        else
        {
            Coefficients = new Vector<T>([solution[0]], NumOps);
        }
    }
}