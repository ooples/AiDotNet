namespace AiDotNet.Regression;

public class MultivariateRegression<T> : RegressionBase<T>
{
    public MultivariateRegression(RegressionOptions<T>? options = null)
        : base(options)
    {
    }

    public override void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization)
    {
        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(regularization.RegularizeMatrix(xTx));
        var xTy = x.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = solution.Slice(1, solution.Length - 1);
        }
        else
        {
            Coefficients = solution;
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Options.UseIntercept)
            input = input.AddConstantColumn(NumOps.One);

        return input.Multiply(Coefficients);
    }
}