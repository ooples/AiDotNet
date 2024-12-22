using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

public class MultivariateRegression<T> : RegressionBase<T>
{
    public MultivariateRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.RegularizeMatrix(xTx));
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