
namespace AiDotNet.Regression;

public class SimpleRegression<T> : RegressionBase<T>
{
    public SimpleRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Columns != 1)
            throw new ArgumentException("Simple regression expects only one feature column.");

        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.RegularizeMatrix(xTx));
        var xTy = x.Transpose().Multiply(y);
        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([solution[1]]);
        }
        else
        {
            Coefficients = new Vector<T>([solution[0]]);
        }
    }

    protected override ModelType GetModelType()
    {
        return ModelType.SimpleRegression;
    }
}