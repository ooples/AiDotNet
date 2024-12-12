namespace AiDotNet.Regression;

public class PolynomialRegression<T> : BaseRegression<T>
{
    private readonly PolynomialRegressionOptions _polyOptions;

    public PolynomialRegression(INumericOperations<T> numOps, PolynomialRegressionOptions options)
        : base(numOps, options)
    {
        _polyOptions = options;
    }

    public override void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization)
    {
        var polyX = CreatePolynomialFeatures(x);

        if (Options.UseIntercept)
            polyX = polyX.AddConstantColumn(NumOps.One);

        var xTx = polyX.Transpose().Multiply(polyX);
        var regularizedXTx = xTx.Add(regularization.GetRegularizationMatrix(xTx.Rows, NumOps.FromDouble(Options.RegularizationStrength)));
        var xTy = polyX.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>(solution.Skip(1).ToArray(), NumOps);
        }
        else
        {
            Coefficients = new Vector<T>(solution, NumOps);
        }
    }

    private Matrix<T> CreatePolynomialFeatures(Matrix<T> x)
    {
        var rows = x.Rows;
        var cols = x.Columns * _polyOptions.Degree;
        var polyX = new Matrix<T>(rows, cols, NumOps);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int d = 0; d < _polyOptions.Degree; d++)
                {
                    polyX[i, j * _polyOptions.Degree + d] = NumOps.Power(x[i, j], NumOps.FromDouble(d + 1));
                }
            }
        }

        return polyX;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        var polyInput = CreatePolynomialFeatures(input);
        return base.Predict(polyInput);
    }
}