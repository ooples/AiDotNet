global using AiDotNet.Extensions;

namespace AiDotNet.Regression;

public class WeightedRegression<T> : RegressionBase<T>
{
    private readonly Vector<T> _weights;
    private readonly int _order;

    public WeightedRegression(WeightedRegressionOptions<T>? options = null)
        : base(options)
    {
        _weights = options?.Weights ?? throw new ArgumentNullException(nameof(options), "Weights must be provided for weighted regression.");
        _order = options.Order;
    }

    public override void Fit(Matrix<T> x, Vector<T> y, IRegularization<T> regularization)
    {
        var expandedX = ExpandFeatures(x);

        if (Options.UseIntercept)
            expandedX = expandedX.AddConstantColumn(NumOps.One);

        var weightMatrix = Matrix<T>.CreateDiagonal(_weights, NumOps);
        var xTWx = expandedX.Transpose().Multiply(weightMatrix).Multiply(expandedX);
        var regularizedXTWx = xTWx.Add(regularization.RegularizeMatrix(xTWx));
        var xTWy = expandedX.Transpose().Multiply(weightMatrix).Multiply(y);

        var solution = SolveSystem(regularizedXTWx, xTWy);

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
        var expandedInput = ExpandFeatures(input);
        return base.Predict(expandedInput);
    }

    private Matrix<T> ExpandFeatures(Matrix<T> x)
    {
        var expandedX = new Matrix<T>(x.Rows, x.Columns * _order, NumOps);

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int k = 0; k < _order; k++)
                {
                    expandedX[i, j * _order + k] = NumOps.Power(x[i, j], NumOps.FromDouble(k + 1));
                }
            }
        }

        return expandedX;
    }
}