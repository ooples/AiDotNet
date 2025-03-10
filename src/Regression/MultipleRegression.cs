﻿namespace AiDotNet.Regression;

public class MultipleRegression<T> : RegressionBase<T>
{
    public MultipleRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
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
            Coefficients = new Vector<T>([.. solution.Skip(1)]);
        }
        else
        {
            Coefficients = new Vector<T>(solution);
        }
    }

    protected override ModelType GetModelType()
    {
        return ModelType.MultipleRegression;
    }
}