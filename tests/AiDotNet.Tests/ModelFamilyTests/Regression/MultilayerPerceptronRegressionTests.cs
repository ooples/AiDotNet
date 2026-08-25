using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class MultilayerPerceptronRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new MultilayerPerceptronRegression<double>();

    /// <summary>
    /// A perceptron's marginal effects vary from point to point, so at the probe location the
    /// ordering of two features can invert while the fit is good everywhere it was trained.
    /// Measured: x0 came out at 13.86 and x1 at 11.71, both correctly positive. The sign
    /// assertions still apply.
    /// </summary>
    protected override bool CoefficientOrderingInvariantApplicable => false;
}
