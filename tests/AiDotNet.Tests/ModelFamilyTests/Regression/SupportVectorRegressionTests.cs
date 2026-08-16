using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class SupportVectorRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SupportVectorRegression<double>();

    /// <summary>
    /// SVR defaults to an RBF kernel (as scikit-learn's SVR does), whose influence decays with
    /// distance from the training points. Probing from -5 to 15 leaves the training support, where
    /// the prediction correctly relaxes toward the bias instead of continuing to rise, so the
    /// linear-extrapolation invariant does not apply.
    /// </summary>
    protected override bool MonotonicResponseInvariantApplicable => false;
}
