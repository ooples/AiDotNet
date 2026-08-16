using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class BetaRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BetaRegression<double>();
    /// <summary>
    /// This is a generalized linear model with a non-identity link, so it is multiplicative rather
    /// than additive and restricts its response domain. The base class's equivariance and
    /// generic-quality invariants assume an additive identity-link estimator over unrestricted
    /// continuous data, which this model is not. Its correctness is established on
    /// domain-appropriate data by GLMFamilyRegressionIntegrationTests.
    /// </summary>
    protected override bool IdentityLinkInvariantsApplicable => false;
}
