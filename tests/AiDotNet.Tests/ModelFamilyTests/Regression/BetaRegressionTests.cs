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
    /// Beta regression models a proportion, so it is trained on a target inside (0,1) rather than on the
    /// unrestricted continuous target this harness generates.
    /// </summary>
    /// <remarks>
    /// The target is standardized and passed through the logistic function, which is strictly increasing
    /// and never reaches the boundary, so the ordering is preserved and the Beta density stays defined.
    /// This used to be unnecessary only because the model silently fitted ordinary least squares whenever
    /// the target left [0,1]; it now rejects such a target, as betareg and statsmodels do.
    /// </remarks>
    protected override Vector<double> ToTarget(Vector<double> y) => ToT(ToProportion(y));

    /// <summary>
    /// This is a generalized linear model with a non-identity link, so it is multiplicative rather
    /// than additive and restricts its response domain. The base class's equivariance and
    /// generic-quality invariants assume an additive identity-link estimator over unrestricted
    /// continuous data, which this model is not. Its correctness is established on
    /// domain-appropriate data by GLMFamilyRegressionIntegrationTests.
    /// </summary>
    protected override bool IdentityLinkInvariantsApplicable => false;
}
