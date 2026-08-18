using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class MultinomialLogisticRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new MultinomialLogisticRegression<double>();

    /// <summary>
    /// Multinomial logistic regression classifies, so it is trained on discrete class labels rather than
    /// on the unrestricted continuous target this harness generates.
    /// </summary>
    /// <remarks>
    /// The target is binned into four equal-width ordered classes, which keeps the label monotone in the
    /// original target so the structural invariants still mean what they meant. The model used to do a
    /// binning of its own, silently and invisibly to the caller, whenever it decided the target had too
    /// many distinct values; that has been removed, so the harness now states the discretization it wants.
    /// </remarks>
    protected override Vector<double> ToTarget(Vector<double> y) => ToT(BinIntoClasses(y, 4));

    /// <summary>
    /// Predictions are class labels from a softmax link, not values on the target's own scale.
    /// </summary>
    /// <remarks>
    /// Shifting or rescaling the target produces the same equal-width bins and therefore the same class
    /// labels, so translation and scaling equivariance cannot hold. Predictions are drawn from the finite
    /// label set, so residual-mean-zero and R-squared against the continuous response measure the
    /// discretization rather than the estimator.
    /// </remarks>
    protected override bool IdentityLinkInvariantsApplicable => false;
}
