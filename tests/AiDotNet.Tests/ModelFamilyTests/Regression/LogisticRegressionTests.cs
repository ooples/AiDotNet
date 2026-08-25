using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class LogisticRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LogisticRegression<double>();

    /// <summary>
    /// Logistic regression classifies, so it is trained on two class labels rather than on the
    /// unrestricted continuous target this harness generates.
    /// </summary>
    /// <remarks>
    /// The split is at the median, which keeps the classes balanced and keeps the label monotone in the
    /// original target, so the structural invariants still mean what they meant. This used to be
    /// unnecessary only because the model silently fitted ordinary least squares whenever the target was
    /// not binary; it now rejects such a target, which is what scikit-learn and statsmodels do.
    /// </remarks>
    protected override Vector<double> ToTarget(Vector<double> y) => ToT(ThresholdAtMedian(y));

    /// <summary>
    /// Predictions are probabilities in (0,1) from a logit link, not values on the target's own scale.
    /// </summary>
    /// <remarks>
    /// The equivariance and generic-quality invariants assume an additive identity-link estimator whose
    /// predictions live on the same scale as the response. A shifted or rescaled target produces the same
    /// median split and therefore the same probabilities, so translation and scaling equivariance cannot
    /// hold; and R-squared of a probability against a 0/1 label is not the quantity this model optimizes.
    /// </remarks>
    protected override bool IdentityLinkInvariantsApplicable => false;
}
