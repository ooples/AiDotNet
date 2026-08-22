using AiDotNet.Interfaces;
using AiDotNet.Classification.SVM;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class NuSupportVectorClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        // The shared classification contract deliberately includes a 90/10 class-prior case.
        // For nu-SVC, feasibility requires nu <= 2 * minorityFraction, so use one explicit
        // paper-valid value for the entire model-family suite instead of silently changing a
        // caller's requested nu during training.
        => new NuSupportVectorClassifier<double>(nu: 0.1);
}
