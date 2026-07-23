using AiDotNet.Interfaces;
using AiDotNet.Classification.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class LabelPropagationTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LabelPropagation<double>();

    // Semi-supervised graph-based model — stores label distributions, not flat parameters
    protected override bool HasFlatParameters => false;
}
