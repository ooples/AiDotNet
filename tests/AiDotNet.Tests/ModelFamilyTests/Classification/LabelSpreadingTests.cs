using AiDotNet.Interfaces;
using AiDotNet.Classification.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class LabelSpreadingTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LabelSpreading<double>();

    // Semi-supervised graph-based model — stores label distributions, not flat parameters
    protected override bool HasFlatParameters => false;
}
