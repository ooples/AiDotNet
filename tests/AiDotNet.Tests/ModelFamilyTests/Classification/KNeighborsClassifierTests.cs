using AiDotNet.Interfaces;
using AiDotNet.Classification.Neighbors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class KNeighborsClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new KNeighborsClassifier<double>();

    // KNN stores training data, not flat parameter vectors
    protected override bool HasFlatParameters => false;
}
