using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TransferLearning.Algorithms;
using AiDotNet.TransferLearning.FeatureMapping;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

/// <summary>
/// Manual test factory for MappedRandomForestModel, which requires constructor arguments
/// (base model, feature mapper, target features) that the auto-generated scaffold cannot provide.
/// </summary>
public class MappedRandomForestModelTests : RegressionModelTestBase
{
    protected override int Features => 5;

    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
    {
        // Create a base random forest model and train it on source domain
        var baseModel = new RandomForestRegression<double>();
        var mapper = new LinearFeatureMapper<double>();
        return new MappedRandomForestModel<double>(baseModel, mapper, Features);
    }
}
