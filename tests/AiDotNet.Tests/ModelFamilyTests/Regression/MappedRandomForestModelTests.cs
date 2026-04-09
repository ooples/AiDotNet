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
    // Use default 3 features to match standard regression test expectations.
    // The wrapper adds no feature-space transformation for same-domain operation.
    // protected override int Features => 5;

    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
    {
        // Create a base random forest model with deterministic seed for reproducible
        // results through the builder pipeline (builder splits data, reducing training
        // size from 100 → 70, so a stable RF is critical for positive R²).
        var options = new AiDotNet.Models.Options.RandomForestRegressionOptions
        {
            Seed = 42,
            NumberOfTrees = 100
        };
        var baseModel = new RandomForestRegression<double>(options);
        var mapper = new LinearFeatureMapper<double>();
        return new MappedRandomForestModel<double>(baseModel, mapper, Features);
    }
}
