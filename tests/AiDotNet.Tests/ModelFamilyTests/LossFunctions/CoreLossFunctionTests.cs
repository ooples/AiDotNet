using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.LossFunctions;

public class MeanSquaredErrorLossTests : LossFunctionTestBase
{
    protected override ILossFunction<double> CreateLoss() => new MeanSquaredErrorLoss<double>();
}

public class MeanAbsoluteErrorLossTests : LossFunctionTestBase
{
    protected override ILossFunction<double> CreateLoss() => new MeanAbsoluteErrorLoss<double>();
}

public class HuberLossTests : LossFunctionTestBase
{
    protected override ILossFunction<double> CreateLoss() => new HuberLoss<double>();
}

// CrossEntropyLoss needs special handling — it's designed for classification
// (softmax outputs) and has different mathematical properties than regression losses.
// Skipping from the generic test base for now; it has its own dedicated tests.

public class BinaryCrossEntropyLossTests : LossFunctionTestBase
{
    protected override ILossFunction<double> CreateLoss() => new BinaryCrossEntropyLoss<double>();
    protected override bool ZeroLossForIdentical => false; // BCE(p, p) = -sum(p*log(p) + (1-p)*log(1-p))
}
