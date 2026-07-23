using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.ActivationFunctions;

public class ReLUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new ReLUActivation<double>();
}

public class SigmoidActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new SigmoidActivation<double>();
    protected override bool ZeroMapsToZero => false; // sigmoid(0) = 0.5
    protected override bool IsBounded => true;
}

public class TanhActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new TanhActivation<double>();
    protected override bool IsBounded => true;
}

public class IdentityActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new IdentityActivation<double>();
}

public class LeakyReLUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new LeakyReLUActivation<double>();
}

public class ELUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new ELUActivation<double>();
}

public class SELUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new SELUActivation<double>();
}

public class SwishActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new SwishActivation<double>();
    protected override bool IsMonotonic => false; // Swish dips below 0 near x ≈ -1.28
}

public class GELUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new GELUActivation<double>();
    protected override bool IsMonotonic => false; // GELU is not monotonic (dips near x ≈ -0.5)
}

public class SiLUActivationTests : ActivationFunctionTestBase
{
    protected override IActivationFunction<double> CreateActivation() => new SiLUActivation<double>();
    protected override bool IsMonotonic => false; // SiLU = x*sigmoid(x), not monotonic
}
