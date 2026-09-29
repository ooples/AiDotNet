using Xunit;

namespace Fixture.Tests;

// Inherited tests run under each concrete subclass.
public abstract class ModelContractTests
{
    protected abstract ModelBase Create();

    [Fact]
    public void PredictIsStable() => System.GC.KeepAlive(Create().Predict());
}

public sealed class AlphaContractTests : ModelContractTests
{
    protected override ModelBase Create() => new Alpha();
}
