using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML;

public sealed class AutoMLCompatibilityBaselineTests
{
    [Fact]
    public void ExistingSearchStrategyNumericValuesRemainStable()
    {
        Assert.Equal(0, (int)AutoMLSearchStrategy.RandomSearch);
        Assert.Equal(1, (int)AutoMLSearchStrategy.BayesianOptimization);
        Assert.Equal(2, (int)AutoMLSearchStrategy.Evolutionary);
        Assert.Equal(3, (int)AutoMLSearchStrategy.MultiFidelity);
        Assert.Equal(4, (int)AutoMLSearchStrategy.NeuralArchitectureSearch);
        Assert.Equal(5, (int)AutoMLSearchStrategy.DARTS);
        Assert.Equal(6, (int)AutoMLSearchStrategy.GDAS);
        Assert.Equal(7, (int)AutoMLSearchStrategy.OnceForAll);
        Assert.Equal(8, (int)AutoMLSearchStrategy.MapElites);
    }

    [Fact]
    public void AutoMLOptionsDefaultSearchStrategyRemainsRandomSearch()
    {
        var options = new AutoMLOptions<double, object, object>();

        Assert.Equal(AutoMLSearchStrategy.RandomSearch, options.SearchStrategy);
    }

    [Fact]
    public void FacadeAcceptsAppendedMapElitesStrategy()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var options = new AutoMLOptions<double, Matrix<double>, Vector<double>>
        {
            SearchStrategy = AutoMLSearchStrategy.MapElites,
            MapElites = new MapElitesAutoMLOptions { Seed = 42 }
        };

        Exception? exception = Record.Exception(() => builder.ConfigureAutoML(options));

        Assert.Null(exception);
    }
}
