using AiDotNet.Configuration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>JIT compilation is on by default for a builder and is switched off only by an explicit opt-out.</summary>
public class JitCompilationDefaultTests
{
    [Fact]
    public void NewBuilder_HasJitCompilationEnabled()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var jit = ((IConfiguredView<double, Matrix<double>, Vector<double>>)builder).ConfiguredJitCompilation;

        Assert.NotNull(jit);
        Assert.True(jit?.Enabled ?? false);
    }

    [Fact]
    public void ConfigureJitCompilation_Disabled_TurnsItOff_AndNullRestoresTheDefault()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var view = (IConfiguredView<double, Matrix<double>, Vector<double>>)builder;

        builder.ConfigureJitCompilation(JitCompilationConfig.Disabled);
        Assert.False(view.ConfiguredJitCompilation?.Enabled ?? true);

        builder.ConfigureJitCompilation(null);
        Assert.True(view.ConfiguredJitCompilation?.Enabled ?? false);
    }
}
