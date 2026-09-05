using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Pins the terminal build calls the library's own documentation uses.
/// </summary>
/// <remarks>
/// <para>
/// 67 doc-comment examples across <c>src/</c> — 16 in <c>IAiModelBuilder</c> alone — end a fluent chain with
/// <c>.Build(X, y)</c>. No such method existed: the only terminal call was <c>BuildAsync()</c>, and data had to be
/// supplied separately through <c>ConfigureDataLoader</c>. Every reader following the documentation failed to
/// compile on their first attempt.
/// </para>
/// <para>
/// The examples were not caught by the documentation gate because of where they live rather than what they say:
/// <c>WikiGenerator.NormalizeBlock</c> rewrites every <c>```csharp</c> fence found in doc-comment prose to
/// <c>```cs</c>, and <c>DocSnippetVerify</c> keys on <c>```csharp</c>. Verified <c>&lt;example&gt;</c> tags are
/// compile-checked; anything in <c>&lt;remarks&gt;</c> is exempt by construction, which is exactly where these
/// examples are.
/// </para>
/// <para>
/// These tests exercise the documented shape literally, including the part that first broke: the call is made
/// mid-chain, where the static type is <see cref="IAiModelBuilder{T, TInput, TOutput}"/> rather than the concrete
/// builder. A terminal method declared only on the concrete type is unreachable there.
/// </para>
/// </remarks>
public sealed class FacadeDocumentedBuildTests
{
    private static (Matrix<double> X, Vector<double> y) LinearData(int rows = 60)
    {
        // y = 2*x0 + 3*x1, exactly representable so a fitted model can be checked against the truth.
        var x = new Matrix<double>(rows, 2);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            double a = (i % 10) * 0.5, b = ((i * 7) % 10) * 0.25;
            x[i, 0] = a; x[i, 1] = b;
            y[i] = (2.0 * a) + (3.0 * b);
        }

        return (x, y);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void The_documented_synchronous_Build_with_data_works_mid_chain()
    {
        var (x, y) = LinearData();

        // Written exactly as the documentation shows it: Configure*, then .Build(X, y).
        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .Build(x, y);

        Assert.NotNull(result);

        var probe = new Matrix<double>(1, 2);
        probe[0, 0] = 1.0; probe[0, 1] = 1.0;
        Assert.Equal(5.0, result.Predict(probe)[0], 6);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task The_asynchronous_Build_with_data_works_mid_chain()
    {
        var (x, y) = LinearData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .BuildAsync(x, y);

        Assert.NotNull(result);

        var probe = new Matrix<double>(1, 2);
        probe[0, 0] = 1.0; probe[0, 1] = 1.0;
        Assert.Equal(5.0, result.Predict(probe)[0], 6);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Supplying_data_twice_is_refused_rather_than_silently_preferring_one()
    {
        // Which dataset trained the model is not something a caller should have to guess.
        var (x, y) = LinearData();

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new MultipleRegression<double>());

        var failure = Assert.Throws<InvalidOperationException>(() => builder.Build(x, y));
        Assert.Contains("data loader", failure.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Null_data_is_refused_by_name()
    {
        var (x, y) = LinearData();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>());

        Assert.Throws<ArgumentNullException>(() => builder.Build(null!, y));
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Both_terminal_build_overloads_are_reachable_through_the_interface()
    {
        // The regression this file exists for: the overloads were first added to the concrete builder only, so a
        // chain that had passed through any Configure* call could not see them.
        var iface = typeof(IAiModelBuilder<,,>);
        Assert.Contains(iface.GetMethods(), m => m.Name == "Build" && m.GetParameters().Length == 2);
        Assert.Contains(iface.GetMethods(), m => m.Name == "BuildAsync" && m.GetParameters().Length == 3);
    }
}
