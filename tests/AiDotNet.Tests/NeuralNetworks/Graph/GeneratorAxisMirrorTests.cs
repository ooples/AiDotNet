using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Generators;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Holds <c>ShapeContractGenerator</c>'s internal <c>Axis</c> mirror in step with the real
/// <see cref="TensorAxis"/>.
/// </summary>
/// <remarks>
/// <para>
/// A source generator targets netstandard2.0 and runs AGAINST the analysed assembly rather than
/// referencing it, so it cannot use <see cref="TensorAxis"/> and mirrors it instead. Roslyn hands back
/// a boxed <c>int</c> for an enum argument, and the generator turns that int into the identifier it
/// writes into generated source - so if the two enums disagree, it emits <c>TensorAxis.Height</c> where
/// the model said <c>Width</c>. That compiles. It is wrong, and nothing else would notice.
/// </para>
/// <para>
/// The mirror was first kept in step by a COMMENT saying the values must match - the same mechanism as
/// AnomalyDetectorBase's instruction telling subclasses to override GetParameters, which all 63 ignored.
/// A written rule with nothing enforcing it has already drifted; you just do not know yet.
/// </para>
/// <para>
/// Compared by REFLECTION over both enums. An earlier version of this test regex-parsed the generator's
/// source text, which was fragile in exactly the way the thing it guards is: it would have passed
/// silently if the declaration were reformatted, and failed noisily on a harmless edit. The generator
/// source is linked into this project (see AiDotNetTests.csproj, the same pattern already used for
/// ModelMetadataValidationGenerator), so the type is simply visible and the comparison is type-safe.
/// </para>
/// </remarks>
public class GeneratorAxisMirrorTests
{
    private readonly ITestOutputHelper _out;
    public GeneratorAxisMirrorTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void GeneratorAxisMirrorMatchesTensorAxisExactly()
    {
        var mirrored = Enum.GetValues(typeof(ShapeContractGenerator.Axis))
            .Cast<ShapeContractGenerator.Axis>()
            .ToDictionary(v => v.ToString(), v => (int)v, StringComparer.Ordinal);

        var real = Enum.GetValues(typeof(TensorAxis))
            .Cast<TensorAxis>()
            .ToDictionary(v => v.ToString(), v => (int)v, StringComparer.Ordinal);

        _out.WriteLine($"generator mirror: {mirrored.Count}   TensorAxis: {real.Count}");

        var missing = real.Keys.Where(k => !mirrored.ContainsKey(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var extra = mirrored.Keys.Where(k => !real.ContainsKey(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var wrong = real
            .Where(kv => mirrored.TryGetValue(kv.Key, out int v) && v != kv.Value)
            .Select(kv => $"{kv.Key}: TensorAxis={kv.Value} but generator mirror={mirrored[kv.Key]}")
            .ToList();

        Assert.True(missing.Count == 0,
            "TensorAxis members absent from the generator's mirror, so a [TensorLayout] using them would "
            + $"emit the wrong identifier: {string.Join(", ", missing)}");

        Assert.True(extra.Count == 0,
            $"generator mirror declares members TensorAxis does not have: {string.Join(", ", extra)}");

        Assert.True(wrong.Count == 0,
            "generator mirror disagrees with TensorAxis on a VALUE. Roslyn passes the underlying int, so "
            + "this silently emits the wrong axis name into generated contracts: "
            + string.Join("; ", wrong));
    }
}
