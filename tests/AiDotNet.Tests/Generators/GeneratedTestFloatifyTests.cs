using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Unit tests for the #1679 float-scaffold rewriter (<see cref="GeneratedTestFloatify"/>), which the
/// TestScaffoldGenerator uses to turn a model's <c>&lt;double&gt;</c> generic arguments into
/// <c>&lt;float&gt;</c>. The old implementation was an untested chain of <c>string.Replace</c> calls
/// that missed <c>&lt;double[]&gt;</c> / <c>&lt;double?&gt;</c> and could in principle mangle the
/// <c>double</c> keyword; these tests pin BOTH that the rewrite fires exactly where a generic argument
/// is, and that non-generic <c>double</c> tokens are left completely alone.
/// </summary>
public class GeneratedTestFloatifyTests
{
    [Theory]
    // The cases the old string.Replace handled...
    [InlineData("IFullModel<double>", "IFullModel<float>")]
    [InlineData("IFullModel<double, Tensor<double>>", "IFullModel<float, Tensor<float>>")]
    [InlineData("Foo<int, double>", "Foo<int, float>")]
    [InlineData("Foo<double, int>", "Foo<float, int>")]
    [InlineData("Foo<double,int>", "Foo<float,int>")]
    [InlineData("Foo<int,double>", "Foo<int,float>")]
    // ...and the ones it missed:
    [InlineData("List<double[]>", "List<float[]>")]                       // array element type arg
    [InlineData("Foo<double?>", "Foo<float?>")]                          // nullable type arg
    [InlineData("List<Tensor<double>>", "List<Tensor<float>>")]          // nested generic
    [InlineData("Dictionary<string, double>", "Dictionary<string, float>")]
    [InlineData("IFullModel<double, Tensor<double>, Vector<double>>",
                "IFullModel<float, Tensor<float>, Vector<float>>")]      // every arg in a 3-arg list
    public void Floatify_RewritesGenericDoubleArguments(string input, string expected)
        => Assert.Equal(expected, GeneratedTestFloatify.Floatify(input));

    [Theory]
    // The `double` KEYWORD (not a generic argument) must never be rewritten — this is the scope the
    // PR claimed but did not test. A regression here would silently mangle the scaffold body.
    [InlineData("double threshold = 0.0;")]
    [InlineData("if (double.IsNaN(x)) return;")]
    [InlineData("var y = (double)value;")]
    [InlineData("double.IsInfinity(z)")]
    [InlineData("Assert.Equal(expected, actual, precision: 6); // double precision")]
    [InlineData("var myDouble = 1.0; var doubleCount = 2;")]            // identifiers containing "double"
    [InlineData("typeof(double)")]
    [InlineData("public double Score { get; set; }")]
    public void Floatify_LeavesNonGenericDoubleUntouched(string input)
        => Assert.Equal(input, GeneratedTestFloatify.Floatify(input));

    [Fact]
    public void Floatify_MixedFragment_RewritesOnlyTheGenericArgument()
    {
        const string input =
            "var m = new Foo<double>(); double threshold = 0.0; if (double.IsNaN(threshold)) {}";
        const string expected =
            "var m = new Foo<float>(); double threshold = 0.0; if (double.IsNaN(threshold)) {}";
        Assert.Equal(expected, GeneratedTestFloatify.Floatify(input));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("no generics here at all")]
    [InlineData("double plain = 1.0;")]
    public void Floatify_NoGenericDouble_ReturnsInputUnchanged(string? input)
        => Assert.Equal(input, GeneratedTestFloatify.Floatify(input!));

    [Theory]
    [InlineData("IFullModel<double>", true)]
    [InlineData("List<double[]>", true)]
    [InlineData("Foo<double?>", true)]
    [InlineData("double threshold = 0.0;", false)]
    [InlineData("double.IsNaN(x)", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void ContainsGenericDoubleArg_DetectsRewritableFragments(string? input, bool expected)
        => Assert.Equal(expected, GeneratedTestFloatify.ContainsGenericDoubleArg(input!));
}
