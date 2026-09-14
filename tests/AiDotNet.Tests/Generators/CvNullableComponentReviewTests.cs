using System.Reflection;
using AiDotNet.Generators;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Executes generated component adapters against the actual model/parameter contracts.</summary>
public sealed class CvNullableComponentReviewTests
{
    [Theory]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void NullableIntent_IsPreservedInBothAnalysisContexts(bool nullableEnabled, bool optional)
    {
        string source = $$"""
            #nullable {{(nullableEnabled ? "enable" : "disable")}}
            using AiDotNet.Interfaces;
            using AiDotNet.LossFunctions;
            using AiDotNet.Models;
            using AiDotNet.Models.Parameters;
            using AiDotNet.Tensors.LinearAlgebra;
            namespace ReviewContracts;
            public partial class NullableComponentModel<T> : ModelBase<T, Tensor<T>, Tensor<T>>
            {
                public IParameterSource<T>{{(optional ? "?" : "")}} Component;
                public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();
                public override Tensor<T> Predict(Tensor<T> input) => input;
                public override void Train(Tensor<T> input, Tensor<T> target) => throw new System.NotSupportedException();
                public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) => throw new System.NotSupportedException();
            }
            """;

        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string trustedAssemblies)
            paths.UnionWith(trustedAssemblies.Split(Path.PathSeparator));
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location)) paths.Add(assembly.Location);
        paths.Add(typeof(ModelBase<,,>).Assembly.Location);
        paths.Add(typeof(Tensor<>).Assembly.Location);
        var references = paths.Select(path => MetadataReference.CreateFromFile(path));
        var compilation = CSharpCompilation.Create(
            "NullableComponentReview_" + Guid.NewGuid().ToString("N"),
            new[] { CSharpSyntaxTree.ParseText(source) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new ModelParameterGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var generatedCompilation, out var diagnostics);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.NotEmpty(driver.GetRunResult().GeneratedTrees);
        using var stream = new MemoryStream();
        var emit = generatedCompilation.Emit(stream);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error)));
        var resultAssembly = Assembly.Load(stream.ToArray());
        var definition = resultAssembly.GetType("ReviewContracts.NullableComponentModel`1")
            ?? throw new InvalidOperationException("The generated model was not emitted.");
        var model = Assert.IsAssignableFrom<ModelBase<double, Tensor<double>, Tensor<double>>>(
            Activator.CreateInstance(definition.MakeGenericType(typeof(double))));
        using (model)
        {
            if (optional)
            {
                Assert.Equal(0, model.ParameterCount);
                Assert.Empty(model.GetParameters());
            }
            else
            {
                Assert.Throws<ParameterLayoutNotReadyException>(() => model.ParameterCount);
                Assert.Throws<ParameterLayoutNotReadyException>(() => model.GetParameters());
            }
        }
    }
}
