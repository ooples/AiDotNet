using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks every shape-declaration build gate at compiler-error severity.</summary>
public class ShapeDeclarationValidationGeneratorTests
{
    private const string Infrastructure = @"
using System;
using System.Collections.Generic;
namespace AiDotNet.Enums
{
    public enum TensorAxis { Batch, Channels, Height, Width, Depth, Time, Length, Features, Frames, Heads, Classes, Other }
}
namespace AiDotNet.Attributes
{
    using AiDotNet.Enums;
    public enum TensorLayoutDirection { Input, Output }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    public sealed class TensorLayoutAttribute : Attribute
    {
        public TensorLayoutAttribute(params TensorAxis[] axes) { }
        public TensorLayoutDirection Direction { get; set; }
        public bool BatchOptional { get; set; }
    }
    [AttributeUsage(AttributeTargets.Class, Inherited = true)]
    public sealed class ElementWiseShapeAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Class, Inherited = true)]
    public sealed class LayerPropertyAttribute : Attribute
    {
        public int ExpectedInputRank { get; set; }
        public string TestInputShape { get; set; }
    }
    [AttributeUsage(AttributeTargets.Class, Inherited = true)]
    public sealed class PreprocessesInputAttribute : Attribute
    {
        public PreprocessesInputAttribute(string reason) { }
    }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    public sealed class StackInputLayoutAttribute : Attribute
    {
        public StackInputLayoutAttribute(params TensorAxis[] axes) { }
        public bool BatchOptional { get; set; }
    }
}
namespace AiDotNet.Interfaces
{
    public interface IShapeContract
    {
        IReadOnlyList<int> OutputAxesFor(int inputRank);
    }
}
namespace AiDotNet.Tensors.LinearAlgebra
{
    public class Tensor<T> { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    using AiDotNet.Tensors.LinearAlgebra;
    public abstract class LayerBase<T>
    {
        public virtual Tensor<T> Forward(Tensor<T> input) => input;
        protected virtual Tensor<T> ForwardTraced(Tensor<T> input) => input;
    }
}
namespace AiDotNet.NeuralNetworks
{
    public abstract class NeuralNetworkBase<T> { }
}";

    private static ImmutableArray<MetadataReference> BaseReferences()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location))
                continue;
            references.Add(MetadataReference.CreateFromFile(assembly.Location));
        }

        return references.ToImmutableArray();
    }

    private static ImmutableArray<Diagnostic> Run(string source)
    {
        var compilation = CSharpCompilation.Create(
            "ShapeDeclarations",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.ShapeDeclarationValidationGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        return diagnostics;
    }

    private static Diagnostic Error(string source, string id)
    {
        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == id));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        return diagnostic;
    }

    [Fact]
    public void ConcreteLayerWithoutContract_IsCompilerError()
    {
        Error(@"
using AiDotNet.NeuralNetworks.Layers;
public sealed class Missing<T> : LayerBase<T> { }", "ADNSHAPE006");
    }

    [Fact]
    public void ContractWithoutInputLayout_IsCompilerError()
    {
        Error(@"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public sealed class MissingInput<T> : LayerBase<T>, IShapeContract
{
    public IReadOnlyList<int> OutputAxesFor(int rank) => new[] { 1 };
}", "ADNSHAPE003");
    }

    [Fact]
    public void LayerThatBypassesForwardTracing_IsCompilerError()
    {
        Error(@"
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
public sealed class Invisible<T> : LayerBase<T>
{
    public override Tensor<T> Forward(Tensor<T> input) => input;
}", "ADNSHAPE004");
    }

    [Fact]
    public void ConcreteModelWithoutContract_IsCompilerError()
    {
        Error(@"
using AiDotNet.NeuralNetworks;
public sealed class MissingModel<T> : NeuralNetworkBase<T> { }", "ADNSHAPE007");
    }

    [Fact]
    public void ModelContractRequiresBothCallerFacingLayouts()
    {
        const string missingOutput = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
public sealed class MissingOutput<T> : NeuralNetworkBase<T>, IShapeContract
{
    public IReadOnlyList<int> OutputAxesFor(int rank) => new[] { 1 };
}";
        const string missingInput = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public sealed class MissingInput<T> : NeuralNetworkBase<T>, IShapeContract
{
    public IReadOnlyList<int> OutputAxesFor(int rank) => new[] { 1 };
}";

        Error(missingInput, "ADNSHAPE008");
        Error(missingOutput, "ADNSHAPE009");
    }

    [Fact]
    public void PreprocessorWithoutStackEntryLayout_IsCompilerError()
    {
        Error(@"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[PreprocessesInput(""tokenizes"")]
public sealed class MissingStackLayout<T> : NeuralNetworkBase<T>, IShapeContract
{
    public IReadOnlyList<int> OutputAxesFor(int rank) => new[] { 1 };
}", "ADNSHAPE010");
    }

    [Fact]
    public void CompletePreprocessingContract_PassesAllModelShapeGates()
    {
        const string source = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes, Direction = TensorLayoutDirection.Output, BatchOptional = true)]
[PreprocessesInput(""tokenizes"")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Time, BatchOptional = true)]
public sealed class Complete<T> : NeuralNetworkBase<T>, IShapeContract
{
    public IReadOnlyList<int> OutputAxesFor(int rank) => new[] { 1 };
}";

        Assert.DoesNotContain(Run(source), diagnostic =>
            diagnostic.Id is "ADNSHAPE007" or "ADNSHAPE008" or "ADNSHAPE009" or "ADNSHAPE010");
    }
}
