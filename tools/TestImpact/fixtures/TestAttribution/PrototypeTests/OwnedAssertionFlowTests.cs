using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class OwnedAssertionFlowTests
{
    public enum Flow { Local, SharedWrite, SharedReference, Swallowed, Hoisted, CompletionCallback, UnreviewedAssertion }
    public static bool SharedFlag;
    public static object? SharedObject;

    [Theory]
    [InlineData(Flow.Local, OwnedReturnUseStatus.ConditionalOnSuccessfulOwner)]
    [InlineData(Flow.SharedWrite, OwnedReturnUseStatus.Escapes)]
    [InlineData(Flow.SharedReference, OwnedReturnUseStatus.Escapes)]
    [InlineData(Flow.Swallowed, OwnedReturnUseStatus.NeedsBoundaryProof)]
    [InlineData(Flow.Hoisted, OwnedReturnUseStatus.Escapes)]
    [InlineData(Flow.CompletionCallback, OwnedReturnUseStatus.NeedsBoundaryProof)]
    [InlineData(Flow.UnreviewedAssertion, OwnedReturnUseStatus.NeedsBoundaryProof)]
    public void OnlyLocalReviewedAssertionFlowsReceiveConditionalStatus(Flow flow, OwnedReturnUseStatus expected)
    {
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(OwnedAssertionFlowTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(OwnedAssertionFlowTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        TypeDefinition type = assembly.MainModule.Types.Single(type => type.FullName == typeof(OwnedAssertionFlowTests).FullName);
        string entryName = flow switch
        {
            Flow.SharedWrite => nameof(WritesState), Flow.SharedReference => nameof(Escapes), Flow.Swallowed => nameof(Swallows),
            Flow.Hoisted => nameof(Hoists), _ => nameof(Local)
        };
        MethodDefinition entry = type.Methods.Single(method => method.Name == entryName);
        CustomAttribute marker = entry.CustomAttributes.Single(attribute => attribute.AttributeType.Name == "AsyncStateMachineAttribute");
        MethodDefinition caller = ((TypeReference)marker.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
        var body = caller.Body.Instructions;
        if (flow == Flow.CompletionCallback)
            body.Insert(body.Count - 1, Instruction.Create(OpCodes.Call, type.Methods.Single(method => method.Name == nameof(Callback))));
        if (flow == Flow.UnreviewedAssertion)
            body.Single(instruction => instruction.Operand is MethodReference reference && reference.DeclaringType.FullName == "Xunit.Assert")
                .Operand = type.Methods.Single(method => method.Name == nameof(Unreviewed));
        int factoryCall = body.Select((instruction, index) => (instruction, index)).Single(site =>
            site.instruction.Operand is MethodReference reference && reference.Name == nameof(Create)).index;
        var factory = ((MethodReference)body[factoryCall].Operand).Resolve();
        Assert.NotNull(new OwnedResultEffects().Read(factory));
        Assert.Equal(expected, OwnedReturnUseReader.Read(caller, factoryCall, entry).Status);
        if (flow == Flow.Local)
            Assert.Equal(OwnedReturnUseStatus.NeedsBoundaryProof, OwnedReturnUseReader.Read(caller, factoryCall).Status);
    }

    private sealed class Configuration { public bool Flag { get; set; } }
    private static Configuration Create() => new() { Flag = true };
    private static void Callback() => SharedFlag = true;
    private static void Unreviewed(bool value, string message) { SharedFlag = value; GC.KeepAlive(message); }
    private static async Task Local() { await Task.Yield(); var config = Create(); Assert.True(config.Flag, "local"); }
    private static async Task WritesState() { await Task.Yield(); var config = Create(); Assert.True(config.Flag, "local"); SharedFlag = true; }
    private static async Task Escapes() { await Task.Yield(); var config = Create(); SharedObject = config; }
    private static async Task Swallows() { await Task.Yield(); var config = Create(); try { Assert.True(config.Flag, "local"); } catch (Xunit.Sdk.TrueException) { } }
    private static async Task Hoists() { await Task.Yield(); var config = Create(); await Task.Yield(); Assert.True(config.Flag, "local"); }
}
