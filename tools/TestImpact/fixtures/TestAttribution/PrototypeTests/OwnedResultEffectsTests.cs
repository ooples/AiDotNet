using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class OwnedResultEffectsTests
{
    public enum Mutation { ConstructorEscape, StaticWrite, SetterEffect, Finalizer, CallbackAfterWrite, MultipleAllocations, MixedReturn, UnsupportedInstruction }
    public enum BundleMutation { AdditionalBody, Metadata, CallbackFile }

    [Theory]
    [InlineData(BundleMutation.AdditionalBody)]
    [InlineData(BundleMutation.Metadata)]
    [InlineData(BundleMutation.CallbackFile)]
    public void ExperimentRejectsChangesOutsideOwnedFlag(BundleMutation mutation)
    {
        using var fixture = new Fixture();
        DirectoryInfo root = Directory.CreateTempSubdirectory("owned-effect-control-");
        try
        {
            string before = Directory.CreateDirectory(Path.Combine(root.FullName, "before")).FullName;
            string after = Directory.CreateDirectory(Path.Combine(root.FullName, "after")).FullName;
            fixture.Write(before);
            fixture.Factory.Body.Instructions.Single(instruction => instruction.OpCode.Code == Code.Ldc_I4_1).OpCode = OpCodes.Ldc_I4_0;
            switch (mutation)
            {
                case BundleMutation.AdditionalBody: fixture.Callback.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
                case BundleMutation.Metadata: fixture.Shared.Name = "changedDeclaration"; break;
                case BundleMutation.CallbackFile: File.WriteAllText(Path.Combine(after, "callback.config"), "changed"); break;
                default: throw new ArgumentOutOfRangeException(nameof(mutation));
            }
            fixture.Write(after);
            var inventory = new DiscoveryManifest(1, "experiment", new("tree", RunnerBinding.HashBundle(after), "profile"), [new("case", "Tests:Tests.Example")]);
            Assert.Throws<InvalidDataException>(() => RuntimeEffectsExperiment.Read(before, after, inventory, fixture.Owner.FullName, fixture.Factory.Name));
        }
        finally { root.Delete(recursive: true); }
    }

    [Fact]
    public void IndependentFullControlRejectsMissedRuntimeDispatch()
    {
        TestCaseResult[] before = [new("direct", CaseOutcome.Passed), new("reflection", CaseOutcome.Passed)];
        TestCaseResult[] after = [new("direct", CaseOutcome.Failed), new("reflection", CaseOutcome.Failed)];
        Assert.Throws<InvalidDataException>(() => RuntimeEffectControl.Compare(["direct", "reflection"], ["direct"], before, after, [after[0]]));
        after[1] = new("reflection", CaseOutcome.Passed);
        RuntimeEffectControl.Compare(["direct", "reflection"], ["direct"], before, after, [after[0]]);
        Assert.Throws<InvalidDataException>(() => RuntimeEffectControl.Compare(["direct", "reflection"], ["direct"], before, after, [before[0]]));
    }

    [Theory]
    [InlineData(CaseOutcome.Skipped)]
    [InlineData(CaseOutcome.Cancelled)]
    [InlineData((CaseOutcome)99)]
    public void IncompleteControlCannotProveSelection(CaseOutcome outcome)
    {
        TestCaseResult[] before = [new("a", CaseOutcome.Passed), new("b", CaseOutcome.Passed)];
        Assert.Throws<InvalidDataException>(() => RuntimeEffectControl.Compare(["a", "b"], ["a"], before, [before[0], new("b", outcome)], [before[0]]));
        Assert.Throws<InvalidDataException>(() => RuntimeEffectControl.Compare(["a", "b"], ["a"], before, [before[0]], [before[0]]));
        Assert.Throws<InvalidDataException>(() => RuntimeEffectControl.Compare(["a", "b"], ["a"], before, [before[0], before[0]], [before[0]]));
    }

    [Fact]
    public void OnlyOwnedBooleanValuesAreMasked()
    {
        using var fixture = new Fixture();
        OwnedResultEffect before = Assert.IsType<OwnedResultEffect>(new OwnedResultEffects().Read(fixture.Factory));
        Instruction literal = fixture.Factory.Body.Instructions.Single(instruction => instruction.OpCode.Code == Code.Ldc_I4_1);
        literal.OpCode = OpCodes.Ldc_I4_0;
        OwnedResultEffect after = Assert.IsType<OwnedResultEffect>(new OwnedResultEffects().Read(fixture.Factory));
        Assert.Equal(before.ShapeHash, after.ShapeHash);
        Assert.True(Assert.Single(before.Writes).Value);
        Assert.False(Assert.Single(after.Writes).Value);
        Assert.Contains(DependencyGraph.Stable(fixture.Constructor), before.OwnershipDependencies);
        Assert.Contains(DependencyGraph.Stable(fixture.Setter), before.OwnershipDependencies);
        fixture.Factory.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop));
        Assert.NotEqual(after.ShapeHash, Assert.IsType<OwnedResultEffect>(new OwnedResultEffects().Read(fixture.Factory)).ShapeHash);
    }

    [Theory]
    [InlineData(Mutation.ConstructorEscape)]
    [InlineData(Mutation.StaticWrite)]
    [InlineData(Mutation.SetterEffect)]
    [InlineData(Mutation.Finalizer)]
    [InlineData(Mutation.CallbackAfterWrite)]
    [InlineData(Mutation.MultipleAllocations)]
    [InlineData(Mutation.MixedReturn)]
    [InlineData(Mutation.UnsupportedInstruction)]
    public void UnsafeOwnershipIsNotSummarized(Mutation mutation)
    {
        using var fixture = new Fixture();
        var body = fixture.Factory.Body.Instructions;
        switch (mutation)
        {
            case Mutation.ConstructorEscape:
                fixture.Constructor.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Ldarg_0));
                fixture.Constructor.Body.Instructions.Insert(1, Instruction.Create(OpCodes.Call, fixture.Escape));
                break;
            case Mutation.StaticWrite:
                body.Insert(0, Instruction.Create(OpCodes.Ldc_I4_0));
                body.Insert(1, Instruction.Create(OpCodes.Stsfld, fixture.Shared));
                break;
            case Mutation.SetterEffect:
                fixture.Setter.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Call, fixture.Callback));
                break;
            case Mutation.Finalizer:
                fixture.Owner.Methods.Add(new MethodDefinition("Finalize", MethodAttributes.Family | MethodAttributes.Virtual, fixture.Module.TypeSystem.Void));
                break;
            case Mutation.CallbackAfterWrite:
                body.Insert(body.Count - 1, Instruction.Create(OpCodes.Call, fixture.Callback));
                break;
            case Mutation.MultipleAllocations:
                body.Insert(0, Instruction.Create(OpCodes.Newobj, fixture.Constructor));
                body.Insert(1, Instruction.Create(OpCodes.Pop));
                break;
            case Mutation.MixedReturn:
                Instruction ret = body[^1];
                Instruction allocation = body[0];
                body.Insert(0, Instruction.Create(OpCodes.Ldc_I4_0));
                body.Insert(1, Instruction.Create(OpCodes.Brtrue, allocation));
                body.Insert(2, Instruction.Create(OpCodes.Ldnull));
                body.Insert(3, Instruction.Create(OpCodes.Br, ret));
                break;
            case Mutation.UnsupportedInstruction:
                body.Insert(0, Instruction.Create(OpCodes.Break));
                break;
            default: throw new ArgumentOutOfRangeException(nameof(mutation));
        }
        Assert.Null(new OwnedResultEffects().Read(fixture.Factory));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly AssemblyDefinition assembly = AssemblyDefinition.CreateAssembly(new("OwnedFixture", new(1, 0)), "OwnedFixture", ModuleKind.Dll);
        internal ModuleDefinition Module => assembly.MainModule;
        internal TypeDefinition Owner { get; }
        internal MethodDefinition Factory { get; }
        internal MethodDefinition Constructor { get; }
        internal MethodDefinition Setter { get; }
        internal MethodDefinition Escape { get; }
        internal MethodDefinition Callback { get; }
        internal FieldDefinition Shared { get; }

        internal Fixture()
        {
            // Synthetic IL avoids coupling negative controls to compiler lowering.
            Owner = new("Fixture", "Owned", TypeAttributes.Public | TypeAttributes.Class, null);
            Module.Types.Add(Owner);
            var field = new FieldDefinition("flag", FieldAttributes.Private, Module.TypeSystem.Boolean);
            Owner.Fields.Add(field);
            Shared = new("shared", FieldAttributes.Static, Module.TypeSystem.Boolean);
            Owner.Fields.Add(Shared);
            Constructor = Method(".ctor", MethodAttributes.Public | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, Module.TypeSystem.Void);
            Constructor.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
            Setter = Method("set_Flag", MethodAttributes.Public | MethodAttributes.SpecialName, Module.TypeSystem.Void);
            Setter.Parameters.Add(new("value", ParameterAttributes.None, Module.TypeSystem.Boolean));
            foreach (Instruction instruction in new[] { Instruction.Create(OpCodes.Ldarg_0), Instruction.Create(OpCodes.Ldarg_1), Instruction.Create(OpCodes.Stfld, field), Instruction.Create(OpCodes.Ret) }) Setter.Body.Instructions.Add(instruction);
            Escape = Method("Escape", MethodAttributes.Public | MethodAttributes.Static, Module.TypeSystem.Void);
            Escape.Parameters.Add(new("value", ParameterAttributes.None, Owner));
            Escape.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
            Callback = Method("Callback", MethodAttributes.Public | MethodAttributes.Static, Module.TypeSystem.Void);
            Callback.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
            Factory = Method("Create", MethodAttributes.Public | MethodAttributes.Static, Owner);
            foreach (Instruction instruction in new[] { Instruction.Create(OpCodes.Newobj, Constructor), Instruction.Create(OpCodes.Dup), Instruction.Create(OpCodes.Ldc_I4_1), Instruction.Create(OpCodes.Callvirt, Setter), Instruction.Create(OpCodes.Ret) }) Factory.Body.Instructions.Add(instruction);
        }

        private MethodDefinition Method(string name, MethodAttributes attributes, TypeReference result)
        {
            var method = new MethodDefinition(name, attributes, result);
            Owner.Methods.Add(method);
            return method;
        }
        public void Dispose() => assembly.Dispose();
        internal void Write(string directory) => assembly.Write(Path.Combine(directory, "OwnedFixture.dll"));
    }
}
