using System.Reflection;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using AttributionRuntime;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;
using Xunit.Sdk;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class TrialHookTests
{
    public enum ObservationMutation { MissingScope, OpenScope, CustomCase, MissingOwner, MissingHook, StaleBundle, DuplicateOwners, WrongAssembly }
    private const string HookedOwner = "PrototypeTests:PrototypeTests.TrialHookTests.HookedFact";

    [Fact, HookFixture]
    public async Task HookedFact() => await Task.Yield();

    [Fact]
    public void ActualLifecycleAndVerifiedObservationAreBothRequired()
    {
        string bundle = Path.GetDirectoryName(typeof(TrialHookTests).Assembly.Location) ?? throw new InvalidOperationException("Missing bundle");
        using var evidence = new ObservedOwnerTests.Evidence([new("one", HookedOwner)], RunnerBinding.HashBundle(bundle));
        evidence.Report = evidence.Report with { TrialScopes = ScopeReport(HookedOwner) };
        ObservedTrialHookAssessment result = Assert.Single(ObservedTrialHookReader.ReadAll(bundle, "PrototypeTests.dll", [HookedOwner], evidence.Verify()));
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? ObservedTrialHookProof.ReviewedHookAndCompletedScope : ObservedTrialHookProof.Unresolved, result.Proof);
        if (supported)
        {
            Assert.DoesNotContain(TrialHookRequirement.ObservedOwner, result.Requirements);
            Assert.Contains(TrialHookRequirement.BodySlotIsolation, result.Requirements);
            Assert.Contains(TrialHookRequirement.BodyFileIsolation, result.Requirements);
            Assert.Contains(TrialHookRequirement.OwnerContextFlow, result.Requirements);
        }
    }

    [Theory]
    [InlineData(ObservationMutation.MissingScope)]
    [InlineData(ObservationMutation.OpenScope)]
    [InlineData(ObservationMutation.CustomCase)]
    [InlineData(ObservationMutation.MissingOwner)]
    [InlineData(ObservationMutation.MissingHook)]
    [InlineData(ObservationMutation.StaleBundle)]
    [InlineData(ObservationMutation.DuplicateOwners)]
    [InlineData(ObservationMutation.WrongAssembly)]
    public void ObservationsCannotSubstituteForMissingBindings(ObservationMutation mutation)
    {
        string bundle = Path.GetDirectoryName(typeof(TrialHookTests).Assembly.Location) ?? throw new InvalidOperationException("Missing bundle");
        string owner = mutation == ObservationMutation.MissingHook
            ? "PrototypeTests:PrototypeTests.ReviewedOwnerCompletionTests.AwaitedFact" : HookedOwner;
        using var evidence = new ObservedOwnerTests.Evidence([new("one", owner)],
            mutation == ObservationMutation.StaleBundle ? new('f', 64) : RunnerBinding.HashBundle(bundle));
        TrialScopeReport scope = ScopeReport(owner);
        if (mutation == ObservationMutation.OpenScope) scope.Scopes[0] = scope.Scopes[0] with { State = TrialScopeState.Open };
        evidence.Report = evidence.Report with { TrialScopes = mutation == ObservationMutation.MissingScope ? null : scope };
        if (mutation == ObservationMutation.CustomCase) evidence.Report.Cases[0] = evidence.Report.Cases[0] with
        { Case = evidence.Report.Cases[0].Case with { Kind = DiscoveredCaseKind.DeferredOrCustom } };
        string[] requested = mutation == ObservationMutation.DuplicateOwners ? [owner, owner]
            : [mutation == ObservationMutation.MissingOwner ? owner + "Missing" : owner];
        var result = ObservedTrialHookReader.ReadAll(bundle, mutation == ObservationMutation.WrongAssembly ? "missing.dll" : "PrototypeTests.dll",
            requested, evidence.Verify());
        Assert.Equal(requested.Length, result.Length);
        Assert.All(result, item => Assert.Equal(ObservedTrialHookProof.Unresolved, item.Proof));
    }

    private static TrialScopeReport ScopeReport(string owner) => new(TrialScopeLedgerState.Recorded,
        [new(owner, new('a', 64), new('b', 64), new('c', 64), new('d', 64), TrialScopeState.Complete)]);

    public enum Mutation { EarlyObservation, WrongObservedPath, WrongPrevious, WrongRestoreGetter, SkippedDispose,
        EarlyCleanup, MissingClear, SharedScope, OtherSlot, DifferentRoot, DifferentExtension, WrongGuidFormat,
        WrongTombstone, ExtraCall, Handler, Synchronized, CustomObserver, WrongGuidLocal, ConstructorCall, CleanupCall, CleanupPath, CleanupFilter,
        CleanupTryBoundary, ForeignGetterParameter, ForeignSetterParameter }

    [Fact]
    public void ExactHookOrderingStillRequiresBodyAndOwnerProof()
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        var (before, after) = Methods(assembly);
        TrialHookAssessment result = TrialHookReader.Read(before, after);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) ==
            ReviewedOwnerCompletion.RuntimeHash;
        Assert.True(result.Contract == (supported ? TrialHookContract.ObservedSaveRestore : TrialHookContract.Unresolved), result.Failure.ToString());
        if (supported) Assert.Equal(Enum.GetValues<TrialHookRequirement>(), result.Requirements);
        else Assert.Empty(result.Requirements);
    }

    [Theory]
    [InlineData(Mutation.EarlyObservation)]
    [InlineData(Mutation.WrongObservedPath)]
    [InlineData(Mutation.WrongPrevious)]
    [InlineData(Mutation.WrongRestoreGetter)]
    [InlineData(Mutation.SkippedDispose)]
    [InlineData(Mutation.EarlyCleanup)]
    [InlineData(Mutation.MissingClear)]
    [InlineData(Mutation.SharedScope)]
    [InlineData(Mutation.OtherSlot)]
    [InlineData(Mutation.DifferentRoot)]
    [InlineData(Mutation.DifferentExtension)]
    [InlineData(Mutation.WrongGuidFormat)]
    [InlineData(Mutation.WrongTombstone)]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.Handler)]
    [InlineData(Mutation.Synchronized)]
    [InlineData(Mutation.CustomObserver)]
    [InlineData(Mutation.WrongGuidLocal)]
    [InlineData(Mutation.ConstructorCall)]
    [InlineData(Mutation.CleanupCall)]
    [InlineData(Mutation.CleanupPath)]
    [InlineData(Mutation.CleanupFilter)]
    [InlineData(Mutation.CleanupTryBoundary)]
    [InlineData(Mutation.ForeignGetterParameter)]
    [InlineData(Mutation.ForeignSetterParameter)]
    public void AlteredHookCannotBorrowTheReviewedObservationOrder(Mutation mutation)
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        var (before, after) = Methods(assembly);
        var first = before.Body.Instructions;
        var last = after.Body.Instructions;
        switch (mutation)
        {
            case Mutation.ForeignGetterParameter:
            case Mutation.ForeignSetterParameter:
                var foreign = new TypeDefinition("Synthetic", "Foreign`1", Mono.Cecil.TypeAttributes.Class, assembly.MainModule.TypeSystem.Object);
                var parameter = new GenericParameter("T", foreign);
                foreign.GenericParameters.Add(parameter);
                assembly.MainModule.Types.Add(foreign);
                var accessor = (MethodReference)(mutation == Mutation.ForeignGetterParameter ? last[3].Operand : first[16].Operand);
                if (mutation == Mutation.ForeignGetterParameter) accessor.ReturnType = parameter;
                else accessor.Parameters[0].ParameterType = parameter;
                break;
            case Mutation.EarlyObservation: (first[15].Operand, first[20].Operand) = (first[20].Operand, first[15].Operand); break;
            case Mutation.WrongObservedPath: first[17].OpCode = OpCodes.Ldloc_1; break;
            case Mutation.WrongPrevious: first[18].OpCode = OpCodes.Ldloc_0; break;
            case Mutation.WrongRestoreGetter: last[13].Operand = first[0].Operand; break;
            case Mutation.SkippedDispose: last[5].Operand = last[9]; break;
            case Mutation.EarlyCleanup: (last[14].Operand, last[16].Operand) = (last[16].Operand, last[14].Operand); break;
            case Mutation.MissingClear: last[10].OpCode = OpCodes.Ldloc_0; break;
            case Mutation.SharedScope: before.DeclaringType.Fields[0].IsStatic = false; break;
            case Mutation.OtherSlot:
                var other = new FieldDefinition("Other", Mono.Cecil.FieldAttributes.Private | Mono.Cecil.FieldAttributes.Static,
                    before.DeclaringType.Fields[0].FieldType);
                before.DeclaringType.Fields.Add(other); last[9].Operand = other; break;
            case Mutation.DifferentRoot: first[1].Operand = "shared-trial-state"; break;
            case Mutation.DifferentExtension: first[7].Operand = ".txt"; break;
            case Mutation.WrongGuidFormat: first[5].Operand = "D"; break;
            case Mutation.WrongTombstone: last[20].Operand = ".shared"; break;
            case Mutation.ExtraCall: first.Insert(0, Instruction.Create(OpCodes.Call, (MethodReference)first[0].Operand)); break;
            case Mutation.Handler: after.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.Synchronized: after.ImplAttributes |= Mono.Cecil.MethodImplAttributes.Synchronized; break;
            case Mutation.CustomObserver: first[20].Operand = first[15].Operand; break;
            case Mutation.WrongGuidLocal: first[4].Operand = before.Body.Variables[0]; break;
            case Mutation.ConstructorCall:
                before.DeclaringType.Methods.Single(method => method.IsConstructor && !method.IsStatic).Body.Instructions[1].Operand = first[0].Operand; break;
            case Mutation.CleanupCall:
                before.DeclaringType.Methods.Single(method => method.Name == "TryDelete").Body.Instructions[9].Operand = first[0].Operand; break;
            case Mutation.CleanupPath:
                var cleanup = before.DeclaringType.Methods.Single(method => method.Name == "TryDelete");
                cleanup.Body.Instructions[8].OpCode = OpCodes.Ldstr; cleanup.Body.Instructions[8].Operand = "outside-scope.json"; break;
            case Mutation.CleanupFilter:
                before.DeclaringType.Methods.Single(method => method.Name == "TryDelete").Body.ExceptionHandlers[0].FilterStart = last[0]; break;
            case Mutation.CleanupTryBoundary:
                var protectedCleanup = before.DeclaringType.Methods.Single(method => method.Name == "TryDelete");
                protectedCleanup.Body.ExceptionHandlers[0].TryStart = protectedCleanup.Body.Instructions[4]; break;
        }
        TrialHookAssessment result = TrialHookReader.Read(before, after);
        Assert.Equal(TrialHookContract.Unresolved, result.Contract);
        Assert.Empty(result.Requirements);
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(TrialHookTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }
    private static AssemblyDefinition Read(DefaultAssemblyResolver resolver) => AssemblyDefinition.ReadAssembly(
        typeof(TrialHookTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
    private static (MethodDefinition Before, MethodDefinition After) Methods(AssemblyDefinition assembly)
    {
        TypeDefinition hook = assembly.MainModule.Types.Single(type => type.FullName == typeof(TrialHookTests).FullName)
            .NestedTypes.Single(type => type.Name == nameof(HookFixture));
        return (hook.Methods.Single(method => method.Name == "Before"), hook.Methods.Single(method => method.Name == "After"));
    }

    // Applied only to the isolated HookedFact control. Other prototype tests
    // retain their existing execution semantics.
    private sealed class HookFixture : BeforeAfterTestAttribute
    {
        private static readonly AsyncLocal<IDisposable?> scope = new();
        public override void Before(MethodInfo methodUnderTest)
        {
            string path = Path.Combine(Path.GetTempPath(), "aidotnet-trial-tests", Guid.NewGuid().ToString("N") + ".json");
            string? previous = GuardFixture.Current;
            scope.Value = GuardFixture.Set(path);
            Tracker.TrialScopeStarted(path, previous, GuardFixture.Current);
        }
        public override void After(MethodInfo methodUnderTest)
        {
            string? path = GuardFixture.Current;
            scope.Value?.Dispose();
            scope.Value = null;
            Tracker.TrialScopeEnded(path, GuardFixture.Current);
            TryDelete(path);
            TryDelete(path is null ? null : path + ".tombstone");
        }
        private static void TryDelete(string? path)
        {
            if (string.IsNullOrEmpty(path)) return;
            try { if (File.Exists(path)) File.Delete(path); }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException) { }
        }
    }

    private static class GuardFixture
    {
        private static readonly AsyncLocal<string?> slot = new();
        internal static string? Current => slot.Value;
        internal static IDisposable Set(string? value)
        {
            string? previous = slot.Value;
            slot.Value = value;
            return new Scope(previous);
        }
        private sealed class Scope : IDisposable
        {
            private readonly string? previous;
            internal Scope(string? previous) { this.previous = previous; }
            public void Dispose() { slot.Value = previous; }
        }
    }
}
