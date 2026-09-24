using System.Security.Cryptography;
using AiDotNet.TestImpact;
using AttributionRuntime;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ReviewedOwnerCompletionTests
{
    public enum Mutation { ExecutionBinary, CoreBinary, FrameworkOverride, DetachedTask, StaleBundle, CustomCase, MissingOwner }
    private const string Owner = "PrototypeTests:PrototypeTests.ReviewedOwnerCompletionTests.AwaitedFact";

    [Fact]
    public async Task AwaitedFact() => await Task.Yield();

    [Fact]
    public void StandardTaskRequiresTheReviewedBinaryRuntimeAndObservedOwner()
    {
        string bundle = Path.GetDirectoryName(typeof(ReviewedOwnerCompletionTests).Assembly.Location)
            ?? throw new InvalidOperationException("Missing bundle.");
        using var evidence = new ObservedOwnerTests.Evidence([new("actual", Owner)], RunnerBinding.HashBundle(bundle));
        OwnerCompletionProof proof = ReviewedOwnerCompletion.Read(bundle, "PrototypeTests.dll", Owner, evidence.Verify());
        string runtimeHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location)));
        Assert.Equal(runtimeHash == ReviewedOwnerCompletion.RuntimeHash ? OwnerCompletionProof.ReviewedStandardTaskObserved :
            OwnerCompletionProof.Unresolved, proof);
    }

    [Theory]
    [InlineData(Mutation.ExecutionBinary)]
    [InlineData(Mutation.CoreBinary)]
    [InlineData(Mutation.FrameworkOverride)]
    [InlineData(Mutation.DetachedTask)]
    [InlineData(Mutation.StaleBundle)]
    [InlineData(Mutation.CustomCase)]
    [InlineData(Mutation.MissingOwner)]
    public void UnsupportedRunnerOrOwnerCannotCloseObservation(Mutation mutation)
    {
        string root = Directory.CreateTempSubdirectory("owner-contract-").FullName;
        try
        {
            string original = Path.GetDirectoryName(typeof(ReviewedOwnerCompletionTests).Assembly.Location)
                ?? throw new InvalidOperationException("Missing bundle.");
            foreach (string file in Directory.EnumerateFiles(original, "*.dll"))
                File.Copy(file, Path.Combine(root, Path.GetFileName(file)));
            if (mutation is Mutation.ExecutionBinary or Mutation.CoreBinary)
            {
                string file = Path.Combine(root, mutation == Mutation.ExecutionBinary ? "xunit.execution.dotnet.dll" : "xunit.core.dll");
                using var stream = new FileStream(file, FileMode.Append);
                stream.WriteByte(0);
            }
            if (mutation is Mutation.FrameworkOverride or Mutation.DetachedTask)
            {
                string file = Path.Combine(root, "PrototypeTests.dll");
                using var assembly = AssemblyDefinition.ReadAssembly(file, new ReaderParameters { InMemory = true });
                if (mutation == Mutation.FrameworkOverride)
                {
                    var type = new TypeDefinition("Mutation", "CustomFramework", TypeAttributes.Public | TypeAttributes.Class,
                        new TypeReference("AiDotNet.TestImpact.Xunit", "AttributionTestFramework", assembly.MainModule,
                            assembly.MainModule.AssemblyReferences.Single(reference => reference.Name == "Attribution.Xunit")));
                    var method = new MethodDefinition("CreateExecutor", MethodAttributes.Public | MethodAttributes.Virtual,
                        assembly.MainModule.TypeSystem.Object);
                    method.Body.Instructions.Add(Instruction.Create(OpCodes.Ldnull));
                    method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
                    type.Methods.Add(method);
                    assembly.MainModule.Types.Add(type);
                    var attribute = assembly.CustomAttributes.Single(item => item.AttributeType.FullName == "Xunit.TestFrameworkAttribute");
                    attribute.ConstructorArguments[0] = new(assembly.MainModule.TypeSystem.String, type.FullName);
                    attribute.ConstructorArguments[1] = new(assembly.MainModule.TypeSystem.String, assembly.Name.Name);
                }
                else
                {
                    MethodDefinition entry = assembly.MainModule.Types.Single(type => type.FullName == typeof(ReviewedOwnerCompletionTests).FullName)
                        .Methods.Single(method => method.Name == nameof(AwaitedFact));
                    entry.Body = new MethodBody(entry);
                    entry.Body.Instructions.Add(Instruction.Create(OpCodes.Call, assembly.MainModule.ImportReference(
                        typeof(Task).GetProperty(nameof(Task.CompletedTask))?.GetMethod ?? throw new InvalidOperationException("Missing task getter."))));
                    entry.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
                }
                assembly.Write(file);
            }
            using var evidence = new ObservedOwnerTests.Evidence([new("actual", Owner)],
                mutation == Mutation.StaleBundle ? new('f', 64) : RunnerBinding.HashBundle(root));
            if (mutation == Mutation.CustomCase)
                evidence.Report.Cases[0] = evidence.Report.Cases[0] with
                {
                    Case = evidence.Report.Cases[0].Case with { Kind = DiscoveredCaseKind.DeferredOrCustom }
                };
            Assert.Equal(OwnerCompletionProof.Unresolved, ReviewedOwnerCompletion.Read(root, "PrototypeTests.dll",
                mutation == Mutation.MissingOwner ? Owner + "Missing" : Owner, evidence.Verify()));
        }
        finally { Directory.Delete(root, recursive: true); }
    }
}
