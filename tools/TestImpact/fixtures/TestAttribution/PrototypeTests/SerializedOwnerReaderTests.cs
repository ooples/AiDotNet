using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class SerializedOwnerReaderTests
{
    public enum Mutation { MissingMembership, DuplicateMembership, UnknownDefinition, DuplicateDefinition, ParallelAllowed,
        MissingParallelFlag, ExtraSetting, ClassFixture, CollectionFixture, MutableState, StaticInitializer, ConstructorCode,
        InheritedClass, MissingMethod, ForeignAssembly, WrongBooleanType, ConstructorArgument,
        MembershipProperty, MembershipField, Finalizer, AssemblyOrderer, DuplicateBehavior, CustomCollectionFactory }

    [Fact]
    public void FixtureFreeSerialCollectionIsBoundToTheReviewedMetadata()
    {
        using var fixture = new Fixture();
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? OwnerConcurrencyContract.FixtureFreeNonparallelCollection : OwnerConcurrencyContract.Unresolved,
            SerializedOwnerReader.Read(fixture.Assembly, fixture.Owner));
    }

    [Theory]
    [InlineData(Mutation.MissingMembership)]
    [InlineData(Mutation.DuplicateMembership)]
    [InlineData(Mutation.UnknownDefinition)]
    [InlineData(Mutation.DuplicateDefinition)]
    [InlineData(Mutation.ParallelAllowed)]
    [InlineData(Mutation.MissingParallelFlag)]
    [InlineData(Mutation.ExtraSetting)]
    [InlineData(Mutation.ClassFixture)]
    [InlineData(Mutation.CollectionFixture)]
    [InlineData(Mutation.MutableState)]
    [InlineData(Mutation.StaticInitializer)]
    [InlineData(Mutation.ConstructorCode)]
    [InlineData(Mutation.InheritedClass)]
    [InlineData(Mutation.MissingMethod)]
    [InlineData(Mutation.ForeignAssembly)]
    [InlineData(Mutation.WrongBooleanType)]
    [InlineData(Mutation.ConstructorArgument)]
    [InlineData(Mutation.MembershipProperty)]
    [InlineData(Mutation.MembershipField)]
    [InlineData(Mutation.Finalizer)]
    [InlineData(Mutation.AssemblyOrderer)]
    [InlineData(Mutation.DuplicateBehavior)]
    [InlineData(Mutation.CustomCollectionFactory)]
    public void ChangedLifecycleDoesNotInheritSerialIsolation(Mutation mutation)
    {
        using var fixture = new Fixture();
        var membership = fixture.Type.CustomAttributes.Single(attribute => attribute.AttributeType.FullName == "Xunit.CollectionAttribute");
        var declaration = fixture.Definition.CustomAttributes.Single(attribute => attribute.AttributeType.FullName == "Xunit.CollectionDefinitionAttribute");
        string owner = fixture.Owner;
        switch (mutation)
        {
            case Mutation.MissingMembership: fixture.Type.CustomAttributes.Remove(membership); break;
            case Mutation.DuplicateMembership: fixture.Type.CustomAttributes.Add(membership); break;
            case Mutation.UnknownDefinition: fixture.Definition.CustomAttributes.Clear(); break;
            case Mutation.DuplicateDefinition: fixture.Definition.CustomAttributes.Add(declaration); break;
            case Mutation.ParallelAllowed: declaration.Properties[0] = new("DisableParallelization", new(fixture.Assembly.MainModule.TypeSystem.Boolean, false)); break;
            case Mutation.MissingParallelFlag: declaration.Properties.Clear(); break;
            case Mutation.ExtraSetting: declaration.Properties.Add(new("Unknown", new(fixture.Assembly.MainModule.TypeSystem.Boolean, true))); break;
            case Mutation.ClassFixture: fixture.Type.Interfaces.Add(new(fixture.Assembly.MainModule.ImportReference(typeof(IDisposable)))); break;
            case Mutation.CollectionFixture: fixture.Definition.Interfaces.Add(new(fixture.Assembly.MainModule.ImportReference(typeof(IDisposable)))); break;
            case Mutation.MutableState: fixture.Type.Fields.Add(new("state", FieldAttributes.Private, fixture.Assembly.MainModule.TypeSystem.Int32)); break;
            case Mutation.StaticInitializer:
                fixture.Type.Methods.Add(new(".cctor", MethodAttributes.Static | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, fixture.Assembly.MainModule.TypeSystem.Void)); break;
            case Mutation.ConstructorCode: fixture.Type.Methods.Single(method => method.IsConstructor).Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.InheritedClass: fixture.Type.BaseType = fixture.Definition; break;
            case Mutation.MissingMethod: owner += "Missing"; break;
            case Mutation.ForeignAssembly: owner = "foreign:" + owner; break;
            case Mutation.WrongBooleanType: declaration.Properties[0] = new("DisableParallelization", new(fixture.Assembly.MainModule.TypeSystem.String, true)); break;
            case Mutation.ConstructorArgument: membership.Constructor.Parameters[0].ParameterType = fixture.Assembly.MainModule.TypeSystem.Object; break;
            case Mutation.MembershipProperty: membership.Properties.Add(new("Unknown", new(fixture.Assembly.MainModule.TypeSystem.Boolean, true))); break;
            case Mutation.MembershipField: membership.Fields.Add(new("Unknown", new(fixture.Assembly.MainModule.TypeSystem.Boolean, true))); break;
            case Mutation.Finalizer: fixture.Type.Methods.Add(new("Finalize", MethodAttributes.Family | MethodAttributes.Virtual, fixture.Assembly.MainModule.TypeSystem.Void)); break;
            case Mutation.AssemblyOrderer:
                fixture.Assembly.CustomAttributes.Add(new(fixture.Assembly.MainModule.ImportReference(
                    typeof(TestCollectionOrdererAttribute).GetConstructor([typeof(string), typeof(string)]) ?? throw new InvalidOperationException()))); break;
            case Mutation.DuplicateBehavior:
            case Mutation.CustomCollectionFactory:
                foreach (var existing in fixture.Assembly.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.CollectionBehaviorAttribute").ToArray())
                    fixture.Assembly.CustomAttributes.Remove(existing);
                var behavior = new CustomAttribute(fixture.Assembly.MainModule.ImportReference(
                    typeof(CollectionBehaviorAttribute).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException()));
                fixture.Assembly.CustomAttributes.Add(behavior);
                if (mutation == Mutation.DuplicateBehavior) fixture.Assembly.CustomAttributes.Add(behavior);
                else behavior.ConstructorArguments.Add(new(fixture.Assembly.MainModule.TypeSystem.String, "custom"));
                break;
        }
        Assert.Equal(OwnerConcurrencyContract.Unresolved, SerializedOwnerReader.Read(fixture.Assembly, owner));
    }

    [Collection(nameof(SerialDefinition))]
    private sealed class TestOwner { internal static void Entry() { } }

    [CollectionDefinition(nameof(SerialDefinition), DisableParallelization = true)]
    public sealed class SerialDefinition { }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal TypeDefinition Type { get; }
        internal TypeDefinition Definition { get; }
        internal string Owner => Assembly.Name.Name + ":" + Type.FullName.Replace('/', '+') + ".Entry";
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(SerializedOwnerReaderTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(SerializedOwnerReaderTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            var parent = Assembly.MainModule.Types.Single(type => type.FullName == typeof(SerializedOwnerReaderTests).FullName);
            Type = parent.NestedTypes.Single(type => type.Name == nameof(TestOwner));
            Definition = parent.NestedTypes.Single(type => type.Name == nameof(SerialDefinition));
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
