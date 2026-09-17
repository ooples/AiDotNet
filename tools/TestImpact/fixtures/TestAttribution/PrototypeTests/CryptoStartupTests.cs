using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests
{
    [Trait("Scenario", "RuntimeEffects")]
    public sealed class CryptoStartupTests
    {
        public enum Mutation { ExtraCall, SuppliedRng, EscapedGenerator, WrongLocal, InstanceFactory, GenericFactory, CallbackInsteadOfCall, Handler }
        public enum InitializerMutation { ExtraCall, MutableSlot, ExtraSlot, ThreadLocalSlot, ForeignFactory, AlternateEncoding, DuplicateStore }

        [Fact]
        public void LookalikeFactoryShapeDoesNotAuthenticateTheCryptoPackage()
        {
            using var fixture = new Fixture();
            Assert.True(ReviewedCryptoStartup.ReadShape(fixture.Factory));
            Assert.Equal(CryptoStartupContract.Unresolved, ReviewedCryptoStartup.Read(fixture.Factory).Contract);
        }

        [Fact]
        public void CompleteHelperInitializerStillRequiresThePinnedCryptoPackage()
        {
            using var fixture = new Fixture();
            Assert.Equal(fixture.Factory, LicenseSupportInitializerReader.ReadShape(fixture.Owner));
            Assert.Equal(LicenseSupportInitializerContract.Unresolved, LicenseSupportInitializerReader.Read(fixture.Owner).Contract);
        }

        [Theory]
        [InlineData(Mutation.ExtraCall)]
        [InlineData(Mutation.SuppliedRng)]
        [InlineData(Mutation.EscapedGenerator)]
        [InlineData(Mutation.WrongLocal)]
        [InlineData(Mutation.InstanceFactory)]
        [InlineData(Mutation.GenericFactory)]
        [InlineData(Mutation.CallbackInsteadOfCall)]
        [InlineData(Mutation.Handler)]
        public void FactoryProtocolRejectsUnknownEffects(Mutation mutation)
        {
            using var fixture = new Fixture();
            var il = fixture.Factory.Body.Instructions;
            switch (mutation)
            {
                case Mutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Nop)); break;
                case Mutation.SuppliedRng: il[3].OpCode = OpCodes.Ldnull; il[3].Operand = null; break;
                case Mutation.EscapedGenerator: il[1].OpCode = OpCodes.Pop; break;
                case Mutation.WrongLocal: fixture.Factory.Body.Variables[0].VariableType = fixture.Assembly.MainModule.TypeSystem.Object; break;
                case Mutation.InstanceFactory: fixture.Factory.IsStatic = false; fixture.Factory.HasThis = true; break;
                case Mutation.GenericFactory: fixture.Factory.GenericParameters.Add(new GenericParameter("T", fixture.Factory)); break;
                case Mutation.CallbackInsteadOfCall: il[7].OpCode = OpCodes.Ldftn; break;
                case Mutation.Handler: fixture.Factory.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            }
            Assert.False(ReviewedCryptoStartup.ReadShape(fixture.Factory));
        }

        [Theory]
        [InlineData(InitializerMutation.ExtraCall)]
        [InlineData(InitializerMutation.MutableSlot)]
        [InlineData(InitializerMutation.ExtraSlot)]
        [InlineData(InitializerMutation.ThreadLocalSlot)]
        [InlineData(InitializerMutation.ForeignFactory)]
        [InlineData(InitializerMutation.AlternateEncoding)]
        [InlineData(InitializerMutation.DuplicateStore)]
        public void StartupCannotHideEffectsOutsideTheFactory(InitializerMutation mutation)
        {
            using var fixture = new Fixture();
            var il = fixture.Owner.Methods.Single(method => method.IsConstructor && method.IsStatic).Body.Instructions;
            var slot = ((FieldReference)il[5].Operand).Resolve();
            switch (mutation)
            {
                case InitializerMutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Nop)); break;
                case InitializerMutation.MutableSlot: slot.IsInitOnly = false; break;
                case InitializerMutation.ExtraSlot: fixture.Owner.Fields.Add(new("hidden", FieldAttributes.Static | FieldAttributes.Private, fixture.Assembly.MainModule.TypeSystem.Object)); break;
                case InitializerMutation.ThreadLocalSlot: slot.CustomAttributes.Add(new(fixture.Assembly.MainModule.ImportReference(typeof(ThreadStaticAttribute).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
                case InitializerMutation.ForeignFactory: il[4].Operand = fixture.Assembly.MainModule.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()); break;
                case InitializerMutation.AlternateEncoding: ((MethodReference)il[0].Operand).Name = "get_Unicode"; break;
                case InitializerMutation.DuplicateStore: il[5].Operand = il[3].Operand; break;
            }
            Assert.Null(LicenseSupportInitializerReader.ReadShape(fixture.Owner));
        }

        private static class Helper
        {
            internal static readonly byte[] Key = System.Text.Encoding.UTF8.GetBytes("public-fixture-key");
            private static readonly Org.BouncyCastle.Crypto.AsymmetricCipherKeyPair Pair = Create();
            private static Org.BouncyCastle.Crypto.AsymmetricCipherKeyPair Create()
            {
                var generator = new Org.BouncyCastle.Crypto.Generators.Ed25519KeyPairGenerator();
                generator.Init(new Org.BouncyCastle.Crypto.Parameters.Ed25519KeyGenerationParameters(new Org.BouncyCastle.Security.SecureRandom()));
                return generator.GenerateKeyPair();
            }
        }

        private sealed class Fixture : IDisposable
        {
            private readonly DefaultAssemblyResolver resolver = new();
            internal AssemblyDefinition Assembly { get; }
            internal TypeDefinition Owner { get; }
            internal MethodDefinition Factory { get; }
            internal Fixture()
            {
                resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(CryptoStartupTests).Assembly.Location));
                resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
                Assembly = AssemblyDefinition.ReadAssembly(typeof(CryptoStartupTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
                Owner = Assembly.MainModule.GetType(typeof(Helper).FullName?.Replace('+', '/'));
                Factory = Owner.Methods.Single(method => method.Name == "Create");
            }
            public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
        }
    }
}

// IL-only hostile lookalikes. Names and valid allocation shapes are insufficient
// to inherit a contract: these implementations are not the reviewed package.
namespace Org.BouncyCastle.Security { public class SecureRandom { public SecureRandom() { GC.Collect(); } } }
namespace Org.BouncyCastle.Crypto
{
    public class AsymmetricCipherKeyPair { }
    public class KeyGenerationParameters { }
}
namespace Org.BouncyCastle.Crypto.Parameters
{
    public class Ed25519KeyGenerationParameters : KeyGenerationParameters { public Ed25519KeyGenerationParameters(Org.BouncyCastle.Security.SecureRandom random) { } }
}
namespace Org.BouncyCastle.Crypto.Generators
{
    public class Ed25519KeyPairGenerator
    {
        public void Init(KeyGenerationParameters parameters) { }
        public AsymmetricCipherKeyPair GenerateKeyPair() => new();
    }
}
