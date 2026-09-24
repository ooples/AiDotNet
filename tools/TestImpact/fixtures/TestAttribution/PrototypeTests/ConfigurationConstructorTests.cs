using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests
{
    [Trait("Scenario", "RuntimeEffects")]
    public sealed class ConfigurationConstructorTests
    {
        public enum Mutation { MissingBackend, InvalidRate, NanRate, InfiniteRate, WrongScalar, StaticProvider,
            ForeignProviderParameter, DifferentLocal, ForeignSetterContext, EscapingWrite, ExtraEffect, ChangedBranch, StaticInitializer, Finalizer }

        [Fact]
        public void ShapeAloneCannotTrustAProviderWithTheRightName()
        {
            using var fixture = new Fixture();
            Assert.NotNull(ConfigurationConstructorReader.ReadShape(fixture.Call, true, 0.001));
            Assert.Equal(ConfigurationConstructorContract.Unresolved, ConfigurationConstructorReader.Read(fixture.Call, true, 0.001).Contract);
        }

        [Theory]
        [InlineData(Mutation.MissingBackend)]
        [InlineData(Mutation.InvalidRate)]
        [InlineData(Mutation.NanRate)]
        [InlineData(Mutation.InfiniteRate)]
        [InlineData(Mutation.WrongScalar)]
        [InlineData(Mutation.StaticProvider)]
        [InlineData(Mutation.ForeignProviderParameter)]
        [InlineData(Mutation.DifferentLocal)]
        [InlineData(Mutation.ForeignSetterContext)]
        [InlineData(Mutation.EscapingWrite)]
        [InlineData(Mutation.ExtraEffect)]
        [InlineData(Mutation.ChangedBranch)]
        [InlineData(Mutation.StaticInitializer)]
        [InlineData(Mutation.Finalizer)]
        public void ConstructorShapeRejectsChangedInputsAndEffects(Mutation mutation)
        {
            using var fixture = new Fixture();
            bool backend = true;
            double rate = 0.001;
            var il = fixture.Constructor.Body.Instructions;
            switch (mutation)
            {
                case Mutation.MissingBackend: backend = false; break;
                case Mutation.InvalidRate: rate = 0; break;
                case Mutation.NanRate: rate = double.NaN; break;
                case Mutation.InfiniteRate: rate = double.PositiveInfinity; break;
                case Mutation.WrongScalar: ((GenericInstanceType)fixture.Call.DeclaringType).GenericArguments[0] = fixture.Assembly.MainModule.ImportReference(typeof(float)); break;
                case Mutation.StaticProvider: ((GenericInstanceMethod)il[24].Operand).ElementMethod.HasThis = true; break;
                case Mutation.ForeignProviderParameter: ((GenericInstanceMethod)il[24].Operand).GenericArguments[0] = new GenericParameter("T", fixture.Constructor); break;
                case Mutation.DifferentLocal: il[27].OpCode = OpCodes.Ldnull; break;
                case Mutation.ForeignSetterContext:
                    var foreign = new GenericInstanceType(fixture.Constructor.DeclaringType);
                    foreign.GenericArguments.Add(fixture.Assembly.MainModule.ImportReference(typeof(float)));
                    ((MethodReference)il[30].Operand).DeclaringType = foreign; break;
                case Mutation.EscapingWrite: il[16].OpCode = OpCodes.Stsfld; break;
                case Mutation.ExtraEffect: il.Insert(0, Instruction.Create(OpCodes.Nop)); break;
                case Mutation.ChangedBranch: il[11].Operand = il[24]; break;
                case Mutation.StaticInitializer: fixture.Constructor.DeclaringType.Methods.Add(new(".cctor", MethodAttributes.Static | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, fixture.Assembly.MainModule.TypeSystem.Void)); break;
                case Mutation.Finalizer: fixture.Constructor.DeclaringType.Methods.Add(new("Finalize", MethodAttributes.Family | MethodAttributes.Virtual, fixture.Assembly.MainModule.TypeSystem.Void)); break;
            }
            Assert.Null(ConfigurationConstructorReader.ReadShape(fixture.Call, backend, rate));
        }

        private sealed class Fixture : IDisposable
        {
            private readonly DefaultAssemblyResolver resolver = new();
            internal AssemblyDefinition Assembly { get; }
            internal MethodDefinition Constructor { get; }
            internal MethodReference Call { get; }
            internal Fixture()
            {
                resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(ConfigurationConstructorTests).Assembly.Location));
                resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
                Assembly = AssemblyDefinition.ReadAssembly(typeof(ConfigurationConstructorTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
                var type = Assembly.MainModule.GetType("AiDotNet.DistributedTraining.ShapeConfiguration`1");
                Constructor = type.Methods.Single(method => method.IsConstructor && !method.IsStatic);
                var concrete = new GenericInstanceType(type); concrete.GenericArguments.Add(Assembly.MainModule.ImportReference(typeof(double)));
                Call = new MethodReference(".ctor", Constructor.ReturnType, concrete) { HasThis = true };
                foreach (var parameter in Constructor.Parameters) Call.Parameters.Add(new(parameter.ParameterType));
            }
            public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
        }
    }
}

// Untrusted look-alikes, used only as IL fixtures. They must never pass the
// package-bound provider check even though their signatures match production.
namespace AiDotNet.Tensors.Interfaces { public interface INumericOperations<T> { T FromDouble(double value); } }
namespace AiDotNet.Tensors.Helpers
{
    public static class MathHelper
    {
        public static AiDotNet.Tensors.Interfaces.INumericOperations<T> GetNumericOperations<T>() => throw new NotSupportedException();
    }
}
namespace AiDotNet.DistributedTraining
{
    public interface ICommunicationBackend<T> { }
    public sealed class ShapeConfiguration<T>
    {
        public bool AutoSyncGradients { get; set; } = true;
        public int MinimumParameterGroupSize { get; set; } = 1024;
        public ICommunicationBackend<T> CommunicationBackend { get; }
        public T LearningRate { get; set; }
        public bool CpuOffloadOptimizer { get; set; }
        public bool CpuOffloadGradients { get; set; }
        public bool CpuOffloadParams { get; set; }
        public ShapeConfiguration(ICommunicationBackend<T> backend, double learningRate)
        {
            CommunicationBackend = backend ?? throw new ArgumentNullException(nameof(backend));
            if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be greater than zero.");
            var provider = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            LearningRate = provider.FromDouble(learningRate);
        }
        public static ShapeConfiguration<T> CreateForZeROOffload(ICommunicationBackend<T>? backend)
        {
            if (backend is null) throw new ArgumentNullException(nameof(backend));
            return new(backend, 0.01) { AutoSyncGradients = true, CpuOffloadOptimizer = true };
        }
    }
}
