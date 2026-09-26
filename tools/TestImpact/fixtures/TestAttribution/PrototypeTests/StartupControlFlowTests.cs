using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class StartupControlFlowTests
{
    public enum LocalShape { Valid, Missing, WrongType }
    public enum Mutation { ExtraCall, ReorderedObservation, CompletionOutsideTry, UnknownLicense, ExistingLicense, UnknownParallelism, OverrideParallelism, MissingResetSuccess, Loop }

    [Fact]
    public void RecognitionKeepsTargetsForIndependentAuthentication()
    {
        using var fixture = new Fixture();
        StartupControlFlow flow = Assert.IsType<StartupControlFlow>(StartupControlFlowReader.ReadShape(fixture.Method, Profile()));
        Assert.Equal(Enum.GetValues<StartupOperation>(), flow.Calls.Select(call => call.Operation));
        Assert.Contains(flow.ReachedMethods, call => call.DeclaringType.FullName == "AiDotNet.Helpers.BuildKeyProvider");
        // These fake methods have the right signatures, but no implementation.
        // Shape recognition is deliberately not a trusted startup certificate.
        Assert.All(flow.Calls.Where(call => call.Method.DeclaringType.Namespace.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal)),
            call => Assert.Null(call.Method.Resolve()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExactNonNullCpuReceiverSupportsBothGetTypeCallEncodings(bool virtualCall)
    {
        using var fixture = new Fixture();
        var module = fixture.Method.Module;
        var il = fixture.Method.Body.GetILProcessor();
        Instruction result = fixture.Method.Body.Instructions[^4];
        var engine = new TypeReference("AiDotNet.Tensors.Engines", "AiDotNetEngine", module, module);
        var cpu = new TypeReference("AiDotNet.Tensors.Engines", "CpuEngine", module, module);
        var current = new MethodReference("get_Current", new TypeReference("AiDotNet.Tensors.Engines", "IEngine", module, module), engine);
        il.InsertBefore(result, Instruction.Create(OpCodes.Call, current));
        il.InsertBefore(result, Instruction.Create(virtualCall ? OpCodes.Callvirt : OpCodes.Call,
            module.ImportReference(typeof(object).GetMethod(nameof(GetType)) ?? throw new InvalidOperationException())));
        il.InsertBefore(result, Instruction.Create(OpCodes.Ldtoken, cpu));
        il.InsertBefore(result, Instruction.Create(OpCodes.Call, module.ImportReference(typeof(Type).GetMethod(nameof(Type.GetTypeFromHandle)) ?? throw new InvalidOperationException())));
        il.InsertBefore(result, Instruction.Create(OpCodes.Call, module.ImportReference(typeof(Type).GetMethod("op_Equality") ?? throw new InvalidOperationException())));
        il.Remove(result);
        Assert.NotNull(StartupControlFlowReader.ReadShape(fixture.Method, Profile()));
    }

    [Theory]
    [InlineData(LocalShape.Valid)]
    [InlineData(LocalShape.Missing)]
    [InlineData(LocalShape.WrongType)]
    public void StartupLocalsMustExistAndMatchTheirValues(LocalShape shape)
    {
        using var fixture = new Fixture();
        var body = fixture.Method.Body;
        if (shape != LocalShape.Missing) body.Variables.Add(new(fixture.Method.Module.ImportReference(shape == LocalShape.Valid ? typeof(int) : typeof(string))));
        var first = body.Instructions[0];
        var il = body.GetILProcessor();
        il.InsertBefore(first, Instruction.Create(OpCodes.Ldc_I4_1));
        il.InsertBefore(first, Instruction.Create(OpCodes.Stloc_0));
        Assert.Equal(shape == LocalShape.Valid, StartupControlFlowReader.ReadShape(fixture.Method, Profile()) is not null);
    }

    [Theory]
    [InlineData(Mutation.ExtraCall)] [InlineData(Mutation.ReorderedObservation)] [InlineData(Mutation.CompletionOutsideTry)]
    [InlineData(Mutation.UnknownLicense)] [InlineData(Mutation.ExistingLicense)] [InlineData(Mutation.UnknownParallelism)]
    [InlineData(Mutation.OverrideParallelism)] [InlineData(Mutation.MissingResetSuccess)] [InlineData(Mutation.Loop)]
    public void UnknownBranchesAndMissingPostCallProofAreRejected(Mutation mutation)
    {
        using var fixture = new Fixture();
        RuntimeContractProfile profile = Profile();
        var startup = profile.Initialization;
        var inputs = startup.Inputs ?? throw new InvalidOperationException();
        var il = fixture.Method.Body.Instructions;
        switch (mutation)
        {
            case Mutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Call, fixture.Method.Module.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.ReorderedObservation: il[0].Operand = fixture.Method.Module.ImportReference(typeof(RuntimeContractInitialization).GetMethod(nameof(RuntimeContractInitialization.RecordCpuResetCompletion)) ?? throw new InvalidOperationException()); break;
            case Mutation.CompletionOutsideTry: fixture.Method.Body.ExceptionHandlers[0].TryStart = fixture.Method.Body.ExceptionHandlers[0].TryEnd; break;
            case Mutation.UnknownLicense: startup = startup with { Inputs = inputs with { LicenseStartup = RuntimeLicenseStartupPolicy.Unknown } }; break;
            case Mutation.ExistingLicense: startup = startup with { Inputs = inputs with { LicenseStartup = RuntimeLicenseStartupPolicy.ExistingLicenseKey } }; break;
            case Mutation.UnknownParallelism: startup = startup with { Inputs = inputs with { CpuParallelism = RuntimeCpuParallelismPolicy.Unknown } }; break;
            case Mutation.OverrideParallelism: startup = startup with { Inputs = inputs with { CpuParallelism = RuntimeCpuParallelismPolicy.OverridePresent } }; break;
            case Mutation.MissingResetSuccess: startup = startup with { ResetOutcome = RuntimeCpuResetOutcome.Unknown }; break;
            case Mutation.Loop: il.Insert(0, Instruction.Create(OpCodes.Br, il[0])); il[0].Operand = il[0]; break;
        }
        Assert.Null(StartupControlFlowReader.ReadShape(fixture.Method, profile with { Initialization = startup }));
    }

    private static RuntimeContractProfile Profile()
    {
        var input = new RuntimeEnvironmentBinding(1, new string('a', 64), RuntimeObserverSignals.NoneReported,
            RuntimeGpuStartupPolicy.Disabled, RuntimeGpuDiagnosticsPolicy.NoDumpRequested, RuntimeLicenseStartupPolicy.DefaultTestLicense,
            RuntimeCpuParallelismPolicy.DefaultSingleThread);
        return new(input, new(RuntimeInitializationStatus.Recorded, input, new(RuntimeCpuMode.Cpu, 1),
            new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed), RuntimeCpuResetOutcome.Completed));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Method { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(RuntimeContractInitialization).Assembly.Location));
            assembly = AssemblyDefinition.CreateAssembly(new("StartupShape", new(1, 0)), "StartupShape", ModuleKind.Dll);
            // Imported CoreLib types resolve through the assembly's resolver.
            // Register the trusted runtime location in its default resolver.
            if (assembly.MainModule.AssemblyResolver is DefaultAssemblyResolver defaults)
                defaults.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            var module = assembly.MainModule;
            var owner = new TypeDefinition("Fixtures", "Startup", TypeAttributes.Abstract | TypeAttributes.Sealed, module.ImportReference(typeof(object)));
            module.Types.Add(owner);
            var flag = new FieldDefinition("initialized", FieldAttributes.Private | FieldAttributes.Static, module.ImportReference(typeof(bool)));
            owner.Fields.Add(flag);
            var helper = new TypeDefinition("AiDotNet.Tests.Helpers", "LicenseTestSupport", TypeAttributes.Abstract | TypeAttributes.Sealed, module.ImportReference(typeof(object)));
            module.Types.Add(helper);
            var key = new FieldDefinition("TestBuildKey", FieldAttributes.Static | FieldAttributes.InitOnly, module.ImportReference(typeof(byte[])));
            helper.Fields.Add(key);
            Method = new("Initialize", MethodAttributes.Static | MethodAttributes.Private, module.ImportReference(typeof(void)));
            owner.Methods.Add(Method);
            var il = Method.Body.GetILProcessor();
            void Record(string name) => il.Emit(OpCodes.Call, module.ImportReference(typeof(RuntimeContractInitialization).GetMethod(name) ?? throw new InvalidOperationException()));
            MethodReference Fake(string type, string name, Type result, params Type[] arguments)
            {
                int split = type.LastIndexOf('.');
                var reference = new MethodReference(name, module.ImportReference(result), new TypeReference(type[..split], type[(split + 1)..], module, module));
                foreach (Type argument in arguments) reference.Parameters.Add(new(module.ImportReference(argument)));
                return reference;
            }
            var setEnvironment = module.ImportReference(typeof(Environment).GetMethod(nameof(Environment.SetEnvironmentVariable), [typeof(string), typeof(string)]) ?? throw new InvalidOperationException());
            Record(nameof(RuntimeContractInitialization.RecordCpuStartup));
            il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Stsfld, flag);
            foreach (string variable in new[] { "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS" })
            { il.Emit(OpCodes.Ldstr, variable); il.Emit(OpCodes.Ldstr, "1"); il.Emit(OpCodes.Call, setEnvironment); }
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Call, Fake("AiDotNet.Tensors.Helpers.CpuParallelSettings", "set_MaxDegreeOfParallelism", typeof(void), typeof(int)));
            var start = Instruction.Create(OpCodes.Ldc_I4_1); il.Append(start);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Newobj, module.ImportReference(typeof(RuntimeCpuResetInput).GetConstructor([typeof(RuntimeCpuEntryMode), typeof(RuntimeCpuLogging)]) ?? throw new InvalidOperationException()));
            Record(nameof(RuntimeContractInitialization.RecordCpuResetInput));
            il.Emit(OpCodes.Call, Fake("AiDotNet.Tensors.Engines.AiDotNetEngine", "ResetToCpu", typeof(void)));
            Record(nameof(RuntimeContractInitialization.RecordCpuResetCompletion));
            var end = Instruction.Create(OpCodes.Ldsfld, key);
            il.Emit(OpCodes.Leave_S, end);
            var handler = Instruction.Create(OpCodes.Pop); il.Append(handler); il.Emit(OpCodes.Leave_S, end); il.Append(end);
            Method.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Catch) { TryStart = start, TryEnd = handler, HandlerStart = handler, HandlerEnd = end, CatchType = module.ImportReference(typeof(object)) });
            il.Emit(OpCodes.Call, Fake("AiDotNet.Helpers.BuildKeyProvider", "OverrideForTesting", typeof(void), typeof(byte[])));
            il.Emit(OpCodes.Ldstr, "AIDOTNET_LICENSE_KEY"); il.Emit(OpCodes.Ldstr, "testdefault1"); il.Emit(OpCodes.Ldnull);
            il.Emit(OpCodes.Call, Fake("AiDotNet.Tests.Helpers.LicenseTestSupport", "SignedKey", typeof(string), typeof(string), typeof(byte[])));
            il.Emit(OpCodes.Call, setEnvironment);
            il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Ldc_I4_1); Record(nameof(RuntimeContractInitialization.RecordCpuCompletion)); il.Emit(OpCodes.Ret);
        }
        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
