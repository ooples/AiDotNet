using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class AssertionFailureReaderTests
{
    public enum Caller { Direct, Swallowed, Async, AsyncSwallowed, AsyncFinally }
    public enum Mutation { AdditionalHandler, Callback, DifferentExceptionLocal, SuccessfulCompletion, ForeignState, WrongExit }

    [Theory]
    [InlineData(Caller.Direct, AssertionFailurePropagation.LeavesMethod)]
    [InlineData(Caller.Swallowed, AssertionFailurePropagation.Unresolved)]
    [InlineData(Caller.Async, AssertionFailurePropagation.ForwardsToTaskBuilder)]
    [InlineData(Caller.AsyncSwallowed, AssertionFailurePropagation.Unresolved)]
    [InlineData(Caller.AsyncFinally, AssertionFailurePropagation.Unresolved)]
    public void OnlyTheReviewedExceptionalExitIsRecognized(Caller kind, AssertionFailurePropagation expected)
    {
        using var fixture = new Fixture(kind);
        Assert.Equal(expected, AssertionFailureReader.Read(fixture.Method, fixture.Index));
    }

    [Theory]
    [InlineData(Mutation.AdditionalHandler)]
    [InlineData(Mutation.Callback)]
    [InlineData(Mutation.DifferentExceptionLocal)]
    [InlineData(Mutation.SuccessfulCompletion)]
    [InlineData(Mutation.ForeignState)]
    [InlineData(Mutation.WrongExit)]
    public void ChangedAsyncCatchCannotInheritPropagation(Mutation mutation)
    {
        using var fixture = new Fixture(Caller.Async);
        MethodDefinition method = fixture.Method;
        ExceptionHandler handler = Assert.Single(method.Body.ExceptionHandlers);
        int start = method.Body.Instructions.IndexOf(handler.HandlerStart);
        var body = method.Body.Instructions;
        switch (mutation)
        {
            case Mutation.AdditionalHandler: method.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.Callback: body.Insert(start + 1, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.DifferentExceptionLocal: body[start + 6] = Instruction.Create(OpCodes.Ldloc_0); break;
            case Mutation.SuccessfulCompletion:
                body[start + 7].Operand = body.Select(item => item.Operand).OfType<MethodReference>().First(reference => reference.Name == "SetResult"); break;
            case Mutation.ForeignState:
                ((FieldReference)body[start + 3].Operand).Resolve().IsStatic = true; break;
            case Mutation.WrongExit: body[start + 8].Operand = body[start + 1]; break;
        }
        Assert.Equal(AssertionFailurePropagation.Unresolved, AssertionFailureReader.Read(method, fixture.Index));
    }

    private static void Direct(bool flag) => Assert.True(flag, "contract");
    private static void Swallowed(bool flag) { try { Assert.True(flag, "contract"); } catch (Xunit.Sdk.TrueException) { } }
    private static async Task Async(bool flag) { await Task.Yield(); Assert.True(flag, "contract"); }
    private static async Task AsyncSwallowed(bool flag) { await Task.Yield(); try { Assert.True(flag, "contract"); } catch (Xunit.Sdk.TrueException) { } }
    private static async Task AsyncFinally(bool flag) { await Task.Yield(); try { Assert.True(flag, "contract"); } finally { GC.KeepAlive(flag); } }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Method { get; }
        internal int Index { get; }

        internal Fixture(Caller kind)
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(AssertionFailureReaderTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            assembly = AssemblyDefinition.ReadAssembly(typeof(AssertionFailureReaderTests).Assembly.Location,
                new ReaderParameters { AssemblyResolver = resolver });
            var owner = assembly.MainModule.Types.Single(type => type.FullName == typeof(AssertionFailureReaderTests).FullName);
            string name = kind switch
            {
                Caller.Direct => nameof(Direct), Caller.Swallowed => nameof(Swallowed), Caller.Async => nameof(Async),
                Caller.AsyncSwallowed => nameof(AsyncSwallowed), Caller.AsyncFinally => nameof(AsyncFinally),
                _ => throw new ArgumentOutOfRangeException(nameof(kind))
            };
            MethodDefinition entry = owner.Methods.Single(method => method.Name == name);
            CustomAttribute? state = entry.CustomAttributes.SingleOrDefault(attribute =>
                attribute.AttributeType.FullName == "System.Runtime.CompilerServices.AsyncStateMachineAttribute");
            Method = state is null ? entry : ((TypeReference)state.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
            Index = Method.Body.Instructions.Select((instruction, index) => (instruction, index)).Single(site =>
                site.instruction.Operand is MethodReference reference && reference.FullName == "System.Void Xunit.Assert::True(System.Boolean,System.String)").index;
        }

        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
