using System.Reflection;
using AttributionRuntime;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace AiDotNet.TestImpact.Xunit;

// Preserve xUnit's actual case runners (including custom facts and deferred
// theories), fixture lifetimes, collection order and parallelism. Only the
// method-runner boundary is decorated; ordinary execution uses the base executor.
public class AttributionTestFramework(IMessageSink diagnosticSink) : XunitTestFramework(diagnosticSink)
{
    protected override ITestFrameworkExecutor CreateExecutor(AssemblyName assemblyName) => Tracker.IsEnabled
        ? new AttributionExecutor(assemblyName, SourceInformationProvider, DiagnosticMessageSink)
        : base.CreateExecutor(assemblyName);

    internal static string Owner(IXunitTestCase testCase)
    {
        if (testCase.TestMethod.TestClass.Class is not IReflectionTypeInfo type)
            throw new InvalidOperationException("Missing concrete reflection test type.");
        string assembly = type.Type.Assembly.GetName().Name ?? throw new InvalidOperationException("Missing test assembly identity.");
        string concreteType = type.Type.FullName ?? throw new InvalidOperationException("Missing concrete test type.");
        return $"{assembly}:{concreteType}.{testCase.TestMethod.Method.Name}";
    }
}

internal sealed class AttributionExecutor(AssemblyName assemblyName, ISourceInformationProvider source,
    IMessageSink diagnostics) : XunitTestFrameworkExecutor(assemblyName, source, diagnostics)
{
    protected override async void RunTestCases(IEnumerable<IXunitTestCase> testCases,
        IMessageSink executionSink, ITestFrameworkExecutionOptions options)
    {
        IXunitTestCase[] inventory = testCases.ToArray();
        Tracker.RegisterCases(inventory.Select(test => new DiscoveredCase(test.UniqueID,
            AttributionTestFramework.Owner(test), test.DisplayName, test.GetType() == typeof(XunitTestCase)
                ? DiscoveredCaseKind.Enumerated : DiscoveredCaseKind.DeferredOrCustom)));
        using var runner = new AttributionAssemblyRunner(TestAssembly, inventory, DiagnosticMessageSink,
            new AttributionExecutionSink(executionSink), options);
        await runner.RunAsync();
    }
}

internal sealed class AttributionExecutionSink(IMessageSink next) : IMessageSink
{
    public bool OnMessage(IMessageSinkMessage message)
    {
        switch (message)
        {
            case ITestPassed passed:
                Tracker.RecordCaseResult(passed.Test.TestCase.UniqueID, passed.Test.DisplayName, ObservedOutcome.Passed);
                break;
            case ITestFailed failed:
                Tracker.RecordCaseResult(failed.Test.TestCase.UniqueID, failed.Test.DisplayName, ObservedOutcome.Failed);
                break;
            case ITestSkipped skipped:
                Tracker.RecordCaseResult(skipped.Test.TestCase.UniqueID, skipped.Test.DisplayName, ObservedOutcome.Skipped);
                break;
            case ITestCaseFinished finished:
                Tracker.FinishCase(finished.TestCase.UniqueID, finished.TestsRun, finished.TestsFailed, finished.TestsSkipped);
                break;
        }
        return next.OnMessage(message);
    }
}

internal sealed class AttributionAssemblyRunner(ITestAssembly assembly, IEnumerable<IXunitTestCase> cases,
    IMessageSink diagnostics, IMessageSink execution, ITestFrameworkExecutionOptions options)
    : XunitTestAssemblyRunner(assembly, cases, diagnostics, execution, options)
{
    protected override Task<RunSummary> RunTestCollectionAsync(IMessageBus bus, ITestCollection collection,
        IEnumerable<IXunitTestCase> cases, CancellationTokenSource cancellation) =>
        new AttributionCollectionRunner(collection, cases, DiagnosticMessageSink, bus, TestCaseOrderer,
            new ExceptionAggregator(Aggregator), cancellation).RunAsync();
}

internal sealed class AttributionCollectionRunner(ITestCollection collection, IEnumerable<IXunitTestCase> cases,
    IMessageSink diagnostics, IMessageBus bus, ITestCaseOrderer orderer, ExceptionAggregator aggregator,
    CancellationTokenSource cancellation)
    : XunitTestCollectionRunner(collection, cases, diagnostics, bus, orderer, aggregator, cancellation)
{
    protected override Task<RunSummary> RunTestClassAsync(ITestClass testClass, IReflectionTypeInfo type,
        IEnumerable<IXunitTestCase> cases) =>
        new AttributionClassRunner(testClass, type, cases, DiagnosticMessageSink, MessageBus, TestCaseOrderer,
            new ExceptionAggregator(Aggregator), CancellationTokenSource, CollectionFixtureMappings).RunAsync();
}

internal sealed class AttributionClassRunner(ITestClass testClass, IReflectionTypeInfo type,
    IEnumerable<IXunitTestCase> cases, IMessageSink diagnostics, IMessageBus bus, ITestCaseOrderer orderer,
    ExceptionAggregator aggregator, CancellationTokenSource cancellation, IDictionary<Type, object> fixtures)
    : XunitTestClassRunner(testClass, type, cases, diagnostics, bus, orderer, aggregator, cancellation, fixtures)
{
    protected override Task<RunSummary> RunTestMethodAsync(ITestMethod method, IReflectionMethodInfo reflectionMethod,
        IEnumerable<IXunitTestCase> cases, object[] constructorArguments) =>
        new AttributionMethodRunner(method, Class, reflectionMethod, cases, DiagnosticMessageSink, MessageBus,
            new ExceptionAggregator(Aggregator), CancellationTokenSource, constructorArguments).RunAsync();
}

internal sealed class AttributionMethodRunner(ITestMethod method, IReflectionTypeInfo type,
    IReflectionMethodInfo reflectionMethod, IEnumerable<IXunitTestCase> cases, IMessageSink diagnostics,
    IMessageBus bus, ExceptionAggregator aggregator, CancellationTokenSource cancellation, object[] constructorArguments)
    : XunitTestMethodRunner(method, type, reflectionMethod, cases, diagnostics, bus, aggregator, cancellation, constructorArguments)
{
    protected override async Task<RunSummary> RunTestCaseAsync(IXunitTestCase testCase)
    {
        string owner = AttributionTestFramework.Owner(testCase);
        Tracker.Begin(owner);
        try { return await base.RunTestCaseAsync(testCase); }
        finally { Tracker.End(owner); }
    }
}
