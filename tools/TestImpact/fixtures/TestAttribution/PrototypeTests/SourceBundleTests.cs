using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class SourceBundleTests
{
    private static readonly TestCaseIdentity[] Inventory = [new("a", "Tests:A"), new("b", "Tests:B")];
    private static SourceSnapshot Assembly(char revision, string file, params SourceMethod[] methods) =>
        new(1, new(revision, 40), file, new('c', 64), new('d', 64), new('e', 64), SourceMapStatus.Verified, methods);
    private static SourceMethod Method(string id, string owner, string[] calls, string file, int line) =>
        new(new(id, calls, [], DependencyBoundary.Closed), owner, new('e', 64), [new(file, line, line)], false);
    private static SourceBundleSnapshot Bundle(char revision) => new(1, new(revision, 40), "Tests.dll",
        [Assembly(revision, "Tests.dll", Method("test-a", "Tests:A", ["prod-a"], "Tests.cs", 10),
            Method("test-b", "Tests:B", ["prod-b"], "Tests.cs", 20)),
         Assembly(revision, "Library.dll", Method("prod-a", "Library:A", [], "Library.cs", 10),
            Method("prod-b", "Library:B", [], "Library.cs", 20))]);
    private static SourceDelta Delta => new(new('a', 40), new('b', 40), [new("Library.cs", 10, 1)], [new("Library.cs", 10, 1)], false);
    private static SourceSelection Select(SourceBundleSnapshot? before = null, SourceBundleSnapshot? after = null) =>
        SourceImpact.Select(before ?? Bundle('a'), after ?? Bundle('b'), Inventory, Inventory, Delta);

    [Fact]
    public void ProductionEditFollowsCrossAssemblyCallWithoutBecomingAGroupRoot()
    {
        Assert.Equal("Tests:A", Assert.Single(Select().Methods).MethodId);
    }

    [Fact]
    public void ProductionSharedStateStillExpandsTheSelectedMethods()
    {
        SourceBundleSnapshot after = Bundle('b');
        foreach (int index in new[] { 0, 1 })
        {
            SourceMethod method = after.Assemblies[1].Methods[index];
            after.Assemblies[1].Methods[index] = method with { Dependency = method.Dependency with { SharedState = ["Library:state"] } };
        }
        Assert.Equal(2, Select(after: after).Methods.Length);
    }

    [Fact]
    public void TestFixtureCallingProductionRemainsGroupWide()
    {
        SourceBundleSnapshot after = Bundle('b');
        SourceSnapshot tests = after.Assemblies[0];
        after.Assemblies[0] = tests with { Methods = [.. tests.Methods, Method("fixture", "Tests:Fixture", ["prod-a"], "Tests.cs", 30)] };
        Assert.Equal(2, Select(after: after).Methods.Length);
    }

    [Fact]
    public void RemovedCrossAssemblyCallStillUsesTheOldGraph()
    {
        SourceBundleSnapshot after = Bundle('b');
        SourceMethod method = after.Assemblies[0].Methods[0];
        after.Assemblies[0].Methods[0] = method with { Dependency = method.Dependency with { Calls = [] } };
        Assert.Equal("Tests:A", Assert.Single(Select(after: after).Methods).MethodId);
    }

    [Fact]
    public void MissingDependencyAssemblyBroadensExecution()
    {
        SourceBundleSnapshot after = Bundle('b');
        Assert.Equal(2, Select(after: after with { Assemblies = [after.Assemblies[0]] }).Methods.Length);
    }

    [Fact]
    public void ForeignRevisionAndDuplicateAssemblyAreRejected()
    {
        SourceBundleSnapshot after = Bundle('b');
        after.Assemblies[1] = after.Assemblies[1] with { SourceTree = new('f', 40) };
        Assert.Throws<EvidenceException>(() => Select(after: after));
        after = Bundle('b');
        Assert.Throws<EvidenceException>(() => Select(after: after with { Assemblies = [.. after.Assemblies, after.Assemblies[0]] }));
    }
}
