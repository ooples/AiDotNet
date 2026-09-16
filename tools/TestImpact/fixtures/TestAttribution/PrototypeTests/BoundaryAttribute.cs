using System.Reflection;
using AttributionRuntime;
using Xunit.Sdk;

namespace PrototypeTests;

// Attribute state is deliberately empty: xUnit may reuse attributes across cases.
public sealed class BoundaryAttribute : BeforeAfterTestAttribute
{
    private static string Owner(MethodInfo method) =>
        $"{method.Module.Assembly.GetName().Name}:{method.DeclaringType?.FullName ?? throw new InvalidOperationException("Missing test type.")}.{method.Name}";

    public override void Before(MethodInfo methodUnderTest) => Tracker.Begin(Owner(methodUnderTest));
    public override void After(MethodInfo methodUnderTest) => Tracker.End(Owner(methodUnderTest));
}
