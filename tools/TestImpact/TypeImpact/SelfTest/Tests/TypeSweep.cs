namespace Fixture.Tests;

// A test-side helper that enumerates types; the tests that call it depend on the enumerated set.
public static class TypeSweep
{
    public static int CountModels() => typeof(ModelBase).Assembly.GetTypes().Length;
}
