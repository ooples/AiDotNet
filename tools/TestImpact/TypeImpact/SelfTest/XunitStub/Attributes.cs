// Stand-ins for xUnit's attributes: TypeImpact recognises tests by attribute name and by a
// reference to an assembly whose name starts with "xunit", so the fixture needs no package.
namespace Xunit;

public class FactAttribute : System.Attribute { }
public class TheoryAttribute : FactAttribute { }

[System.AttributeUsage(System.AttributeTargets.All, AllowMultiple = true)]
public sealed class TraitAttribute(string name, string value) : System.Attribute
{
    public string Name { get; } = name;
    public string Value { get; } = value;
}
