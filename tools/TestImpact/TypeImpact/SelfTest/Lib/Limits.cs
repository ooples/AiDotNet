namespace Fixture;

// A visible const: consumers inline the value and keep no reference to this type.
public static class Limits
{
    public const int MaxDepth = 3;

    public static int Describe() => MaxDepth;
}
