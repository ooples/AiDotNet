namespace ManagedLibrary;

public static class Scalar
{
    public static T Identity<T>(T value) => value;
    public static T Identity<T, TUnused>(T value) => value;
}
