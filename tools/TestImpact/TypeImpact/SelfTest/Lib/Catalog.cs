namespace Fixture;

// Names every model and is reached through one entry point: a dispatch table.
public static class Catalog
{
    public static ModelBase Create(string name) => name switch
    {
        "alpha" => new Alpha(),
        "beta" => new Beta(),
        _ => throw new System.ArgumentException(name),
    };
}
