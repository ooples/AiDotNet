using System;
using System.Linq;
using System.Reflection;

namespace AiDotNetTestConsole;

internal static class VecInspect
{
    public static void Run()
    {
        var t = typeof(AiDotNet.Tensors.LinearAlgebra.Vector<double>);
        Console.WriteLine($"Type: {t.FullName}");
        Console.WriteLine("PUBLIC PROPERTIES:");
        foreach (var p in t.GetProperties(BindingFlags.Public | BindingFlags.Instance))
            Console.WriteLine($"  {p.PropertyType.Name} {p.Name}");
        Console.WriteLine("PUBLIC METHODS:");
        foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => !m.IsSpecialName).Take(40))
            Console.WriteLine($"  {m.ReturnType.Name} {m.Name}({string.Join(",", m.GetParameters().Select(p => p.ParameterType.Name))})");
        Console.WriteLine("FIELDS (incl non-public):");
        foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            Console.WriteLine($"  {f.FieldType.Name} {f.Name}");

        // Check the base class too
        var bt = t.BaseType;
        if (bt is not null)
        {
            Console.WriteLine($"BASE TYPE: {bt.FullName}");
            foreach (var p in bt.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                Console.WriteLine($"  PROP {p.PropertyType.Name} {p.Name}");
            foreach (var f in bt.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                Console.WriteLine($"  FIELD {f.FieldType.Name} {f.Name}");
        }
    }
}
