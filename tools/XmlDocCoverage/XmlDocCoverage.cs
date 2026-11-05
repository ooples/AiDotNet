using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Xml.Linq;

namespace AiDotNet.Tools.XmlDocCoverage;

/// <summary>
/// Finds the externally visible members of an assembly that have no entry in its generated XML documentation file:
/// the members the compiler would report as CS1591 if the warning were on.
/// </summary>
/// <remarks>
/// Member names are written in the compiler's documentation-ID format (ECMA-334 annex D), so a member is undocumented
/// exactly when its ID is missing from the file. Members the compiler never reports are left out: the properties a
/// positional record synthesizes from its parameters (documented by the record's own param tags) and the runtime
/// methods of delegate types. A type's single parameterless constructor is left out too, because reflection cannot
/// tell the compiler's implicit one, which is never reported, from a declared one.
/// </remarks>
internal static class XmlDocCoverage
{
    /// <summary>
    /// Returns every visible member of <paramref name="assembly"/> missing from <paramref name="documentation"/>, as
    /// its documentation ID with the type that declares it, sorted by ID.
    /// </summary>
    internal static IReadOnlyList<(Type DeclaringType, string Id)> FindUndocumented(Assembly assembly, XDocument documentation)
    {
        var documented = new HashSet<string>(
            documentation.Descendants("member").Select(member => (string?)member.Attribute("name") ?? string.Empty),
            StringComparer.Ordinal);

        var missing = new List<(Type DeclaringType, string Id)>();
        foreach (Type type in LoadableTypes(assembly).Where(IsVisible).Where(type => !IsCompilerGenerated(type)))
        {
            foreach (string id in VisibleMemberIds(type).Where(id => !documented.Contains(id)))
            {
                missing.Add((type, id));
            }
        }

        missing.Sort((a, b) => StringComparer.Ordinal.Compare(a.Id, b.Id));
        return missing;
    }

    private static IEnumerable<Type> LoadableTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException exception)
        {
            return exception.Types.OfType<Type>();
        }
    }

    private static bool IsVisible(Type type)
    {
        if (!type.IsNested)
        {
            return type.IsPublic;
        }

        Type? declaring = type.DeclaringType;
        return declaring != null && IsVisible(declaring)
            && (type.IsNestedPublic || type.IsNestedFamily || type.IsNestedFamORAssem);
    }

    private static bool IsCompilerGenerated(MemberInfo member)
        => member.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false);

    private static bool IsVisible(MethodBase? method)
        => method != null && (method.IsPublic || method.IsFamily || method.IsFamilyOrAssembly);

    private static IEnumerable<string> VisibleMemberIds(Type type)
    {
        string typeName = TypeName(type);
        yield return "T:" + typeName;

        if (typeof(Delegate).IsAssignableFrom(type))
        {
            yield break;
        }

        const BindingFlags declared = BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.NonPublic
            | BindingFlags.Instance | BindingFlags.Static;

        ConstructorInfo[] constructors = type.GetConstructors(declared);
        bool onlyParameterless = constructors.Length == 1 && constructors[0].GetParameters().Length == 0;
        foreach (ConstructorInfo constructor in constructors)
        {
            if (constructor.IsStatic || !IsVisible(constructor) || IsCompilerGenerated(constructor) || onlyParameterless)
            {
                continue;
            }

            yield return "M:" + typeName + ".#ctor" + Parameters(constructor.GetParameters());
        }

        foreach (MethodInfo method in type.GetMethods(declared))
        {
            if (!IsVisible(method) || IsCompilerGenerated(method)
                || (method.IsSpecialName && !method.Name.StartsWith("op_", StringComparison.Ordinal)))
            {
                continue;
            }

            string id = "M:" + typeName + "." + method.Name.Replace('.', '#');
            if (method.IsGenericMethodDefinition)
            {
                id += "``" + method.GetGenericArguments().Length;
            }

            id += Parameters(method.GetParameters());
            if (method.Name == "op_Implicit" || method.Name == "op_Explicit")
            {
                id += "~" + ParameterType(method.ReturnType);
            }

            yield return id;
        }

        var positional = new HashSet<string>(StringComparer.Ordinal);
        if (type.GetMethods(declared).Any(method => method.Name == "PrintMembers" && IsCompilerGenerated(method)))
        {
            foreach (ConstructorInfo constructor in constructors)
            {
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    positional.Add(parameter.Name ?? string.Empty);
                }
            }
        }

        foreach (PropertyInfo property in type.GetProperties(declared))
        {
            if ((!IsVisible(property.GetMethod) && !IsVisible(property.SetMethod)) || IsCompilerGenerated(property)
                || (positional.Contains(property.Name) && property.GetMethod is MethodInfo getter && IsCompilerGenerated(getter)))
            {
                continue;
            }

            yield return "P:" + typeName + "." + property.Name.Replace('.', '#') + Parameters(property.GetIndexParameters());
        }

        foreach (EventInfo @event in type.GetEvents(declared))
        {
            if (!IsVisible(@event.AddMethod) || IsCompilerGenerated(@event))
            {
                continue;
            }

            yield return "E:" + typeName + "." + @event.Name;
        }

        foreach (FieldInfo field in type.GetFields(declared))
        {
            if (field.IsSpecialName || !(field.IsPublic || field.IsFamily || field.IsFamilyOrAssembly) || IsCompilerGenerated(field))
            {
                continue;
            }

            yield return "F:" + typeName + "." + field.Name;
        }
    }

    /// <summary>The ID form of a type definition: namespace, then each enclosing type, each with its own `arity.</summary>
    private static string TypeName(Type type)
    {
        if (type.IsNested && type.DeclaringType is Type declaring)
        {
            return TypeName(declaring) + "." + type.Name;
        }

        return string.IsNullOrEmpty(type.Namespace) ? type.Name : type.Namespace + "." + type.Name;
    }

    private static string Parameters(ParameterInfo[] parameters)
        => parameters.Length == 0
            ? string.Empty
            : "(" + string.Join(",", parameters.Select(parameter => ParameterType(parameter.ParameterType))) + ")";

    private static string ParameterType(Type type)
    {
        if (type.IsByRef)
        {
            return ParameterType(ElementType(type)) + "@";
        }

        if (type.IsPointer)
        {
            return ParameterType(ElementType(type)) + "*";
        }

        if (type.IsArray)
        {
            Type element = ElementType(type);
            int rank = type.GetArrayRank();
            return ParameterType(element) + (rank == 1 && type == element.MakeArrayType()
                ? "[]"
                : "[" + string.Join(",", Enumerable.Repeat("0:", rank)) + "]");
        }

        if (type.IsGenericParameter)
        {
            return (type.DeclaringMethod != null ? "``" : "`") + type.GenericParameterPosition;
        }

        if (type.IsGenericType)
        {
            Type[] arguments = type.GetGenericArguments();
            int consumed = 0;
            return Constructed(type.GetGenericTypeDefinition(), arguments, ref consumed);
        }

        return TypeName(type);
    }

    private static Type ElementType(Type type)
        => type.GetElementType() ?? throw new InvalidOperationException($"{type} has no element type.");

    /// <summary>
    /// The ID form of a constructed generic type. The type arguments of the whole nesting chain arrive in one array,
    /// outermost first, and each level takes as many as its own arity.
    /// </summary>
    private static string Constructed(Type definition, Type[] arguments, ref int consumed)
    {
        string prefix = definition.IsNested && definition.DeclaringType is Type declaring
            ? Constructed(declaring, arguments, ref consumed) + "."
            : string.IsNullOrEmpty(definition.Namespace) ? string.Empty : definition.Namespace + ".";

        string name = definition.Name;
        int tick = name.IndexOf('`');
        if (tick < 0)
        {
            return prefix + name;
        }

        int arity = int.Parse(name.Substring(tick + 1), System.Globalization.CultureInfo.InvariantCulture);
        string own = string.Join(",", arguments.Skip(consumed).Take(arity).Select(ParameterType));
        consumed += arity;
        return prefix + name.Substring(0, tick) + "{" + own + "}";
    }
}
