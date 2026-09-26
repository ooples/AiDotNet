using Mono.Cecil;
using Mono.Cecil.Cil;

// Resolving a field erases the closed declaring type. A static Map<float> and
// Map<double> share the same definition but are different storage locations.
internal static class OwnedFieldBinding
{
    internal static FieldDefinition? Read(Instruction instruction, TypeDefinition owner)
    {
        if (instruction.Operand is not FieldReference reference || reference.Resolve() is not FieldDefinition field ||
            field.DeclaringType != owner || !SameType(reference.FieldType, field.FieldType)) return null;
        if (reference is FieldDefinition) return field;
        return SelfType(reference.DeclaringType, owner) ? field : null;
    }

    internal static bool SelfType(TypeReference reference, TypeDefinition owner)
    {
        if (reference is TypeDefinition) return reference == owner;
        if (!owner.HasGenericParameters)
            return reference is not TypeSpecification && reference.Resolve() == owner;
        if (reference is not GenericInstanceType context || context.ElementType.Resolve() != owner ||
            context.GenericArguments.Count != owner.GenericParameters.Count) return false;
        for (int index = 0; index < context.GenericArguments.Count; index++)
            if (context.GenericArguments[index] is not GenericParameter parameter || parameter.Type != GenericParameterType.Type ||
                parameter.Position != index || parameter.Owner is not TypeReference declared || declared.Resolve() != owner) return false;
        return true;
    }

    internal static bool SameType(TypeReference left, TypeReference right)
    {
        if (left is ArrayType array)
            return right is ArrayType otherArray && array.IsVector == otherArray.IsVector && array.Rank == otherArray.Rank &&
                array.Dimensions.Select((dimension, index) => dimension.LowerBound == otherArray.Dimensions[index].LowerBound &&
                    dimension.UpperBound == otherArray.Dimensions[index].UpperBound).All(equal => equal) && SameType(array.ElementType, otherArray.ElementType);
        if (left is GenericParameter first)
            return right is GenericParameter second && first.Type == second.Type && first.Position == second.Position &&
                first.Owner is TypeReference firstOwner && second.Owner is TypeReference secondOwner &&
                firstOwner.Resolve() is TypeDefinition parameterOwner && parameterOwner == secondOwner.Resolve();
        if (left is GenericInstanceType instance)
            return right is GenericInstanceType other && instance.ElementType.Resolve() is TypeDefinition genericOwner && genericOwner == other.ElementType.Resolve() &&
                instance.GenericArguments.Count == other.GenericArguments.Count &&
                instance.GenericArguments.Select((argument, index) => SameType(argument, other.GenericArguments[index])).All(value => value);
        return left is not TypeSpecification && right is not (TypeSpecification or GenericParameter) &&
            left.Resolve() is TypeDefinition concrete && concrete == right.Resolve();
    }
}
