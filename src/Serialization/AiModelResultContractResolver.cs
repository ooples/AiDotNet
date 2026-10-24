using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.Serialization;

internal sealed class AiModelResultContractResolver : DefaultContractResolver
{
    protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
    {
        var properties = base.CreateProperties(type, memberSerialization);

        if (IsNeuralNetworkArchitecture(type) || IsOrDerivesFromRawGeneric(type, typeof(NeuralNetworkBase<>)))
        {
            properties = properties
                .Where(p => !string.Equals(p.PropertyName, "Layers", StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        return properties;
    }

    private static bool IsNeuralNetworkArchitecture(Type type)
    {
        return type.IsGenericType && type.GetGenericTypeDefinition() == typeof(NeuralNetworkArchitecture<>);
    }

    private static bool IsOrDerivesFromRawGeneric(Type type, Type rawGeneric)
    {
        // BaseType is null once the walk reaches the top of the chain, which is what the loop's null
        // check has always been for: object and every interface end it. Treating that as an error turns
        // a normal termination into a serialization failure.
        Type? candidate = type;
        while (candidate is not null && candidate != typeof(object))
        {
            var current = candidate.IsGenericType ? candidate.GetGenericTypeDefinition() : candidate;
            if (current == rawGeneric)
            {
                return true;
            }

            candidate = candidate.BaseType;
        }

        return false;
    }
}

