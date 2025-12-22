using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.Serialization;

internal sealed class PredictionModelResultContractResolver : DefaultContractResolver
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
        while (type != null && type != typeof(object))
        {
            var current = type.IsGenericType ? type.GetGenericTypeDefinition() : type;
            if (current == rawGeneric)
            {
                return true;
            }

            type = type.BaseType!;
        }

        return false;
    }
}

