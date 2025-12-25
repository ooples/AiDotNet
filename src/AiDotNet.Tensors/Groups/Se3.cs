using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Represents an element of SE(3) with rotation and translation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class Se3<T>
{
    public So3<T> Rotation { get; }
    public Vector<T> Translation { get; }

    public Se3(So3<T> rotation, Vector<T> translation)
    {
        if (rotation.Matrix is null)
            throw new ArgumentException("Rotation matrix cannot be null (default-initialized struct).", nameof(rotation));
        Rotation = rotation;
        Translation = translation ?? throw new ArgumentNullException(nameof(translation));
        if (translation.Length != 3)
            throw new ArgumentException("SE(3) translation must be length 3.", nameof(translation));
    }
}
