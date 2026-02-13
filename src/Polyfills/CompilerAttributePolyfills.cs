// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

// Polyfills for compiler-required attributes to support .NET Framework 4.6.2 and 4.7.1
// These attributes are required by the C# compiler for modern language features

#if !NETCOREAPP3_0_OR_GREATER && !NETSTANDARD2_1_OR_GREATER

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Polyfill for MethodImplOptions.AggressiveOptimization which was introduced in .NET Core 3.0.
    /// This provides the constant value (512) that can be used with [MethodImpl] attribute.
    /// </summary>
    /// <remarks>
    /// In .NET Framework, this flag has no effect at runtime, but it allows code to compile.
    /// The JIT compiler in .NET Framework will simply ignore this flag.
    /// </remarks>
    public static class MethodImplOptionsEx
    {
        /// <summary>
        /// Specifies that the method should be optimized aggressively by the JIT compiler.
        /// Value: 512 (0x200). Only effective in .NET Core 3.0+; ignored in .NET Framework.
        /// </summary>
        public const MethodImplOptions AggressiveOptimization = (MethodImplOptions)512;
    }
    /// <summary>
    /// Reserved for use by a compiler for tracking metadata.
    /// This class should not be used by developers in source code.
    /// Used to mark init-only setters.
    /// </summary>
    [AttributeUsage(AttributeTargets.All, Inherited = false)]
    internal sealed class IsExternalInit : Attribute
    {
    }

    /// <summary>
    /// Specifies that a type has required members or that a member is required.
    /// Used by the C# compiler for the 'required' keyword (C# 11).
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    internal sealed class RequiredMemberAttribute : Attribute
    {
    }

    /// <summary>
    /// Indicates the attributed type is to be used in a compiler-generated state machine.
    /// Used by async methods.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
    internal sealed class CompilerFeatureRequiredAttribute : Attribute
    {
        public CompilerFeatureRequiredAttribute(string featureName)
        {
            FeatureName = featureName;
        }

        public string FeatureName { get; }
        public bool IsOptional { get; set; }
    }
}

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Polyfill for CallerArgumentExpressionAttribute, introduced in C# 10 / .NET 6.
    /// Allows methods to capture the expression passed to a parameter as a string.
    /// When using a C# compiler that supports CallerArgumentExpression (for example, Roslyn with
    /// <c>LangVersion</c> 10 or later), the compiler will populate this automatically even when
    /// targeting older frameworks such as .NET Framework. With older compilers or language versions
    /// that do not support CallerArgumentExpression, callers must pass <c>nameof(param)</c> explicitly.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
    internal sealed class CallerArgumentExpressionAttribute : Attribute
    {
        public CallerArgumentExpressionAttribute(string parameterName)
        {
            ParameterName = parameterName;
        }

        public string ParameterName { get; }
    }
}

namespace System.Diagnostics.CodeAnalysis
{
    /// <summary>
    /// Specifies that this constructor sets all required members for the current type,
    /// and callers do not need to set any required members themselves.
    /// </summary>
    [AttributeUsage(AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    internal sealed class SetsRequiredMembersAttribute : Attribute
    {
    }

    /// <summary>
    /// Indicates that the specified parameter will be non-null when the method returns the specified value.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
    public sealed class NotNullWhenAttribute : Attribute
    {
        public NotNullWhenAttribute(bool returnValue)
        {
            ReturnValue = returnValue;
        }

        public bool ReturnValue { get; }
    }

    /// <summary>
    /// Specifies that an output is not null even if the corresponding type allows it.
    /// Applied to Guard.NotNull's parameter to tell the compiler the value is non-null after the call.
    /// </summary>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter | AttributeTargets.ReturnValue, Inherited = false)]
    internal sealed class NotNullAttribute : Attribute
    {
    }
}

#endif
