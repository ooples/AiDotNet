using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace AiDotNet.Serialization;

/// <summary>
/// A diagnostic converter for regression models that provides detailed debugging
/// about the serialization process.
/// </summary>
/// <remarks>
/// <para>
/// This converter extends the diagnostic base converter to provide regression model-specific
/// type checking and naming.
/// </para>
/// <para><b>For Beginners:</b> This is a special debug-focused converter for regression models.
/// It helps identify serialization issues by logging detailed information about what's happening
/// during the save/load process.
/// </para>
/// </remarks>
public class DiagnosticRegressionModelConverter : DiagnosticModelConverter
{
    /// <summary>
    /// Initializes a new instance of the DiagnosticRegressionModelConverter class.
    /// </summary>
    public DiagnosticRegressionModelConverter()
    {
        // Set a model-specific debug log path
        DebugLogPath = "regression_model_debug.log";
        Console.WriteLine("========= REGRESSION MODEL CONVERTER INITIALIZED =========");
    }

    /// <summary>
    /// Determines if a type is a regression model or derives from RegressionModelBase.
    /// </summary>
    protected override bool IsTypeSupported(Type objectType)
    {
        if (objectType == null)
            return false;

        // Check all base types
        Type? currentType = objectType;
        while (currentType != null && currentType != typeof(object))
        {
            // Check if this is a generic type
            if (currentType.IsGenericType)
            {
                Type genericTypeDef = currentType.GetGenericTypeDefinition();
                string fullName = genericTypeDef.FullName ?? string.Empty;

                // Check if this is a RegressionModelBase
                if (fullName.Contains("RegressionModelBase"))
                {
                    Console.WriteLine($"Type {objectType.Name} is a RegressionModelBase derivative");
                    return true;
                }
            }

            // Check interfaces for IRegressionModel
            foreach (var iface in currentType.GetInterfaces())
            {
                if (iface.IsGenericType)
                {
                    Type genericIFaceDef = iface.GetGenericTypeDefinition();
                    string fullName = genericIFaceDef.FullName ?? string.Empty;

                    if (fullName.Contains("IRegressionModel"))
                    {
                        Console.WriteLine($"Type {objectType.Name} implements IRegressionModel");
                        return true;
                    }
                }
            }

            // Check base type
            currentType = currentType.BaseType;
        }

        return false;
    }

    /// <summary>
    /// Gets the base model type name for diagnostic purposes.
    /// </summary>
    protected override string GetBaseModelTypeName()
    {
        return "RegressionModel";
    }
}