using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace AiDotNet.Serialization;

/// <summary>
/// A diagnostic converter for time series models that provides detailed debugging
/// about the serialization process.
/// </summary>
/// <remarks>
/// <para>
/// This converter extends the diagnostic base converter to provide time series model-specific
/// type checking and naming.
/// </para>
/// <para><b>For Beginners:</b> This is a special debug-focused converter for time series models.
/// It helps identify serialization issues by logging detailed information about what's happening
/// during the save/load process.
/// </para>
/// </remarks>
public class DiagnosticTimeSeriesModelConverter : DiagnosticModelConverter
{
    /// <summary>
    /// Initializes a new instance of the DiagnosticTimeSeriesModelConverter class.
    /// </summary>
    public DiagnosticTimeSeriesModelConverter()
    {
        // Set a model-specific debug log path
        DebugLogPath = "time_series_model_debug.log";
        Console.WriteLine("========= TIME SERIES MODEL CONVERTER INITIALIZED =========");
    }

    /// <summary>
    /// Determines if a type is a time series model or derives from TimeSeriesModelBase.
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

                // Check if this is a TimeSeriesModelBase
                if (fullName.Contains("TimeSeriesModelBase"))
                {
                    Console.WriteLine($"Type {objectType.Name} is a TimeSeriesModelBase derivative");
                    return true;
                }
            }

            // Check interfaces for ITimeSeriesModel
            foreach (var iface in currentType.GetInterfaces())
            {
                if (iface.IsGenericType)
                {
                    Type genericIFaceDef = iface.GetGenericTypeDefinition();
                    string fullName = genericIFaceDef.FullName ?? string.Empty;

                    if (fullName.Contains("ITimeSeriesModel"))
                    {
                        Console.WriteLine($"Type {objectType.Name} implements ITimeSeriesModel");
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
        return "TimeSeriesModel";
    }
}