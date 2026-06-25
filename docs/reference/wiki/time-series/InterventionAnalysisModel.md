---
title: "InterventionAnalysisModel"
description: "Represents a model that analyzes and forecasts time series data with interventions or structural changes."
section: "Reference"
---

_Time-Series Models_

Represents a model that analyzes and forecasts time series data with interventions or structural changes.

## For Beginners

Intervention analysis helps understand how specific events affect your data patterns. Think of it like analyzing sales data: - You've been tracking monthly sales that follow a regular pattern - Then you run a major marketing campaign in July - Sales jump significantly and stay higher for several months This model helps you: - Measure exactly how much the marketing campaign boosted sales - Understand how long the effect lasted - Make better predictions by accounting for these special events Other examples of interventions include policy changes, natural disasters, product launches, or any significant event that changes the normal pattern of your data.

## How It Works

Intervention analysis combines ARIMA (AutoRegressive Integrated Moving Average) modeling with the ability to account for external events or interventions that cause structural changes in the time series. These interventions can be temporary or permanent and can have various effects on the level, trend, or seasonality of the data.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new InterventionAnalysisModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"InterventionAnalysisModel: forecast {forecast.Length} steps.");
```

