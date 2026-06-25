---
title: "SpikingNeuronType"
description: "Specifies the type of spiking neuron model to use in neuromorphic computing simulations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of spiking neuron model to use in neuromorphic computing simulations.

## How It Works

**For Beginners:** Spiking neurons are AI components that work more like real brain cells.

Traditional AI neurons output continuous values (like 0.7), but spiking neurons work with
discrete "spikes" or pulses of activity (like a real neuron firing). This makes them more
biologically realistic and potentially more efficient for certain tasks.

Think of regular AI neurons as light bulbs with dimmers that can be set to any brightness,
while spiking neurons are more like light bulbs that either flash brightly or stay off.

Different spiking neuron types represent different mathematical models of how real neurons work,
with varying levels of biological accuracy and computational complexity.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdaptiveExponential` | A model that combines exponential spike generation with adaptive threshold mechanisms. |
| `HodgkinHuxley` | A detailed biophysical model that accurately represents ion channel dynamics in neurons. |
| `IntegrateAndFire` | A basic neuron model that accumulates input until reaching a threshold, then fires. |
| `Izhikevich` | A computationally efficient model that can reproduce many behaviors of biological neurons. |
| `LeakyIntegrateAndFire` | A simplified neuron model that accumulates input and "leaks" voltage over time. |

