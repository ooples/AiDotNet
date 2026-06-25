---
title: "ChannelDropout<T>"
description: "Randomly zeros out one or more color channels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Randomly zeros out one or more color channels.

## How It Works

ChannelDropout sets entire color channels to a fill value (default 0), forcing the
model to learn from the remaining channels. This prevents over-reliance on any single
color channel.

