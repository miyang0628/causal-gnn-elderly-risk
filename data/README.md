# Data

The study uses a public in-home monitoring dataset of **single-occupancy
elderly households**. Each household contributes a structured multi-sensor
time series at a 10-minute interval:

- **Environmental (EM)**: temperature, humidity, illuminance, activity/motion IR, CO2, TVOC
- **Physiological (SM)**: heart rate, breathing rate, SpO2, sleep phase, sleep score, stress index
- **Emergency (ER)**: button, shout
- **Risk label** per timestep (mapped to a binary risk target).

## Layout

```
data/
└── processed/
    └── df_new.pkl      # parsed table: 27 households, 109,819 rows
```

`processed/df_new.pkl` is the parsed, cached table consumed by every notebook.
The raw source files are **not redistributed** here due to licensing; obtain the
dataset from its original provider if you need to regenerate `df_new.pkl`.
