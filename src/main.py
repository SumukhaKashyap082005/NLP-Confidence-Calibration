from calibration_analysis import run_calibration_analysis

results = run_calibration_analysis(
    "../data/train.csv",
    sample_size=1000
)

print("Calibration Analysis Results:")
for k, v in results.items():
    print(f"{k}: {v}")
