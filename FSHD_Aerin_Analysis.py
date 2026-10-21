
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv("C:\\Users\\aerin\\OneDrive\\Desktop\\COMP BME 2315 Fall 2026\\Module 2\\BME2315_Module2\\Module2_FSHD_Dataset.csv")

# Display dataset information
print("Number of subjects:", df["subject_id"].nunique())
print("Dataset shape:", df.shape)

df.head()


# Define visits and their corresponding columns
visits = {
    "Baseline": {
        "ff": "ff_baseline_R",
        "tug": "tug_time_baseline",
        "months": "lag_months_baseline"
    },
    "Year 1": {
        "ff": "ff_y1_R",
        "tug": "tug_time_y1",
        "months": "lag_months_y1"
    },
    "Year 2": {
        "ff": "ff_y2_R",
        "tug": "tug_time_y2",
        "months": "lag_months_y2"
    }
}

# Create a longitudinal dataset
long_data = []

for _, subject in df.iterrows():
    for visit, columns in visits.items():
        long_data.append({
            "subject_id": subject["subject_id"],
            "visit": visit,
            "months": subject[columns["months"]],
            "fat_fraction": subject[columns["ff"]],
            "tug_time": subject[columns["tug"]]
        })

long_df = pd.DataFrame(long_data)

# Sort by subject and actual time
long_df = long_df.sort_values(
    ["subject_id", "months"]
).reset_index(drop=True)

print(long_df.head(9))



# Plot individual subject trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for subject_id, group in long_df.groupby("subject_id"):

    # Right-limb fat fraction
    axes[0].plot(
        group["months"],
        group["fat_fraction"],
        marker="o",
        alpha=0.4
    )

    # TUG time
    axes[1].plot(
        group["months"],
        group["tug_time"],
        marker="o",
        alpha=0.4
    )

axes[0].set_title("Right-Limb Fat Fraction Over Time")
axes[0].set_xlabel("Months Since Baseline")
axes[0].set_ylabel("Fat Fraction")

axes[1].set_title("TUG Time Over Time")
axes[1].set_xlabel("Months Since Baseline")
axes[1].set_ylabel("TUG Time (seconds)")

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Common time points: baseline through 24 months
common_months = np.arange(0, 25, 1)

interpolated_data = []

for subject_id, group in long_df.groupby("subject_id"):

    group = group.sort_values("months")

    for variable in ["fat_fraction", "tug_time"]:

        # Keep only valid measurements
        valid = group[["months", variable]].dropna()
        valid = valid.drop_duplicates(subset="months")

        # Need at least 2 points for interpolation
        if len(valid) < 2:
            continue

        x = valid["months"].to_numpy()
        y = valid[variable].to_numpy()

        # Interpolate only within observed time range
        valid_months = common_months[
            (common_months >= x.min()) &
            (common_months <= x.max())
        ]

        estimated_values = np.interp(
            valid_months, x, y
        )

        for month, value in zip(
            valid_months, estimated_values
        ):
            interpolated_data.append({
                "subject_id": subject_id,
                "months": month,
                "variable": variable,
                "value": value
            })

interp_df = pd.DataFrame(interpolated_data)

# Convert back to paired FF and TUG columns
interp_wide = interp_df.pivot(
    index=["subject_id", "months"],
    columns="variable",
    values="value"
).reset_index()

# Only retain months with both measurements
interp_wide = interp_wide.dropna(
    subset=["fat_fraction", "tug_time"]
)

print(interp_wide.head(10))



# Select one subject as an example
subject_id = df["subject_id"].iloc[0]

original = long_df[
    long_df["subject_id"] == subject_id
]

interpolated = interp_wide[
    interp_wide["subject_id"] == subject_id
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

variables = [
    ("fat_fraction", "Right-Limb Fat Fraction"),
    ("tug_time", "TUG Time (seconds)")
]

for ax, (variable, title) in zip(axes, variables):

    ax.plot(
        interpolated["months"],
        interpolated[variable],
        "-",
        label="Interpolated",
        color="steelblue"
    )

    ax.scatter(
        original["months"],
        original[variable],
        color="darkorange",
        s=65,
        label="Observed",
        zorder=3
    )

    ax.set_title(title + " - " + str(subject_id))
    ax.set_xlabel("Months Since Baseline")
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()




correlation_results = []

for subject_id, group in long_df.groupby("subject_id"):

    # Use actual paired observations, not interpolated points
    paired = group[
        ["fat_fraction", "tug_time"]
    ].dropna()

    # At least 3 paired observations are needed
    if len(paired) < 3:
        continue

    # Check that both variables vary
    if (
        paired["fat_fraction"].nunique() < 2 or
        paired["tug_time"].nunique() < 2
    ):
        continue

    r, p = pearsonr(
        paired["fat_fraction"],
        paired["tug_time"]
    )

    correlation_results.append({
        "subject_id": subject_id,
        "pearson_r": r,
        "p_value": p,
        "n_observations": len(paired)
    })

correlation_df = pd.DataFrame(correlation_results)

print(correlation_df)



# Sort subjects by correlation strength
correlation_sorted = correlation_df.sort_values(
    "pearson_r"
)

plt.figure(figsize=(12, 7))

plt.barh(
    correlation_sorted["subject_id"],
    correlation_sorted["pearson_r"],
    color="steelblue"
)

plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Pearson Correlation Coefficient (r)")
plt.ylabel("Subject ID")
plt.title("Individual Correlations: Right-Limb FF vs. TUG")

plt.xlim(-1.1, 1.1)
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()



# Extract baseline measurements
baseline = interp_wide[
    interp_wide["months"] == 0
][["subject_id", "fat_fraction", "tug_time"]]

baseline = baseline.rename(columns={
    "fat_fraction": "baseline_ff",
    "tug_time": "baseline_tug"
})

# Extract follow-up measurements at months 12 and 24
followup = interp_wide[
    interp_wide["months"].isin([12, 24])
].copy()

# Merge with baseline values
change_df = followup.merge(
    baseline,
    on="subject_id",
    how="inner"
)

# Calculate changes relative to baseline
change_df["delta_ff"] = (
    change_df["fat_fraction"] -
    change_df["baseline_ff"]
)

change_df["delta_tug"] = (
    change_df["tug_time"] -
    change_df["baseline_tug"]
)

print(change_df[
    ["subject_id", "months", "delta_ff", "delta_tug"]
].head(10))




#Correlate changes in right-limb fat fraction with changes in TUG time at 12 and 24 months
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, month in zip(axes, [12, 24]):

    subset = change_df[
        change_df["months"] == month
    ].dropna(subset=["delta_ff", "delta_tug"])

    ax.scatter(
        subset["delta_ff"],
        subset["delta_tug"],
        alpha=0.7,
        color="steelblue"
    )

    if (
        len(subset) >= 3 and
        subset["delta_ff"].nunique() > 1 and
        subset["delta_tug"].nunique() > 1
    ):
        r, p = pearsonr(
            subset["delta_ff"],
            subset["delta_tug"]
        )

        # Regression line for visualization
        slope, intercept = np.polyfit(
            subset["delta_ff"],
            subset["delta_tug"],
            1
        )

        x_line = np.linspace(
            subset["delta_ff"].min(),
            subset["delta_ff"].max(),
            100
        )

        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="darkorange"
        )

        ax.set_title(
            f"Month {month}: r={r:.3f}, p={p:.3f}, n={len(subset)}"
        )

        print(
            f"Month {month}: r={r:.3f}, "
            f"p={p:.4f}, n={len(subset)}"
        )

    ax.set_xlabel("Change in Right-Limb Fat Fraction")
    ax.set_ylabel("Change in TUG Time (seconds)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
