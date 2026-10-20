import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display

df = pd.read_csv("/Users/williambarlow/Library/CloudStorage/OneDrive-UniversityofVirginia/Second Year/First Semester/BME 2315/Module 2/Untitled/Module2_FSHD_Dataset.csv")

visits = ["baseline", "y1", "y2"]

time_cols = [f"lag_months_{v}" for v in visits]

measure_cols = {
    "FF": [f"ff_{v}_R" for v in visits],
    "TUG": [f"tug_time_{v}" for v in visits],
}

target_times = np.array([0.0, 11.15, 22.13])
time_labels = ["Baseline", "T1", "T2"]

measure_names = {
    "FF": "Right-leg fat fraction",
    "TUG": "TUG time",
}

axis_labels = {
    "FF": "Right-leg fat fraction (%)",
    "TUG": "TUG time (s)",
}

print(f"Number of subjects: {len(df)}")
display(df.head())


required_cols = (
    ["subject_id"]
    + time_cols
    + measure_cols["FF"]
    + measure_cols["TUG"]
)

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

if df["subject_id"].isna().any():
    raise ValueError("Some subjects have no subject_id.")

if df["subject_id"].duplicated().any():
    raise ValueError("Duplicate subject IDs found.")

# Convert measurement and time columns to numbers.
# Missing values remain NaN.
for col in required_cols[1:]:
    df[col] = pd.to_numeric(df[col], errors="raise")

availability = []

for visit in visits:
    ff_col = f"ff_{visit}_R"
    tug_col = f"tug_time_{visit}"

    availability.append({
        "Visit": visit,
        "FF available": df[ff_col].notna().sum(),
        "TUG available": df[tug_col].notna().sum(),
        "Both available": df[[ff_col, tug_col]].notna().all(axis=1).sum(),
    })

display(pd.DataFrame(availability))

# Flag duplicate recorded visit times.
duplicate_time_mask = df[time_cols].apply(
    lambda row: row.dropna().duplicated().any(),
    axis=1
)

timing_issues = df.loc[duplicate_time_mask].copy()

print("Subjects excluded because of ambiguous visit times:")
display(
    timing_issues[
        ["subject_id"] + time_cols + measure_cols["TUG"]
    ]
)

# Preserve the original data and filter the analysis copy.
df_original = df.copy()
df = df.loc[~duplicate_time_mask].copy()

print(f"Subjects remaining after timing check: {len(df)}")







def get_observed_values(row, measure):
    """Return this subject's valid times and measurements in time order."""
    t = row[time_cols].to_numpy(dtype=float)
    y = row[measure_cols[measure]].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError(
            f"Repeated visit times for subject {row['subject_id']}"
        )

    return t, y


def interpolate_measure(row, measure):
    """Estimate common-time values without extrapolating."""
    t, y = get_observed_values(row, measure)
    estimates = np.full(len(target_times), np.nan)

    # Preserve a measurement exactly at a target, even with one observation.
    for j, target in enumerate(target_times):
        exact = np.flatnonzero(np.isclose(t, target, rtol=0, atol=1e-8))

        if len(exact):
            estimates[j] = y[exact[0]]
        elif len(t) >= 2 and t[0] < target < t[-1]:
            estimates[j] = np.interp(target, t, y)

    return estimates


aligned = pd.DataFrame({"subject_id": df["subject_id"]})

for measure in ["FF", "TUG"]:
    estimates = np.vstack([
        interpolate_measure(row, measure)
        for _, row in df.iterrows()
    ])

    for j, label in enumerate(time_labels):
        aligned[f"{measure}_{label}"] = estimates[:, j]

display(aligned.head())

analysis_cols = [
    f"{measure}_{label}"
    for measure in ["FF", "TUG"]
    for label in time_labels
]

aligned["Included"] = aligned[analysis_cols].notna().all(axis=1)

cohort = aligned.loc[aligned["Included"]].copy()

print(f"Total subjects: {len(df)}")
print(f"Included in the shared cohort: {len(cohort)}")
print(f"Excluded from the shared cohort: {len(df) - len(cohort)}")

excluded = aligned.loc[~aligned["Included"], ["subject_id"]].copy()

excluded["Unavailable values"] = aligned.loc[
    ~aligned["Included"], analysis_cols
].apply(
    lambda row: ", ".join(row.index[row.isna()]),
    axis=1
)

display(excluded)

# Identify included subjects whose missing middle visit was bridged.
included_original = df[df["subject_id"].isin(cohort["subject_id"])]

for measure, cols in measure_cols.items():
    bridged = (
        included_original[cols[0]].notna()
        & included_original[cols[1]].isna()
        & included_original[cols[2]].notna()
    )
    print(f"{measure}: {bridged.sum()} subjects with a missing middle value")


for measure in ["FF", "TUG"]:
    fig, axes = plt.subplots(
        6, 5, figsize=(16, 15), sharex=True
    )
    axes = axes.ravel()

    for ax, subject_id in zip(axes, cohort["subject_id"]):
        row = df.loc[df["subject_id"] == subject_id].iloc[0]
        estimates = cohort.loc[
            cohort["subject_id"] == subject_id
        ].iloc[0]

        t, y = get_observed_values(row, measure)

        ax.plot(t, y, "o-", color="steelblue", label="Observed")
        ax.scatter(
            target_times,
            [estimates[f"{measure}_{label}"] for label in time_labels],
            color="darkorange",
            marker="x",
            s=55,
            zorder=3,
            label="Common-time value"
        )

        ax.set_title(str(subject_id), fontsize=10)
        ax.grid(alpha=0.2)

    for ax in axes[len(cohort):]:
        ax.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right")
    fig.supxlabel("Months since baseline")
    fig.supylabel(axis_labels[measure])
    fig.suptitle(
        f"{measure_names[measure]}: observed and common-time values",
        fontsize=16
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    plt.show()



descriptive_rows = []

for measure in ["FF", "TUG"]:
    for label, month in zip(time_labels, target_times):
        values = cohort[f"{measure}_{label}"]

        descriptive_rows.append({
            "Measure": measure,
            "Time": label,
            "Month": month,
            "n": len(values),
            "Mean": values.mean(),
            "SD": values.std(ddof=1),
            "Median": values.median(),
            "Minimum": values.min(),
            "Maximum": values.max(),
        })

descriptive_table = pd.DataFrame(descriptive_rows)
display(descriptive_table.round(3))



change_rows = []

for measure in ["FF", "TUG"]:
    start = cohort[f"{measure}_T1"]
    end = cohort[f"{measure}_T2"]

    change = end - start
    pct_change = 100 * change / start.where(start != 0)

    cohort[f"{measure}_change"] = change
    cohort[f"{measure}_pct_change"] = pct_change

    # Paired t-test on common-time values.
    test = stats.ttest_rel(end, start)

    # 95% confidence interval for mean absolute change.
    n = len(change)
    margin = stats.t.ppf(0.975, df=n - 1) * stats.sem(change)
    mean_change = change.mean()

    mean_pct = pct_change.mean()
    sd_pct = pct_change.std(ddof=1)

    change_rows.append({
        "Measure": measure,
        "n": n,
        "Mean absolute change": mean_change,
        "95% CI lower": mean_change - margin,
        "95% CI upper": mean_change + margin,
        "Mean % change": mean_pct,
        "SD of % change": sd_pct,
        "Mean % / SD": mean_pct / sd_pct if sd_pct > 0 else np.nan,
        "% with an increase": 100 * (change > 0).mean(),
        "t statistic": test.statistic,
        "p value": test.pvalue,
    })

change_table = pd.DataFrame(change_rows)
display(change_table.round(4))



fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for j, measure in enumerate(["FF", "TUG"]):
    differences = cohort[f"{measure}_change"]

    axes[0, j].hist(
        differences, bins=8,
        color="steelblue", edgecolor="white"
    )
    axes[0, j].axvline(0, color="black", linestyle="--")
    axes[0, j].set_title(f"{measure}: T2 minus T1")
    axes[0, j].set_xlabel(
        "FF change (percentage points)" if measure == "FF"
        else "TUG change (s)"
    )
    axes[0, j].set_ylabel("Number of subjects")

    stats.probplot(differences, dist="norm", plot=axes[1, j])
    axes[1, j].set_title(f"{measure}: Q–Q plot of paired differences")

plt.tight_layout()
plt.show()

# Compare percentage changes on the same scale.
fig, ax = plt.subplots(figsize=(7, 5))
rng = np.random.default_rng(42)

for position, measure in enumerate(["FF", "TUG"]):
    values = cohort[f"{measure}_pct_change"].dropna()

    ax.scatter(
        position + rng.uniform(-0.08, 0.08, len(values)),
        values,
        alpha=0.65
    )
    ax.errorbar(
        position, values.mean(),
        yerr=values.std(ddof=1),
        fmt="ks", capsize=7
    )

ax.axhline(0, color="gray", linestyle="--")
ax.set_xticks([0, 1], ["Right-leg FF", "TUG"])
ax.set_ylabel("Change from T1 to T2 (%)")
ax.set_title("Individual percentage changes and mean ± SD")
plt.tight_layout()
plt.show()


def regression_summary(x, y):
    """Summarize a regression using valid paired observations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n": len(x), "Pearson r": np.nan,
            "R squared": np.nan, "Slope": np.nan,
            "Intercept": np.nan, "p value": np.nan
        }

    fit = stats.linregress(x, y)

    return {
        "n": len(x),
        "Pearson r": fit.rvalue,
        "R squared": fit.rvalue**2,
        "Slope": fit.slope,
        "Intercept": fit.intercept,
        "p value": fit.pvalue,
    }


correlation_rows = []
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4), sharex=True, sharey=True
)

for ax, label, month in zip(axes, time_labels, target_times):
    x = cohort[f"FF_{label}"]
    y = cohort[f"TUG_{label}"]

    result = regression_summary(x, y)
    correlation_rows.append({"Month": month, **result})

    ax.scatter(x, y, alpha=0.75)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        grid,
        result["Intercept"] + result["Slope"] * grid,
        color="red"
    )
    ax.set_title(
        f"{month:g} months: r = {result['Pearson r']:.2f}"
    )
    ax.set_xlabel("Right-leg fat fraction (%)")

axes[0].set_ylabel("TUG time (s)")
plt.tight_layout()
plt.show()

correlation_table = pd.DataFrame(correlation_rows)
display(correlation_table.round(4))



change_correlation = regression_summary(
    cohort["FF_change"],
    cohort["TUG_change"]
)

display(pd.DataFrame([change_correlation]).round(4))

x = cohort["FF_change"]
y = cohort["TUG_change"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, alpha=0.8)

grid = np.linspace(x.min(), x.max(), 100)
ax.plot(
    grid,
    change_correlation["Intercept"]
    + change_correlation["Slope"] * grid,
    color="red"
)

ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Right-leg FF change (percentage points)")
ax.set_ylabel("TUG change (s)")
ax.set_title("Association between changes: 11.15 to 22.13 months")
plt.tight_layout()
plt.show()



subject_rows = []

for _, row in df.iterrows():
    ff = row[measure_cols["FF"]].to_numpy(dtype=float)
    tug = row[measure_cols["TUG"]].to_numpy(dtype=float)
    t = row[time_cols].to_numpy(dtype=float)

    valid = np.isfinite(ff) & np.isfinite(tug) & np.isfinite(t)
    n = int(valid.sum())

    r = np.nan

    # Two points would give a trivial correlation of +1 or -1.
    # Require three original paired observations.
    if n == 3 and np.ptp(ff[valid]) > 0 and np.ptp(tug[valid]) > 0:
        r = stats.pearsonr(ff[valid], tug[valid]).statistic

    subject_rows.append({
        "subject_id": row["subject_id"],
        "Original paired observations": n,
        "Descriptive Pearson r": r,
    })

subject_correlation_table = pd.DataFrame(subject_rows)
display(subject_correlation_table.round(3))
