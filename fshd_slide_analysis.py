# %% 1. Imports and settings
"""Run in VS Code as a script or use # %% cells with Run Cell.
Install: python -m pip install pandas numpy matplotlib scipy
Place the original CSV beside this script, or pass its path on the command line.
Outputs: annotated PNG/PDF figures and CSV result tables in fshd_results/.
All p-values are two-sided and exploratory. Interpolation uncertainty is not
propagated. FF units are assumed percent and TUG units seconds; verify metadata.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('csv', nargs='?', default=None)
args, _ = parser.parse_known_args()
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
CSV = Path("/Users/williambarlow/Library/CloudStorage/OneDrive-UniversityofVirginia/Second Year/First Semester/BME 2315/Module 2/Untitled/Module2_FSHD_Dataset.csv")
OUT = HERE / 'fshd_results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12})
VISITS = ['baseline', 'y1', 'y2']
TC = [f'lag_months_{v}' for v in VISITS]
COLS = {'FF': [f'ff_{v}_R' for v in VISITS],
        'TUG': [f'tug_time_{v}' for v in VISITS]}
TARGETS = [0., 11.15, 22.13]
LABELS = ['Baseline', 'T1', 'T2']
AXLABEL = {'FF': 'Right-leg fat fraction (%)', 'TUG': 'TUG time (s)'}
UNITS = {'FF': 'percentage points', 'TUG': 's'}

def pf(p):
    return f'{p:.2e}' if p < .001 else f'{p:.4f}'

def save(fig, name, caption):
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    with (OUT / 'figure_captions.txt').open('a') as f:
        f.write(f'{name}: {caption}\n\n')
    plt.show()
    plt.close(fig)

def note(ax, text):
    ax.axis('off')
    ax.text(0, 1, text, va='top', fontsize=12, linespacing=1.55)

(OUT / 'figure_captions.txt').write_text('')

# %% 2. Load, describe, and flag timing issues
raw = pd.read_csv(CSV)
if raw.subject_id.isna().any() or raw.subject_id.duplicated().any():
    raise ValueError('Missing or duplicate subject identifiers')
for col in TC + COLS['FF'] + COLS['TUG']:
    raw[col] = pd.to_numeric(raw[col], errors='raise')
print('Original cohort:', len(raw))
print(raw.sex.value_counts())
print(raw.age_baseline.agg(['min', 'max', 'mean', 'median']))
ambiguous = raw[TC].apply(lambda r: r.dropna().duplicated().any(), axis=1)
raw.loc[ambiguous].to_csv(OUT / 'excluded_ambiguous_times.csv', index=False)
df = raw.loc[~ambiguous].copy()

# %% 3. Linear interpolation and shared cohort

def observed(row, measure):
    t = row[TC].to_numpy(float)
    y = row[COLS[measure]].to_numpy(float)
    k = np.isfinite(t) & np.isfinite(y)
    t, y = t[k], y[k]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError(f'Ambiguous times: {row.subject_id}')
    return t, y

aligned = pd.DataFrame({'subject_id': df.subject_id.to_numpy()})
for measure in COLS:
    values = []
    for _, row in df.iterrows():
        t, y = observed(row, measure)
        est = np.full(3, np.nan)
        for j, target in enumerate(TARGETS):
            exact = np.flatnonzero(np.isclose(t, target, rtol=0, atol=1e-8))
            if len(exact):
                est[j] = y[exact[0]]
            elif len(t) >= 2 and t[0] < target < t[-1]:
                est[j] = np.interp(target, t, y)
        values.append(est)
    for j, label in enumerate(LABELS):
        aligned[f'{measure}_{label}'] = np.asarray(values)[:, j]
needed = [f'{m}_{l}' for m in COLS for l in LABELS]
aligned['included'] = aligned[needed].notna().all(axis=1)
cohort = aligned.loc[aligned.included].copy()
aligned.to_csv(OUT / 'aligned_values_and_inclusion.csv', index=False)
print('Shared cohort:', len(cohort))

# %% 4. Verify interpolation with measured points and estimates
sid = cohort.subject_id.iloc[0]
row = df.loc[df.subject_id == sid].iloc[0]
est = cohort.loc[cohort.subject_id == sid].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, m in zip(axes, COLS):
    t, y = observed(row, m)
    ax.plot(t, y, 'o-', label='Observed measurements')
    yy = [est[f'{m}_{l}'] for l in LABELS]
    ax.scatter(TARGETS, yy, marker='x', s=100, color='darkorange',
               label='Common-time values', zorder=3)
    for x, val in zip(TARGETS, yy):
        ax.annotate(f'{val:.2f}', (x, val), xytext=(5, 9), textcoords='offset points')
    ax.set(xlabel='Months since baseline', ylabel=AXLABEL[m], title=f'Subject {sid}: {m}')
    ax.legend(fontsize=9)
fig.suptitle('Linear interpolation: 0, 11.15, and 22.13 months')
fig.tight_layout()
save(fig, '01_interpolation_example', 'Crosses are estimated common-time values; circles are original measurements. Straight segments assume linear change; no extrapolation is used.')

# %% 5. Paired tests, confidence intervals, changes, and Holm adjustment
rows = []
for start, interval in [('T1', '11.15 to 22.13 months'), ('Baseline', '0 to 22.13 months')]:
    for m in COLS:
        a, b = cohort[f'{m}_{start}'], cohort[f'{m}_T2']
        d = b - a
        pc = 100 * d / a.where(a != 0)
        n = len(d)
        margin = stats.t.ppf(.975, n-1) * stats.sem(d)
        test = stats.ttest_rel(b, a)
        rows.append(dict(measure=m, interval=interval, n=n,
            mean_start=a.mean(), mean_end=b.mean(), mean_change=d.mean(),
            sd_change=d.std(ddof=1), ci_low=d.mean()-margin, ci_high=d.mean()+margin,
            mean_pct=pc.mean(), sd_pct=pc.std(ddof=1),
            mean_pct_div_sd=pc.mean()/pc.std(ddof=1),
            t=test.statistic, df=n-1, p=test.pvalue,
            pct_increased=100*(d>0).mean()))
        if start == 'T1':
            cohort[f'{m}_change'] = d
            cohort[f'{m}_pct_change'] = pc
summary = pd.DataFrame(rows)
# Explicit family: all four paired t-tests shown here.
pvals = summary.p.to_numpy()
order = np.argsort(pvals)
adjusted = np.minimum(1, np.maximum.accumulate((len(pvals)-np.arange(len(pvals)))*pvals[order]))
holm = np.empty(len(pvals)); holm[order] = adjusted
summary['p_holm_four_tests'] = holm
summary.to_csv(OUT / 'paired_test_results.csv', index=False)
cohort.to_csv(OUT / 'shared_cohort_changes.csv', index=False)
for m in COLS:
    sub = summary[summary.measure == m].reset_index(drop=True)
    fig, (ax, box) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios':[1.1, 1.2]})
    x = sub.mean_change.to_numpy()
    ax.errorbar(x, [1, 0], xerr=np.vstack([x-sub.ci_low, sub.ci_high-x]), fmt='o', capsize=7, color='steelblue')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set(yticks=[1, 0], yticklabels=['T1 to T2', 'Baseline to T2'], ylim=(-.6,1.6),
           xlabel=f'Mean change ({UNITS[m]})', title=f'{m}: mean change and 95% CI')
    lines = []
    for _, r in sub.iterrows():
        lines.append(f'{r.interval}\nn = {r.n}; t({r.df}) = {r.t:.3f}\n'
            f'Mean change = {r.mean_change:.3f}\n95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]\n'
            f'p = {pf(r.p)}; Holm p = {pf(r.p_holm_four_tests)}')
    note(box, '\n\n'.join(lines)+'\n\nPaired t-tests; Holm family = 4 tests.\nBaseline comparison is additional/exploratory.')
    fig.tight_layout()
    save(fig, f'02_{m}_paired_changes', 'Mean absolute changes with 95% t confidence intervals for the shared cohort. Positive values indicate increasing FF or longer TUG. Tests use interpolated values without propagating interpolation uncertainty.')

# %% 6. Percentage-change distributions and lecture consistency statistic
rng = np.random.default_rng(42)
fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax, m in zip(axes, COLS):
    r = summary[(summary.measure==m)&(summary.interval=='11.15 to 22.13 months')].iloc[0]
    vals = cohort[f'{m}_pct_change']
    ax.scatter(rng.uniform(-.07,.07,len(vals)), vals, alpha=.7)
    ax.errorbar(.25, vals.mean(), yerr=vals.std(ddof=1), fmt='ks', capsize=7)
    ax.axhline(0,color='gray',linestyle='--')
    ax.set(xticks=[0,.25],xticklabels=['Subjects','Mean +/- SD'],ylabel='Change (%)',title=f'{m}: T1 to T2',xlim=(-.18,1.0))
    ax.text(.98,.97,f'n = {r.n}\nMean = {r.mean_pct:.2f}%\nSD = {r.sd_pct:.2f}%\nMean/SD = {r.mean_pct_div_sd:.3f}\nIncreased: {r.pct_increased:.0f}%', transform=ax.transAxes,ha='right',va='top',fontsize=11)
fig.tight_layout()
save(fig,'03_percentage_changes','Dots are subject percentage changes. Squares and bars show mean +/- sample SD, not confidence intervals. Mean/SD is a descriptive consistency statistic; it is not a reliability estimate.')

# %% 7. Regression charts: equation, r, R-squared, p, slope CI, Spearman
regrows = []
def regression_plot(x, y, title, xlabel, ylabel, name):
    pair = pd.DataFrame({'x':np.asarray(x),'y':np.asarray(y)}).dropna()
    x,y = pair.x.to_numpy(),pair.y.to_numpy()
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        print('Skipping regression with insufficient variation:', name); return
    fit = stats.linregress(x,y)
    rho = stats.spearmanr(x,y)
    crit = stats.t.ppf(.975,len(x)-2)
    slo,shi = fit.slope-crit*fit.stderr,fit.slope+crit*fit.stderr
    grid = np.linspace(x.min(),x.max(),150)
    fitted = fit.intercept+fit.slope*grid
    residual_sd = np.sqrt(np.sum((y-fit.intercept-fit.slope*x)**2)/(len(x)-2))
    se = residual_sd*np.sqrt(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2))
    fig,(ax,box)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.6,1]})
    ax.scatter(x,y,alpha=.8)
    ax.plot(grid,fitted,color='firebrick',label='OLS fit')
    ax.fill_between(grid,fitted-crit*se,fitted+crit*se,color='firebrick',alpha=.13,label='95% CI for mean response')
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.legend(fontsize=9)
    note(box,f'n = {len(x)}\ny = {fit.slope:.4f}x {fit.intercept:+.4f}\n'
        f'Pearson r = {fit.rvalue:.3f}\nR-squared = {fit.rvalue**2:.3f}\n'
        f'Slope p = {pf(fit.pvalue)}\nSlope 95% CI:\n[{slo:.4f}, {shi:.4f}]\n\n'
        f'Spearman rho = {rho.statistic:.3f}\nSpearman p = {pf(rho.pvalue)}\n\n'
        'Slope p also tests Pearson r = 0.\nUnadjusted exploratory p-values.')
    fig.tight_layout()
    save(fig,name,'Each point represents one subject. Ordinary least-squares line with a pointwise 95% confidence band for the mean response, not a prediction interval. Equation uses the units on the axes.')
    regrows.append(dict(analysis=name,n=len(x),slope=fit.slope,intercept=fit.intercept,
        pearson_r=fit.rvalue,r_squared=fit.rvalue**2,p=fit.pvalue,
        slope_ci_low=slo,slope_ci_high=shi,spearman_rho=rho.statistic,spearman_p=rho.pvalue))
for label,month in zip(LABELS,TARGETS):
    regression_plot(cohort[f'FF_{label}'],cohort[f'TUG_{label}'],f'FF and TUG at {month:g} months',AXLABEL['FF'],AXLABEL['TUG'],f'04_regression_{label}')
regression_plot(cohort.FF_change,cohort.TUG_change,'Association between T1-to-T2 changes','FF change (percentage points)','TUG change (s)','05_change_regression')
pd.DataFrame(regrows).to_csv(OUT/'regression_results.csv',index=False)

# %% 8. Paired-difference diagnostics and influence checks
fig,axes=plt.subplots(2,2,figsize=(11,8))
for j,m in enumerate(COLS):
    d=cohort[f'{m}_change']
    axes[0,j].hist(d,bins=8,edgecolor='white')
    axes[0,j].set(title=f'{m}: paired differences',xlabel=f'Change ({UNITS[m]})',ylabel='Subjects')
    axes[0,j].axvline(0,color='gray',linestyle='--')
    stats.probplot(d,dist='norm',plot=axes[1,j])
    axes[1,j].set_title(f'{m}: Q-Q diagnostic')
fig.tight_layout()
save(fig,'06_diagnostics','Diagnostics concern within-subject differences. Q-Q reference line is diagnostic, not the scientific FF-TUG regression; its p-value is not an endpoint test.')
influence=[]
for idx,r in cohort.iterrows():
    sub=cohort.drop(index=idx)
    influence.append({'omitted_subject':r.subject_id,'pearson_r':stats.pearsonr(sub.FF_change,sub.TUG_change).statistic})
pd.DataFrame(influence).to_csv(OUT/'leave_one_out_correlations.csv',index=False)

# %% 9. Complete-visit sensitivity and descriptive individual correlations
complete_ids=df.dropna(subset=TC+COLS['FF']+COLS['TUG']).subject_id
complete=cohort[cohort.subject_id.isin(complete_ids)]
sensitivity=[]
for m in COLS:
    test=stats.ttest_rel(complete[f'{m}_T2'],complete[f'{m}_T1'])
    sensitivity.append({'measure':m,'n':len(complete),'t':test.statistic,'p':test.pvalue})
pd.DataFrame(sensitivity).to_csv(OUT/'complete_visit_sensitivity.csv',index=False)
individual=[]
for _,r in df.iterrows():
    x=r[COLS['FF']].to_numpy(float);y=r[COLS['TUG']].to_numpy(float);t=r[TC].to_numpy(float)
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(t)
    rr=np.nan
    if valid.sum()==3 and np.ptp(x[valid])>0 and np.ptp(y[valid])>0:
        rr=stats.pearsonr(x[valid],y[valid]).statistic
    individual.append({'subject_id':r.subject_id,'original_pairs':valid.sum(),'descriptive_r':rr})
pd.DataFrame(individual).to_csv(OUT/'individual_descriptive_correlations.csv',index=False)
print(summary.round(4).to_string(index=False))
print('Charts and tables saved in:',OUT)
