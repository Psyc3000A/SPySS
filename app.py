import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import streamlit as st
from streamlit_ace import st_ace
import hashlib
import io
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import copy
from bs4 import BeautifulSoup

st.set_page_config(page_title="Student Statistics Lab", layout="wide")
st.title("Student Statistics Lab")
st.caption("pandas for descriptives • seaborn/matplotlib for plots • Pingouin and Statsmodels for inference")

# Load data
DATASETS = {
    "General teaching data": (
        "general_data.csv",
        "Descriptives, normality, correlation, t-tests, ANOVA, ANCOVA, and regression."
    ),
    "Repeated-measures data": (
        "repeated_measures_data.csv",
        "Repeated-measures ANOVA and Friedman: score, time, participant."
    ),
    "Reliability data": (
        "reliability_data.csv",
        "Cronbach alpha: select item1 through item8."
    ),
    "Categorical data": (
        "categorical_data.csv",
        "Chi-square: treatment × outcome, adherence × outcome, or smoker × outcome."
    ),
    "Nonparametric data": (
        "nonparametric_data.csv",
        "Mann-Whitney, Kruskal-Wallis, and Wilcoxon."
    ),
    "Penguins (Pingouin)": (
        None,
        "A familiar dataset for descriptives, correlations, ANOVA, and regression."
    ),
}

dataset_name = st.sidebar.selectbox("Built-in teaching dataset", list(DATASETS))
filename, dataset_note = DATASETS[dataset_name]
st.sidebar.caption(dataset_note)

uploaded = st.sidebar.file_uploader(
    "Or upload CSV, Excel, or SAV", type=["csv", "xlsx", "xls", "sav"]
)

if uploaded:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_spss(uploaded)
elif filename:
    df = pd.read_csv(Path(__file__).with_name(filename))
else:
    df = pg.read_dataset("penguins")

df.columns = [str(c).strip() for c in df.columns]
all_cols = list(df.columns)
num_cols = list(df.select_dtypes(include=np.number).columns)
st.sidebar.write(f"{len(df)} rows × {len(df.columns)} columns")

if st.checkbox("Show data preview", value=True):
    st.dataframe(df.head(50), width="stretch")

analysis = st.sidebar.selectbox("Analysis", [
    "1. Descriptive statistics",
    "2. Normality and Q-Q plot",
    "3. Correlation",
    "4. Partial correlation",
    "5. T-test",
    "6. One-way ANOVA",
    "7. Repeated-measures ANOVA",
    "8. ANCOVA",
    "9. Nonparametric test",
    "10. Linear regression",
    "11. Cronbach alpha",
    "12. Chi-square test",
    "13. Exam generator",
    "14. Statsmodels regression models"
], index=0)
st.header(analysis)
code = ""

if analysis.startswith("1."):
    vars_ = st.multiselect("Numeric variables", num_cols, default=num_cols[:4])
    code = f'''x = df[{vars_!r}]
result = pd.DataFrame({{
    "N": x.count(), "Missing": x.isna().sum(), "Mean": x.mean(),
    "SD": x.std(), "Median": x.median(), "Minimum": x.min(),
    "Maximum": x.max(), "Skewness": x.skew(), "Kurtosis": x.kurt()
}})'''

elif analysis.startswith("2."):
    x = st.selectbox("Variable", num_cols)
    code = f'''result = pg.normality(df[{x!r}].dropna())
sns.histplot(df[{x!r}].dropna(), kde=True)
plt.title("Distribution of {x}")
plt.show()
pg.qqplot(df[{x!r}].dropna())
plt.title("Q-Q plot of {x}")
plt.show()'''

elif analysis.startswith("3."):
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", num_cols)
    y = c2.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1))
    method = c3.selectbox("Method", ["pearson", "spearman", "kendall", "bicor"])
    code = f'''result = pg.corr(x=df[{x!r}], y=df[{y!r}], method={method!r})
sns.regplot(data=df, x={x!r}, y={y!r})
plt.show()'''

elif analysis.startswith("4."):
    x = st.selectbox("X", num_cols)
    y = st.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1))
    covars = st.multiselect("Covariates", [c for c in num_cols if c not in [x, y]])
    code = f'''result = pg.partial_corr(data=df, x={x!r}, y={y!r}, covar={covars!r})'''

elif analysis.startswith("5."):
    design = st.radio("Design", ["One sample", "Paired", "Independent"], horizontal=True)
    if design == "One sample":
        x = st.selectbox("Variable", num_cols)
        mu = st.number_input("Test value", value=0.0)
        code = f'''result = pg.ttest(df[{x!r}].dropna(), {mu})'''
    elif design == "Paired":
        x_default = num_cols.index("pre_score") if "pre_score" in num_cols else 0
        x = st.selectbox("Variable 1", num_cols, index=x_default)
        y_default = num_cols.index("post_score") if "post_score" in num_cols else min(1, len(num_cols) - 1)
        y = st.selectbox("Variable 2", num_cols, index=y_default)
        code = f'''d = df[[{x!r}, {y!r}]].dropna()
result = pg.ttest(d[{x!r}], d[{y!r}], paired=True)'''
    else:
        outcome = st.selectbox("Outcome", num_cols)
        group = st.selectbox("Grouping variable", [c for c in all_cols if c != outcome])
        levels = list(df[group].dropna().unique())
        chosen = st.multiselect("Choose two groups", levels, default=levels[:2])
        code = "# Choose exactly two groups."
        if len(chosen) == 2:
            g1, g2 = chosen
            code = f'''x = df.loc[df[{group!r}] == {g1!r}, {outcome!r}].dropna()
y = df.loc[df[{group!r}] == {g2!r}, {outcome!r}].dropna()
result = pg.ttest(x, y)'''

elif analysis.startswith("6."):
    dv = st.selectbox("Dependent variable", num_cols)
    between = st.selectbox("Factor", [c for c in all_cols if c != dv])
    kind = st.radio("Type", ["Classical", "Welch"], horizontal=True)
    posthoc = st.selectbox("Post hoc", ["None", "Tukey", "Games-Howell"])
    code = (f'''result = pg.anova(data=df, dv={dv!r}, between={between!r}, detailed=True)'''
            if kind == "Classical" else
            f'''result = pg.welch_anova(data=df, dv={dv!r}, between={between!r})''')
    if posthoc == "Tukey":
        code += f'''\nposthoc = pg.pairwise_tukey(data=df, dv={dv!r}, between={between!r})'''
    elif posthoc == "Games-Howell":
        code += f'''\nposthoc = pg.pairwise_gameshowell(data=df, dv={dv!r}, between={between!r})'''

elif analysis.startswith("7."):
    # Prefer plausible defaults in teaching datasets: score, time, participant.
    dv_default = num_cols.index("score") if "score" in num_cols else 0
    dv = st.selectbox("Dependent variable", num_cols, index=dv_default)

    within_options = [c for c in all_cols if c != dv]
    within_default = within_options.index("time") if "time" in within_options else 0
    within = st.selectbox("Within-subject factor", within_options, index=within_default)

    subject_options = [c for c in all_cols if c not in [dv, within]]
    subject_default = subject_options.index("participant") if "participant" in subject_options else 0
    subject = st.selectbox("Subject ID", subject_options, index=subject_default)

    code = f'''d = df[[{subject!r}, {within!r}, {dv!r}]].dropna().copy()

# Each subject must contribute data at two or more within-subject levels.
levels_per_subject = d.groupby({subject!r})[{within!r}].nunique()
valid_subjects = levels_per_subject[levels_per_subject >= 2].index
d = d[d[{subject!r}].isin(valid_subjects)]

if d.empty or d[{within!r}].nunique() < 2:
    raise ValueError(
        "Repeated-measures ANOVA requires each subject to have observations "
        "at two or more levels of the within-subject factor."
    )

result = pg.rm_anova(
    data=d, dv={dv!r}, within={within!r}, subject={subject!r}, detailed=True
)
posthoc = pg.pairwise_tests(
    data=d, dv={dv!r}, within={within!r}, subject={subject!r},
    padjust="holm"
)'''

elif analysis.startswith("8."):
    dv = st.selectbox("Dependent variable", num_cols)
    between = st.selectbox("Factor", [c for c in all_cols if c != dv])
    covars = st.multiselect("Covariates", [c for c in num_cols if c != dv])
    code = f'''result = pg.ancova(data=df, dv={dv!r}, between={between!r}, covar={covars!r})'''

elif analysis.startswith("9."):
    test = st.selectbox("Test", ["Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "Friedman"])
    if test == "Mann-Whitney":
        outcome = st.selectbox("Outcome", num_cols)
        group = st.selectbox("Grouping variable", [c for c in all_cols if c != outcome])
        levels = list(df[group].dropna().unique())
        chosen = st.multiselect("Choose two groups", levels, default=levels[:2])
        code = "# Choose exactly two groups."
        if len(chosen) == 2:
            g1, g2 = chosen
            code = f'''x = df.loc[df[{group!r}] == {g1!r}, {outcome!r}].dropna()
y = df.loc[df[{group!r}] == {g2!r}, {outcome!r}].dropna()
result = pg.mwu(x, y)'''
    elif test == "Wilcoxon":
        x = st.selectbox("Variable 1", num_cols)
        y = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols) - 1))
        code = f'''d = df[[{x!r}, {y!r}]].dropna()
result = pg.wilcoxon(d[{x!r}], d[{y!r}])'''
    elif test == "Kruskal-Wallis":
        dv = st.selectbox("Dependent variable", num_cols)
        between = st.selectbox("Factor", [c for c in all_cols if c != dv])
        code = f'''result = pg.kruskal(data=df, dv={dv!r}, between={between!r})'''
    else:
        dv = st.selectbox("Dependent variable", num_cols)
        within = st.selectbox("Within factor", [c for c in all_cols if c != dv])
        subject = st.selectbox("Subject ID", [c for c in all_cols if c not in [dv, within]])
        code = f'''result = pg.friedman(data=df, dv={dv!r}, within={within!r}, subject={subject!r})'''

elif analysis.startswith("10."):
    y = st.selectbox("Outcome", num_cols)
    xs = st.multiselect("Predictors", [c for c in num_cols if c != y])
    code = f'''result = pg.linear_regression(X=df[{xs!r}], y=df[{y!r}], remove_na=True)'''

elif analysis.startswith("11."):
    item_defaults = [c for c in num_cols if c.lower().startswith("item")]
    items = st.multiselect(
        "Scale items", num_cols,
        default=item_defaults if item_defaults else num_cols[:3]
    )
    code = f'''alpha, ci = pg.cronbach_alpha(data=df[{items!r}])
result = pd.DataFrame({{"Cronbach alpha": [alpha], "CI lower": [ci[0]], "CI upper": [ci[1]]}})'''

elif analysis.startswith("12."):
    x = st.selectbox("Row variable", all_cols)
    y = st.selectbox("Column variable", [c for c in all_cols if c != x])
    code = f'''expected, observed, result = pg.chi2_independence(data=df, x={x!r}, y={y!r})'''

elif analysis.startswith("14."):
    model_family = st.selectbox("Model family", [
        "OLS (ANOVA / ANCOVA)",
        "Logit (binary)",
        "Probit (binary)",
        "Ordinal logit",
        "GLM"
    ])

    glm_family = None
    if model_family == "OLS (ANOVA / ANCOVA)":
        outcome_options = num_cols
    elif model_family in ["Logit (binary)", "Probit (binary)"]:
        outcome_options = [c for c in all_cols if df[c].nunique(dropna=True) == 2]
    elif model_family == "Ordinal logit":
        outcome_options = [c for c in all_cols if df[c].nunique(dropna=True) >= 2]
    else:
        glm_family = st.selectbox("GLM distribution", [
            "Poisson", "Negative binomial", "Gaussian", "Binomial", "Gamma"
        ])
        outcome_options = (
            [c for c in all_cols if df[c].nunique(dropna=True) == 2]
            if glm_family == "Binomial" else num_cols
        )

    if not outcome_options:
        st.warning("No suitable outcome variable is available for this model.")
        code = "raise ValueError('Choose a dataset with a suitable outcome variable.')"
    else:
        outcome_default = outcome_options.index("outcome") if "outcome" in outcome_options else 0
        outcome = st.selectbox("Outcome", outcome_options, index=outcome_default)
        predictor_options = [c for c in all_cols if c != outcome]
        predictor_defaults = [
            c for c in predictor_options
            if c.lower() not in {"id", "participant", "subject"}
            and not c.lower().endswith("_id")
        ][:3]
        predictors = st.multiselect(
            "Predictors", predictor_options, default=predictor_defaults
        )
        predictor_terms = [
            f"C(Q({column!r}))"
            if not pd.api.types.is_numeric_dtype(df[column])
            else f"Q({column!r})"
            for column in predictors
        ]
        rhs = " + ".join(predictor_terms) or "1"
        formula = f"Q({outcome!r}) ~ {rhs}"
        selected_columns = [outcome, *predictors]

        if model_family == "OLS (ANOVA / ANCOVA)":
            code = f'''import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

d = df[{selected_columns!r}].dropna().copy()
model = smf.ols(formula={formula!r}, data=d).fit()
result = model.summary2().tables[1]
overall = pd.DataFrame({{"F": [model.fvalue], "df_model": [model.df_model], "p-value": [model.f_pvalue]}})
anova = anova_lm(model, typ=2)'''

        elif model_family in ["Logit (binary)", "Probit (binary)"]:
            fit_function = "logit" if model_family == "Logit (binary)" else "probit"
            binary_formula = f"__outcome ~ {rhs}"
            code = f'''import statsmodels.formula.api as smf
from scipy.stats import chi2

d = df[{selected_columns!r}].dropna().copy()
d["__outcome"] = pd.Categorical(d[{outcome!r}]).codes
if d["__outcome"].nunique() != 2:
    raise ValueError("The selected outcome must have exactly two levels.")
model = smf.{fit_function}(formula={binary_formula!r}, data=d).fit(disp=False)
result = pd.DataFrame({{"coef": model.params, "SE": model.bse, "z": model.tvalues, "p-value": model.pvalues}})
overall = pd.DataFrame({{"LR chi2": [model.llr], "df": [model.df_model], "p-value": [model.llr_pvalue]}})'''

        elif model_family == "Ordinal logit":
            outcome_levels = list(df[outcome].dropna().unique())
            ordered_levels = st.multiselect(
                "Outcome order (first to last)", outcome_levels, default=outcome_levels
            )
            st.caption("Select levels in the intended ordinal order.")
            ordinal_rhs = " + ".join(predictor_terms)
            if not ordinal_rhs:
                st.warning("Ordinal logit requires at least one predictor.")
                code = "raise ValueError('Select at least one predictor for ordinal logit.')"
            else:
                design_formula = ordinal_rhs
                code = f'''import numpy as np
import patsy
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel

d = df[{selected_columns!r}].dropna().copy()
d["__outcome"] = pd.Categorical(
    d[{outcome!r}], categories={ordered_levels!r}, ordered=True
).codes
if (d["__outcome"] < 0).any() or d["__outcome"].nunique() < 2:
    raise ValueError("Select all outcome levels in their intended order.")
X = patsy.dmatrix({design_formula!r}, d, return_type="dataframe").drop(columns="Intercept")
model = OrderedModel(d["__outcome"], X, distr="logit").fit(method="bfgs", disp=False)
result = pd.DataFrame({{"coef": model.params.loc[X.columns], "SE": model.bse.loc[X.columns], "z": model.tvalues.loc[X.columns], "p-value": model.pvalues.loc[X.columns]}})
null_model = OrderedModel(d["__outcome"], np.empty((len(d), 0)), distr="logit").fit(method="bfgs", disp=False)
lr_chi2 = 2 * (model.llf - null_model.llf)
overall = pd.DataFrame({{"LR chi2": [lr_chi2], "df": [len(X.columns)], "p-value": [chi2.sf(lr_chi2, len(X.columns))]}})'''

        else:
            glm_family_expression = {
                "Poisson": "sm.families.Poisson()",
                "Negative binomial": "sm.families.NegativeBinomial()",
                "Gaussian": "sm.families.Gaussian()",
                "Binomial": "sm.families.Binomial()",
                "Gamma": "sm.families.Gamma()",
            }[glm_family]
            glm_formula = formula
            if glm_family == "Binomial":
                glm_formula = f"__outcome ~ {rhs}"
                response_setup = f'd["__outcome"] = pd.Categorical(d[{outcome!r}]).codes\n'
            else:
                response_setup = ""
            null_formula = f"{'__outcome' if glm_family == 'Binomial' else f'Q({outcome!r})'} ~ 1"
            code = f'''import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

d = df[{selected_columns!r}].dropna().copy()
{response_setup}family = {glm_family_expression}
model = smf.glm(formula={glm_formula!r}, data=d, family=family).fit()
result = pd.DataFrame({{"coef": model.params, "SE": model.bse, "z": model.tvalues, "p-value": model.pvalues}})
null_model = smf.glm(formula={null_formula!r}, data=d, family=family).fit()
lr_chi2 = max(0, 2 * (model.llf - null_model.llf))
lr_df = model.df_model - null_model.df_model
overall = pd.DataFrame({{"LR chi2": [lr_chi2], "df": [lr_df], "p-value": [chi2.sf(lr_chi2, lr_df)]}})'''

elif analysis.startswith("13."):

    st.subheader("Generate Multiple Exam Versions")

    uploaded_qti = st.file_uploader(
        "Upload QTI XML",
        type=["xml"],
        key="exam_generator"
    )

    num_tests = st.number_input(
        "Number of test versions",
        min_value=1,
        max_value=100,
        value=4,
        step=1
    )

    compact = st.checkbox(
        "Compact one-line format",
        value=True
    )

    code = "# Exam generator does not use the code editor."

    if uploaded_qti is not None:

        st.success(
            f"Uploaded: {uploaded_qti.name}"
        )

        if st.button("Generate Exams"):

            questions = []

            with tempfile.TemporaryDirectory() as tmpdir:

                xml_file = Path(tmpdir) / uploaded_qti.name

                with open(xml_file, "wb") as f:
                    f.write(uploaded_qti.getbuffer())

                tree = ET.parse(xml_file)
                root = tree.getroot()

                for item in root.iter():

                    if not item.tag.endswith("item"):
                        continue

                    question_text = ""

                    mt = item.find(".//{*}presentation//{*}mattext")

                    if mt is not None and mt.text:

                        question_text = BeautifulSoup(
                            mt.text,
                            "html.parser"
                        ).get_text(" ", strip=True)

                    options = []
                    id_to_text = {}

                    for rl in item.findall(".//{*}response_label"):

                        ident = rl.attrib.get("ident", "")

                        option_mt = rl.find(".//{*}mattext")

                        option_text = ""

                        if option_mt is not None and option_mt.text:

                            option_text = BeautifulSoup(
                                option_mt.text,
                                "html.parser"
                            ).get_text(" ", strip=True)

                        options.append(option_text)
                        id_to_text[ident] = option_text

                    correct_text = ""

                    for rc in item.findall(".//{*}respcondition"):

                        sv = rc.find(".//{*}setvar")

                        if sv is None:
                            continue

                        if float((sv.text or "0").strip()) != 100:
                            continue

                        v = rc.find(".//{*}varequal")

                        if v is not None:

                            ident = (v.text or "").strip()

                            if ident in id_to_text:

                                correct_text = id_to_text[ident]
                                break

                    if question_text:

                        questions.append({
                            "question": question_text,
                            "options": options,
                            "correct": correct_text
                        })

            # -------------------------------------
            # Create ZIP file
            # -------------------------------------

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zipf:

                for test_num in range(1, num_tests + 1):

                    rng = np.random.default_rng(test_num)

                    qlist = copy.deepcopy(questions)
                    rng.shuffle(qlist)

                    test_lines = []
                    answer_key = []

                    for qnum, q in enumerate(qlist, start=1):

                        options = q["options"].copy()

                        rng.shuffle(options)

                        correct_letter = ""
                        option_strings = []

                        for idx, option in enumerate(options):

                            letter = chr(65 + idx)

                            option_strings.append(
                                f"{letter}) {option}"
                            )

                            if option == q["correct"]:
                                correct_letter = letter

                        if compact:

                            test_lines.append(
                                f"{qnum}. {q['question']}    "
                                + "    ".join(option_strings)
                            )

                        else:

                            test_lines.append(
                                f"{qnum}. {q['question']}\n"
                                + "\n".join(option_strings)
                            )

                        answer_key.append({
                            "Question": qnum,
                            "Answer": correct_letter,
                            "Correct_Text": q["correct"]
                        })

                    test_text = "\n\n".join(test_lines)

                    zipf.writestr(
                        f"test{test_num}.txt",
                        test_text
                    )

                    key_csv = pd.DataFrame(
                        answer_key
                    ).to_csv(index=False)

                    zipf.writestr(
                    f"key{test_num}.csv",
                        key_csv
                    )

            zip_buffer.seek(0)

            st.success(
                f"Generated {num_tests} versions from "
                f"{len(questions)} questions."
            )

            st.download_button(
                "Download ZIP Package",
                data=zip_buffer,
                file_name="exam_versions.zip",
                mime="application/zip"
            )
if not analysis.startswith("13."):
    st.subheader("Editable code")

    editor_key = hashlib.md5(code.encode()).hexdigest()

    code = st_ace(
    value=code.strip(),
    language="python",
    theme="github",
    key=f"code_{editor_key}",
    height=300,
    font_size=14,
    tab_size=4,
    wrap=True,
    auto_update=True,)

if not analysis.startswith("13."):

    if st.button("Run analysis", type="primary"):
        env = {"df": df.copy(), "pd": pd, "np": np, "pg": pg, "sns": sns, "plt": plt}
        try:
            plt.close("all")
            exec(code, env)
            for name in ["result", "posthoc", "observed", "expected", "overall", "anova"]:
                if name in env:
                    st.subheader(name.capitalize())
                    value = env[name]
                    st.dataframe(value, width="stretch") if isinstance(value, (pd.DataFrame, pd.Series)) else st.write(value)
            for number in plt.get_fignums():
                st.pyplot(plt.figure(number))
        except Exception:
            st.error("Analysis failed")
            st.code(traceback.format_exc())
