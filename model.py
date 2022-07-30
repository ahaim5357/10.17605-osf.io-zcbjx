from io import FileIO
import pandas as pd
import numpy as np
import statsmodels.api as sm
import construct as ct
from statsmodels.stats.multitest import multipletests
from aos_constants import T, R, drop_not_columns
from typing import Any, Callable, Optional, Tuple, Type

def backward_selection_stepwise_ols(posttest: pd.DataFrame, dependent_measure: str, features: "list[str]" = [], fixed_features: "list[str]" = [], use_intercept: bool = True, correlation_threshold: Optional[float] = 0.95) -> sm.regression.linear_model.RegressionResults:
    if not features:
        features = posttest.columns.tolist()
        features.remove(dependent_measure)
    y: pd.Series = posttest[dependent_measure]
    x: pd.DataFrame = drop_not_columns(posttest, features)
    
    if use_intercept:
        print("Using Intercept")
        x = sm.add_constant(x)
    
    break_correlations: bool = False
    while True:
        if correlation_threshold is not None:
            corr_mat: pd.DataFrame = x.corr().abs()
            upper: pd.DataFrame = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            to_drop: list[str] = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            to_drop = list(set(to_drop).difference(set(fixed_features)))
            if not to_drop:
                break_correlations = True
            else:
                print(f"Dropping {to_drop}")
                x = x.drop(columns=to_drop)
        else:
            break_correlations = True

        while True:
            ols: sm.regression.linear_model.RegressionResults = sm.OLS(y, x).fit(cov_type='HC1')
            ols_pvalues: pd.Series = ols.pvalues

            for column in fixed_features:
                ols_pvalues = ols_pvalues.drop(column)
            if use_intercept:
                ols_pvalues = ols_pvalues.drop("const")

            if ols_pvalues.empty:
                break

            ols_max: str = ols_pvalues.idxmax()
            if ols_pvalues.loc[ols_max] < 0.05:
                break
            else:
                print(f"Dropping {ols_max}")
                x = x.drop(columns=ols_max)
        
        if break_correlations:
            break
    
    return sm.OLS(y, x).fit(cov_type='HC1')

## OLS Take average posttest score for each student (prior stats) and run ols model (do not use student_sample_count)
## Backwards stepwise regression (remove insignificant covariates)
## Add interaction effects, repeat and remove insignificant (final model)
def construct_initial_model(posttest: pd.DataFrame, dependent_measure: str, fixed_features: "list[str]" = [], excluded_interactors: "list[str]" = [], excluded_interaction_pairs: "list[list[str]]" = [], use_intercept: bool = True, correlation_threshold: Optional[float] = 0.95) -> sm.regression.linear_model.RegressionResults:
    posttest = posttest.copy(deep=True)

    # Generate initial model
    print("Running Initial Model")
    initial_model: sm.regression.linear_model.RegressionResults = backward_selection_stepwise_ols(
        posttest,
        dependent_measure,
        fixed_features=fixed_features,
        use_intercept=use_intercept,
        correlation_threshold=correlation_threshold
    )

    # Generate interactions
    columns: list[str] = initial_model.params.keys().tolist()
    if use_intercept:
        columns.remove('const')
    excluded_columns: list[str] = list(columns)
    
    for interactor in excluded_interactors:
        columns.remove(interactor)
    print(f"Generating Interactions: {columns}")

    interaction_columns: list[str] = list(excluded_interactors)
    interaction_columns.extend(columns)

    num_of_columns: int = len(columns)

    for i in range(0, num_of_columns):
        for j in range(i + 1, num_of_columns):
            col_i: str = columns[i]
            col_j: str = columns[j]

            if ":" in col_i or ":" in col_j: # Don't create interaction effects from interaction effects
                continue

            skip: bool = False
            for excluded_combination in excluded_interaction_pairs:
                if col_i in excluded_combination and col_j in excluded_combination:
                    skip = True
                    break
            if skip:
                continue

            name = f'{col_i}:{col_j}'
            posttest[name] = pd.Series(posttest[col_i] * posttest[col_j], name=name)
            interaction_columns.append(name)

    print("Creating Final Model")
    final_model: sm.regression.linear_model.RegressionResults = backward_selection_stepwise_ols(
        posttest,
        dependent_measure,
        features=interaction_columns,
        fixed_features=excluded_columns,
        use_intercept=use_intercept,
        correlation_threshold=correlation_threshold
    )
    return final_model

def features(results: sm.regression.linear_model.RegressionResults, excluded_features: "list[str]" = []) -> "list[str]":
    features: list[str] = results.params.keys().tolist()

    if "const" in features:
        features.remove("const")
    
    if excluded_features:
        for column in excluded_features:
            features.remove(column)
    return features

def write(data: str, path: str):
    data_file: FileIO = open(path, "w")
    data_file.write(data)
    data_file.close()

def initial_model_features(posttest: pd.DataFrame, dependent_measure: str, fixed_features: "list[str]" = [], use_intercept: bool = True, correlation_threshold: Optional[float] = 0.95, excluded_features: "list[str]" = [], excluded_interactors: "list[str]" = [], excluded_interaction_pairs: "list[list[str]]" = [], directory: str = ".", name: str = "results") -> "list[str]":
    results: sm.regression.linear_model.RegressionResults = construct_initial_model(
        posttest, dependent_measure,
        fixed_features = fixed_features,
        excluded_interactors = excluded_interactors,
        excluded_interaction_pairs = excluded_interaction_pairs,
        use_intercept = use_intercept,
        correlation_threshold = correlation_threshold
    )
    write(results.summary().as_text(), f"{directory}/{name}.txt")
    return features(results, excluded_features=excluded_features)

def is_type(val: Any, tval: Type[T], func: Callable[[T], R]):
    if type(val) is not tval:
        return val
    else:
        return func(val)

def generate_model_results(study_data: pd.DataFrame, pretest_data: pd.DataFrame, posttest_data: pd.DataFrame, stats: pd.DataFrame, interaction_map: pd.DataFrame, independent_measure: str, dependent_measure: str, participated: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, features: "list[str]" = [], interactions: "list[tuple[str, str]]" = [], directory_mod: Callable[[str], str] = ct.mkdir_if_absent, path: str = "results", save_results: bool = True) -> pd.DataFrame:
    model_results: pd.DataFrame = pd.DataFrame(
        index = pd.MultiIndex.from_tuples(interaction_map.index.values, names=interaction_map.index.names),
        columns = [f'{independent_measure}_coef', f'{independent_measure}_std_err', f'{independent_measure}_conf_int', f'{independent_measure}_pvalue', f'{independent_measure}_corrected_pvalue', f'{independent_measure}_top', 'can_analyze']
    )
    model_results['can_analyze'] = interaction_map['can_analyze']

    model_results['results'] = model_results.index.map(
        lambda i: handle_model(
            study_data, pretest_data, posttest_data, stats,
            independent_measure, dependent_measure, interaction_map.loc[i],
            participated = participated, features = features, interactions = interactions,
            directory_mod = directory_mod,
            path = path,
            save_results = save_results
        )
    )
    model_results[f'{independent_measure}_coef'] = model_results['results'].map(
        lambda tup: is_type(tup, tuple,
            lambda t: t[0]
        )
    )
    model_results[f'{independent_measure}_std_err'] = model_results['results'].map(
        lambda tup: is_type(tup, tuple,
            lambda t: t[1]
        )
    )
    model_results[f'{independent_measure}_conf_int'] = model_results['results'].map(
        lambda tup: is_type(tup, tuple,
            lambda t: t[2]
        )
    )
    model_results[f'{independent_measure}_pvalue'] = model_results['results'].map(
        lambda tup: is_type(tup, tuple,
            lambda t: t[3]
        )
    )

    model_results_analyzed = model_results[model_results['can_analyze']].copy(deep=True)
    model_results_analyzed[f'{independent_measure}_corrected_pvalue'] = multipletests(model_results_analyzed[f'{independent_measure}_pvalue'], method="fdr_bh")[1]
    model_results[f'{independent_measure}_corrected_pvalue'] = model_results.index.map(
        lambda i: model_results_analyzed.loc[i][f'{independent_measure}_corrected_pvalue'] if i in model_results_analyzed.index else np.nan
    )

    model_results[f'{independent_measure}_top'] = model_results.apply(
        lambda row: np.nan if (not row['can_analyze']) or (row[f'{independent_measure}_corrected_pvalue'] > 0.05) or ((row[f'{independent_measure}_conf_int'][0] < 0) != (row[f'{independent_measure}_conf_int'][1] < 0)) else int(row[f'{independent_measure}_conf_int'][0] > 0),
    axis = 1)
    
    return model_results.drop(columns='results')

def handle_model(study_data: pd.DataFrame, pretest_data: pd.DataFrame, posttest_data: pd.DataFrame, stats: pd.DataFrame, independent_measure: str, dependent_measure: str, row: pd.Series, participated: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, features: "list[str]" = [], interactions: "list[tuple[str, str]]" = [], directory_mod: Callable[[str], str] = ct.mkdir_if_absent, path: str = "results", save_results: bool = True) -> Tuple[float, float, Tuple[float, float], float]:
    if not row['can_analyze']:
        return np.nan
    _, _, posttest, directory = ct.construct_study_data(study_data, pretest_data, posttest_data, stats, row.name[0], row.name[1], participated = participated, directory_mod = directory_mod)
    posttest = posttest.reset_index()
    if features is None:
        features = posttest.columns.tolist()
        features.remove(dependent_measure)
    y: pd.Series = posttest[dependent_measure]
    x: pd.DataFrame = drop_not_columns(posttest, features)

    for pair in interactions:
        name = f'{pair[0]}:{pair[1]}'
        x[name] = pd.Series(x[pair[0]] * x[pair[1]], name=name)

    x = sm.add_constant(x)
    model_results: sm.regression.linear_model.RegressionResults = sm.OLS(y, x).fit(cov_type='HC1')
    if save_results:
        write(model_results.summary().as_text(), f"{directory}/{path}.txt")
    return (
        model_results.params[independent_measure],
        model_results.bse[independent_measure],
        tuple(model_results.conf_int().loc[independent_measure]),
        abs(model_results.pvalues[independent_measure])
    )
