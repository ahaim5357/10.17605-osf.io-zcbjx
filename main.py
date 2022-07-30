# Import necessary libraries

# Pandas warning about numpy ignoring
import warnings
warnings.filterwarnings("ignore")

from typing import Callable, List, Union
import pandas as pd
from pathlib import Path
import sys
import preprocessor as pp
import interactions as it
import construct as ct
import model as ml
import plot as plt
import aos_constants as aosc
import numpy as np

def main() -> int:
    # Load in data
    support_logs: pd.DataFrame

    print("Loading Student Support Features")
    support_features: pd.DataFrame = pd.read_csv("student_support_features.csv", index_col="student_support_id")

    print("Loading Star Authors")
    star_authors: pd.DataFrame = pd.read_csv("star_authors.csv", index_col="author_id")

    # Run preprocessor
    if Path("student_support_creator_logs.csv").is_file():
        print("Preprocessed Data Found: student_support_creator_logs.csv")
        support_logs = pd.read_csv("student_support_creator_logs.csv")
    else:
        support_logs = pd.read_csv("student_support_logs.csv")
        print("Preprocessing Data: student_support_logs.csv")
        support_logs = pp.preprocess(support_logs, support_features)
        support_logs.to_csv("student_support_creator_logs.csv", index=False)

    print("Loading Student Prior Statistics")
    student_prior_stats: pd.DataFrame = pd.read_csv("student_prior_stats.csv", index_col="student_id").drop(columns="student_num_of_problems")

    print("Separating Testing Phases")

    ## Pretest (Februrary 16th - March 1st)
    pre_test = support_logs.query('timestamp >= 1644969600 and timestamp < 1646092800').copy(deep=True)

    ## Study 1 (March 1st - April 1st)
    study_1 = support_logs.query('timestamp >= 1646092800 and timestamp < 1648771200').copy(deep=True)

    ## Midtest (April 1st - April 16th)
    mid_test = support_logs.query('timestamp >= 1648771200 and timestamp < 1650067200').copy(deep=True)

    ## Study 2 (April 16th - May 16th)
    study_2 = support_logs.query('timestamp >= 1650067200 and timestamp < 1652659200').copy(deep=True)

    ## Posttest (May 16th - June 1st)
    post_test = support_logs.query('timestamp >= 1652659200 and timestamp < 1654041600').copy(deep=True)

    print("Indexing Author Identifiers")

    # Create Creator Id Map
    creator_ids: list[int] = support_features['student_support_content_creator_id'].unique().tolist()
    creator_ids.sort()

    # Participated filter
    participated: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df[df['tutoring_observed'] == 1]
    ct.mkdir_if_absent(aosc.RESULTS)

    # Study 1

    ## Generate interaction map
    study_1_interactions: pd.DataFrame
    if Path("study_1_interactions.csv").is_file():
        print("Interaction Data Found: study_1_interactions.csv")
        study_1_interactions = pd.read_csv("study_1_interactions.csv", index_col=['author_a', 'author_b'])
    else:
        print("Generating Interaction Data: study_1_interactions.csv")
        study_1_interactions = it.generate_interaction_map(study_1, support_features, creator_ids)
        study_1_interactions.to_csv("study_1_interactions.csv")
    
    print("Process Interaction Rows: study_1_interactions.csv")
    study_1_interactions = it.mark_viable_rows(study_1_interactions, as_treated=True)

    ## Get pairwise data to burn to generate model
    burned_comparison, study_1_interactions = it.burn_data(study_1_interactions)
    print(f"Burning {burned_comparison.name[0]} vs {burned_comparison.name[1]}")
    _, _, burned_posttest, burned_directory = ct.construct_study_data(
        study_1, pre_test, mid_test, student_prior_stats,
        burned_comparison.name[0], burned_comparison.name[1],
        participated = participated,
        directory_mod = lambda s: ct.mkdir_if_absent(f"{aosc.RESULTS}/{s}_burned")
    )
    burned_comparison.to_csv(f"{aosc.RESULTS}/{burned_comparison.name[0]}_{burned_comparison.name[1]}_burned/burned_comparison.csv", header = True)

    ## Get features to use for models
    features: list[str] = ml.initial_model_features(
        burned_posttest, 'average_problem_accuracy',
        fixed_features = ['author', 'pretest_avg_problem_accuracy'],
        excluded_interactors = ['author'],
        directory = burned_directory)
    print(f"Using Features: {features}")
    ml.write(features.__repr__(), f"{aosc.RESULTS}/{burned_comparison.name[0]}_{burned_comparison.name[1]}_burned/features.txt")

    study_1_model_results: pd.DataFrame = ml.generate_model_results(
        study_1, pre_test, mid_test, student_prior_stats, study_1_interactions,
        'author', 'average_problem_accuracy',
        participated = participated, features = features,
        directory_mod = lambda s: ct.mkdir_if_absent(f"{ct.mkdir_if_absent(aosc.STUDY_1)}/{s}")
    )
    study_1_model_results.to_csv(f"{aosc.STUDY_1}/model_results.csv")

    ## Generate demographic model
    student_knowledge_groups: pd.DataFrame
    if Path("student_knowledge_groups.csv").is_file():
        print("Preprocessed Data Found: student_knowledge_groups.csv")
        student_knowledge_groups = pd.read_csv("student_knowledge_groups.csv", index_col="student_id")
    else:
        print("Preprocessing Data: student_knowledge_groups.csv")
        student_knowledge_groups = pp.knowledge_separation(student_prior_stats)
        student_knowledge_groups.to_csv("student_knowledge_groups.csv")

    student_gender_info: pd.DataFrame
    if Path("student_gender_info.csv").is_file():
        print("Preprocessed Data Found: student_gender_info.csv")
        student_gender_info = pd.read_csv("student_gender_info.csv", index_col="student_id")
    else:
        student_gender_info = pd.read_csv("infered_student_gender_ASSISTments_2021_Aug_15_2022_Jul_15.csv", index_col="user_id")
        print("Preprocessing Data: student_gender_info.csv")
        student_gender_info = pp.preprocess_gender_info(student_gender_info)
        student_gender_info.to_csv("student_gender_info.csv")
    
    ## Locale Codes: https://nces.ed.gov/ccd/CCDLocaleCode.asp
    student_locale_info: pd.DataFrame = pd.read_csv("student_locale_info.csv", index_col="student_id")
    
    sgic = student_gender_info.columns.to_list()
    skqc = student_knowledge_groups.columns.to_list()
    slic = student_locale_info.columns.to_list()
    demographic_interactions = [
        ("author", "rural"),
        ("author", "suburban"),
        ("author", "low_knowledge"),
        ("author", "mid_knowledge"),
        ("author", "gender")
    ]

    skqc.remove("high_knowledge")
    slic.remove("urban")

    demographic_independent_measures: list[tuple[str, str, str, str]] = [
        ('author', 'Author', 'Author A > Author B', 'Author A < Author B'),
        ('gender', 'Gender', 'Male', 'Female'),
        ('low_knowledge', 'Low Knowledge', 'Low Knowledge', 'Not Low Knowledge'),
        ('mid_knowledge', 'Mid Knowledge', 'Mid Knowledge', 'Not Mid Knowledge'),
        ('rural', 'Rural', 'Rural', 'Not Rural'),
        ('suburban', 'Suburban', 'Suburban', 'Not Suburban'),
        ('author:gender', 'Author x Gender', 'Positive Interaction Between Author and Gender', 'Negative Interaction Between Author and Gender'),
        ('author:low_knowledge', 'Author x Low Knowledge', 'Positive Interaction Between Author and Low Knowledge', 'Negative Interaction Between Author and Low Knowledge'),
        ('author:mid_knowledge', 'Author x Mid Knowledge', 'Positive Interaction Between Author and Mid Knowledge', 'Negative Interaction Between Author and Mid Knowledge'),
        ('author:rural', 'Author x Rural', 'Positive Interaction Between Author and Rural', 'Negative Interaction Between Author and Rural'),
        ('author:suburban', 'Author x Suburban', 'Positive Interaction Between Author and Suburban', 'Negative Interaction Between Author and Suburban')
    ]

    ## Draw to graph

    # -6: Identity
    # -5: Burned data
    # -4: No interactions
    # -3: Not enough data to analyze
    # -2: P-value insignificant
    # -1: Confidence interval includes 0
    #  0: author B performs better
    #  1: author A performs better
    def index_value(independent_measure: str, model_results: pd.DataFrame, author_a: int, author_b: int, no_flip: bool = False) -> int:
        if author_a == author_b:
            return  -6
        elif (author_a == burned_comparison.name[0] and author_b == burned_comparison.name[1]) \
            or (author_b == burned_comparison.name[0] and author_a == burned_comparison.name[1]):
            return -5
        try:
            author_pair: list[int] = [author_a, author_b]
            author_pair.sort()
            results: pd.Series = model_results.loc[author_pair[0], author_pair[1]]
            if not results['can_analyze']:
                return -3
            elif results[f'{independent_measure}_corrected_pvalue'] > 0.05:
                return -2
            elif np.isnan(results[f'{independent_measure}_top']):
                return -1
            if no_flip:
                return results[f'{independent_measure}_top']
            elif author_a < author_b: # Check if author_a is the author with the smallest id
                return  1 - results[f'{independent_measure}_top'] # If so, reverse the value
            else:
                return results[f'{independent_measure}_top'] # Else, return the actual value
        except:
            return -4

    result_indexer: Callable[[str, pd.DataFrame, bool], Callable[[int, int], int]] = lambda independent_measure, model_results, no_flip: lambda a, b: index_value(independent_measure, model_results, a, b, no_flip = no_flip)

    colormap: list[str] = [
        'black',
        'orange',
        'silver',
        'gold',
        'tomato',
        'orchid',
        'royalblue',
        'lawngreen'
    ]
    ticks: Callable[[str, str, str], List[List[Union[float, list[str]]]]] = lambda feature, positive, negative: [
        [-5.55,-4.65,-3.85,-2.95,-2.05,-1.15,-.35,0.55],
        [
            'Identity',
            'Authors Used to Select Model Features',
            'No Valid Pairwise Comparison Between Authors',
            'Not Enough Data to Compare Authors',
            f'{feature} Feature in Model had an Insignificant P-Value',
            f'Confidence Interval of {feature} Feature Includes Zero',
            negative,
            positive
        ]
    ]
    vmin: int = -6
    vmax: int = 1
    xlabel: str = "Author A Identifiers"
    ylabel: str = "Author B Identifiers"

    ## Generate Figure
    plt.save_author_matrix(
        star_authors, result_indexer('author', study_1_model_results, False),
        cmap = colormap, vmin = vmin, vmax = vmax,
        xlabel = xlabel, ylabel = ylabel,
        ticks = ticks(demographic_independent_measures[0][1], demographic_independent_measures[0][2], demographic_independent_measures[0][3]),
        directory = ct.mkdir_if_absent(aosc.STUDY_1)
    )

    ## Draw and Generate demographic features

    for metadata in demographic_independent_measures:
        feature_observation: str = metadata[0].replace(":", "_")
        demographic_results: pd.DataFrame = ml.generate_model_results(
            study_1, pre_test, mid_test, student_prior_stats.merge(student_gender_info, left_index = True, right_index = True)
                .merge(student_knowledge_groups, left_index = True, right_index = True)
                .merge(student_locale_info, left_index = True, right_index = True)
            , study_1_interactions,
            metadata[0], 'average_problem_accuracy',
            participated = participated, features = features + sgic + skqc + slic,
            interactions = demographic_interactions,
            directory_mod = lambda s: ct.mkdir_if_absent(f"{ct.mkdir_if_absent(aosc.STUDY_1)}/{s}"),
            path = f"demographic_results_{feature_observation}",
            save_results = metadata[0] == 'author'
        )
        demographic_results.to_csv(f"{aosc.STUDY_1}/demographic_results_{feature_observation}.csv")

        plt.save_author_matrix(
            star_authors, result_indexer(metadata[0], demographic_results, metadata[0] != 'author'),
            cmap = colormap, vmin = vmin, vmax = vmax,
            xlabel = xlabel, ylabel = ylabel,
            ticks = ticks(metadata[1], metadata[2], metadata[3]),
            directory = ct.mkdir_if_absent(aosc.STUDY_1),
            path = f"demographic_results_{feature_observation}"
        )

    # Study 2

    ## Generate interaction map
    study_2_interactions: pd.DataFrame
    if Path("study_2_interactions.csv").is_file():
        print("Interaction Data Found: study_2_interactions.csv")
        study_2_interactions = pd.read_csv("study_2_interactions.csv", index_col=['author_a', 'author_b'])
    else:
        print("Generating Interaction Data: study_2_interactions.csv")
        study_2_interactions = it.generate_interaction_map(study_2, support_features, creator_ids)
        study_2_interactions.to_csv("study_2_interactions.csv")

    print("Process Interaction Rows: study_2_interactions.csv")
    study_2_interactions = study_2_interactions
    study_2_interactions = it.mark_viable_rows(study_2_interactions, as_treated=True)
    
    study_2_model_results: pd.DataFrame = ml.generate_model_results(
        study_2, mid_test, post_test, student_prior_stats, study_2_interactions,
        'author', 'average_problem_accuracy',
        participated = participated, features = features,
        directory_mod = lambda s: ct.mkdir_if_absent(f"{ct.mkdir_if_absent(aosc.STUDY_2)}/{s}")
    )
    study_2_model_results.to_csv(f"{aosc.STUDY_2}/model_results.csv")

    ## Generate Figure
    plt.save_author_matrix(
        star_authors, result_indexer('author', study_2_model_results, False),
        cmap = colormap, vmin = vmin, vmax = vmax,
        xlabel = xlabel, ylabel = ylabel,
        ticks = ticks(demographic_independent_measures[0][1], demographic_independent_measures[0][2], demographic_independent_measures[0][3]),
        directory = ct.mkdir_if_absent(aosc.STUDY_2)
    )

    for metadata in demographic_independent_measures:
        feature_observation: str = metadata[0].replace(":", "_")
        demographic_results: pd.DataFrame = ml.generate_model_results(
            study_2, mid_test, post_test, student_prior_stats.merge(student_gender_info, left_index = True, right_index = True)
                .merge(student_knowledge_groups, left_index = True, right_index = True)
                .merge(student_locale_info, left_index = True, right_index = True)
            , study_2_interactions,
            metadata[0], 'average_problem_accuracy',
            participated = participated, features = features + sgic + skqc + slic,
            interactions = demographic_interactions,
            directory_mod = lambda s: ct.mkdir_if_absent(f"{ct.mkdir_if_absent(aosc.STUDY_2)}/{s}"),
            path = f"demographic_results_{feature_observation}",
            save_results = metadata[0] == 'author'
        )
        demographic_results.to_csv(f"{aosc.STUDY_2}/demographic_results_{feature_observation}.csv")

        plt.save_author_matrix(
            star_authors, result_indexer(metadata[0], demographic_results, metadata[0] != 'author'),
            cmap = colormap, vmin = vmin, vmax = vmax,
            xlabel = xlabel, ylabel = ylabel,
            ticks = ticks(metadata[1], metadata[2], metadata[3]),
            directory = ct.mkdir_if_absent(aosc.STUDY_2),
            path = f"demographic_results_{feature_observation}"
        )

    return 0

if __name__ == '__main__':
    sys.exit(main())
