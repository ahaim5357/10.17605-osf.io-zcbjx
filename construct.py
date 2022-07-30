# Import necessary libraries

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Callable
import os

# Compare two authors (conditions which each student is within)
def treatment(study: pd.DataFrame, author_a: int, author_b: int, participated: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    if participated is not None:
        study = participated(study)
    treatment_results: pd.DataFrame = study.query(f'in_treatment == 1 and \
                                  (selected_student_support_creator_id == {author_a} and \
                                  (alternative_student_support_creator_id_1 == {author_b} \
                                  or alternative_student_support_creator_id_2 == {author_b} \
                                  or alternative_student_support_creator_id_3 == {author_b} \
                                  or alternative_student_support_creator_id_4 == {author_b})) \
                                  or (selected_student_support_creator_id == {author_b} and \
                                  (alternative_student_support_creator_id_1 == {author_a} \
                                  or alternative_student_support_creator_id_2 == {author_a} \
                                  or alternative_student_support_creator_id_3 == {author_a} \
                                  or alternative_student_support_creator_id_4 == {author_a}))')

    study_student_conditions: pd.DataFrame = pd.DataFrame(columns = ['student_id', f'author_{author_a}', f'author_{author_b}'])
    for student_id, data in treatment_results.groupby('student_id'):
        study_student_conditions = study_student_conditions.append({
            'student_id': student_id,
            f'author_{author_a}': author_a in data['selected_student_support_creator_id'].values,
            f'author_{author_b}': author_b in data['selected_student_support_creator_id'].values
        }, ignore_index=True)
    study_student_conditions = study_student_conditions.set_index('student_id')
    students_xor: pd.Index = (study_student_conditions[f'author_{author_a}'] ^ study_student_conditions[f'author_{author_b}']).index

    ## Filter students that are in one or the other condition
    return study_student_conditions[study_student_conditions.index.isin(students_xor)]

## Create pretest conditions (Column for authors [a=1,b=0])
def pretest(pretest_data: pd.DataFrame, study_conditions: pd.DataFrame, author_a: int) -> pd.DataFrame:
    pretest: pd.DataFrame = pd.DataFrame(columns = ['student_id', 'average_problem_accuracy'])
    for student_id, log in pretest_data[pretest_data['student_id'].isin(study_conditions.index)].groupby('student_id'):
        pretest = pretest.append({
            'student_id': int(student_id),
            'average_problem_accuracy': log['problem_accuracy'].mean()
        }, ignore_index=True)
    pretest['author'] = pretest['student_id'].map(lambda x: 1 if study_conditions.loc[x][f'author_{author_a}'] else 0)

    pretest['student_id'] = pretest['student_id'].astype(int)
    return pretest.set_index('student_id')

## Create posttest conditions (Column for authors [a=1,b=0])
def posttest(posttest_data: pd.DataFrame, study_conditions: pd.DataFrame, author_a: int) -> pd.DataFrame:
    posttest: pd.DataFrame = pd.DataFrame(columns = ['student_id', 'average_problem_accuracy'])
    for student_id, log in posttest_data[posttest_data['student_id'].isin(study_conditions.index)].groupby('student_id'):
        posttest = posttest.append({
            'student_id': int(student_id),
            'average_problem_accuracy': log['problem_accuracy'].mean()
        }, ignore_index=True)
    posttest['author'] = posttest['student_id'].map(lambda x: 1 if study_conditions.loc[x][f'author_{author_a}'] else 0)
    posttest['student_id'] = posttest['student_id'].astype(int)
    return posttest.set_index('student_id')

# Checks if students in study has pretest and posttest entries and discards missing data if necessary
def student_condition_check(study_conditions: pd.DataFrame, pretest: pd.DataFrame, posttest: pd.DataFrame, stats: pd.DataFrame, directory: str = ".") -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    students_in_study: pd.Index = study_conditions.index
    students_with_pretest: np.ndarray[int] = np.intersect1d(students_in_study, pretest.index.values)
    students_with_posttest: np.ndarray[int] = np.intersect1d(students_in_study, posttest.index.values)
    students_with_complete: np.ndarray[int] = np.intersect1d(students_with_pretest, students_with_posttest)
    
    student_data_summary: pd.DataFrame = pd.DataFrame(
        index=students_in_study,
        columns=['has_all_data', 'has_pretest', 'has_posttest']
    )
    student_data_summary['has_all_data'] = student_data_summary.index.isin(students_with_complete)
    student_data_summary['has_pretest'] = student_data_summary.index.isin(students_with_pretest)
    student_data_summary['has_posttest'] = student_data_summary.index.isin(students_with_posttest)
    student_data_summary.to_csv(f"{directory}/student_data_summary.csv")

    sizes: tuple[int, int, int, int] = (len(students_in_study), len(students_with_complete), len(students_with_pretest), len(students_with_posttest))
    if sizes[0] > sizes[1]:
        pretest_only: int = sizes[2] - sizes[1]
        posttest_only: int = sizes[3] - sizes[1]
        missing_data: int = sizes[0] - sizes[1]
        no_data: int = missing_data - pretest_only - posttest_only
        print(f"Missing Data for {missing_data} of {sizes[0]} Students: (Pretest Only: {pretest_only}, Posttest Only : {posttest_only}, No Data: {no_data})")

        study_conditions = study_conditions[study_conditions.index.isin(students_with_complete)]
        pretest = pretest[pretest.index.isin(students_with_complete)]
        posttest = posttest[posttest.index.isin(students_with_complete)]
    
    posttest.loc[:, 'pretest_avg_problem_accuracy'] = posttest.index.map(lambda i: pretest.loc[i]["average_problem_accuracy"])
    return study_conditions, pretest, merge_stats(posttest, stats)

def merge_stats(post_test: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    return post_test.merge(stats, left_index=True, right_index=True)

def mkdir_if_absent(path: str) -> str:
    if not Path(path).is_dir():
        os.mkdir(path)
    return path

# Constructs all data for a given study
def construct_study_data(study_data: pd.DataFrame, pretest_data: pd.DataFrame, posttest_data: pd.DataFrame, stats: pd.DataFrame, author_a: int, author_b: int, participated: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, directory_mod: Callable[[str], str] = mkdir_if_absent) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]":
    study_conditions: pd.DataFrame = treatment(study_data, author_a, author_b, participated=participated)
    directory: str = directory_mod(f"{author_a}_{author_b}")
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = student_condition_check(
        study_conditions,
        pretest(pretest_data, study_conditions, author_a),
        posttest(posttest_data, study_conditions, author_a),
        stats,
        directory = directory
    )

    return result + (directory,)
