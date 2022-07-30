# Import necessary libraries

import pandas as pd
import numpy as np
from aos_constants import T
from typing import Dict, List, Tuple, Generic, Callable, Set
import sys

class PotentialViewed(Generic[T]):
    def __init__(self, factory: Callable[[], T]):
        self.potential = factory()
        self.viewed = factory()
    
    def update(self, update_func: Callable[[T], T], viewed: bool):
        self.potential = update_func(self.potential)
        if (viewed):
            self.viewed = update_func(self.viewed)
        

def pv_int() -> PotentialViewed[int]:
    return PotentialViewed(lambda: 0)

def pv_set() -> PotentialViewed[Set[int]]:
    return PotentialViewed(set)

class authorInfo:
    def __init__(self, author: int):
        self.author = author
        self.students = pv_set()
        self.hints = pv_int()
        self.explanations = pv_int()
    
    def update(self, author_data: pd.Series, viewed: bool):
        def add_to_set(x):
            x.add(author_data['student_id'])
            return x

        if author_data['support_selected']:
            self.students.update(add_to_set, viewed)
        self.hints.update(lambda x: x + author_data['student_support_is_hint'], viewed)
        self.explanations.update(lambda x: x + author_data['student_support_is_explanation'], viewed)
    
    def __repr__(self):
        return f"{self.author}"

class authorInteraction:
    def __init__(self, author_a: int, author_b: int):
        self.author_a = authorInfo(author_a)
        self.author_b = authorInfo(author_b)
        self.interactions = pv_int()
    
    def update(self, author_a_data: pd.Series, author_b_data: pd.Series, viewed: bool):
        self.interactions.update(lambda x: x + 1, viewed)
        self.author_a.update(author_a_data, viewed)
        self.author_b.update(author_b_data, viewed)
    
    def __repr__(self):
        return f"{self.author_a.author} vs {self.author_b.author}"

def generate_interaction_map(study_data: pd.DataFrame, support_features: pd.DataFrame, creator_ids: List[int]) -> pd.DataFrame:
    # Create index dictionary
    id_index_map: Dict[int, int] = {creator_ids[i]: i for i in range(0, len(creator_ids))}

    # Create default interaction map
    interactions: np.ndarray = np.array([
        [
            authorInteraction(a, b)
            for b in creator_ids
        ]
        for a in creator_ids
    ])

    def preprocess_row(row: pd.Series, support_id: int) -> pd.Series:
        data: pd.Series = support_features.loc[support_id]
        data['student_id'] = row['student_id']
        data['support_selected'] = support_id == row['selected_student_support_id']
        return data

    for _, row in study_data.iterrows():
        support_ids: List[int] = []

        support_ids.append(int(row['selected_student_support_id']))
        support_ids.append(int(row['alternative_student_support_id_1']))
        support_ids.append(int(row['alternative_student_support_id_2']))
        support_ids.append(int(row['alternative_student_support_id_3']))
        support_ids.append(int(row['alternative_student_support_id_4']))
        support_ids = list(filter((-1).__ne__, support_ids))
        support_pairs: List[Tuple[int, int]] = [(a, b) for idx, a in enumerate(support_ids) for b in support_ids[idx + 1:]]

        for support_pair in support_pairs:
            pair_0_data: pd.Series = preprocess_row(row, support_pair[0])
            pair_1_data: pd.Series = preprocess_row(row, support_pair[1])

            interaction: authorInteraction = interactions[tuple(map(id_index_map.__getitem__, sorted((int(pair_0_data['student_support_content_creator_id']), int(pair_1_data['student_support_content_creator_id'])))))]
            if (pair_0_data['student_support_content_creator_id'] < pair_1_data['student_support_content_creator_id']):
                interaction.update(pair_0_data, pair_1_data, row['tutoring_observed'] == 1)
            else:
                interaction.update(pair_1_data, pair_0_data, row['tutoring_observed'] == 1)

    # Write interaction map to DataFrame
    interaction_map: pd.DataFrame = pd.DataFrame(columns=['author_a', 'author_b',
        'interactions', 'viewed_interactions',
        'author_a_students', 'viewed_author_a_students',
        'author_b_students', 'viewed_author_b_students',
        'author_a_hints', 'viewed_author_a_hints',
        'author_a_explanations', 'viewed_author_a_explanations',
        'author_b_hints', 'viewed_author_b_hints',
        'author_b_explanations', 'viewed_author_b_explanations'
    ])

    for interaction_array in interactions:
        for interaction in interaction_array:
            if interaction.interactions.potential == 0:
                continue
            interaction_map = interaction_map.append({
                'author_a': interaction.author_a.author,
                'author_b': interaction.author_b.author,
                'interactions': interaction.interactions.potential,
                'viewed_interactions': interaction.interactions.viewed,
                'author_a_students': len(interaction.author_a.students.potential.difference(interaction.author_b.students.potential)),
                'viewed_author_a_students': len(interaction.author_a.students.viewed.difference(interaction.author_b.students.viewed)),
                'author_b_students': len(interaction.author_b.students.potential.difference(interaction.author_a.students.potential)),
                'viewed_author_b_students': len(interaction.author_b.students.viewed.difference(interaction.author_a.students.viewed)),
                'author_a_hints': interaction.author_a.hints.potential,
                'viewed_author_a_hints': interaction.author_a.hints.viewed,
                'author_a_explanations': interaction.author_a.explanations.potential,
                'viewed_author_a_explanations': interaction.author_a.explanations.viewed,
                'author_b_hints': interaction.author_b.hints.potential,
                'viewed_author_b_hints': interaction.author_b.hints.viewed,
                'author_b_explanations': interaction.author_b.explanations.potential,
                'viewed_author_b_explanations': interaction.author_b.explanations.viewed
            }, ignore_index=True)

    interaction_map = interaction_map.astype(int)
    interaction_map['students'] = interaction_map['author_a_students'] + interaction_map['author_b_students']
    interaction_map['viewed_students'] = interaction_map['viewed_author_a_students'] + interaction_map['viewed_author_b_students']

    return interaction_map.set_index(['author_a', 'author_b'])

def mark_viable_rows(interaction_map: pd.DataFrame, student_threshold: int = 1000, as_treated: bool = False) -> pd.DataFrame:
    interaction_map['can_analyze'] = (interaction_map['students'] > student_threshold) & (interaction_map['author_a_hints'] == 0) & (interaction_map['author_b_hints'] == 0) \
        if as_treated else interaction_map['students'] > student_threshold
    return interaction_map

def burn_data(interaction_map: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    temp_map = interaction_map[interaction_map['can_analyze']]
    burned_series: pd.Series = temp_map.loc[temp_map['students'].idxmin()]
    return burned_series, interaction_map.drop(burned_series.name)

def main() -> int:
    # Load in data
    support_creator_logs: pd.DataFrame = pd.read_csv("student_support_creator_logs.csv")
    support_features: pd.DataFrame = pd.read_csv("student_support_features.csv", index_col="student_support_id")

    # Get Test Period Data

    ## Study 1 (March 1st - April 1st)
    study_1: pd.DataFrame = support_creator_logs.query('timestamp >= 1646092800 and timestamp < 1648771200').copy(deep=True)
    
    ## Study 2 (April 16th - May 16th)
    study_2: pd.DataFrame = support_creator_logs.query('timestamp >= 1650067200 and timestamp < 1652659200').copy(deep=True)

    # Create Creator Id Map
    creator_ids: List[int] = support_features['student_support_content_creator_id'].unique().tolist()
    creator_ids.sort()

    # Generate and Save Interaction Maps
    study_1_interactions: pd.DataFrame = generate_interaction_map(study_1, support_features, creator_ids)
    study_1_interactions.to_csv("study_1_interactions.csv", index=False)
    
    study_2_interactions: pd.DataFrame = generate_interaction_map(study_2, support_features, creator_ids)
    study_2_interactions.to_csv("study_2_interactions.csv", index=False)

    return 0

if __name__ == '__main__':
    sys.exit(main())
