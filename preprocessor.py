# Import necessary libraries
import pandas as pd
import sys

from aos_constants import drop_not_columns

# Preprocess data into correct format
def preprocess(support_logs: pd.DataFrame, support_features: pd.DataFrame) -> pd.DataFrame:

    # Map Support Id to its Content Creator
    def support_to_creator(index: int) -> int:
        if index == -1:
            return -1
        return support_features.loc[index]['student_support_content_creator_id']

    # Filter support logs
    support_logs = support_logs[support_logs['ambiguous_problem_log'] == 0]
    support_logs = support_logs[support_logs['no_problem_log'] == 0]

    # If there is a log but the student did not complete the assignment, return 0
    support_logs['problem_accuracy'] = support_logs['problem_accuracy'].fillna(0)

    # Set NaN Support Ids to -1
    support_logs['selected_student_support_id'] = support_logs['selected_student_support_id'].fillna(-1)
    support_logs['alternative_student_support_id_1'] = support_logs['alternative_student_support_id_1'].fillna(-1)
    support_logs['alternative_student_support_id_2'] = support_logs['alternative_student_support_id_2'].fillna(-1)
    support_logs['alternative_student_support_id_3'] = support_logs['alternative_student_support_id_3'].fillna(-1)
    support_logs['alternative_student_support_id_4'] = support_logs['alternative_student_support_id_4'].fillna(-1)

    # Add Creators for Each Support
    support_logs['selected_student_support_creator_id'] = support_logs['selected_student_support_id'].map(support_to_creator)
    support_logs['alternative_student_support_creator_id_1'] = support_logs['alternative_student_support_id_1'].map(support_to_creator)
    support_logs['alternative_student_support_creator_id_2'] = support_logs['alternative_student_support_id_2'].map(support_to_creator)
    support_logs['alternative_student_support_creator_id_3'] = support_logs['alternative_student_support_id_3'].map(support_to_creator)
    support_logs['alternative_student_support_creator_id_4'] = support_logs['alternative_student_support_id_4'].map(support_to_creator)

    return support_logs

def knowledge_separation(stats: pd.DataFrame) -> pd.DataFrame:
    student_rank: pd.Series = stats.sort_values('student_avg_accuracy')['student_avg_accuracy'].groupby(lambda _: True).cumcount()
    n_students: int = len(student_rank) / 3

    knowledge_groups: pd.DataFrame = pd.DataFrame(index=student_rank.index)

    knowledge_groups['low_knowledge'] = knowledge_groups.index.map(lambda i: student_rank.loc[i] < n_students).astype(int)
    knowledge_groups['mid_knowledge'] = knowledge_groups.index.map(lambda i: student_rank.loc[i] >= n_students and student_rank.loc[i] < 2 * n_students).astype(int)
    knowledge_groups['high_knowledge'] = knowledge_groups.index.map(lambda i: student_rank.loc[i] >= 2 * n_students).astype(int)
    return knowledge_groups

# 1 is Male
# 0 is Female
def preprocess_gender_info(gender_info: pd.DataFrame) -> pd.DataFrame:
    gender_info = gender_info[(gender_info['gender'] == "M") | (gender_info['gender'] == "F")]
    gender_info.loc[:, 'gender'] = gender_info['gender'].map(lambda g: 1 if g == "M" else 0)
    gender_info.index = gender_info.index.rename('student_id')

    return drop_not_columns(gender_info, ['gender'])

def main() -> int:
    # Load in data
    support_logs: pd.DataFrame = pd.read_csv("student_support_logs.csv")
    support_features: pd.DataFrame  = pd.read_csv("student_support_features.csv", index_col="student_support_id")
    student_prior_stats: pd.DataFrame = pd.read_csv("student_prior_stats.csv", index_col="student_id")
    infered_student_gender: pd.DataFrame = pd.read_csv("infered_student_gender.csv", index_col="user_id")

    # Run Preprocessor
    filtered_logs: pd.DataFrame = preprocess(support_logs, support_features)

    # Write creator logs to output
    filtered_logs.to_csv("student_support_creator_logs.csv", index=False)

    # Create knowledge groups
    knowledge_separation(student_prior_stats).to_csv("student_knowledge_groups.csv")

    # Process gender information
    preprocess_gender_info(infered_student_gender).to_csv("student_gender_info.csv")

    return 0

if __name__ == '__main__':
    sys.exit(main())
