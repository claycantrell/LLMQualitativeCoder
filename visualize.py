# visualize.py

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Connect to the SQLite database
    conn = sqlite3.connect('meaning_units.db')

    try:
        # Read MeaningUnit table into a pandas DataFrame
        meaning_units_df = pd.read_sql_query("SELECT * FROM MeaningUnit", conn)

        # Read CodeAssigned table into a pandas DataFrame
        code_assigned_df = pd.read_sql_query("SELECT * FROM CodeAssigned", conn)

        # Merge the DataFrames on the unique_id and meaning_unit_id columns
        merged_df = pd.merge(meaning_units_df, code_assigned_df, left_on='unique_id', right_on='meaning_unit_id')

        # Visualization 1: For each speaker, display the most commonly assigned codes
        # Group by speaker_id and code_name to get counts
        speaker_code_counts = merged_df.groupby(['speaker_id', 'code_name']).size().reset_index(name='counts')

        # For each speaker, find their most commonly assigned codes
        # We'll consider all codes assigned to the speaker and sort them by count
        speakers = speaker_code_counts['speaker_id'].unique()
        num_speakers = len(speakers)

        # Set up subplots for each speaker
        fig, axes = plt.subplots(nrows=num_speakers, ncols=1, figsize=(10, 5 * num_speakers))
        if num_speakers == 1:
            axes = [axes]  # Ensure axes is iterable

        for ax, speaker in zip(axes, speakers):
            # Get data for the current speaker
            data = speaker_code_counts[speaker_code_counts['speaker_id'] == speaker]
            # Sort codes by counts in descending order
            data = data.sort_values('counts', ascending=False)
            # Plot the codes and their counts
            ax.bar(data['code_name'], data['counts'], color='skyblue')
            ax.set_title(f"Most Commonly Assigned Codes for Speaker {speaker}")
            ax.set_xlabel('Code Name')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(data['code_name'])))
            ax.set_xticklabels(data['code_name'], rotation=45, ha='right')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

        # Visualization 2: Pie chart showing all assigned codes and their frequency
        code_frequencies = merged_df['code_name'].value_counts().reset_index()
        code_frequencies.columns = ['code_name', 'counts']

        # Handle too many slices in the pie chart by grouping less frequent codes
        max_slices = 10
        if len(code_frequencies) > max_slices:
            top_codes = code_frequencies[:max_slices - 1]
            other_counts = code_frequencies['counts'][max_slices - 1:].sum()
            # Create a DataFrame for 'Other' category
            other_row = pd.DataFrame({'code_name': ['Other'], 'counts': [other_counts]})
            # Concatenate the top codes with the 'Other' row
            top_codes = pd.concat([top_codes, other_row], ignore_index=True)
            code_frequencies = top_codes

        plt.figure(figsize=(8, 8))
        plt.pie(code_frequencies['counts'], labels=code_frequencies['code_name'], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Codes Assigned')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    main()

