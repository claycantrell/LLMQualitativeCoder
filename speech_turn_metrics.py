# speech_turn_metrics.py

import json
import sys
from collections import defaultdict
from typing import List, Dict
import os


class Cue:
    """
    Represents a single cue from the JSON file.
    """

    def __init__(self, cue_id: int, duration: float, text: str, speaker: str):
        self.id = cue_id
        self.duration = duration
        self.text = text
        self.speaker = speaker

    @classmethod
    def from_json(cls, data: Dict):
        """
        Creates a Cue instance from a JSON object.
        """
        try:
            cue_id = data.get('id')
            duration = float(data.get('length_of_time_spoken_seconds', 0.0))
            text = data.get('text_context', '')
            speaker = data.get('speaker_name', 'Unknown')
            return cls(cue_id, duration, text, speaker)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in cue ID {data.get('id', 'Unknown')}: {e}")


class Speaker:
    """
    Represents a speaker and aggregates metrics.
    """

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0  # in seconds
        self.speaking_turns = 0
        self.word_count = 0

    def add_cue(self, cue: Cue):
        """
        Adds a cue's data to the speaker's metrics.
        """
        self.total_time += cue.duration
        self.speaking_turns += 1
        self.word_count += len(cue.text.split())

    def calculate_metrics(self):
        """
        Calculates additional metrics for the speaker.
        """
        self.avg_words_per_turn = self.word_count / self.speaking_turns if self.speaking_turns > 0 else 0
        self.avg_duration_per_turn = self.total_time / self.speaking_turns if self.speaking_turns > 0 else 0
        self.words_per_minute = (self.word_count / (self.total_time / 60)) if self.total_time > 0 else 0

    def get_metrics(self, total_talking_time: float):
        """
        Returns the calculated metrics as a dictionary.
        """
        percentage_time = (self.total_time / total_talking_time) * 100 if total_talking_time > 0 else 0
        return {
            'Participant': self.name,
            'Total Time(s)': round(self.total_time, 2),
            '% Time': round(percentage_time, 2),
            'Turns': self.speaking_turns,
            'Word Count': self.word_count,
            'Avg Words/Turn': round(self.avg_words_per_turn, 2),
            'Avg Duration/Turn(s)': round(self.avg_duration_per_turn, 2),
            'WPM': round(self.words_per_minute, 2)
        }


class TranscriptAnalyzer:
    """
    Analyzes the transcript from a JSON file and computes metrics per speaker.
    """

    def __init__(self, json_file: str):
        self.json_file = json_file
        self.cues: List[Cue] = []
        self.speakers: Dict[str, Speaker] = {}
        self.total_talking_time = 0.0

    def load_cues(self):
        """
        Loads cues from the JSON file.
        """
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of cue objects.")
            for item in data:
                cue = Cue.from_json(item)
                self.cues.append(cue)
        except FileNotFoundError:
            print(f"Error: File '{self.json_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON. {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def process_cues(self):
        """
        Processes each cue and aggregates data per speaker.
        """
        for cue in self.cues:
            speaker_name = cue.speaker if cue.speaker else 'Unknown'
            if speaker_name not in self.speakers:
                self.speakers[speaker_name] = Speaker(speaker_name)
            self.speakers[speaker_name].add_cue(cue)
            self.total_talking_time += cue.duration

    def calculate_metrics(self):
        """
        Calculates additional metrics for each speaker.
        """
        for speaker in self.speakers.values():
            speaker.calculate_metrics()

    def display_results(self):
        """
        Displays the calculated metrics for each speaker.
        """
        if self.total_talking_time == 0:
            print("No talking time recorded.")
            return

        print("\n=== Interview Transcript Analysis ===\n")
        print(f"Total Talking Time: {round(self.total_talking_time, 2)} seconds\n")

        header = (
            f"{'Participant':<20} {'Total Time(s)':<15} {'% Time':<10} "
            f"{'Turns':<10} {'Word Count':<12} {'Avg Words/Turn':<16} "
            f"{'Avg Duration/Turn(s)':<22} {'WPM':<10}"
        )
        print(header)
        print("-" * len(header))

        for speaker in self.speakers.values():
            metrics = speaker.get_metrics(self.total_talking_time)
            print(
                f"{metrics['Participant']:<20} {metrics['Total Time(s)']:<15} {metrics['% Time']:<10} "
                f"{metrics['Turns']:<10} {metrics['Word Count']:<12} {metrics['Avg Words/Turn']:<16} "
                f"{metrics['Avg Duration/Turn(s)']:<22} {metrics['WPM']:<10}"
            )

        print("\n=== End of Analysis ===\n")

    def run_analysis(self):
        """
        Executes the full analysis workflow.
        """
        self.load_cues()
        self.process_cues()
        self.calculate_metrics()
        self.display_results()


def main():
    """
    Main function to execute the speech time and metrics calculator.
    """
    # Use the default filename "output_cues.json" in the json_transcripts folder
    filename = "output_cues.json"
    json_file = os.path.join('json_transcripts', filename)

    # Check if the file exists
    if not os.path.isfile(json_file):
        print(f"Error: File '{json_file}' not found in the 'json_transcripts' folder.")
        sys.exit(1)

    analyzer = TranscriptAnalyzer(json_file)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
