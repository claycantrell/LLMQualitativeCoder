# database.py
import sqlite3
from typing import List
from qual_functions import MeaningUnit, CodeAssigned

def get_connection(db_path: str = 'meaning_units.db') -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    # Enable foreign key support
    conn.execute("PRAGMA foreign_keys = 1")
    return conn

def create_tables(conn: sqlite3.Connection):
    """
    Create MeaningUnit and CodeAssigned tables in the SQLite database.
    """
    cursor = conn.cursor()
    # Create MeaningUnit table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS MeaningUnit (
            unique_id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id TEXT NOT NULL,
            meaning_unit_string TEXT NOT NULL
        )
    ''')
    # Create CodeAssigned table with a foreign key to MeaningUnit
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CodeAssigned (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meaning_unit_id INTEGER NOT NULL,
            code_name TEXT NOT NULL,
            code_justification TEXT NOT NULL,
            FOREIGN KEY (meaning_unit_id) REFERENCES MeaningUnit(unique_id)
                ON DELETE CASCADE
        )
    ''')
    conn.commit()

def insert_meaning_unit(conn: sqlite3.Connection, mu: MeaningUnit):
    """
    Insert a MeaningUnit instance and its associated CodeAssigned instances into the database.
    """
    cursor = conn.cursor()
    # Insert into MeaningUnit
    cursor.execute('''
        INSERT INTO MeaningUnit (speaker_id, meaning_unit_string)
        VALUES (?, ?)
    ''', (mu.speaker_id, mu.meaning_unit_string))
    # Get the generated unique_id
    mu_id = cursor.lastrowid
    # Insert associated CodeAssigned entries
    for code in mu.assigned_code_list:
        cursor.execute('''
            INSERT INTO CodeAssigned (meaning_unit_id, code_name, code_justification)
            VALUES (?, ?, ?)
        ''', (mu_id, code.code_name, code.code_justification))
    conn.commit()

def bulk_insert_meaning_units(conn: sqlite3.Connection, meaning_units: List[MeaningUnit]):
    """
    Insert multiple MeaningUnit instances into the database.
    """
    cursor = conn.cursor()
    for mu in meaning_units:
        # Insert into MeaningUnit
        cursor.execute('''
            INSERT INTO MeaningUnit (speaker_id, meaning_unit_string)
            VALUES (?, ?)
        ''', (mu.speaker_id, mu.meaning_unit_string))
        # Get the generated unique_id
        mu_id = cursor.lastrowid
        # Insert associated CodeAssigned entries
        code_entries = [
            (mu_id, code.code_name, code.code_justification)
            for code in mu.assigned_code_list
        ]
        cursor.executemany('''
            INSERT INTO CodeAssigned (meaning_unit_id, code_name, code_justification)
            VALUES (?, ?, ?)
        ''', code_entries)
    conn.commit()

def fetch_all_meaning_units(conn: sqlite3.Connection) -> List[MeaningUnit]:
    """
    Fetch all MeaningUnit instances from the database along with their associated CodeAssigned entries.
    """
    cursor = conn.cursor()
    # Fetch MeaningUnits
    cursor.execute('SELECT unique_id, speaker_id, meaning_unit_string FROM MeaningUnit')
    meaning_units_data = cursor.fetchall()

    meaning_units = []
    for mu_row in meaning_units_data:
        unique_id, speaker_id, meaning_unit_string = mu_row
        # Fetch associated CodeAssigned entries
        cursor.execute('''
            SELECT code_name, code_justification FROM CodeAssigned
            WHERE meaning_unit_id = ?
        ''', (unique_id,))
        codes_data = cursor.fetchall()
        assigned_codes = [CodeAssigned(code_name=code[0], code_justification=code[1]) for code in codes_data]
        # Create MeaningUnit instance
        mu = MeaningUnit(
            speaker_id=speaker_id,
            meaning_unit_string=meaning_unit_string,
            assigned_code_list=assigned_codes
        )
        mu.unique_id = unique_id  # Manually set the unique_id
        meaning_units.append(mu)

    return meaning_units
