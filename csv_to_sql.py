import csv
import sqlite3

connection = sqlite3.connect("steam_games_recommendations.db")

# Create games table.
with open("dataset/games.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    columns = ", ".join(header)
    create_table_query = f"CREATE TABLE IF NOT EXISTS games ({columns})"
    connection.execute(create_table_query)

# Create users table.
with open("dataset/users.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    columns = ", ".join(header)
    create_table_query = f"CREATE TABLE IF NOT EXISTS users ({columns})"
    connection.execute(create_table_query)

# Create recommendations table.
with open("dataset/recommendations.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    columns = ", ".join(header)
    create_table_query = f"CREATE TABLE IF NOT EXISTS recommendations ({columns})"
    connection.execute(create_table_query)

# 

connection.commit()
connection.close()
