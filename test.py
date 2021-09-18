import sqlite3

conn = sqlite3.connect('database.db')

cur = conn.cursor()
# cur.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT,image_path TEXT NOT NULL,date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP);")
# cur.execute("CREATE TABLE IF NOT EXISTS captions (id INTEGER PRIMARY KEY AUTOINCREMENT,caption TEXT NOT NULL,accuracy INTEGER NOT NULL ,image_id INTEGER NOT NULL,FOREIGN KEY (image_id) REFERENCES images (id));")
# cur.execute("CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY AUTOINCREMENT,object_name TEXT NOT NULL,image_id INTEGER NOT NULL,FOREIGN KEY (image_id) REFERENCES images (id));")

# conn.commit()