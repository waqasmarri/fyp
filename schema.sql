DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS captions;
DROP TABLE IF EXISTS objects;

CREATE TABLE images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_path TEXT NOT NULL,
  date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
);

CREATE TABLE IF NOT EXISTS captions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  caption TEXT NOT NULL,
  accuracy INTEGER NOT NULL ,
  image_id INTEGER NOT NULL,
  FOREIGN KEY (image_id) REFERENCES images (id)
);

CREATE TABLE IF NOT EXISTS objects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  object_name TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  FOREIGN KEY (image_id) REFERENCES images (id)
);