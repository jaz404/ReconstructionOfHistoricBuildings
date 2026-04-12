import sqlite3
import numpy as np

"""
Reusable Helper Class
JOB: Translating Python data into C++ memory formats.

    create_tables(): 
        Sets up the exact SQL schema COLMAP expects (cameras, images, keypoints, two_view_geometries).
    add_keypoints() & add_matches(): 
        They take standard Python numpy arrays, cast them into highly specific data types 
        (like np.float32 for coordinates and np.uint32 for indices), 
        and convert them into binary strings using .tobytes().

This class hides all the ugly binary memory management.
"""

class COLMAPDatabase:
    def __init__(self, database_path):
        self.db = sqlite3.connect(database_path)
        self.cursor = self.db.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER);
            CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, camera_id INTEGER, prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL, prior_ty REAL, prior_tz REAL);
            CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB, PRIMARY KEY(image_id));
            CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB, config INTEGER, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB, PRIMARY KEY(pair_id));
        """)

    def add_camera(self, model, width, height, params):
        params = np.asarray(params, np.float64)
        self.cursor.execute("INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                            (model, width, height, params.tobytes(), False))
        return self.cursor.lastrowid

    def add_image(self, name, camera_id):
        self.cursor.execute("INSERT INTO images (name, camera_id) VALUES (?, ?)", (name, camera_id))
        return self.cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        # COLMAP expects an Nx2 or Nx6 float32 matrix
        keypoints = np.asarray(keypoints, np.float32)
        self.cursor.execute("INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints.tobytes()))

    def add_matches(self, image_id1, image_id2, matches, F_matrix=None):
        # pair_id is a specific bitwise operation required by COLMAP
        pair_id = image_id2 * 2147483647 + image_id1 if image_id1 > image_id2 else image_id1 * 2147483647 + image_id2
        matches = np.asarray(matches, np.uint32)
        F_blob = np.asarray(F_matrix, np.float64).tobytes() if F_matrix is not None else np.zeros((3, 3), np.float64).tobytes()
        self.cursor.execute("INSERT INTO two_view_geometries (pair_id, rows, cols, data, F, config) VALUES (?, ?, ?, ?, ?, ?)",
                            (pair_id, matches.shape[0], matches.shape[1], matches.tobytes(), F_blob, 2)) # config 2 = verified

    def commit(self):
        self.db.commit()