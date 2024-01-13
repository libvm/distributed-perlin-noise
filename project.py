import noise
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import sqlite3
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


database_name = 'heydb.db'
width = 100
height = 100

def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=seed)
    return world

def save_to_database(data, seed):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM perlin_noise WHERE seed = ?", (seed,))
    existing_id = cursor.fetchone()

    if existing_id:
        print(f"Запись с seed {seed} уже существует.")
    else:
        cursor.execute("INSERT INTO perlin_noise (data, seed) VALUES (?, ?)", (data.tobytes(), seed))
        print(f"Запись с seed {seed} добавлена в базу данных.")

    conn.commit()
    conn.close()

def initialize_database():
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS perlin_noise (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data BLOB,
            seed INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def generate_and_save_perlin_noise(seed):
    scale = 50.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM perlin_noise WHERE seed = ?", (seed,))
    existing_id = cursor.fetchone()

    if existing_id:
        print(f"Запись с seed {seed} уже существует.")
    else:
        data = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed)
        cursor.execute("INSERT INTO perlin_noise (data, seed) VALUES (?, ?)", (data.tobytes(), seed))
        print(f"Запись с seed {seed} добавлена в базу данных.")

    conn.commit()
    conn.close()

def read_from_database():
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM perlin_noise ORDER BY id")
    results = cursor.fetchall()
    res = [np.frombuffer(result[0], dtype=np.float64).reshape((width, height)) for result in results]
    r = np.block([[res[i+10*j] for j in range(10)]for i in range(10)])
    conn.close()
    return r

def visualize_height_map(height_map):
    x, y = np.meshgrid(np.arange(height_map.shape[1]), np.arange(height_map.shape[0]))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, height_map, cmap='terrain')
    plt.show()

def main():
    cluster = LocalCluster()
    client = Client(cluster)

    initialize_database()
    futures = [dask.delayed(generate_and_save_perlin_noise)(seed) for seed in range(100)]
    dask.compute(*futures)

    client.close()
    cluster.close()

    data = read_from_database()
    smoothed_data = gaussian_filter(data, sigma=50)

    visualize_height_map(data)

if __name__ == "__main__":
    main()