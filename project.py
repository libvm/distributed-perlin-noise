import noise
import numpy as np
import dask
from dask.distributed import Client, SSHCluster
import sqlite3
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

database_name = 'heydb.db'
width = 5000
height = 5000

def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=width,
                                        repeaty=height,
                                        base=seed)
    return world

def save_to_database(final_data):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    for block, seed in final_data:
        cursor.execute("SELECT id FROM perlin_noise WHERE seed = ?", (seed,))
        existing_id = cursor.fetchone()
        if existing_id:
            print(f"Запись с seed {seed} уже существует.")
        else:
            cursor.execute("INSERT INTO perlin_noise (data, seed) VALUES (?, ?)", (block.tobytes(), seed))
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
    scale = 100.0
    octaves = 6
    persistence = 0.0
    lacunarity = 5.0

    data = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed)
    return data, seed

def read_from_database(step: int):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM perlin_noise ORDER BY id")
    results = cursor.fetchall()
    res = [np.frombuffer(result[0], dtype=np.float64).reshape((width, height)) for result in results]
    r = np.block([[res[i+step*j] for j in range(step)]for i in range(step)])
    conn.close()
    return r

def visualize_height_map(height_map):
    x, y = np.meshgrid(np.arange(height_map.shape[1]), np.arange(height_map.shape[0]))
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, height_map, cmap='terrain')
    plt.show()

def main():
    dask.config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler
    cluster = SSHCluster(
    ["localhost", "192.168.61.129", "192.168.61.128"],
    connect_options={"known_hosts": None},
    worker_options={"nthreads": 2, "n_workers": 1},
    scheduler_options={"port":0, "dashboard_address": ":8797"}
)
    client = Client(cluster)
    print(client.dashboard_link)

    t1 = time.time()

    initialize_database()

    block_count = 4
    futures = [dask.delayed(generate_and_save_perlin_noise)(seed) for seed in range(block_count)]
    results = dask.compute(*futures)

    final_data = client.gather(results)

    save_to_database(final_data)

    client.close()
    cluster.close()

    data = read_from_database(2)
    img = np.floor((data+1) * 127).astype(np.uint8)
    smoothed_data = gaussian_filter(img, sigma=10)

    t2 = time.time()

    print(t2-t1)

    visualize_height_map(smoothed_data)

if __name__ == "__main__":
    main()