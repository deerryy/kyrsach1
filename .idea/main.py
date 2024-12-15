import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time
import tracemalloc
import gc
import random  # Добавлено для генерации случайных чисел

# Функция для генерации случайных координат и установки начального и конечного города
def generate_random_data():
    try:
        n = int(entry_cities.get())
        if n < 2 or n > 101:
            raise ValueError("Количество городов должно быть от 2 до 100.")

        global points
        points.clear()  # Очистка списка координат

        for _ in range(n):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            points.append([x, y])

        start_city = random.randint(0, n - 1)
        end_city = random.randint(0, n - 1)
        while end_city == start_city:  # Убедимся, что конечный город отличается от начального
            end_city = random.randint(0, n - 1)

        # Отображение данных в интерфейсе
        create_city_inputs(n)
        for i in range(n):
            entries_coords[i].delete(0, tk.END)
            entries_coords[i].insert(0, f"{points[i][0]:.2f},{points[i][1]:.2f}")

        entry_start.delete(0, tk.END)
        entry_start.insert(0, str(start_city))

        entry_end.delete(0, tk.END)
        entry_end.insert(0, str(end_city))

        messagebox.showinfo("Генерация завершена", "Случайные данные успешно сгенерированы!")

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для вычисления расстояния
def calculate_distance(points):
    return np.linalg.norm(points[0] - points[1])

# Функция для вычисления длины маршрута
def route_length(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

# Основная функция для решения задачи коммивояжера
def tsp_start_end(start, end, distance_matrix):
    n = len(distance_matrix)
    best_route = []
    min_length = float('inf')

    # Перебор всех маршрутов, начиная с start и заканчивая end
    for perm in permutations([i for i in range(n) if i != start and i != end]):
        current_route = [start] + list(perm) + [end]
        current_length = route_length(current_route, distance_matrix)

        if current_length < min_length:
            min_length = current_length
            best_route = current_route

    return min_length, best_route

# Функция для отображения маршрута
def plot_route(points, route, distance_matrix):
    plt.figure(figsize=(10, 8))

    # Отображение всех городов
    for i, (x, y) in enumerate(points):
        city_name = city_names.get(i, f'Город {i}')
        plt.scatter(x, y, label=city_name, s=100)
        plt.annotate(city_name, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    # Отображение всех дорог и их расстояний
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = distance_matrix[i][j]
            plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'grey', linestyle='--')
            mid_x = (points[i][0] + points[j][0]) / 2
            mid_y = (points[i][1] + points[j][1]) / 2
            plt.text(mid_x, mid_y, f'{distance:.2f}', fontsize=8, ha='center')

    # Преобразование маршрута в координаты
    route_coords = [points[i] for i in route]

    # Прорисовка минимального маршрута
    plt.plot(*zip(*route_coords), marker='o', color='red', linewidth=2, label='Оптимальный путь')

    plt.title('Графическое изображение лучшего пути')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid()
    plt.legend()
    plt.axis('equal')
    plt.show()

# Функция обработки ввода через GUI
def solve_tsp():
    try:
        n = int(entry_cities.get())
        if n < 2 or n > 101:
            raise ValueError("Количество городов должно быть от 2 до 100.")

        points.clear()
        for i in range(n):
            coords = entries_coords[i].get().split(',')
            if len(coords) != 2:
                raise ValueError(f"Неверный формат координат для города {i}.")
            x, y = map(float, coords)
            if not (-100 <= x <= 100) or not (-100 <= y <= 100):
                raise ValueError(f"Координаты города {i} должны быть в диапазоне от -100 до 100.")
            points.append([x, y])

        # Проверка на дублирование координат
        unique_points = set(tuple(point) for point in points)
        if len(unique_points) != len(points):
            raise ValueError("Обнаружены дублирующиеся координаты у разных городов.")

        points_array = np.array(points)
        start_city = int(entry_start.get())
        end_city = int(entry_end.get())

        if start_city < 0 or start_city >= n or end_city < 0 or end_city >= n:
            raise ValueError("Индексы городов должны быть в пределах 0 до n-1.")

        # Создание матрицы расстояний
        distance_matrix.clear()
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = calculate_distance([points_array[i], points_array[j]])
        distance_matrix.append(matrix)

        # Решение задачи TSP
        global min_length, best_route
        min_length, best_route = tsp_start_end(start_city, end_city, matrix)

        # Показ результата в интерфейсе
        result_var.set(f"Минимальный путь: {min_length:.2f}\nЛучший маршрут: {best_route}")

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для создания полей ввода с прокруткой
def create_city_inputs(n):
    try:
        if n < 2 or n > 101:
            raise ValueError("Количество городов должно быть от 2 до 100.")

        global entries_coords, city_labels
        entries_coords = []
        city_labels = []

        # Очистка предыдущих элементов в scrollable_frame
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        for i in range(n):
            city_name = city_names.get(i, f'Город {i}')
            label = tk.Label(scrollable_frame, text=f"Координаты {city_name} (x,y):")
            label.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            city_labels.append(label)
            entry = tk.Entry(scrollable_frame)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            entries_coords.append(entry)
            tk.Button(scrollable_frame, text="Назвать город", command=lambda i=i: name_city(i)).grid(row=i, column=2, padx=5, pady=2)

        tk.Label(scrollable_frame, text="Начальный город:").grid(row=n, column=0, padx=5, pady=2, sticky="w")
        global entry_start
        entry_start = tk.Entry(scrollable_frame)
        entry_start.grid(row=n, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(scrollable_frame, text="Конечный город:").grid(row=n+1, column=0, padx=5, pady=2, sticky="w")
        global entry_end
        entry_end = tk.Entry(scrollable_frame)
        entry_end.grid(row=n+1, column=1, padx=5, pady=2, sticky="ew")

        tk.Button(scrollable_frame, text="Рассчитать маршрут", command=solve_tsp).grid(row=n+2, column=0, columnspan=2, pady=10)

        global result_var
        result_var = tk.StringVar()
        tk.Label(scrollable_frame, textvariable=result_var, fg="blue").grid(row=n+3, column=0, columnspan=2)

        tk.Button(scrollable_frame, text="Показать график", command=show_graph).grid(row=n+4, column=0, columnspan=2, pady=5)

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для показа графика
def show_graph():
    try:
        if len(points) >= 10:
            raise ValueError("Построение графика доступно только для количества городов меньше 10.")

        # Проверка индексов маршрута
        if not best_route:
            raise ValueError("Сначала рассчитайте маршрут!")

        for index in best_route:
            if index < 0 or index >= len(points):
                raise ValueError(f"Индекс {index} из маршрута выходит за пределы допустимого диапазона.")

        # Построение графика
        plot_route(points, best_route, distance_matrix[0])
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для анализа производительности
def analyze_performance():
    try:
        times = []
        memories = []
        city_counts = range(2, 8)
        fixed_points = [[1, 3], [2, 4]]  # Начальные точки для первых двух городов

        print("Анализ производительности:")
        for n in city_counts:
            # Если количество городов больше текущих точек, добавляем новые случайные координаты
            while len(fixed_points) < n:
                new_point = np.random.uniform(-100, 100, size=(1, 2)).tolist()[0]
                fixed_points.append(new_point)

            # Подготовка данных для текущего числа городов
            points = np.array(fixed_points[:n])

            # Создание матрицы расстояний
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        distance_matrix[i][j] = calculate_distance([points[i], points[j]])

            # Установка начального и конечного города
            start_city = 0
            end_city = n - 1

            # Принудительный сбор мусора перед измерениями
            gc.collect()

            # Измерение времени
            tracemalloc.start()
            initial_memory, _ = tracemalloc.get_traced_memory()  # Память до запуска алгоритма
            start_time = time.perf_counter()  # Используем perf_counter для более точного времени

            tsp_start_end(start_city, end_city, distance_matrix)

            elapsed_time = (time.perf_counter() - start_time) * 1000  # Перевод времени в миллисекунды
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Логирование данных
            print(f"Города: {n}")
            print(f"  Время выполнения: {elapsed_time:.4f} мс")
            print(f"  Используемая память (до алгоритма): {initial_memory / 1024:.2f} КБ")
            print(f"  Используемая память (после алгоритма): {current_memory / 1024:.2f} КБ")
            print(f"  Пиковая память: {peak_memory / 1024:.2f} КБ")

            # Сохраняем результаты
            times.append(elapsed_time)
            memories.append(current_memory / 1024)  # Память в КБ

            # Полный сброс данных для текущего количества городов
            del points
            del distance_matrix
            gc.collect()

        # Построение графиков
        plt.figure(figsize=(12, 6))

        # График времени
        plt.subplot(1, 2, 1)
        plt.plot(city_counts, times, marker='o', label="Время выполнения")
        plt.xlabel("Количество городов")
        plt.ylabel("Время (мс)")
        plt.title("Время выполнения")
        plt.grid()
        plt.legend()

        # График памяти
        plt.subplot(1, 2, 2)
        plt.plot(city_counts, memories, marker='o', color='red', label="Используемая память")
        plt.xlabel("Количество городов")
        plt.ylabel("Память (КБ)")
        plt.title("Использование памяти")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для загрузки данных из файла
def load_from_file():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return

        with open(file_path, 'r') as file:
            lines = file.readlines()

        n = int(lines[0].strip())  # Читаем количество городов
        if n < 2 or n > 101:
            raise ValueError("Количество городов должно быть от 2 до 100.")

        points.clear()
        for i in range(1, n+1):
            x, y = map(float, lines[i].strip().split(','))
            if not (-100 <= x <= 100) or not (-100 <= y <= 100):
                raise ValueError(f"Координаты города {i-1} должны быть в диапазоне от -100 до 100.")
            points.append([x, y])

        # Чтение начального и конечного города
        start_city, end_city = map(int, lines[n+1].strip().split(','))
        if start_city < 0 or start_city >= n or end_city < 0 or end_city >= n:
            raise ValueError("Индексы городов должны быть в пределах от 0 до n-1.")

        # Отображение данных в интерфейсе
        entry_cities.delete(0, tk.END)
        entry_cities.insert(0, str(n))

        # Обновление списка координат
        create_city_inputs(n)
        for i in range(n):
            entries_coords[i].delete(0, tk.END)
            entries_coords[i].insert(0, f"{points[i][0]},{points[i][1]}")

        entry_start.delete(0, tk.END)
        entry_start.insert(0, str(start_city))

        entry_end.delete(0, tk.END)
        entry_end.insert(0, str(end_city))

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Функция для именования города
def name_city(index):
    try:
        city_name = simpledialog.askstring("Имя города", f"Введите имя для города {index}:")
        if city_name:
            city_names[index] = city_name
            city_labels[index].config(text=f"Координаты {city_name} (x,y):")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Создание GUI
root = tk.Tk()
root.title("TSP Решение")

# Добавляем прокрутку
frame_canvas = tk.Frame(root)
frame_canvas.grid(row=1, column=0, columnspan=6, sticky="nsew")  # Изменено: Создан контейнер для Canvas и Scrollbar

# Создаем Canvas и Scrollbar сразу после основного окна root
frame_canvas = tk.Frame(root)
frame_canvas.grid(row=1, column=0, columnspan=6, sticky="nsew")  # Контейнер для Canvas и Scrollbar

canvas = tk.Canvas(frame_canvas)  # Canvas для отображения содержимого
scrollbar = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)  # Вертикальная прокрутка
scrollable_frame = tk.Frame(canvas)  # Фрейм для размещения всех элементов внутри Canvas

# Добавление кнопки для генерации случайных данных в GUI
tk.Button(root, text="Сгенерировать случайные данные", command=generate_random_data).grid(row=0, column=5)

# Настройка Canvas и прокрутки
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  # Обновляем область прокрутки
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")  # Добавляем scrollable_frame в Canvas
canvas.configure(yscrollcommand=scrollbar.set)  # Связываем Scrollbar с Canvas

# Размещение Canvas и Scrollbar
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Обновление весов для адаптивной прокрутки
frame_canvas.grid_rowconfigure(0, weight=1)
frame_canvas.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Поле для ввода количества городов
tk.Label(root, text="Количество городов:").grid(row=0, column=0)
entry_cities = tk.Entry(root)
entry_cities.grid(row=0, column=1)
tk.Button(root, text="Установить", command=lambda: create_city_inputs(int(entry_cities.get()))).grid(row=0, column=2)

# Кнопка для загрузки данных из файла и анализа
tk.Button(root, text="Загрузить данные из файла", command=load_from_file).grid(row=0, column=3)
tk.Button(root, text="Анализ производительности", command=analyze_performance).grid(row=0, column=4)

entries_coords = []
points = []
distance_matrix = []
min_length = None
best_route = []
city_names = {}  # Словарь для хранения имен городов
city_labels = []  # Список для хранения меток городов

root.mainloop()
