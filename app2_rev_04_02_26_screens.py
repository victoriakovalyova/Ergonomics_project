import serial
import struct
import numpy as np
import cv2
import math
from scipy import ndimage
import os
from datetime import datetime
from pathlib import Path

#BLUE_THRESHOLD = 5000 

FRAME_WIDTH = 18
FRAME_HEIGHT = 10
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT
BYTES_PER_FRAME = FRAME_SIZE * 4
START_MARKER = b'\xAA\x55\xAA'
MAX_BUFFER = 4096

EXTRA_COLS = 1
ROWS = FRAME_HEIGHT
COLS = FRAME_WIDTH + 2 * EXTRA_COLS

# Получаем разрешение экрана для полноэкранного режима
screen_info = {}
try:
    # Способ 1: Через X11 (для Linux)
    if os.name == 'posix':
        try:
            import subprocess
            output = subprocess.check_output(['xrandr']).decode('utf-8')
            for line in output.splitlines():
                if ' connected primary' in line or ' connected ' in line:
                    import re
                    match = re.search(r'(\d+)x(\d+)', line)
                    if match:
                        screen_info['width'] = int(match.group(1))
                        screen_info['height'] = int(match.group(2))
                        print(f"Определено разрешение экрана: {screen_info['width']}x{screen_info['height']}")
        except:
            pass
    
    # Способ 2: Резервный вариант
    if not screen_info:
        import tkinter as tk
        root = tk.Tk()
        screen_info['width'] = root.winfo_screenwidth()
        screen_info['height'] = root.winfo_screenheight()
        root.destroy()
        print(f"Определено разрешение экрана (через tkinter): {screen_info['width']}x{screen_info['height']}")
except Exception as e:
    print(f"Не удалось определить разрешение экрана, использую 1920x1080. Ошибка: {e}")
    screen_info = {'width': 1920, 'height': 1080}

# Устанавливаем размер окна равным разрешению экрана для полноэкранного режима
WINDOW_WIDTH = screen_info['width']
WINDOW_HEIGHT = screen_info['height']
CELL_SIZE = max(70, min(80, WINDOW_HEIGHT // 20))  # Адаптивный размер ячейки
current_interpolation_method = cv2.INTER_LANCZOS4

ser = serial.Serial("/dev/ttyS3", 115200, timeout=1)
print("Waiting for frame start...")

buffer = b""

# Глобальные переменные для управления
recorded_screenshots = []  # Список для хранения путей к скриншотам за сеанс
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Уникальный ID сеанса
output_dir = Path("thermal_screenshots")  # Директория для сохранения

# Создаем директорию если не существует
output_dir.mkdir(exist_ok=True)

# Константы для кнопок (адаптируем под новый размер окна)
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 180
BUTTON_MARGIN = 20
BUTTON_Y = WINDOW_HEIGHT - 70  # Позиция Y для кнопок
BUTTON1_X = WINDOW_WIDTH // 2 - BUTTON_WIDTH - BUTTON_MARGIN
BUTTON2_X = WINDOW_WIDTH // 2 + BUTTON_MARGIN

# Состояния кнопок (нажата/не нажата)
button_save_state = False
button_compare_state = False
button_save_clicked = False
button_compare_clicked = False

GL_COUNTER = 0
timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
DIR_NAME = f"{timestamp1}"
os.mkdir(f"thermal_screenshots/{DIR_NAME}")

# Список для хранения имен окон сравнения
comparison_windows = []



def get_screen_resolution():
    """Возвращает разрешение экрана без использования tkinter"""
    try:
        # Способ 1: Через X11 (для Linux)
        if os.name == 'posix':
            try:
                import subprocess
                output = subprocess.check_output(['xrandr']).decode('utf-8')
                for line in output.splitlines():
                    if ' connected primary' in line or ' connected ' in line:
                        # Ищем разрешение в формате "1920x1080"
                        import re
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            width = int(match.group(1))
                            height = int(match.group(2))
                            return {'width': width, 'height': height}
            except:
                pass
        
        # Способ 2: Через ctypes (работает на Windows и Linux)
        try:
            import ctypes
            # Для Windows
            if os.name == 'nt':
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return {'width': width, 'height': height}
            # Для Linux с X11
            elif os.name == 'posix':
                # Пробуем через Xlib
                try:
                    from Xlib import display
                    d = display.Display()
                    s = d.screen()
                    width = s.width_in_pixels
                    height = s.height_in_pixels
                    return {'width': width, 'height': height}
                except ImportError:
                    # Если Xlib не установлен, пробуем другой способ
                    pass
        except:
            pass
        
        # Способ 3: Через переменные окружения (может работать в некоторых окружениях)
        try:
            if 'DISPLAY' in os.environ:
                # Пробуем использовать xdpyinfo
                import subprocess
                result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if 'dimensions:' in line:
                        import re
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            width = int(match.group(1))
                            height = int(match.group(2))
                            return {'width': width, 'height': height}
        except:
            pass
        
    except Exception as e:
        print(f"Ошибка при определении разрешения экрана: {e}")
    
    # Способ 4: Значение по умолчанию
    print("Не удалось определить разрешение экрана, использую 1920x1080")
    return {'width': 1920, 'height': 1080}



def combine_matrix_parts(upper_part, lower_part):
    """Combine upper and lower parts into one 10x18 matrix"""
    combined_matrix = np.zeros((10, 18), dtype=int)
    combined_matrix[:, :9] = upper_part  # Left half
    combined_matrix[:, 9:] = lower_part  # Right half
    return combined_matrix


def extend_matrix_with_zeros(original_matrix):
    """Add zero columns to the sides of the original matrix"""
    extended_matrix = np.zeros((ROWS, COLS))
    start_col = EXTRA_COLS
    extended_matrix[:, start_col:start_col + FRAME_WIDTH] = original_matrix
    return extended_matrix


#SENSIVITY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
BLUE_THRESHOLD = 5000

def value_to_color(adc, adc_min=500, adc_max=19500):
    # 1. Жесткая отсечка фона
    if adc <= BLUE_THRESHOLD:
        return (140, 0, 0) # Dark Blue

    # 2. Ограничиваем сверху, чтобы не выходить за красный
    if adc >= adc_max:
        return (0, 0, 255) # Чистый красный

    # 3. Нормализация внутри активного диапазона [BLUE_THRESHOLD ... 30000]
    # Именно это меняет "плавность": чем меньше разница, тем резче градиент
    active_range = adc_max - BLUE_THRESHOLD
    norm = (adc - BLUE_THRESHOLD) / active_range
    
    # 4. Расчет плавных переходов (BGR)
    # Мы делим спектр на 5 равных отрезков внутри активного диапазона
    
    if norm < 0.2:
        # Темно-синий -> Голубой
        seg = norm / 0.2
        return (int(140 + 115 * seg), int(200 * seg), 0)
    
    elif norm < 0.4:
        # Голубой -> Зеленый
        seg = (norm - 0.2) / 0.2
        return (int(255 - 255 * seg), int(200 + 55 * seg), 0)
    
    elif norm < 0.6:
        # Зеленый -> Желтый
        seg = (norm - 0.4) / 0.2
        return (0, 255, int(255 * seg))
    
    elif norm < 0.8:
        # Желтый -> Оранжевый
        seg = (norm - 0.6) / 0.2
        return (0, int(255 - 105 * seg), 255)
    
    else:
        # Оранжевый -> Красный
        seg = (norm - 0.8) / 0.2
        return (0, int(150 - 150 * seg), 255)


def create_smooth_visualization(extended_matrix, original_matrix):
    """Interpolate temperatures, then convert to colors"""
    img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 50

    # STEP 1: Mirror the matrix vertically
    mirrored_matrix = np.flipud(extended_matrix)
    mirrored_original = np.flipud(original_matrix)

    # STEP 2: Interpolate the temperature matrix
    SCALE_FACTOR = 6

    # Use scipy for numerical data interpolation
    enlarged_temperatures = ndimage.zoom(mirrored_matrix, SCALE_FACTOR, order=3)

    # STEP 3: Convert interpolated temperatures to colors
    enlarged_rows = ROWS * SCALE_FACTOR
    enlarged_cols = COLS * SCALE_FACTOR
    enlarged_colored = np.zeros((enlarged_rows, enlarged_cols, 3), dtype=np.uint8)

    for i in range(enlarged_rows):
        for j in range(enlarged_cols):
            temp_value = enlarged_temperatures[i, j]
            enlarged_colored[i, j] = value_to_color(temp_value)

    # STEP 4: Final stretching
    matrix_display_width = COLS * CELL_SIZE
    matrix_display_height = ROWS * CELL_SIZE

    display_matrix = cv2.resize(enlarged_colored,
                                (matrix_display_width, matrix_display_height),
                                interpolation=current_interpolation_method)

    start_x = (WINDOW_WIDTH - matrix_display_width) // 2
    start_y = (WINDOW_HEIGHT - matrix_display_height) // 2 - 50  # Увеличили отступ для кнопок

    # Insert matrix into main image
    end_y = start_y + matrix_display_height
    end_x = start_x + matrix_display_width

    # Make sure we don't go out of image bounds
    if start_y >= 0 and start_x >= 0 and end_y <= WINDOW_HEIGHT and end_x <= WINDOW_WIDTH:
        img[start_y:end_y, start_x:end_x] = display_matrix

    # Add title and information
    method_names = {
        cv2.INTER_NEAREST: 'NEAREST',
        cv2.INTER_LINEAR: 'LINEAR',
        cv2.INTER_CUBIC: 'CUBIC',
        cv2.INTER_LANCZOS4: 'LANCZOS4'
    }
    method_name = method_names.get(current_interpolation_method, 'UNKNOWN')

    title = f"Temperature Matrix - OpenCV (Interpolation: {method_name})"
    cv2.putText(img, title, (WINDOW_WIDTH // 2 - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add statistics
    max_val = np.max(mirrored_original)
    min_val = np.min(mirrored_original)
    avg_val = np.mean(mirrored_original)

    stats_text = f"Max: {max_val:.0f}  Min: {min_val:.0f}  Avg: {avg_val:.1f}"
    cv2.putText(img, stats_text, (WINDOW_WIDTH // 2 - 100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Add range information
    range_text = f"Range: 1000 - 30000"
    cv2.putText(img, range_text, (WINDOW_WIDTH // 2 - 100, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Add legend
    add_legend(img, start_y + matrix_display_height + 10)

    # Добавляем кнопки
    draw_buttons(img)

    # Add info about saved screenshots
    if recorded_screenshots:
        cv2.putText(img, f"Saved: {len(recorded_screenshots)}", 
                    (WINDOW_WIDTH - 1200, 800), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 1)

    return img


def create_basic_visualization(extended_matrix, original_matrix):
    """Basic visualization without interpolation"""
    img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 50

    # Mirror the matrix vertically
    mirrored_matrix = np.flipud(extended_matrix)
    mirrored_original = np.flipud(original_matrix)

    # Calculate matrix area dimensions
    matrix_width = COLS * CELL_SIZE
    matrix_height = ROWS * CELL_SIZE

    # Position matrix at center
    start_x = (WINDOW_WIDTH - matrix_width) // 2
    start_y = (WINDOW_HEIGHT - matrix_height) // 2 - 40

    # Draw matrix
    for i in range(ROWS):
        for j in range(COLS):
            value = mirrored_matrix[i, j]
            color = value_to_color(value)

            # Cell coordinates
            x1 = start_x + j * CELL_SIZE
            y1 = start_y + i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            # Draw cell fill
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            # Draw cell border
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Add text for original matrix
            start_col = EXTRA_COLS
            if start_col <= j < start_col + FRAME_WIDTH:
                orig_i = i
                orig_j = j - start_col
                orig_value = mirrored_original[orig_i, orig_j]

                # Text color based on background
                text_color = (255, 255, 255) if value < 20000 else (0, 0, 0)

                # Add value
                cv2.putText(img, str(orig_value),
                            (x1 + CELL_SIZE // 2 - 10, y1 + CELL_SIZE // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Add title
    cv2.putText(img, "Temperature Matrix - OpenCV (No Interpolation)",
                (WINDOW_WIDTH // 2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add statistics
    max_val = np.max(mirrored_original)
    min_val = np.min(mirrored_original)
    avg_val = np.mean(mirrored_original)

    stats_text = f"Max: {max_val:.0f}  Min: {min_val:.0f}  Avg: {avg_val:.1f}"
    cv2.putText(img, stats_text, (WINDOW_WIDTH // 2 - 100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Add range information
    range_text = f"Range: 1000 - 30000"
    cv2.putText(img, range_text, (WINDOW_WIDTH // 2 - 100, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Add legend
    add_legend(img, start_y + matrix_height + 10)

    # Добавляем кнопки
    draw_buttons(img)

    # Add info about saved screenshots
    if recorded_screenshots:
        cv2.putText(img, f"Saved: {len(recorded_screenshots)}", 
                    (WINDOW_WIDTH - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 1)

    return img


def create_visualization(extended_matrix, original_matrix, use_interpolation=True):
    """Creates matrix visualization with optional interpolation"""
    if use_interpolation:
        return create_smooth_visualization(extended_matrix, original_matrix)
    else:
        return create_basic_visualization(extended_matrix, original_matrix)


def draw_buttons(img):
    """Рисует кнопки в интерфейсе"""
    global button_save_state, button_compare_state
    
    # Кнопка "Сохранить"
    save_color = (0, 200, 100) if button_save_state else (0, 150, 0)
    cv2.rectangle(img, (BUTTON1_X, BUTTON_Y), 
                  (BUTTON1_X + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT), 
                  save_color, -1)
    cv2.rectangle(img, (BUTTON1_X, BUTTON_Y), 
                  (BUTTON1_X + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT), 
                  (255, 255, 255), 2)
    
    # Текст кнопки "Сохранить"
    text_size = cv2.getTextSize("Save", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = BUTTON1_X + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = BUTTON_Y + BUTTON_HEIGHT // 2 + text_size[1] // 2
    cv2.putText(img, "Save", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Кнопка "Сравнить"
    compare_color = (200, 100, 0) if button_compare_state else (200, 150, 0)
    cv2.rectangle(img, (BUTTON2_X, BUTTON_Y), 
                  (BUTTON2_X + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT), 
                  compare_color, -1)
    cv2.rectangle(img, (BUTTON2_X, BUTTON_Y), 
                  (BUTTON2_X + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT), 
                  (255, 255, 255), 2)
    
    # Текст кнопки "Сравнить"
    text_size = cv2.getTextSize("Compare", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = BUTTON2_X + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = BUTTON_Y + BUTTON_HEIGHT // 2 + text_size[1] // 2
    cv2.putText(img, "Compare", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def add_legend(img, legend_y):
    """Adds color legend with correct color order"""
    legend_width = 500
    legend_height = 25
    legend_x = (WINDOW_WIDTH - legend_width) // 2

    # Draw gradient legend WITH CORRECT ORDER
    for x in range(legend_width):
        # Convert x position to value in range 1000..30000
        value = 1000 + (x / legend_width) * 29000  # 30000 - 1000 = 29000
        color = value_to_color(value)
        cv2.line(img,
                 (legend_x + x, legend_y),
                 (legend_x + x, legend_y + legend_height),
                 color, 1)

    # Legend frame
    cv2.rectangle(img,
                  (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (255, 255, 255), 2)

    # Legend labels (properly distributed)
    labels = [
        (1000, legend_x - 30, legend_y + legend_height + 25),
        (7000, legend_x + legend_width // 5 - 20, legend_y + legend_height + 25),   # 20%
        (13000, legend_x + 2*legend_width // 5 - 20, legend_y + legend_height + 25), # 40%
        (19000, legend_x + 3*legend_width // 5 - 20, legend_y + legend_height + 25), # 60%
        (25000, legend_x + 4*legend_width // 5 - 20, legend_y + legend_height + 25), # 80%
        (30000, legend_x + legend_width - 30, legend_y + legend_height + 25)        # 100%
    ]

    for value, x_pos, y_pos in labels:
        # Format value for display
        if value >= 1000:
            label_text = f"{int(value/1000)}k" if value >= 1000 else str(value)
        else:
            label_text = str(value)
        
        cv2.putText(img, label_text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Legend title
    cv2.putText(img, "Temperature Scale (ADC values)", 
                (legend_x + legend_width // 2 - 110, legend_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)


def save_screenshot(frame_img, frame_matrix):
    """Сохраняет текущий скриншот и возвращает информацию о нем"""
    global GL_COUNTER
    if GL_COUNTER == 4:
        return 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"thermal_{timestamp}.png"
    filepath = output_dir / DIR_NAME / filename
    # Сохраняем изображение
    cv2.imwrite(str(filepath), frame_img)
    
    # Сохраняем данные матрицы в текстовый файл
    data_filename = f"thermal_{timestamp}.txt"
    data_filepath = output_dir / DIR_NAME / data_filename
    
    with open(data_filepath, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Matrix Shape: {frame_matrix.shape}\n")
        f.write("Matrix Data:\n")
        np.savetxt(f, frame_matrix, fmt='%d')
    
    screenshot_info = {
        'filename': filename,
        'filepath': str(filepath),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'matrix_data': frame_matrix.copy()
    }
    
    recorded_screenshots.append(screenshot_info)
    print(f"✓ Сохранен скриншот: {filename}")
    GL_COUNTER +=1
    return screenshot_info


def compare_all_screenshots():
    """Открывает все сохраненные скриншоты за сеанс в отдельных окнах с адаптацией к разрешению экрана"""
    if not recorded_screenshots:
        print("Нет сохраненных скриншотов для сравнения")
        return
    global GL_COUNTER, comparison_windows
    GL_COUNTER = 0
    # Очищаем список окон сравнения перед созданием новых
    close_comparison_windows()
    
    # Получаем разрешение экрана
    screen_info = get_screen_resolution()
    screen_width = screen_info['width']
    screen_height = screen_info['height']
    
    print(f"\n=== Сравнение {len(recorded_screenshots)} скриншотов ===")
    print(f"Разрешение экрана: {screen_width}x{screen_height}")
    
    # Параметры расположения окон (можно настраивать)
    MAX_WINDOWS_X = 2  # Максимальное количество окон по горизонтали
    MAX_WINDOWS_Y = 2  # Максимальное количество окон по вертикали
    MARGIN = 10        # Отступ между окнами
    TOP_MARGIN = 30    # Отступ сверху для панели задач
    BOTTOM_MARGIN = 20 # Отступ снизу
    
    # Рассчитываем количество окон в сетке
    num_windows = len(recorded_screenshots)
    windows_x = min(MAX_WINDOWS_X, num_windows)
    windows_y = min(MAX_WINDOWS_Y, math.ceil(num_windows / MAX_WINDOWS_X))
    
    # Рассчитываем размеры окон на основе разрешения экрана и количества окон
    available_width = screen_width - MARGIN * (windows_x + 1)
    available_height = screen_height - TOP_MARGIN - BOTTOM_MARGIN - MARGIN * (windows_y + 1)
    
    # Размеры каждого окна
    window_width = available_width // windows_x + 38
    window_height = available_height // windows_y + 38
    
    # Корректируем соотношение сторон (сохраняем пропорции оригинального изображения)
    original_ratio = WINDOW_WIDTH / WINDOW_HEIGHT
    if window_width / window_height > original_ratio:
        window_width = int(window_height * original_ratio)
    else:
        window_height = int(window_width / original_ratio)
    
    print(f"Расположение: {windows_x}x{windows_y} окон")
    print(f"Размер окна: {window_width}x{window_height}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    global DIR_NAME
    DIR_NAME = f"{timestamp}"
    os.mkdir(f"thermal_screenshots/{DIR_NAME}")
    
    # Открываем каждое изображение в отдельном окне
    for idx, screenshot in enumerate(recorded_screenshots):
        try:
            # Загружаем изображение из файла
            img = cv2.imread(screenshot['filepath'])
            
            if img is None:
                print(f"Не удалось загрузить: {screenshot['filename']}")
                continue
            
            # Создаем уникальное имя окна
            window_name = f"Скриншот {idx+1}: {screenshot['timestamp']}"
            
            # Сохраняем имя окна для возможности закрытия
            comparison_windows.append(window_name)
            
            # Рассчитываем позицию в сетке
            grid_x = idx % windows_x
            grid_y = idx // windows_x
            
            # Позиционируем окно в сетке
            x_offset = MARGIN + grid_x * (window_width + MARGIN)
            y_offset = TOP_MARGIN + grid_y * (window_height + MARGIN)
            
            # Создаем окно
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)
            cv2.moveWindow(window_name, x_offset, y_offset)
            
            # Масштабируем изображение под размер окна
            resized_img = cv2.resize(img, (window_width, window_height), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Добавляем информацию на изображение
            overlay = resized_img.copy()
            
            # Полупрозрачный фон для текста вверху
            text_bg_height = 40
            text_bg = np.zeros((text_bg_height, window_width, 3), dtype=np.uint8)
            text_bg[:, :] = (40, 40, 40)
            
            # Создаем маску для прозрачности
            alpha = 0.7
            overlay[0:text_bg_height, 0:window_width] = cv2.addWeighted(
                text_bg, alpha, overlay[0:text_bg_height, 0:window_width], 1 - alpha, 0)
            
            # Добавляем текст
            title = f"Screenshot {idx+1}/{len(recorded_screenshots)}"
            cv2.putText(overlay, title, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Добавляем информацию о матрице
            matrix_info = screenshot['matrix_data']
            max_val = np.max(matrix_info)
            min_val = np.min(matrix_info)
            avg_val = np.mean(matrix_info)
            
            stats_text = f"Max: {max_val:.0f} | Min: {min_val:.0f} | Avg: {avg_val:.1f}"
            cv2.putText(overlay, stats_text, (window_width - 250, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Инструкция по закрытию (только в первом окне)
            if idx == 0:
                instruction_text = "Press SPACE to close all comparison windows"
                cv2.putText(overlay, instruction_text, 
                           (window_width // 2 - 180, window_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Отображаем
            cv2.imshow(window_name, overlay)
            
            print(f"✓ Открыто окно {idx+1} в позиции ({grid_x}, {grid_y}): {screenshot['filename']}")
            
            
        except Exception as e:
            print(f"Ошибка при открытии {screenshot['filename']}: {e}")
    
    print(f"\nОткрыто {min(len(recorded_screenshots), windows_x * windows_y)} окон.")
    print("Нажмите SPACE для закрытия всех окон сравнения.")
    
    # Очищаем список скриншотов
    recorded_screenshots.clear()


def close_comparison_windows():
    """Закрывает все окна сравнения"""
    global comparison_windows
    for window_name in comparison_windows:
        try:
            cv2.destroyWindow(window_name)
        except:
            pass  # Окно могло быть уже закрыто
    comparison_windows.clear()
    print("Все окна сравнения закрыты")


def handle_mouse_click(event, x, y, flags, param):
    """Обработчик кликов мыши для кнопок"""
    global button_save_state, button_compare_state
    global button_save_clicked, button_compare_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Проверяем клик по кнопке "Сохранить"
        if (BUTTON1_X <= x <= BUTTON1_X + BUTTON_WIDTH and 
            BUTTON_Y <= y <= BUTTON_Y + BUTTON_HEIGHT):
            button_save_clicked = True
            button_save_state = True
        
        # Проверяем клик по кнопке "Сравнить"
        elif (BUTTON2_X <= x <= BUTTON2_X + BUTTON_WIDTH and 
              BUTTON_Y <= y <= BUTTON_Y + BUTTON_HEIGHT):
            button_compare_clicked = True
            button_compare_state = True
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Сбрасываем состояние кнопок
        button_save_state = False
        button_compare_state = False


# Основной цикл обработки
current_frame = None
paused = False
use_interpolation = True

# Устанавливаем обработчик мыши
cv2.namedWindow("Thermal Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Thermal Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Thermal Camera", handle_mouse_click)

print("=== Thermal Camera Application ===")
print(f"Разрешение окна: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
print("Режим: Полноэкранный")
print("Горячие клавиши:")
print("  SPACE - Закрыть все окна сравнения")
print("  P     - Пауза/продолжить (только главное окно)")
print("  I     - Вкл/выкл интерполяцию")
print("  F     - Переключить полноэкранный режим")
print("  Q/ESC - Выход из программы")
print("")
print("Используйте мышь для нажатия кнопок:")
print("  [Сохранить] - Сохранить текущий скриншот (1 клик = 1 скрин)")
print("  [Сравнить]  - Открыть все сохраненные скриншоты в разных окнах")
print(f"Скриншоты сохраняются в: {output_dir.absolute()}")
print("=" * 40)

while True:
    # Если не на паузе, читаем данные с порта
    if not paused:
        data = ser.read(1024)
        if not data:
            continue

        buffer += data

        # hard limit
        if len(buffer) > MAX_BUFFER:
            buffer = buffer[-MAX_BUFFER:]

        start_idx = buffer.find(START_MARKER)

        if start_idx == -1:
            # no marker - clear all except possible marker tail
            buffer = buffer[-len(START_MARKER) + 1:]
            ser.reset_input_buffer()
            continue

        if len(buffer) >= start_idx + len(START_MARKER) + BYTES_PER_FRAME:
            frame_data = buffer[start_idx + len(START_MARKER):
                                start_idx + len(START_MARKER) + BYTES_PER_FRAME]
            buffer = buffer[start_idx + len(START_MARKER) + BYTES_PER_FRAME:]

            frame = struct.unpack("<" + "i" * FRAME_SIZE, frame_data)
            frame_np = np.array(frame, dtype=np.int32).reshape(FRAME_HEIGHT, FRAME_WIDTH)
            extended_matrix = extend_matrix_with_zeros(frame_np)
            
            # Создаем визуализацию
            current_frame = create_visualization(extended_matrix, frame_np, use_interpolation)
    
    # Отображаем текущий кадр
    if current_frame is not None:
        cv2.imshow("Thermal Camera", current_frame)
    
    # Обработка кликов по кнопкам
    if button_save_clicked:
        if current_frame is not None:
            save_screenshot(current_frame, frame_np)
        button_save_clicked = False
    
    if button_compare_clicked:
        compare_all_screenshots()
        button_compare_clicked = False
    
    # Обработка клавиатуры
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:  # 'q' или ESC
        break
    
    elif key == ord(' '):  # Пробел - закрыть все окна сравнения
        close_comparison_windows()
        print("Все окна сравнения закрыты")

    elif key == ord('i'):  # Переключение интерполяции
        use_interpolation = not use_interpolation
        print(f"Интерполяция: {'Вкл' if use_interpolation else 'Выкл'}")
    
    elif key == ord('f'):  # Переключение полноэкранного режима
        current_fullscreen = cv2.getWindowProperty("Thermal Camera", cv2.WND_PROP_FULLSCREEN)
        if current_fullscreen == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty("Thermal Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Полноэкранный режим выключен")
        else:
            cv2.setWindowProperty("Thermal Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Полноэкранный режим включен")
    
    elif key == ord('s'):  # Сохранение текущего кадра (горячая клавиша)
        if current_frame is not None:
            save_screenshot(current_frame, frame_np)
    
    elif key == ord('c'):  # Сравнение всех скриншотов (горячая клавиша)
        compare_all_screenshots()
    
    elif key == ord('d'):  # Показать информацию о сохраненных скриншотах
        print(f"\n=== Информация о сеансе ===")
        print(f"ID сеанса: {session_id}")
        print(f"Директория: {output_dir}")
        print(f"Сохранено скриншотов: {len(recorded_screenshots)}")
        print(f"Открыто окон сравнения: {len(comparison_windows)}")
        
        if recorded_screenshots:
            print("\nПоследние 5 скриншотов:")
            for i, ss in enumerate(recorded_screenshots[-5:]):
                print(f"  {i+1}. {ss['filename']} - {ss['timestamp']}")
    
    elif key == ord('h'):  # Помощь
        print("\n=== Горячие клавиши ===")
        print("SPACE - Закрыть все окна сравнения")
        print("I     - Вкл/Выкл интерполяцию")
        print("F     - Переключить полноэкранный режим")
        print("S     - Сохранить текущий кадр (горячая клавиша)")
        print("C     - Сравнить все сохраненные скриншоты (горячая клавиша)")
        print("D     - Показать информацию о сеансе")
        print("H     - Помощь (этот текст)")
        print("Q/ESC - Выход из программы")
        print("\nИли используйте кнопки мышью:")
        print("  [Сохранить] - Сохранить текущий скриншот")
        print("  [Сравнить]  - Открыть окна сравнения")


# Завершение работы
print(f"\n=== Завершение сеанса ===")
print(f"Всего сохранено скриншотов: {len(recorded_screenshots)}")
print(f"Директория с данными: {output_dir.absolute()}")

# Закрываем все окна сравнения перед выходом
close_comparison_windows()

# Создаем отчет о сеансе
if recorded_screenshots:
    report_file = output_dir / f"session_report_{session_id}.txt"
    with open(report_file, 'w') as f:
        f.write(f"Отчет о сеансе тепловизора\n")
        f.write(f"===========================\n\n")
        f.write(f"ID сеанса: {session_id}\n")
        f.write(f"Время начала: {session_id}\n")
        f.write(f"Время окончания: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Всего скриншотов: {len(recorded_screenshots)}\n\n")
        
        f.write("Список скриншотов:\n")
        for i, ss in enumerate(recorded_screenshots):
            f.write(f"{i+1:3d}. {ss['filename']} - {ss['timestamp']}\n")
    
    print(f"Отчет сеанса сохранен в: {report_file}")

cv2.destroyAllWindows()
ser.close()