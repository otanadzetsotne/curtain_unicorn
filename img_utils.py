from PIL import Image
import numpy as np

def create_multi_gradient_image(width, height, points):
    # Создаем новое изображение с поддержкой прозрачности
    image = Image.new("RGBA", (width, height))
    result = np.zeros((height, width, 4), dtype=float)

    # Структура для хранения весов каждого цвета
    weights = np.zeros((height, width, len(points)))

    # Создаем массивы для координат x и y
    x_indices = np.tile(np.arange(width), (height, 1))
    y_indices = np.repeat(np.arange(height), width).reshape(height, width)

    # Рассчитываем веса для каждой точки
    for i, (point, color) in enumerate(points):
        x0, y0 = point
        distance = np.sqrt((x_indices - x0)**2 + (y_indices - y0)**2)
        weights[:, :, i] = np.exp(-distance / 50)  # Эмпирическая константа для сглаживания

    # Нормализация весов
    weight_sums = np.sum(weights, axis=2, keepdims=True)
    normalized_weights = weights / weight_sums

    # Применяем веса для интерполяции цвета
    for i, (point, color) in enumerate(points):
        color_array = np.array(color).reshape(1, 1, 4)
        result += color_array * normalized_weights[:, :, i:i+1]

    # Конвертируем результат в изображение
    image = Image.fromarray(np.uint8(result), 'RGBA')
    return image


if __name__ == '__main__':
    # Пример использования
    width, height = 500, 300
    points = [
        ((0, 0), (0, 0, 0, 0)),
        ((200, 250), (0, 0, 0, 100)),
        ((500, 0), (0, 0, 0, 0)),
    ]

    image = create_multi_gradient_image(width, height, points)
    # image.show()
    image.save('grad.png')
