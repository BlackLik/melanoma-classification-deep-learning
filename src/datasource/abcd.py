import math

import cv2
import numpy as np


def compute_abcd_features(image) -> dict[str, float]:
    """
    Вычисляет параметры ABCD-теста:
    - A: Асимметрия
    - B: Нерегулярность границ
    - C: Вариация цвета
    - D: Диаметр
    Также вычисляется суммарный abcd_score.
    """
    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Применяем пороговую обработку (Otsu) для выделения области родинки
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Если контуры не найдены, возвращаем нулевые значения
        return {
            "asymmetry": 0.0,
            "border_irregularity": 0.0,
            "color_variation": 0.0,
            "diameter": 0.0,
            "abcd_score": 0.0,
        }

    # Выбираем самый большой контур (предполагаем, что это родинка)
    contour = max(contours, key=cv2.contourArea)

    # 1. Асимметрия (A)
    max_pointer_ellipse = 5
    if len(contour) < max_pointer_ellipse:
        asymmetry = 0.0  # Недостаточно точек для аппроксимации эллипса
    else:
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        asymmetry = (major_axis - minor_axis) / major_axis if major_axis != 0 else 0.0

    # 2. Нерегулярность границ (B)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    border_irregularity = (hull_area - area) / hull_area if hull_area != 0 else 0.0

    # 3. Вариация цвета (C)
    # Создаем маску по контуру
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Вычисляем стандартное отклонение цветов в области родинки
    _, std_dev = cv2.meanStdDev(image, mask=mask)
    color_variation = float(np.mean(std_dev))

    # 4. Диаметр (D)
    diameter = math.sqrt(4 * area / math.pi) if area > 0 else 0.0

    # Итоговый ABCD-счет (пример объединения, можно задавать веса)
    # (A×1.3) + (B×0.1) + (C×0.5) + (D×0.5)
    abcd_score = (asymmetry * 1.3) + (border_irregularity * 0.1) + (color_variation * 0.5) + (diameter * 0.5)

    return {
        "asymmetry": asymmetry,
        "border_irregularity": border_irregularity,
        "color_variation": color_variation,
        "diameter": diameter,
        "abcd_score": abcd_score,
    }
