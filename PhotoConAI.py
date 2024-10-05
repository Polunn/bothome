from PIL import Image, ImageEnhance
import pytesseract  # Используем pytesseract
import numpy as np
import cv2

# Открытие изображения
image = Image.open('photoai4.png')

# Увеличение разрешения
image = image.resize((image.width * 6, image.height * 6), Image.LANCZOS)

# Увеличение контрастности
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.0)

# Преобразование в массив NumPy и конвертация в градации серого
image_np = np.array(image)
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# Применение гауссовского размытия для уменьшения шума
image_np = cv2.GaussianBlur(image_np, (7, 7), 0)

# Применение адаптивного порога
image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Вернуть в PIL формат
image = Image.fromarray(image_np)

# Распознавание текста
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, lang='rus', config=custom_config)

# Отображение результата
image.show()

# Вывод результата
print('Распознанный текст:')
print(text)

# Рисование квадратов вокруг текста (с использованием pytesseract.image_to_boxes)
boxes = pytesseract.image_to_boxes(image, config=custom_config)
for box in boxes.splitlines():
    x, y, w, h = map(int, box.split())
    cv2.rectangle(image_np, (x, image.height - y), (w, image.height - h), (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Result', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows() 

