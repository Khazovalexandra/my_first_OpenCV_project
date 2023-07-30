# Программа, определяющее лицо человека с камеры компьютера в реальном времени
> Создана с помощью библиотеки компьютерного зрения OpenCV

**Определяет:**
> на постоянной основе

**улыбается ли человек(Smiling) или нет(Not smiling)**

> при определении улыбки над лицом пишет что человек улыбается(Smiling)

- лицо (зеленый прямоугольник)
- глаза/очки (фиолетовые прямоугольники)
- улыбку(голубой прямоугольник)
  
## Для корректной работы программы 
- **Сохраните данный репозиторий на свою машину**
  > xml файлы содержат необходимые данные для библиотеки OpenCV
- **Установите**

модуль cv2 (для импорта библиотеки OpenCV)
  
```
pip install cv2
```

**Все, теперь можно запускать файл faceinstream.py и наслаждаться его работой=)**

____________
___ВАЖНО!___
если вы хотите добится хорошего результата в распозновании лица и его элементов программой, то следует настроить камеру на своем компьютере, освещение в помещении и отдавать предпочтение однотонному фону на котором будет лучше выделяться распозноваемое лицо/лица.

В репозитории храняться 2 файла для распознования улыбки: haarcascade_smile.xml и haarcascade_smile1.xml, в коде используется второй файл, но возможно вам больше подойдет первый файл.

Поэксперементируйте для более корректной работы программы.

____
__БОНУС__

В папке cats храниться программа для определения кота/нескольких котов на картинке и их подсчет. Так же в ней хранятся картинки для примера и xml-файл для работы cv2.

УДАЧИ
