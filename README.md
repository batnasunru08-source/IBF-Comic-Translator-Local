IBF Comic Translator Local

Проект состоит из двух частей:
Chrome extension — находит изображения на странице, добавляет кнопку IBF, отправляет картинку на локальный API и подставляет переведённую версию обратно.
Локальный FastAPI-сервер — получает изображение, распознаёт текст, переводит его и рендерит перевод поверх исходного изображения. FastAPI подходит для такого API-сервера и поддерживает загрузку файлов через UploadFile.

Для OCR используются:
EasyOCR — для английского и большинства неяпонских языков; у него есть lang_list и поддержка GPU через параметр gpu.
MangaOCR — для японского текста, особенно в манге.

Для перевода используется tencent/HY-MT1.5-7B. Согласно карточке модели, HY-MT 1.5 поддерживает взаимный перевод между 33 языками, а для transformers рекомендуемая версия — 4.56.0.

Установка
1. Клонирование проекта
git clone 
cd comic_translator/server

2. Создание виртуального окружения
Linux
python -m venv .venv
source .venv/bin/activate
Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Установка базовых зависимостей
pip install -r requirements.txt
________________________________________
Установка PyTorch
Вариант 1. CPU
Если у вас нет CUDA-совместимой NVIDIA GPU, можно использовать CPU-режим.
pip install torch torchvision torchaudio
Проверка:
python -c "import torch; print(torch.cuda.is_available())"
Ожидаемый результат:
False
________________________________________
Вариант 2. GPU (CUDA)
Если у вас есть NVIDIA GPU, установите CUDA-сборку PyTorch.
Пример установки:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Проверка:
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
Ожидаемый результат:
•	True
•	версия CUDA, например 12.8
•	название вашей видеокарты
________________________________________
Опционально: bitsandbytes для 4-bit загрузки модели
Если вы хотите уменьшить потребление VRAM при загрузке HY-MT1.5-7B, установите bitsandbytes:
pip install -U "bitsandbytes>=0.46.1"
Если bitsandbytes не установлен, проект может использовать обычную загрузку модели без 4-bit-квантования.
________________________________________
Запуск сервера
Из папки server:
python -m uvicorn main:app --host 127.0.0.1 --port 8000
Проверка:
curl http://127.0.0.1:8000/health
Ожидаемый ответ:
{"ok": true}
________________________________________
Загрузка расширения в Chrome
1.	Откройте chrome://extensions/
2.	Включите Режим разработчика
3.	Нажмите Загрузить распакованное расширение
4.	Выберите папку extension
После этого на изображениях на странице должна появиться кнопка IBF.
________________________________________
Настройки расширения
В popup расширения доступны:
•	Включить / выключить расширение
•	Режим отображения
o	replace — заменить исходное изображение
o	overlay — показать перевод поверх оригинала
•	Язык текста (OCR)
•	Язык перевода
Рекомендуемые OCR-режимы
•	en, ru, de, fr, es, ko, ch_sim, ch_tra → EasyOCR
•	ja → MangaOCR
________________________________________
Как включать CPU или GPU
Проект выбирает устройство автоматически:
•	если torch.cuda.is_available() возвращает True, OCR и перевод пытаются использовать GPU
•	если False, используется CPU
Практически это означает:
•	установили CPU-версию PyTorch → проект работает на CPU
•	установили CUDA-версию PyTorch → проект работает на GPU
Проверка текущего режима
python -c "import torch; print(torch.cuda.is_available())"
Принудительно CPU
Установите CPU-сборку PyTorch и не ставьте CUDA-колёса.
Принудительно GPU
Установите CUDA-сборку PyTorch и при необходимости bitsandbytes.
________________________________________
Использование переводчика
Модель tencent/HY-MT1.5-7B предназначена именно для перевода.
Рекомендуемый шаблон промпта:
Translate the following segment into {target_language}, without additional explanation.

{source_text}
________________________________________
Ограничения текущего MVP
•	OCR может резать один speech bubble на несколько отдельных строк
•	на сложных комиксах возможна фрагментация текста
•	MangaOCR хорошо подходит для японского текста, но может потребовать дополнительной логики группировки
•	некоторые сайты не дают скачать картинку напрямую по URL; для них лучше использовать upload-режим
•	HY-MT1.5-7B — тяжёлая модель, поэтому для GPU желательно 4-bit загрузка или достаточный объём VRAM
________________________________________
Полезные команды для диагностики
Проверка EasyOCR
python -c "import easyocr; print('easyocr ok')"
Проверка MangaOCR
python -c "from manga_ocr import MangaOcr; print('manga_ocr ok')"
Проверка PyTorch / CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
Проверка bitsandbytes
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
________________________________________
Планы по улучшению
•	автоопределение языка текста
•	лучшее объединение строк в speech bubbles
•	кэширование результатов OCR и перевода
•	ручное редактирование перевода перед рендером
•	поддержка большего количества сайтов и режимов вставки изображения
•	улучшенный японский pipeline


## Лицензия
Проект распространяется под лицензией MIT.