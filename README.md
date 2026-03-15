# IBF Comic Translator Local

Локальный переводчик комиксов из браузера.

Проект состоит из двух частей:
- `extension/` — Chrome extension, добавляет кнопку `IBF` на изображения на странице, отправляет изображение на локальный API и подставляет переведенную версию обратно.
- `server/` — FastAPI сервер, который делает OCR, перевод и рендер текста поверх исходной картинки.

Текущий стек:
- OCR: `PaddleOCR`
- Translation: `tencent/HY-MT1.5-7B`
- Runtime: `Linux x86_64` или `WSL2`

## Поддерживаемые сценарии

README описывает три сценария:

- основной: обычный `Linux x86_64 + NVIDIA GPU`
- совместимый: `Windows + WSL2 + NVIDIA GPU`
- fallback: `Linux x86_64` или `WSL2` без GPU, только `CPU`

`WSL2` здесь рассматривается как частный случай Linux-окружения. Windows-native запуск в этом README не описывается.

## Требования

Нужно:
- `Linux x86_64` или `WSL2`
- Python `3.12`
- Google Chrome или Chromium

Для GPU-режима дополнительно нужно:
- NVIDIA GPU
- рабочий `nvidia-smi`

Перед установкой проверьте базовое окружение:

```bash
python3 --version
```

Если планируется GPU-запуск, проверьте еще и:

```bash
nvidia-smi
```

## Установка

Клонируйте проект:

```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local
```

Установите системные пакеты:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

Создайте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Установка Python-зависимостей

Из папки `server/`:

### GPU

```bash
cd server
bash install-linux-gpu.sh
```

`install-linux-gpu.sh`:
- ставит базовые зависимости сервера
- ставит `paddlepaddle-gpu`
- ставит совместимый `torch`
- выравнивает CUDA runtime-пакеты для текущего `cu130` стека

`install-wsl-gpu.sh` оставлен как совместимый alias и вызывает тот же сценарий установки.

Из-за текущих upstream wheel-метаданных `pip check` может ругаться на часть CUDA runtime-пакетов. Для этого проекта это не главный критерий. Важнее, чтобы проходили импорты `torch` и `paddle`, и сервер стартовал.

### CPU

Если GPU нет, ставьте CPU-стек:

```bash
cd server
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-cpu.txt
```

`requirements-cpu.txt`:
- ставит базовые зависимости сервера
- ставит CPU-версию `paddlepaddle`
- ставит обычный `torch`

CPU-режим поддерживается, но будет заметно медленнее GPU-варианта, особенно на больших страницах и на этапе перевода.

## Проверка GPU

Проверьте из активированного `.venv`:

```bash
python - <<'PY'
import torch, paddle
print("torch:", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("paddle:", paddle.__version__, paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())
PY
```

Ожидаемо:
- `torch.cuda.is_available()` -> `True`
- `paddle.device.is_compiled_with_cuda()` -> `True`
- `paddle.device.cuda.device_count()` -> `>= 1`

## Проверка CPU

Если вы запускаете без GPU, достаточно, чтобы импорты проходили и оба фреймворка работали в CPU-режиме:

```bash
python - <<'PY'
import torch, paddle
print("torch:", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("paddle:", paddle.__version__, paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())
PY
```

Ожидаемо:
- `torch.cuda.is_available()` -> `False`
- `paddle.device.is_compiled_with_cuda()` -> `False`
  или `True`, но без доступных GPU устройств
- `paddle.device.cuda.device_count()` -> `0`

## Запуск сервера

Из папки `server/`:

```bash
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Проверьте health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"ok": true}
```

Во время старта полезно увидеть:
- `[STARTUP] Translator warm-up complete`
- `[STARTUP] PaddleOCR en ready`
- `[OCR] ... paddle_gpu_available=True` для GPU
- `[OCR] ... paddle_gpu_available=False` для CPU

## Установка Chrome extension

В Chrome или Chromium:

1. Откройте `chrome://extensions/`
2. Включите `Developer mode`
3. Нажмите `Load unpacked`
4. Выберите папку `extension/`

После загрузки extension:
- откройте popup расширения
- убедитесь, что extension включено
- при необходимости выберите `OCR language` и `Target language`

Если вы меняли файлы в `extension/`, перезагрузите extension на странице `chrome://extensions/`.

Для `WSL2` сервер обычно доступен из Chrome на Windows по `http://127.0.0.1:8000`.

## Как это работает

Основной путь сейчас такой:

1. Extension сама скачивает исходную картинку из браузерного контекста.
2. Если картинка слишком большая, extension уменьшает ее перед upload.
3. Extension отправляет файл на `POST /translate-upload`.
4. Сервер делает OCR, перевод и рендерит итоговую картинку.
5. Extension получает результат и подставляет переведенную версию на страницу.

`/translate-from-url` остается запасным путем, если upload-first путь не сработал.

## Настройки расширения

В popup доступны:
- включение и выключение extension
- `replace` — заменить исходное изображение
- `overlay` — показать перевод поверх оригинала
- язык OCR
- язык перевода

## API

Основные эндпоинты:
- `GET /health`
- `POST /translate-upload`
- `POST /translate-from-url`
- `GET /results/{filename}`

### `POST /translate-upload`

Multipart form-data:
- `file`
- `source_ocr_lang`
- `target_lang`

### `POST /translate-from-url`

JSON body:

```json
{
  "image_url": "https://example.com/page.jpg",
  "page_url": "https://example.com/chapter/1",
  "referer": "https://example.com/chapter/1",
  "source_ocr_lang": "en",
  "target_lang": "Russian"
}
```

## Поддерживаемые OCR language значения

Основные значения:
- `en`
- `ru`
- `de`
- `fr`
- `es`
- `ko`
- `ja`
- `ch_sim`
- `ch_tra`

## Диагностика

Проверка `torch`:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Проверка `paddle`:

```bash
python - <<'PY'
import paddle
print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.cuda.device_count())
PY
```

Проверка `PaddleOCR`:

```bash
python - <<'PY'
from app.ocr import get_paddleocr_engine
ocr = get_paddleocr_engine("en")
print(type(ocr).__name__)
PY
```

Если OCR не находит текст:
- смотрите `debug_dir` в `meta`
- откройте `grouped_blocks.png`
- проверьте `crop_*.png`

## Ограничения

- `HY-MT1.5-7B` тяжелая модель и заметно влияет на общую задержку
- сложные страницы манги могут давать шумные OCR-блоки
- некоторые сайты плохо отдают исходные изображения по прямому URL
- на очень больших страницах рендер и OCR все еще занимают заметное время
- в CPU-режиме OCR и перевод работают существенно медленнее, чем на GPU

## Опционально можете использовать 'PatchMatch'

'PatchMatch' — это дополнительная не нейронная функция заполнения пропущенных областей для текстурированных фонов. 

Она не устанавливается по умолчанию.

Linux or WSL2:

```bash
sudo apt install -y build-essential git pkg-config libopencv-dev
cd server
bash install-patchmatch.sh
export COMIC_TRANSLATOR_INPAINT_BACKEND=patchmatch
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Примечания:
- Нативная библиотека устанавливается в 'server/data/libs/libpatchmatch.so'
- Вы можете переопределить путь к библиотеке с помощью 'COMIC_TRANSLATOR_PATCHMATCH_LIB'
- Если PatchMatch недоступен, сервер переключается на 'telea'

## License

MIT
