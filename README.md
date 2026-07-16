# Imagenette Stage 0

## Результат

| Модель | Инициализация | Параметры | Эпох на этапе | Input | Clean validation accuracy |
|---|---|---:|---:|---:|---:|
| ConvNeXt Tiny, scratch v1 | random initialization | 27.83M | 30 | 160×160 | 72.96% (2288/3136) |
| ConvNeXt Tiny, scratch v2 | random initialization | 27.83M | 80 | 160×160 | 83.16% (2608/3136) |
| **ConvNeXt Tiny, progressive resize** | **v2 checkpoint; originally random** | **27.83M** | **25** | **224×224** | **85.97% (2696/3136)** |

Сначала ConvNeXt
Tiny обучался со случайной инициализацией на Imagenette2-160, затем лучший
чекпоинт продолжил обучение на Imagenette2-320 с
кропом `224×224`. Это progressive resizing, а не перенос весов из внешнего
датасета.

Лучший финальный чекпоинт получен на эпохе 20 этапа progressive resizing.
Графики: [финальный этап 224×224](runs/stage0_convnext_scratch_v3_224/training_curves.png),
[scratch v2 160×160](runs/stage0_convnext_scratch_v2/training_curves.png),
[scratch v1](runs/stage0_convnext_scratch/training_curves.png).

## Данные и разбиение

Используются две версии одних и тех же изображений Imagenette:

- Imagenette2-160 для основного scratch-обучения;
- Imagenette2-320 для финального дообучения с кропом `224×224`.

В обеих версиях разбиение классов одинаковое:

- 8 training: `tench`, `English springer`, `cassette player`,
  `chain saw`, `church`, `French horn`, `garbage truck`, `gas pump`;
- 2 отложенных: `golf ball`, `parachute`.


### Экспериментальный протокол

Скрипты используют `--no-pretrained`: backbone и голова исходно
инициализируются случайно. Финальный `stage0_convnext_scratch_v3_224.sh`
загружает только собственный чекпоинт v2.

## Установка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Воспроизведение результата

Финальный результат воспроизводится двумя последовательными этапами:

```bash
source .venv/bin/activate
bash scripts/stage0_convnext_scratch_v2.sh
bash scripts/stage0_convnext_scratch_v3_224.sh
```

Первый скрипт обучает ConvNeXt с нуля 80 эпох на кропе `160×160` и достигает
83.16%. Второй загружает `runs/stage0_convnext_scratch_v2/best.pt`, создаёт
новые optimizer и scheduler и выполняет 25 эпох на Imagenette2-320 с кропом
`224×224`. Перед файнтюнингом отдельно измеряет исходный чекпоинт в новом
разрешении: без адаптации accuracy составляет 81.25%, после обучения — 85.97%.

Контрольный файнтюнинг на прежнем разрешении ничего не улучшил и остановился
после 9 эпох:

```bash
bash scripts/stage0_convnext_scratch_v3_160.sh
```

## Ручная оценка чекпоинта

```bash
python -m imagenette_stage0.evaluate \
  --checkpoint runs/stage0_convnext_scratch_v3_224/best.pt \
  --dataset-size 320 \
  --image-size 224 \
  --output-csv runs/stage0_convnext_scratch_v3_224/eval_clean.csv
```

Повторная генерация графика:

```bash
python -m imagenette_stage0.plot_training \
  --jsonl runs/stage0_convnext_scratch_v3_224/metrics.jsonl \
  --title "Stage 0: ConvNeXt Tiny progressive resize 224" \
  --output runs/stage0_convnext_scratch_v3_224/training_curves.png \
  --output-csv runs/stage0_convnext_scratch_v3_224/training_history.csv
```


## Проверка кода

```bash
pytest -q
ruff check src tests
```
