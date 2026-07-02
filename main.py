import pandas as pd
import numpy as np
import httpx
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from io import BytesIO

#dowlond
def download_images(df):
    for index, row in df.iterrows():
        file_path = Path(row["file_path"])
        if file_path.exists():
            print(f"Уже есть: {row['image_name']}")
            continue

        try:
            print(f"Скачиваю: {row['image_name']}...")
            response = httpx.get(row['url'], timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")

            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path, "PNG")
            print(f"✓ Сохранено: {file_path}")


        except Exception as e:
            print(f"✗ Ошибка {row['image_name']}: {e}")

    print("\nГотово ")

## Конвертация

def convert_to_grayscale():

    print("\nКонвертация")
    raw_folder = Path("images/raw")
    gray_folder = Path("images/gray")
    gray_folder.mkdir(parents=True, exist_ok=True)

    for image_path in raw_folder.glob("*.png"):

         gray_path = gray_folder / image_path.name

         if gray_path.exists():
             print(f"✓ Уже есть: {image_path.name}")
             continue

         image = Image.open(image_path)

         gray_image = image.convert("L")
         gray_image.save(gray_path)
         print(f"✓ Grayscale: {image_path.name}")

    print("\nГотово!")


def calculate_brightness(df):
    print("\n--- Расчёт яркости ---")

    result = []

    for index, row in df.iterrows():

        gray_path = Path("images/gray") / f"{row['image_name']}.png"

        if not gray_path.exists():
            print(f"Нет фалйа {gray_path}")
            continue

        image = Image.open(gray_path)

        pixels = np.array(image)
        mean_brightness = pixels.mean()
        std_brightness = pixels.std()
        result.append({
            'image_name': row['image_name'],
            'category': row['category'],
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
        })
        print(f"✓ {row['image_name']}: mean={mean_brightness:.1f}, std={std_brightness:.1f}")

    brightness_df = pd.DataFrame(result)
    brightness_df.to_csv("data/brightness.csv", index=False)
    print("\nСохранено в data/brightness.csv")
    print("\nАнализ по категориям")

    summary = brightness_df.groupby("category").agg({
        "mean_brightness": ["mean", "std"],
        "std_brightness": ["mean", "std"],
    }).round(2)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    print(summary)

    summary.to_csv("data/brightness_summary.csv", index=False, )
    print("\nСохранено в data/brightness_summary.csv")

    ## Графики
    category_means = brightness_df.groupby("category")["mean_brightness"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(category_means.index, category_means.values, color=["gray", "orange", "yellow"])
    plt.xlabel("Категория")
    plt.ylabel("Средняя яркость")
    plt.title("Средняя яркость по категориям")
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/brightness_by_category.png")
    plt.close()
    print("✓ Сохранено: plots/brightness_by_category.png")

    # График 2: Яркость отдельных изображений
    plt.figure(figsize=(12, 5))
    colors = {"dark": "gray", "medium": "orange", "bright": "yellow"}
    bar_colors = [colors[cat] for cat in brightness_df["category"]]

    plt.bar(brightness_df["image_name"], brightness_df["mean_brightness"], color=bar_colors)
    plt.xlabel("Изображение")
    plt.ylabel("Яркость")
    plt.title("Яркость отдельных изображений")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/brightness_by_image.png")
    plt.close()
    print("✓ Сохранено: plots/brightness_by_image.png")




if __name__ == "__main__":
    df = pd.read_csv("data/images.csv")
    # download_images(df)
    # convert_to_grayscale()
    calculate_brightness(df)



