import io
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageOps


def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def compute_histograms(rgb: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = np.clip(np.rint(0.2126 * r + 0.7152 * g + 0.0722 * b), 0, 255).astype(np.uint8)

    return {
        "L": np.histogram(lum, bins=256, range=(0, 256))[0],
        "R": np.histogram(r, bins=256, range=(0, 256))[0],
        "G": np.histogram(g, bins=256, range=(0, 256))[0],
        "B": np.histogram(b, bins=256, range=(0, 256))[0],
    }


def estimate_quality(hists: dict[str, np.ndarray], total_pixels: int) -> tuple[float, str, list[str]]:
    lum = hists["L"]
    bins = np.arange(256)
    weights = lum / total_pixels

    shadows_pct = float(np.sum(lum[:6]) / total_pixels * 100)
    highlights_pct = float(np.sum(lum[250:]) / total_pixels * 100)
    mean_lum = float(np.sum(bins * weights))
    std_lum = float(np.sqrt(np.sum(((bins - mean_lum) ** 2) * weights)))

    cdf = np.cumsum(weights)
    span = int(np.searchsorted(cdf, 0.98) - np.searchsorted(cdf, 0.02))

    score = 100.0
    notes = []

    if highlights_pct > 2.0:
        score -= min(35.0, highlights_pct * 2.0)
        notes.append(f"Przepalone światła ({highlights_pct:.1f}% pikseli >= 250)")
    else:
        notes.append(f"Światła bez większych przycięć ({highlights_pct:.1f}%)")

    if shadows_pct > 2.0:
        score -= min(35.0, shadows_pct * 2.0)
        notes.append(f"Zbita czerń ({shadows_pct:.1f}% pikseli <= 5)")
    else:
        notes.append(f"Cienie z zachowanymi detalami ({shadows_pct:.1f}%)")

    if span < 90:
        score -= (90 - span) * 0.25
        notes.append(f"Niski kontrast / wąski zakres tonalny (szerokość {span}/255)")
    else:
        notes.append(f"Dobra rozpiętość tonalna (szerokość {span}/255)")

    if mean_lum < 45 or mean_lum > 210:
        score -= 15.0
        notes.append(f"Przesunięta średnia jasność ({mean_lum:.1f}/255)")
    else:
        notes.append(f"Średnia jasność: {mean_lum:.1f}/255, odchylenie: {std_lum:.1f}")

    score = max(0.0, min(100.0, round(score, 1)))
    if score >= 80:
        label = "Dobra"
    elif score >= 55:
        label = "Przeciętna"
    else:
        label = "Słaba"

    return score, label, notes


def show_results(img: Image.Image, hists: dict[str, np.ndarray], score: float, label: str) -> None:
    x = np.arange(256)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Jakość (z histogramu): {score}/100 ({label})", fontsize=13, fontweight="bold")

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Zdjęcie")
    axes[0, 0].axis("off")

    axes[0, 1].fill_between(x, hists["L"], color="gray", alpha=0.8)
    axes[0, 1].set_title("Histogram ogólny (Luminancja)")
    axes[0, 1].set_xlim(0, 255)

    for ax, key, title, color in [
        (axes[1, 0], "R", "Kanał R", "crimson"),
        (axes[1, 1], "G", "Kanał G", "forestgreen"),
    ]:
        ax.fill_between(x, hists[key], color=color, alpha=0.65)
        ax.set_title(title)
        ax.set_xlim(0, 255)

    # Dodaje osobne okno dla kanału B, aby zachować przejrzysty układ 2x2 z podglądem i luminancją
    fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
    ax_b.fill_between(x, hists["B"], color="royalblue", alpha=0.65)
    ax_b.set_title("Kanał B (Niebieski)")
    ax_b.set_xlim(0, 255)

    plt.tight_layout()
    plt.show()


def main() -> None:
    default_url = "https://i1.jbzd.com.pl/contents/2026/11/normal/hjh1PVxpEvdd46tUh0j4qTe1FRmaeqwL.jpg"
    url = input("URL zdjęcia [Enter = Obrazek GTA 6 z JBZD]: ").strip() or default_url

    img = load_image_from_url(url)
    rgb = np.asarray(img, dtype=np.uint8)
    hists = compute_histograms(rgb)
    score, label, notes = estimate_quality(hists, rgb.shape[0] * rgb.shape[1])

    print(f"\nOcena: {score}/100 ({label})")
    for note in notes:
        print("-", note)

    show_results(img, hists, score, label)


if __name__ == "__main__":
    main()