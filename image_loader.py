from pathlib import Path
from PIL import Image, ImageFilter

def get_image_part(img_name: str, sigma: float, png: bool) -> dict:
    img_type = "png" if png else "jpg"
    blurred = Image.open(f"data/samples/{img_name}").convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=sigma)
    )
    sigma_str = str(sigma).replace(".", ",")  # for folder name
    blurred.save(f"data/blur{sigma_str}/{img_name}")  # keeps your original file safe

    return {
        "mime_type": f"image/{img_type}",  # jpeg works too (jpg?)
        "samples": Path(f"data/blur{sigma_str}/{img_name}").read_bytes()
    }
