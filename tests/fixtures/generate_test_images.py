"""Generates a small set of real, purpose-built synthetic images for
manual and automated testing of the Image Parsing pipeline (redact_photo,
analyze_image) and the chat frontend's photo upload -- run once, not part
of the pytest suite itself (these are static fixtures, not something to
regenerate on every test run). No real customer photos or real PII exist
anywhere in this repo; every "detectable text"/"face" element here is
synthetic, drawn specifically to be detectable, not a real person's data.

Run with: uv run python tests/fixtures/generate_test_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "images"
OUT_DIR.mkdir(exist_ok=True)


def _font(size: int) -> ImageFont.ImageFont:
    # A real system font renders as actual glyphs OCR can detect -- PIL's
    # default bitmap font is deliberately avoided here since EasyOCR can't
    # reliably read it (confirmed while building this: the default font's
    # low resolution produces no real text detections).
    for candidate in ("arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_intact_item() -> None:
    """A plain, undamaged 'product' -- consistency baseline."""
    img = Image.new("RGB", (600, 400), (235, 235, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 100, 480, 300], fill=(70, 110, 180), outline=(20, 20, 20), width=4)
    draw.text((160, 180), "BLUE FROCK", font=_font(28), fill=(255, 255, 255))
    img.save(OUT_DIR / "intact_item.jpg", quality=90)


def make_damaged_item() -> None:
    """Same base product, with visible damage marks -- for a Damaged in
    Transit / Defective-DOA claim's photo evidence."""
    img = Image.new("RGB", (600, 400), (235, 235, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 100, 480, 300], fill=(70, 110, 180), outline=(20, 20, 20), width=4)
    draw.text((160, 180), "BLUE FROCK", font=_font(28), fill=(255, 255, 255))
    # Real, visible "crack"/tear marks across the product.
    for start, end in [((140, 110), (300, 260)), ((300, 260), (460, 130)), ((200, 290), (350, 150))]:
        draw.line([start, end], fill=(220, 30, 30), width=6)
    draw.text((150, 320), "DAMAGED ON ARRIVAL", font=_font(24), fill=(180, 0, 0))
    img.save(OUT_DIR / "damaged_item.jpg", quality=90)


def make_wrong_item() -> None:
    """A visibly different product -- for testing image_consistency=
    'inconsistent' (customer claims one product, photo shows another)."""
    img = Image.new("RGB", (600, 400), (235, 235, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([160, 90, 440, 310], fill=(210, 170, 40), outline=(20, 20, 20), width=4)
    draw.text((230, 190), "RED MUG", font=_font(28), fill=(60, 30, 0))
    img.save(OUT_DIR / "wrong_item.jpg", quality=90)


def make_shipping_label_photo() -> None:
    """A product photo with an incidental shipping label caught in frame
    -- real, OCR-detectable text (a synthetic name/address, not a real
    person's), for testing redact_photo's real EasyOCR-based redaction."""
    img = Image.new("RGB", (600, 400), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 60, 520, 220], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    draw.text((100, 80), "SHIP TO:", font=_font(22), fill=(0, 0, 0))
    draw.text((100, 115), "Jordan Rivera", font=_font(26), fill=(0, 0, 0))
    draw.text((100, 150), "482 Willow Creek Lane", font=_font(22), fill=(0, 0, 0))
    draw.text((100, 180), "Springfield, IL 62704", font=_font(22), fill=(0, 0, 0))
    draw.rectangle([150, 260, 450, 340], fill=(70, 110, 180))
    img.save(OUT_DIR / "shipping_label_photo.jpg", quality=90)


if __name__ == "__main__":
    make_intact_item()
    make_damaged_item()
    make_wrong_item()
    make_shipping_label_photo()
    print(f"Generated 4 test images in {OUT_DIR}")
