"""Generates a small set of real, purpose-built synthetic images for
manual and automated testing of the Image Parsing pipeline (redact_photo,
analyze_image) and the chat frontend's photo upload -- run once, not part
of the pytest suite itself (these are static fixtures, not something to
regenerate on every test run). No real customer photos or real PII exist
anywhere in this repo; every "detectable text"/"face" element here is
synthetic, drawn specifically to be detectable, not a real person's data.

The 3 damaged_shoes/clothes/headphones images replaced an earlier,
too-abstract rectangle/ellipse version at the user's request -- drafted to
a scratch location, shown to the user, and only saved into
tests/fixtures/images/ (and merged into this real generator script) after
explicit approval.

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


def make_damaged_shoes() -> None:
    """A recognizable sneaker-profile silhouette with a torn upper and the
    sole visibly separated at the heel -- more specific than the generic
    rectangle/ellipse products above, for Damaged in Transit/Defective-DOA
    claims about footwear specifically. User-requested (over the earlier,
    too-abstract shapes), reviewed and approved before being saved here."""
    img = Image.new("RGB", (600, 400), (240, 240, 235))
    draw = ImageDraw.Draw(img)

    sole_top_y = 300
    outline = [
        (70, sole_top_y), (70, 275), (95, 235), (140, 205), (230, 175),
        (300, 178), (330, 200), (420, 210), (480, 235), (505, 258),
        (495, 285), (450, sole_top_y),
    ]
    draw.polygon(outline, fill=(190, 60, 60), outline=(15, 15, 15), width=3)
    draw.polygon(
        [(60, sole_top_y), (460, sole_top_y), (515, sole_top_y + 22), (460, sole_top_y + 34),
         (75, sole_top_y + 34), (55, sole_top_y + 18)],
        fill=(45, 45, 50), outline=(15, 15, 15), width=2,
    )
    for i in range(5):
        x0, y0 = 240 + i * 22, 192 + i * 3
        draw.line([(x0, y0), (x0 + 16, y0 + 26)], fill=(235, 235, 235), width=4)
    draw.polygon([(225, 178), (255, 172), (262, 205), (232, 210)], fill=(230, 230, 230), outline=(15, 15, 15))

    draw.line([(330, 205), (395, 245)], fill=(10, 10, 10), width=4)
    draw.line([(340, 198), (400, 240)], fill=(10, 10, 10), width=2)
    draw.polygon([(70, 275), (95, 260), (100, 300), (72, 300)], fill=(240, 240, 235))
    draw.line([(70, 275), (100, 300)], fill=(200, 20, 20), width=3)

    draw.text((30, 30), "DAMAGED: sole separated at heel, upper torn", font=_font(20), fill=(150, 0, 0))
    img.save(OUT_DIR / "damaged_shoes.jpg", quality=90)


def make_damaged_clothes() -> None:
    """A hoodie silhouette with a torn shoulder seam and a stain -- for
    Damaged in Transit/Not as Described claims about apparel. User-
    requested, reviewed and approved before being saved here."""
    img = Image.new("RGB", (600, 400), (240, 240, 235))
    draw = ImageDraw.Draw(img)

    draw.polygon(
        [(220, 90), (380, 90), (400, 130), (460, 150), (450, 210), (410, 195),
         (410, 340), (190, 340), (190, 195), (150, 210), (140, 150), (200, 130)],
        fill=(70, 110, 160), outline=(20, 20, 20), width=3,
    )
    draw.ellipse([(250, 60), (350, 130)], fill=(60, 95, 140), outline=(20, 20, 20), width=3)
    draw.rectangle([(250, 260), (350, 300)], outline=(20, 20, 20), width=2)

    draw.line([(230, 150), (270, 190)], fill=(20, 20, 20), width=3)
    draw.line([(235, 145), (275, 185)], fill=(20, 20, 20), width=2)
    draw.polygon([(245, 165), (255, 155), (260, 175), (248, 178)], fill=(240, 240, 235))
    draw.ellipse([(320, 220), (380, 270)], fill=(90, 60, 30), outline=None)

    draw.text((30, 30), "DAMAGED: torn shoulder seam, stain", font=_font(20), fill=(150, 0, 0))
    img.save(OUT_DIR / "damaged_clothes.jpg", quality=90)


def make_damaged_headphones() -> None:
    """Over-ear headphones with one visibly cracked ear cup and a missing
    chunk -- for Defective/DOA claims about electronics. User-requested,
    reviewed and approved before being saved here."""
    img = Image.new("RGB", (600, 400), (240, 240, 235))
    draw = ImageDraw.Draw(img)

    draw.arc([(150, 60), (450, 340)], start=200, end=340, fill=(30, 30, 30), width=14)
    draw.ellipse([(120, 160), (220, 280)], fill=(35, 35, 40), outline=(10, 10, 10), width=3)
    draw.ellipse([(140, 180), (200, 260)], fill=(60, 60, 65))
    draw.ellipse([(380, 160), (480, 280)], fill=(35, 35, 40), outline=(10, 10, 10), width=3)
    draw.ellipse([(400, 180), (460, 260)], fill=(60, 60, 65))

    draw.line([(390, 175), (450, 260)], fill=(0, 0, 0), width=3)
    draw.line([(420, 170), (395, 240)], fill=(0, 0, 0), width=2)
    draw.pieslice([(440, 165), (480, 205)], start=0, end=180, fill=(240, 240, 235))

    draw.text((30, 30), "DAMAGED: right ear cup cracked, piece missing", font=_font(18), fill=(150, 0, 0))
    img.save(OUT_DIR / "damaged_headphones.jpg", quality=90)


if __name__ == "__main__":
    make_intact_item()
    make_damaged_item()
    make_wrong_item()
    make_shipping_label_photo()
    make_damaged_shoes()
    make_damaged_clothes()
    make_damaged_headphones()
    print(f"Generated 7 test images in {OUT_DIR}")
