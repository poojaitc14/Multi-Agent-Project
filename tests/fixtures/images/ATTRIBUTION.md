# Third-party image attribution

`roboflow_damaged_box_1.jpg`, `roboflow_damaged_box_2.jpg`,
`roboflow_damaged_food_box_1.jpg`, `roboflow_damaged_food_box_2.jpg`,
`roboflow_intact_food_box_1.jpg`, `roboflow_intact_food_box_2.jpg`:

A curated 6-image sample from **Damaged Package Detection**, provided by a
Roboflow user via [Roboflow Universe](https://universe.roboflow.com/iot-project/damaged-package-detection),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Used here as real (not synthetic) test fixtures for the Image Parsing
Agent's "Damaged in Transit" claim category.

All other images in this directory are synthetic (PIL-generated, see
`tests/fixtures/generate_test_images.py`) or pulled directly from
DummyJSON's public product catalog (no attribution required, public demo
API this project already depends on).
