from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Disease:
    code: str
    name: str
    temp_min: float
    temp_max: float
    rh_min: float
    wetness_hours: float
    advice: str


@dataclass(frozen=True)
class CropProfile:
    code: str
    name: str
    kc: float  # FAO-56 crop coefficient
    humidity_min: float
    humidity_max: float
    ph_min: float
    ph_max: float
    disease: Disease


# "generic" uses the same default limits as the backend (humidity 40-80 %, pH 5.5-7.0).
CROPS: dict[str, CropProfile] = {
    "generic": CropProfile(
        "generic", "Cultivo genérico", 1.00, 40, 80, 5.5, 7.0,
        Disease("foliar_fungi", "Hongos foliares", 18, 28, 88, 8,
                "Revisa las hojas bajas y mejora la ventilación entre plantas."),
    ),
    "coffee": CropProfile(
        "coffee", "Café", 0.95, 50, 75, 5.0, 6.5,
        Disease("coffee_rust", "Roya del café (Hemileia vastatrix)", 21, 25, 85, 6,
                "Inspecciona el envés de las hojas buscando polvo naranja y considera un fungicida cúprico."),
    ),
    "plantain": CropProfile(
        "plantain", "Plátano", 1.10, 60, 85, 5.5, 7.0,
        Disease("black_sigatoka", "Sigatoka negra (Pseudocercospora fijiensis)", 25, 29, 90, 10,
                "Deshoja las hojas con rayas negras y mantén el drenaje del lote."),
    ),
    "potato": CropProfile(
        "potato", "Papa", 1.05, 60, 80, 5.0, 6.5,
        Disease("late_blight", "Gota o tizón tardío (Phytophthora infestans)", 10, 22, 90, 11,
                "Aplica fungicida preventivo y evita regar por aspersión en la tarde."),
    ),
    "corn": CropProfile(
        "corn", "Maíz", 1.15, 50, 80, 5.8, 7.0,
        Disease("leaf_blight", "Tizón foliar del maíz (Exserohilum turcicum)", 18, 27, 90, 6,
                "Vigila lesiones alargadas color gris en las hojas y rota el cultivo la próxima siembra."),
    ),
    "cocoa": CropProfile(
        "cocoa", "Cacao", 1.00, 55, 80, 5.0, 7.5,
        Disease("frosty_pod_rot", "Moniliasis (Moniliophthora roreri)", 22, 28, 85, 8,
                "Retira y entierra las mazorcas enfermas cada semana."),
    ),
}

CROP_CODES = list(CROPS)


def crop_index(code: str | None) -> int:
    return CROP_CODES.index(code) if code in CROPS else 0


def profile(code: str | None) -> CropProfile:
    return CROPS.get(code or "generic", CROPS["generic"])
