"""
InstaFlow - Analizador de personalidad Big Five (OCEAN)
=======================================================
Evalúa los cinco rasgos del modelo Big Five (Openness, Conscientiousness,
Extraversion, Agreeableness, Neuroticism) a partir de los textos públicos de un
perfil scrapeado (biografía, descripciones/captions de publicaciones, etc.)
usando la API REST real de Google Gemini.

La API key se lee SIEMPRE desde ``env/api_key.txt``; nunca se escribe en el código.
"""

import json
import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger("instaflow.bigfive_analyzer")

BASE_DIR = Path(__file__).resolve().parent
API_KEY_FILE = BASE_DIR / "env" / "api_key.txt"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Modelos a intentar en orden. "gemini-pro" es el endpoint original pedido; los
# modelos más recientes van primero porque el antiguo puede estar retirado.
MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-pro"]

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

MAX_POSTS = 60          # máximo de publicaciones incluidas en el prompt
MAX_CAPTION_CHARS = 600  # recorte por caption
REQUEST_TIMEOUT = 60


class BigFiveError(Exception):
    """Error descriptivo del análisis Big Five (configuración, API o respuesta)."""


class BigFiveAnalyzer:
    """Analiza un perfil de Instagram con Google Gemini y devuelve puntajes OCEAN 0-100."""

    def __init__(self, api_key_path: Path | str | None = None, models: list[str] | None = None):
        self.api_key_path = Path(api_key_path) if api_key_path else API_KEY_FILE
        self.models = models or list(MODEL_CANDIDATES)
        self.api_key = self._load_api_key()

    # ------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------
    def _load_api_key(self) -> str:
        """Lee la API key desde env/api_key.txt (cadena vacía si no existe)."""
        try:
            if self.api_key_path.is_file():
                return self.api_key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("No se pudo leer %s: %s", self.api_key_path, exc)
        return ""

    # ------------------------------------------------------------------
    # Construcción del prompt
    # ------------------------------------------------------------------
    @staticmethod
    def _split_input(data: dict) -> tuple[dict, list[dict]]:
        """
        Acepta el archivo completo de resultados ({"profile": {...}, "posts": [...]})
        o directamente el dict del perfil con una clave opcional "posts".
        """
        if not isinstance(data, dict):
            raise BigFiveError("El perfil debe ser un objeto JSON.")
        inner = data.get("profile")
        profile = inner if isinstance(inner, dict) else data
        posts = data.get("posts")
        if posts is None:
            posts = profile.get("posts")
        return profile, [p for p in (posts or []) if isinstance(p, dict)]

    def build_prompt(self, data: dict) -> str:
        """Construye el prompt con biografía, captions y demás texto disponible."""
        profile, posts = self._split_input(data)

        lines = ["DATOS DEL PERFIL"]
        for label, key in (
            ("Usuario", "username"),
            ("Nombre", "full_name"),
            ("Biografía", "biography"),
            ("Descripción", "description"),
            ("Categoría", "category"),
            ("Enlace externo", "external_url"),
        ):
            value = profile.get(key)
            if value:
                lines.append(f"- {label}: {value}")
        for label, key in (
            ("Seguidores", "followers_count"),
            ("Seguidos", "following_count"),
            ("Publicaciones", "posts_count"),
        ):
            if profile.get(key) is not None:
                lines.append(f"- {label}: {profile[key]}")

        texts = []
        for i, post in enumerate(posts[:MAX_POSTS], 1):
            caption = (post.get("caption") or "").strip()
            extra = []
            for key in ("text", "description", "alt_text"):
                if post.get(key) and post.get(key) != caption:
                    extra.append(str(post[key]).strip())
            comments = post.get("comments")
            if isinstance(comments, list):
                for c in comments[:10]:
                    txt = c.get("text") if isinstance(c, dict) else c
                    if txt:
                        extra.append(f"comentario: {str(txt).strip()}")
            body = " | ".join(t for t in [caption, *extra] if t)[:MAX_CAPTION_CHARS]
            if not body:
                continue
            kind = post.get("media_type") or post.get("post_type") or "publicación"
            likes = post.get("like_count", post.get("likes"))
            n_comments = post.get("comment_count", post.get("comments_count"))
            stats = f" (likes: {likes}, comentarios: {n_comments})" if likes is not None or n_comments is not None else ""
            texts.append(f"{i}. [{kind}]{stats} {body}")

        lines.append("")
        lines.append(f"TEXTOS DE PUBLICACIONES ({len(texts)} con texto de {len(posts)} analizadas)")
        lines.extend(texts or ["(Ninguna publicación contiene texto)"])
        corpus = "\n".join(lines)

        return (
            "Eres un psicólogo especialista en psicometría y en el modelo de personalidad Big Five (OCEAN). "
            "Analiza el siguiente contenido público de un perfil de Instagram y estima los cinco rasgos "
            "en una escala de 0 a 100 (0 = muy bajo, 100 = muy alto): Openness (apertura a la experiencia), "
            "Conscientiousness (responsabilidad), Extraversion (extraversión), Agreeableness (amabilidad) "
            "y Neuroticism (neuroticismo). Si hay poco texto, estima con prudencia (valores cercanos a 50).\n\n"
            f"{corpus}\n\n"
            "Responde ÚNICAMENTE con un objeto JSON válido, sin markdown ni texto adicional, con exactamente "
            'estas claves numéricas: {"openness": 0, "conscientiousness": 0, "extraversion": 0, '
            '"agreeableness": 0, "neuroticism": 0}'
        )

    # ------------------------------------------------------------------
    # Llamada a Gemini y parseo
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_scores(raw_text: str) -> dict:
        """Extrae el JSON de la respuesta del modelo y normaliza los 5 puntajes a 0-100."""
        text = (raw_text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise BigFiveError("La respuesta del modelo no contiene un objeto JSON.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise BigFiveError(f"JSON inválido en la respuesta del modelo: {exc}") from exc

        scores = {}
        for trait in TRAITS:
            value = data.get(trait)
            if value is None:  # tolerar mayúsculas o nombres cortos
                value = next((v for k, v in data.items() if str(k).lower() == trait), None)
            try:
                scores[trait] = round(max(0.0, min(100.0, float(value))), 1)
            except (TypeError, ValueError):
                raise BigFiveError(f"Falta o es inválido el puntaje de '{trait}' en la respuesta del modelo.")
        return scores

    def analyze_profile(self, profile: dict) -> dict:
        """
        Analiza el perfil (o el archivo de resultados completo) con Gemini.

        Devuelve {"openness", "conscientiousness", "extraversion", "agreeableness",
        "neuroticism"} con valores 0-100. Lanza BigFiveError con un mensaje descriptivo
        si la API key falta o la API falla.
        """
        if not self.api_key:
            raise BigFiveError(f"No hay API key configurada. Escríbala en {self.api_key_path}.")

        payload = {
            "contents": [{"parts": [{"text": self.build_prompt(profile)}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        errors = []
        for model in self.models:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent"
            try:
                logger.info("Solicitando análisis Big Five a Gemini (%s)...", model)
                resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                errors.append(f"{model}: error de red ({exc.__class__.__name__})")
                continue

            if resp.status_code != 200:
                try:
                    detail = resp.json().get("error", {}).get("message", "")
                except ValueError:
                    detail = resp.text[:200]
                errors.append(f"{model}: HTTP {resp.status_code} {detail}".strip())
                if resp.status_code in (400, 401, 403) and "API key" in detail:
                    break  # la clave es inválida: probar otro modelo no ayuda
                continue

            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                raw_text = "".join(p.get("text", "") for p in parts)
            except (KeyError, IndexError, TypeError, ValueError):
                errors.append(f"{model}: respuesta sin contenido (¿bloqueada por filtros de seguridad?)")
                continue
            try:
                return self._parse_scores(raw_text)
            except BigFiveError as exc:
                errors.append(f"{model}: {exc}")

        raise BigFiveError("No se pudo completar el análisis con Gemini. " + " | ".join(errors))
