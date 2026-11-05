"""Transcription vocale (optionnelle) avec faster-whisper.

Installation : pip install faster-whisper   (modèle téléchargé au premier appel)
Limite connue : la Darija est mal couverte par Whisper -> mesurer le WER (voir README).
"""
from __future__ import annotations

import os

from . import config

# Vocabulaire du domaine : biaise Whisper vers les noms de médicaments usuels.
DOMAIN_PROMPT = ("صيدلية، دوليبران، أوغمنتين، سباسفون، سميكطا، فولتارين، فنتولين، "
                 "pharmacie, Doliprane, Augmentin, Spasfon, Smecta, Voltarène, Ventoline")


class ASRUnavailable(RuntimeError):
    pass


_model = None


def transcribe(path: str, language: str | None = os.getenv("ASR_LANGUAGE") or None):
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ASRUnavailable("faster-whisper n'est pas installé : pip install faster-whisper") from e
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL, compute_type="int8")
    segments, info = _model.transcribe(path, language=language, beam_size=5, vad_filter=True,
                                       initial_prompt=DOMAIN_PROMPT)
    return " ".join(s.text.strip() for s in segments).strip(), info.language
