from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "memoire" / "memoire.md"
ANNEX_SOURCE = ROOT / "memoire" / "annexe_dataset_et_protocole.md"
OUTPUT = ROOT / "livrables" / "memoire_sujet9_bases_vectorielles.pdf"


def register_fonts() -> tuple[str, str, str]:
    """Register Times New Roman on Windows, with a safe fallback."""

    candidates = [
        (
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbd.ttf"),
            Path("C:/Windows/Fonts/timesi.ttf"),
            "TimesNewRoman",
        ),
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
            "Arial",
        ),
    ]
    for regular, bold, italic, name in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont(name, regular))
            pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold))
            pdfmetrics.registerFont(TTFont(f"{name}-Italic", italic))
            return name, f"{name}-Bold", f"{name}-Italic"
    return "Times-Roman", "Times-Bold", "Times-Italic"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=23,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
            spaceAfter=24,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#172033"),
            spaceBefore=12,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=14.5,
            leading=18,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=7,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "Heading3Custom",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=12.2,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=5,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=3,
        ),
        "body_left": ParagraphStyle(
            "BodyLeftCustom",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=12,
            leading=18,
            alignment=TA_LEFT,
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=11.5,
            leading=17.25,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName=FONT_ITALIC,
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.3,
            leading=10.4,
            leftIndent=7,
            rightIndent=7,
            spaceBefore=4,
            spaceAfter=7,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#e2e8f0"),
            borderWidth=0.5,
            borderPadding=5,
        ),
    }
    return styles


STYLES = make_styles()


def inline_markup(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    return text


def wrap_code(lines: Iterable[str], width: int = 92) -> str:
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
            continue
        chunks = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped.extend(chunks or [""])
    return "\n".join(wrapped)


def image_flowable(markdown_image: str, markdown_dir: Path) -> KeepTogether:
    match = re.match(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)", markdown_image)
    if not match:
        raise ValueError(f"Image Markdown invalide: {markdown_image}")

    alt = match.group("alt").strip()
    raw_path = match.group("path").strip()
    image_path = (markdown_dir / raw_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Figure introuvable: {image_path}")

    with PILImage.open(image_path) as img:
        width_px, height_px = img.size

    max_width = 14.5 * cm
    max_height = 6.5 * cm
    ratio = min(max_width / width_px, max_height / height_px)
    width = width_px * ratio
    height = height_px * ratio

    flowables = [
        Spacer(1, 4),
        Image(str(image_path), width=width, height=height, hAlign="CENTER"),
        Paragraph(inline_markup(alt), STYLES["caption"]),
    ]
    return KeepTogether(flowables)


def flush_paragraph(buffer: list[str], story: list, style_name: str = "body") -> None:
    if not buffer:
        return
    paragraph = " ".join(part.strip() for part in buffer).strip()
    if paragraph:
        story.append(Paragraph(inline_markup(paragraph), STYLES[style_name]))
    buffer.clear()


def build_story(markdown: str, markdown_dir: Path) -> list:
    story: list = []
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False
    skip_first_h1 = True
    paragraph_style = "body"

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(
                    Preformatted(
                        escape(wrap_code(code_buffer)),
                        STYLES["code"],
                        dedent=0,
                    )
                )
                code_buffer.clear()
                in_code = False
            else:
                flush_paragraph(paragraph_buffer, story, paragraph_style)
                in_code = True
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if not line.strip():
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            continue

        if line.startswith("!["):
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            story.append(image_flowable(line, markdown_dir))
            continue

        if line.startswith("# "):
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            if skip_first_h1:
                skip_first_h1 = False
                continue
            story.append(Paragraph(inline_markup(line[2:].strip()), STYLES["h1"]))
            continue

        if line.startswith("## "):
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            title = line[3:].strip()
            if title == "Bibliographie":
                story.append(PageBreak())
            paragraph_style = "body_left" if title in {"Bibliographie", "Outils utilisés"} else "body"
            story.append(Paragraph(inline_markup(title), STYLES["h2"]))
            continue

        if line.startswith("### "):
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            story.append(Paragraph(inline_markup(line[4:].strip()), STYLES["h3"]))
            continue

        if line.startswith("- "):
            flush_paragraph(paragraph_buffer, story, paragraph_style)
            story.append(
                Paragraph(
                    inline_markup(line[2:].strip()),
                    STYLES["bullet"],
                    bulletText="•",
                )
            )
            continue

        paragraph_buffer.append(line)

    flush_paragraph(paragraph_buffer, story, paragraph_style)
    return story


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 9)
    canvas.setFillColor(colors.HexColor("#64748b"))
    footer = f"Sujet 9 - Bases vectorielles et recherche sémantique | page {doc.page}"
    canvas.drawCentredString(A4[0] / 2, 1.25 * cm, footer)
    canvas.restoreState()


def make_cover(markdown: str) -> list:
    title = markdown.splitlines()[0].lstrip("# ").strip()
    return [
        Spacer(1, 3.0 * cm),
        Paragraph(inline_markup(title), STYLES["cover_title"]),
        Spacer(1, 0.8 * cm),
        Paragraph("Mini-mémoire - Sujet 9", STYLES["cover_meta"]),
        Paragraph("Bases vectorielles, embeddings, BM25 et HNSW", STYLES["cover_meta"]),
        Paragraph("M1 Algorithmique et ingénierie de données", STYLES["cover_meta"]),
        Spacer(1, 0.55 * cm),
        Paragraph("Nemo MULLER et Yoan VIOLLET", STYLES["cover_meta"]),
        Paragraph("Enseignant : Jean Delpech", STYLES["cover_meta"]),
        Paragraph("Année universitaire 2025-2026", STYLES["cover_meta"]),
        Spacer(1, 1.0 * cm),
        Paragraph(
            "Projet reproductible : corpus synthétique de 2 000 résumés, "
            "20 requêtes évaluées, démo Streamlit et index Faiss HNSW.",
            STYLES["cover_meta"],
        ),
        PageBreak(),
    ]


def build_pdf() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    annex_markdown = ANNEX_SOURCE.read_text(encoding="utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title="Mémoire sujet 9 - Bases vectorielles et recherche sémantique",
        author="Nemo MULLER et Yoan VIOLLET",
    )

    story = make_cover(markdown)
    story.extend(build_story(markdown, SOURCE.parent))
    story.append(PageBreak())
    annex_title, annex_body = annex_markdown.split("\n", maxsplit=1)
    story.append(Paragraph(inline_markup(annex_title.lstrip("# ").strip()), STYLES["h1"]))
    story.extend(build_story(annex_body, ANNEX_SOURCE.parent))
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
