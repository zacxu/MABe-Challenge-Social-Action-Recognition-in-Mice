"""Build the phase 6--13 manuscript PDF from its checked-in Markdown.

Run from the project root with:
    .venv\\Scripts\\python.exe report\\build_phase6_13_pdf.py

ReportLab supplies the text/table layout. PyMuPDF places the original vector
figure and equation PDFs in the reserved areas without rasterizing them.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "figures" / "phase6_13" / "_mplcache"))
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stix"})
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
import pymupdf
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BalancedColumns,
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPORT = Path(__file__).resolve().parent
SOURCE = REPORT / "phase9_13_research_section.md"
OUTPUT = REPORT / "phase6_13_research_section.pdf"
BUILD = REPORT / "build" / "phase6_13"
WIDTH, HEIGHT = 612.0, 792.0
BODY_WIDTH = 504.0
COLUMN_WIDTH = (BODY_WIDTH - 27.0) / 2.0


def register_fonts() -> None:
    fonts = Path("C:/Windows/Fonts")
    for name, filename in (
        ("TNR", "times.ttf"),
        ("TNR-Bold", "timesbd.ttf"),
        ("TNR-Italic", "timesi.ttf"),
        ("TNR-BoldItalic", "timesbi.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic",
        boldItalic="TNR-BoldItalic",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    common = dict(fontName="TNR", fontSize=10, leading=12, textColor=colors.black)
    body = ParagraphStyle(
        "Body", alignment=TA_JUSTIFY, spaceAfter=3, splitLongWords=True,
        allowWidows=0, allowOrphans=0, **common,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "Title", parent=body, fontName="TNR-Bold", fontSize=14.5,
            leading=17, alignment=TA_CENTER, spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "Section", parent=body, fontName="TNR-Bold", fontSize=12,
            leading=14, alignment=TA_CENTER, spaceBefore=9, spaceAfter=7,
            keepWithNext=False,
        ),
        "subsection": ParagraphStyle(
            "Subsection", parent=body, fontName="TNR-Bold", fontSize=11,
            leading=13, alignment=TA_LEFT, spaceBefore=7, spaceAfter=4,
            keepWithNext=True,
        ),
        "subsubsection": ParagraphStyle(
            "Subsubsection", parent=body, fontName="TNR-Bold", fontSize=10,
            leading=12, alignment=TA_LEFT, spaceBefore=5, spaceAfter=3,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=body, fontSize=9, leading=11, alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontSize=9, leading=10.5, alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=body, fontName="TNR-Bold", fontSize=9,
            leading=10.5, alignment=TA_LEFT, spaceAfter=0,
        ),
        "reference": ParagraphStyle(
            "Reference", parent=body, fontSize=9, leading=10, alignment=TA_LEFT,
            firstLineIndent=-10, leftIndent=10, spaceAfter=3,
        ),
    }


def inline_markup(text: str, *, print_urls: bool = False) -> str:
    # The author instructions disallow active PDF links: URLs remain plain text.
    def link(match: re.Match[str]) -> str:
        return match.group(2) if print_urls else match.group(1)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<i>\1</i>", text)
    return text


class UnbrokenParagraph(Flowable):
    """Keep an individual bibliography entry within one column."""

    def __init__(self, paragraph: Paragraph):
        super().__init__()
        self.paragraph = paragraph
        self.spaceAfter = paragraph.getSpaceAfter()

    def wrap(self, available_width: float, available_height: float):
        self.width, self.height = self.paragraph.wrap(available_width, available_height)
        return self.width, self.height

    def draw(self):
        self.paragraph.drawOn(self.canv, 0, 0)


class VectorAsset(Flowable):
    """Reserve a figure rectangle and record its physical PDF coordinates."""

    def __init__(self, path: Path, placements: list[dict], *, max_width: float,
                 max_height: float = 540, label: str = "figure"):
        super().__init__()
        with pymupdf.open(path) as source_pdf:
            rect = source_pdf[0].rect
        scale = min(max_width / rect.width, max_height / rect.height)
        self.width = rect.width * scale
        self.height = rect.height * scale
        self.hAlign = "CENTER"
        self.path = path
        self.placements = placements
        self.label = label

    def wrap(self, available_width: float, available_height: float):
        return self.width, self.height

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.placements.append({
            "page": self.canv.getPageNumber() - 1,
            "x": x, "y": y, "width": self.width, "height": self.height,
            "path": str(self.path), "label": self.label,
        })


class ManuscriptCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        kwargs.update(initialFontName="TNR", initialFontSize=10,
                      initialLeading=12, pdfVersion=(1, 4))
        super().__init__(*args, **kwargs)


def parse_blocks(markdown: str) -> list[tuple[str, object]]:
    lines = markdown.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "$$":
            equation = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                equation.append(lines[index].strip())
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed display equation")
            blocks.append(("equation", " ".join(equation)))
            index += 1
        elif line.startswith("# "):
            blocks.append(("section", line[2:]))
            index += 1
        elif line.startswith("## "):
            blocks.append(("subsection", line[3:]))
            index += 1
        elif line.startswith("### "):
            blocks.append(("subsubsection", line[4:]))
            index += 1
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Unsupported figure markup: {line}")
            blocks.append(("image", match.group(1)))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            blocks.append(("table", rows))
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                paragraph.append(lines[index].strip())
                index += 1
            blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


def make_table(rows: list[list[str]], number: int, styles) -> Table:
    fractions = {
        1: (0.245, 0.115, 0.640),
        2: (0.235, 0.370, 0.395),
        3: (0.165, 0.495, 0.170, 0.170),
        4: (0.365, 0.155, 0.205, 0.275),
        5: (0.590, 0.410),
    }[number]
    if len(fractions) != len(rows[0]):
        raise ValueError(f"Table {number}: unexpected column count {len(rows[0])}")
    contents = [
        [Paragraph(inline_markup(cell), styles["cell_header" if index == 0 else "cell"])
         for cell in row]
        for index, row in enumerate(rows)
    ]
    table = Table(contents, colWidths=[BODY_WIDTH * x for x in fractions],
                  hAlign="CENTER", repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "TNR"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
    ]))
    return table


def make_story(blocks, styles, placements):
    story = []
    column_content = []
    in_references = False
    equation_count = 0

    def flush_columns():
        nonlocal column_content
        if column_content:
            story.append(BalancedColumns(
                column_content, nCols=2, needed=48, leftPadding=0,
                innerPadding=27, rightPadding=0, topPadding=0, bottomPadding=0,
                spaceAfter=5,
            ))
            column_content = []

    index = 0
    while index < len(blocks):
        kind, value = blocks[index]
        if index == 0:
            story.append(Paragraph(inline_markup(value.strip("*")), styles["title"]))
        elif kind == "section":
            flush_columns()
            story.append(CondPageBreak(72))
            story.append(Paragraph(inline_markup(value), styles["section"]))
        elif kind == "subsection":
            if value == "References":
                flush_columns()
                in_references = True
                story.append(CondPageBreak(72))
                story.append(Paragraph(value, styles["section"]))
            else:
                column_content.append(Paragraph(inline_markup(value), styles["subsection"]))
        elif kind == "subsubsection":
            column_content.append(Paragraph(inline_markup(value), styles["subsubsection"]))
        elif kind == "image":
            flush_columns()
            source = (REPORT / value).with_suffix(".pdf")
            caption_kind, caption = blocks[index + 1]
            if caption_kind != "paragraph" or not caption.startswith("**Figure "):
                raise ValueError("Every figure must be followed by a caption")
            figure_number = re.search(r"Figure (\d+)", caption).group(1)
            story.append(KeepTogether([
                Spacer(1, 5),
                VectorAsset(source, placements, max_width=BODY_WIDTH - 10,
                            max_height=530, label=f"Figure {figure_number}"),
                Spacer(1, 9),
                Paragraph(inline_markup(caption), styles["caption"]),
                Spacer(1, 12),
            ]))
            index += 1
        elif kind == "equation":
            flush_columns()
            equation_group = []
            while index < len(blocks) and blocks[index][0] == "equation":
                equation_count += 1
                equation_path = BUILD / f"equation{equation_count}.pdf"
                expression = blocks[index][1].rstrip(".")
                math_to_image(f"${expression}$", str(equation_path),
                              prop=FontProperties(size=11), format="pdf", color="black")
                with pymupdf.open(equation_path) as equation_pdf:
                    actual_width = equation_pdf[0].rect.width
                equation_group.extend([
                    Spacer(1, 12),
                    VectorAsset(equation_path, placements,
                                max_width=min(actual_width, BODY_WIDTH - 16),
                                max_height=38, label=f"Equation {equation_count}"),
                    Spacer(1, 7),
                ])
                index += 1
            index -= 1
            story.append(KeepTogether(equation_group))
        elif kind == "paragraph" and value.startswith("**Table "):
            flush_columns()
            number = int(re.search(r"Table (\d+)", value).group(1))
            table_kind, rows = blocks[index + 1]
            if table_kind != "table":
                raise ValueError("Every table caption must precede a table")
            story.append(KeepTogether([
                Paragraph(inline_markup(value), styles["caption"]),
                Spacer(1, 5), make_table(rows, number, styles), Spacer(1, 12),
            ]))
            index += 1
        elif kind == "paragraph":
            style = styles["reference" if in_references else "body"]
            paragraph = Paragraph(
                inline_markup(value, print_urls=in_references), style,
            )
            column_content.append(UnbrokenParagraph(paragraph) if in_references else paragraph)
        else:
            raise ValueError(f"Unconsumed block {kind}: {value!r}")
        index += 1
    flush_columns()
    return story


def build_pdf() -> list[dict]:
    BUILD.mkdir(parents=True, exist_ok=True)
    register_fonts()
    styles = make_styles()
    placements = []
    blocks = parse_blocks(SOURCE.read_text(encoding="utf-8"))
    document = BaseDocTemplate(
        str(BUILD / "text_layout.pdf"), pagesize=(WIDTH, HEIGHT),
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=90,
        title="Pose-Based Mouse Behavior Recognition: Phases 6–13",
        author="", subject="Methodology, experimental setup, and results",
        creator="Reproducible manuscript builder",
    )
    document.addPageTemplates([
        PageTemplate(
            id="First", frames=[Frame(54, 90, BODY_WIDTH, HEIGHT - 180, id="first_full",
                                      leftPadding=0, rightPadding=0,
                                      topPadding=0, bottomPadding=0)],
            autoNextPageTemplate="Body",
        ),
        PageTemplate(
            id="Body", frames=[Frame(54, 90, BODY_WIDTH, HEIGHT - 144, id="body_full",
                                     leftPadding=0, rightPadding=0,
                                     topPadding=0, bottomPadding=0)],
        ),
    ])
    document.build(make_story(blocks, styles, placements), canvasmaker=ManuscriptCanvas)
    with pymupdf.open(BUILD / "text_layout.pdf") as final:
        for placement in placements:
            page = final[placement["page"]]
            rect = pymupdf.Rect(
                placement["x"], HEIGHT - placement["y"] - placement["height"],
                placement["x"] + placement["width"], HEIGHT - placement["y"],
            )
            with pymupdf.open(placement["path"]) as asset:
                page.show_pdf_page(rect, asset, 0, keep_proportion=True)
        final.save(OUTPUT, garbage=4, deflate=True)
    (BUILD / "vector_placements.json").write_text(
        json.dumps(placements, indent=2), encoding="utf-8",
    )
    return placements


def validate_and_preview(placements: list[dict]) -> dict:
    markdown = SOURCE.read_text(encoding="utf-8")
    table_numbers = [int(number) for number in re.findall(r"^\*\*Table (\d+)\.", markdown, re.MULTILINE)]
    figure_numbers = [int(number) for number in re.findall(r"^\*\*Figure (\d+)\.", markdown, re.MULTILINE)]
    if table_numbers != list(range(1, len(table_numbers) + 1)):
        raise ValueError("Tables must be numbered consecutively in source order")
    if figure_numbers != list(range(1, len(figure_numbers) + 1)):
        raise ValueError("Figures must be numbered consecutively in source order")
    with pymupdf.open(OUTPUT) as doc:
        text = "\n".join(page.get_text(sort=True) for page in doc)
        expected = (
            "1. Methodology", "2. Experimental Setup",
            "3. Experimental Results and Discussion", "References",
            "0.470192", "0.376471", "1,915,749", "9,026.91",
            "Phase 6", "Phase 7", "Phase 8", "Phase 9", "Phase 10",
            "Phase 11", "Phase 12", "Phase 13",
            "Training Objective", "1e-6", "PyTorch",
        )
        missing = [item for item in expected if item not in text]
        figures = [f"Figure {number}." for number in figure_numbers]
        tables = [f"Table {number}." for number in table_numbers]
        missing += [item for item in figures + tables if item not in text]
        fonts = {}
        bounds_issues = []
        for page_number, page in enumerate(doc):
            for font in page.get_fonts(full=True):
                xref, extension, kind, name, encoding, *_ = font
                key = f"{name}/{kind}"
                fonts[key] = {"name": name, "type": kind,
                              "embedded": bool(doc.extract_font(xref)[3]),
                              "encoding": encoding}
            top = 90 if page_number == 0 else 54
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        # Glyph ascenders/descenders can extend a little beyond
                        # ReportLab's nominal line box; allow two points.
                        if x0 < 52 or x1 > 560 or y0 < top - 2 or y1 > 704:
                            bounds_issues.append({"page": page_number + 1,
                                                  "text": span["text"],
                                                  "bbox": span["bbox"]})
        report = {
            "pdf": str(OUTPUT.relative_to(REPORT.parent)),
            "source": str(SOURCE.relative_to(REPORT.parent)),
            "pages": len(doc), "file_bytes": OUTPUT.stat().st_size,
            "page_size": "US Letter (612 x 792 pt)",
            "two_column_body": True, "body_font_pt": 10, "body_leading_pt": 12,
            "main_sections": list(expected[:3]),
            "figure_count": len([p for p in placements if p["label"].startswith("Figure")]),
            "table_count": len(table_numbers),
            "display_equation_count": len([p for p in placements if p["label"].startswith("Equation")]),
            "all_vector_assets": True, "active_link_count": sum(len(p.get_links()) for p in doc),
            "bookmark_count": len(doc.get_toc()),
            "fonts": list(fonts.values()),
            "all_fonts_embedded": all(f["embedded"] for f in fonts.values()),
            "type3_font_count": sum(f["type"] == "Type3" for f in fonts.values()),
            "missing_required_text": missing, "margin_issues": bounds_issues,
            "replacement_glyph_count": text.count("\ufffd"),
            "formal_submission_template": False,
            "note": "Journal-style section draft with AAAI-like text/layout; no fabricated author block, abstract, copyright, or conference submission claim.",
        }
        thumb_width, thumb_height = 306, 396
        columns = 3
        sheet = Image.new("RGB", (columns * (thumb_width + 16),
                                   ((len(doc) + columns - 1) // columns) * (thumb_height + 34)),
                          "#d9dde1")
        drawing = ImageDraw.Draw(sheet)
        for number, page in enumerate(doc):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)
            page_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            page_image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            x = (number % columns) * (thumb_width + 16) + 8
            y = (number // columns) * (thumb_height + 34) + 22
            sheet.paste(page_image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
            drawing.text((x, y - 17), f"Page {number + 1}", fill="black")
            preview_path = BUILD / f"page_{number + 1:02}.png"
            page_image.save(preview_path)
        sheet.save(BUILD / "contact_sheet.png")
        (BUILD / "extracted_text.txt").write_text(text, encoding="utf-8")
    (REPORT / "phase6_13_pdf_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    if missing or bounds_issues or not report["all_fonts_embedded"] or report["type3_font_count"]:
        raise RuntimeError(f"PDF validation needs attention: {json.dumps(report, ensure_ascii=False)}")
    return report


if __name__ == "__main__":
    result = validate_and_preview(build_pdf())
    print(json.dumps({key: result[key] for key in (
        "pdf", "pages", "file_bytes", "figure_count", "table_count",
        "display_equation_count", "all_fonts_embedded", "margin_issues",
    )}, indent=2))
