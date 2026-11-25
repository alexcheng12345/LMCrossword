from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
MD = ROOT / "Group_03_Complete_Project_Plan.md"
PDF = ROOT / "Group_03_Complete_Project_Plan.pdf"


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "title", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=19, leading=23, textColor=colors.HexColor("#18324A"),
)
heading_style = ParagraphStyle(
    "heading", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=13, leading=16, textColor=colors.HexColor("#18324A"),
    spaceBefore=8, spaceAfter=5,
)
subheading_style = ParagraphStyle(
    "subheading", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=10.5, leading=13, textColor=colors.HexColor("#315F7C"),
    spaceBefore=7, spaceAfter=3,
)
body_style = ParagraphStyle(
    "body", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.3, leading=10.5, textColor=colors.HexColor("#20262C"),
    spaceAfter=4,
)
code_style = ParagraphStyle(
    "code", parent=styles["Code"], fontName="Courier",
    fontSize=7, leading=8.5, backColor=colors.HexColor("#F2F4F6"),
    borderPadding=5, leftIndent=4, rightIndent=4, spaceAfter=5,
)


story = []
in_code = False
code_lines = []


def flush_code():
    if code_lines:
        story.append(Paragraph("<br/>".join(inline(line) for line in code_lines), code_style))
        code_lines.clear()


lines = MD.read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i]

    if line.startswith("```"):
        if in_code:
            flush_code()
            in_code = False
        else:
            in_code = True
        i += 1
        continue

    if in_code:
        code_lines.append(line)
        i += 1
        continue

    if not line.strip():
        i += 1
        continue

    if line.startswith("# "):
        story.append(Paragraph(inline(line[2:].strip()), title_style))
    elif line.startswith("## "):
        story.append(Paragraph(inline(line[3:].strip()), heading_style))
    elif line.startswith("### "):
        story.append(Paragraph(inline(line[4:].strip()), subheading_style))
    elif line.startswith("---"):
        story.append(Spacer(1, 4))
    elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| :]+\|$", lines[i + 1]):
        headers = row(line)
        i += 2
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(row(lines[i]))
            i += 1
        data = [[Paragraph(inline(cell), body_style) for cell in headers]]
        data += [[Paragraph(inline(cell), body_style) for cell in values] for values in rows]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315F7C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AEB8BF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F4F7F9")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        story.append(Spacer(1, 5))
        continue
    elif re.match(r"^\d+\.\s+", line):
        story.append(Paragraph(inline(line), body_style))
    elif line.startswith("- "):
        story.append(Paragraph("• " + inline(line[2:].strip()), body_style))
    else:
        story.append(Paragraph(inline(line), body_style))
    i += 1

flush_code()

doc = SimpleDocTemplate(
    str(PDF), pagesize=A4,
    leftMargin=16 * mm, rightMargin=16 * mm,
    topMargin=14 * mm, bottomMargin=14 * mm,
    title="Improving Length-Constrained Crossword Solving by LLMs",
)
doc.build(story)
print(PDF)
