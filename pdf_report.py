"""PDF reports for SpecWorth (ReportLab).

The rupee sign is not in the standard PDF fonts, so prices are written as 'Rs.'.
Author: Rishi Mishra, BPIT Rohini.
"""
import io
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INDIGO = colors.HexColor("#4f46e5")
DARK = colors.HexColor("#0f172a")
GREY = colors.HexColor("#64748b")
LIGHT = colors.HexColor("#f1f5f9")


def rs(n):
    """Indian digit grouping with 'Rs.' prefix."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return "Rs. " + s
    head, tail, groups = s[:-3], s[-3:], []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "Rs. " + ",".join(groups) + "," + tail


def _styles():
    base = getSampleStyleSheet()
    return dict(
        title=ParagraphStyle("t", parent=base["Title"], fontSize=20, textColor=colors.white, alignment=0, leading=24),
        sub=ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=colors.HexColor("#c7d2fe")),
        h=ParagraphStyle("h", parent=base["Heading2"], fontSize=12, textColor=DARK, spaceBefore=10, spaceAfter=4),
        p=ParagraphStyle("p", parent=base["Normal"], fontSize=9.5, textColor=DARK, leading=13),
        small=ParagraphStyle("sm", parent=base["Normal"], fontSize=8, textColor=GREY, leading=11),
        big=ParagraphStyle("b", parent=base["Normal"], fontSize=26, textColor=INDIGO, leading=30, fontName="Helvetica-Bold"),
    )


def _header(st, title, subtitle):
    t = Table([[Paragraph(title, st["title"])], [Paragraph(subtitle, st["sub"])]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), DARK), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
    return t


def _table(rows, widths, header=True, align_right_from=1):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
             ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
             ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build_estimate_report(specs, est, low, high, segment, base, contrib, phones, error_note):
    """specs: list of (label, value) | contrib: {feature: rupees} | phones: list of dicts."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="SpecWorth price estimate", author="SpecWorth")
    e = [_header(st, "SpecWorth - Price Estimate", f"Generated {date.today():%d %b %Y} - Know what your phone is worth"),
         Spacer(1, 10), Paragraph("Estimated market price", st["h"]),
         Paragraph(rs(round(est, -2)), st["big"]),
         Paragraph(f"{segment} segment - 80% prediction interval {rs(round(low, -2))} to {rs(round(high, -2))}", st["p"]),
         Paragraph("Specifications entered", st["h"]),
         _table([["Specification", "Value"]] + [[a, str(b)] for a, b in specs], [110 * mm, 64 * mm])]
    if contrib:
        rows = [["Specification", "Effect on price"]] + [[k, ("+" if v >= 0 else "-") + rs(abs(v)).replace("Rs. ", "Rs. ")]
                                                          for k, v in contrib.items()]
        e += [Paragraph("What drives this estimate", st["h"]),
              Paragraph(f"A typical phone in the catalog is estimated at {rs(base)}. Each spec moves your estimate "
                        f"up or down from there (Shapley values).", st["p"]), Spacer(1, 3),
              _table(rows, [110 * mm, 64 * mm])]
    if phones:
        rows = [["Phone", "Price", "RAM / Storage", "Battery"]] + [
            [p["name"], rs(p["price"]), p["mem"], p["battery"]] for p in phones]
        e += [Paragraph("Phones in India around this price", st["h"]),
              _table(rows, [66 * mm, 32 * mm, 40 * mm, 36 * mm], align_right_from=1)]
    e += [Spacer(1, 12), Paragraph(error_note, st["small"]),
          Paragraph("Prices are approximate (Oct 2026) and vary by variant and seller. Verify before buying.", st["small"])]
    doc.build(e)
    return buf.getvalue()


def build_compare_report(table_rows, picks, best_lines, winner):
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="SpecWorth phone comparison", author="SpecWorth")
    w0 = 42 * mm
    wn = (174 * mm - w0) / len(picks)
    e = [_header(st, "SpecWorth - Phone Comparison", f"Generated {date.today():%d %b %Y} - Know what your phone is worth"), Spacer(1, 10),
         _table([["Spec"] + picks] + table_rows, [w0] + [wn] * len(picks)), Spacer(1, 10),
         Paragraph("Highlights", st["h"])]
    e += [Paragraph(line, st["p"]) for line in best_lines]
    e += [Spacer(1, 6), Paragraph(f"<b>Highest overall score:</b> {escape(str(winner))}", st["p"]), Spacer(1, 12),
          Paragraph("The score is spec-based (AnTuTu, RAM, storage, battery, charging, camera MP, display, price). "
                    "Megapixels do not measure camera quality - read reviews before buying. Prices are approximate.", st["small"])]
    doc.build(e)
    return buf.getvalue()
